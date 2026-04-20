"""Supabase async client — singleton with lazy init.

Design principles (mirrored from reference repo):
- Service-role key is used server-side to bypass RLS on all writes.
- All public functions degrade gracefully when Supabase is not configured.
- Joins use Supabase PostgREST syntax: table!inner(col).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

from supabase import AsyncClient, acreate_client

from src.infra.config import get_settings
from src.infra.logger import setup_logger

logger = setup_logger(__name__)

_client: AsyncClient | None = None
_lock = asyncio.Lock()


async def get_client() -> AsyncClient | None:
    """Return the shared Supabase async client, creating it on first call."""
    global _client
    settings = get_settings()
    if not settings.is_supabase_configured:
        return None
    if _client is not None:
        return _client
    async with _lock:
        if _client is None:
            _client = await acreate_client(
                settings.supabase_url,
                settings.supabase_service_role_key,
            )
    return _client


# ── User profile ──────────────────────────────────────────────────────────────

async def get_user_profile(account_id: str) -> dict[str, Any]:
    """Return the user's profile row, or {} if not found / unconfigured."""
    if not account_id:
        return {}
    client = await get_client()
    if client is None:
        return {}
    try:
        # limit(1) + list avoids 406 that .single() raises when no row exists
        result = (
            await client.table("user_profiles")
            .select("*")
            .eq("user_id", account_id)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        return rows[0] if rows else {}
    except Exception:
        logger.warning("Failed to fetch user profile for %s", account_id)
        return {}


async def upsert_user_profile(account_id: str, delta: dict[str, Any]) -> None:
    """Merge a partial profile update into the user_profiles table."""
    if not account_id:
        return
    client = await get_client()
    if client is None:
        return
    try:
        await client.table("user_profiles").upsert(
            {"user_id": account_id, **delta, "updated_at": datetime.now(timezone.utc).isoformat()},
            on_conflict="user_id",
        ).execute()
    except Exception:
        logger.exception("Failed to upsert user profile for %s", account_id)


# ── Conversations & messages ───────────────────────────────────────────────────

async def get_or_create_conversation(account_id: str, conversation_id: str) -> str:
    """Ensure the conversation row exists; return its id.

    Uses account_id to match the existing Supabase schema.
    """
    if not account_id:
        return conversation_id
    client = await get_client()
    if client is None:
        return conversation_id
    now = datetime.now(timezone.utc).isoformat()
    try:
        await client.table("conversations").upsert(
            {
                "id": conversation_id,
                "account_id": account_id,
                "last_activity": now,
            },
            on_conflict="id",
        ).execute()
    except Exception as exc:
        logger.warning("Failed to upsert conversation %s: %s", conversation_id, exc)
    return conversation_id


async def get_conversation_history(
    conversation_id: str, limit: int | None = None
) -> list[dict[str, Any]]:
    """Return the last `limit` messages for a conversation, oldest first."""
    if not conversation_id:
        return []
    client = await get_client()
    if client is None:
        return []
    lim = limit or get_settings().context_message_limit
    try:
        result = (
            await client.table("messages")
            .select("role, content, created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=True)
            .limit(lim)
            .execute()
        )
        rows = result.data or []
        rows.reverse()  # oldest first
        return rows
    except Exception:
        logger.exception("Failed to fetch history for conversation %s", conversation_id)
        return []


async def get_conversations_for_user(account_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Return recent conversations for a user (uses account_id column)."""
    if not account_id:
        return []
    client = await get_client()
    if client is None:
        return []
    try:
        result = (
            await client.table("conversations")
            .select("id, name, last_activity, created_at")
            .eq("account_id", account_id)
            .order("last_activity", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception:
        logger.exception("Failed to fetch conversations for %s", account_id)
        return []


async def persist_message(
    conversation_id: str, role: str, content: str, metadata: dict[str, Any] | None = None
) -> None:
    """Append a message to the messages table.

    Tries with metadata first; falls back without it if the column doesn't exist.
    """
    if not conversation_id:
        return
    client = await get_client()
    if client is None:
        return
    base_payload: dict[str, Any] = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }
    payloads = [{**base_payload, "metadata": metadata}, base_payload] if metadata else [base_payload]
    for payload in payloads:
        try:
            await client.table("messages").insert(payload).execute()
            return
        except Exception as exc:
            if "metadata" in str(exc) and len(payloads) > 1 and payload is not base_payload:
                continue  # retry without metadata (column missing in pre-existing table)
            logger.warning("Failed to persist message to conversation %s: %s", conversation_id, exc)
            return


async def trim_conversation_history(conversation_id: str, keep_last: int = 10) -> None:
    """Delete old messages from a conversation, keeping only the most recent N.

    Prevents stale tool-call exchanges from flooding the agent's context on
    subsequent queries within the same conversation thread.
    """
    if not conversation_id:
        return
    client = await get_client()
    if client is None:
        return
    try:
        # Get all message IDs ordered by creation time
        result = await (
            client.table("messages")
            .select("id")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        )
        rows = result.data or []
        if len(rows) <= keep_last:
            return
        # Delete everything except the last N
        ids_to_delete = [r["id"] for r in rows[:-keep_last]]
        await (
            client.table("messages")
            .delete()
            .in_("id", ids_to_delete)
            .execute()
        )
        logger.info("Trimmed %d old messages from conversation %s", len(ids_to_delete), conversation_id)
    except Exception as exc:
        logger.warning("Failed to trim conversation %s: %s", conversation_id, exc)


# ── Memory / semantic facts ────────────────────────────────────────────────────

async def get_recent_memories(account_id: str, days: int | None = None) -> list[str]:
    """Return extracted fact strings from the last `days` days."""
    if not account_id:
        return []
    client = await get_client()
    if client is None:
        return []
    lookback = days or get_settings().memory_lookback_days
    since = (datetime.now(timezone.utc) - timedelta(days=lookback)).isoformat()
    try:
        # Support both table names (our migration uses user_memories; some deployments use memories)
        for table in ("user_memories", "memories"):
            try:
                result = (
                    await client.table(table)
                    .select("fact")
                    .eq("user_id", account_id)
                    .gte("created_at", since)
                    .order("created_at", desc=True)
                    .limit(20)
                    .execute()
                )
                return [row["fact"] for row in (result.data or [])]
            except Exception as exc:
                if "schema cache" in str(exc):
                    continue
                raise
        return []
    except Exception:
        logger.warning("Failed to fetch memories for %s — returning empty", account_id)
        return []


async def upsert_memory_fact(account_id: str, fact: str) -> None:
    """Insert a new extracted fact into user_memories (or memories)."""
    if not account_id or not fact:
        return
    client = await get_client()
    if client is None:
        return
    for table in ("user_memories", "memories"):
        try:
            await client.table(table).insert(
                {"user_id": account_id, "fact": fact}
            ).execute()
            return
        except Exception as exc:
            if "schema cache" in str(exc):
                continue
            logger.warning("Failed to upsert memory fact for %s: %s", account_id, exc)
            return


# ── Domain data (used by agents) ──────────────────────────────────────────────

async def get_meal_items(
    account_id: str,
    date: str | None = None,
    meal_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return meal_items for a user, optionally filtered by date (YYYY-MM-DD) and meal_type."""
    client = await get_client()
    if client is None:
        return []
    try:
        query = (
            client.table("meal_items")
            .select("id, food_name, description, occasion, calories_kcal, protein_g, carbs_g, fat_g, portion, logged_at")
            .eq("account_id", account_id)
            .order("logged_at", desc=True)
            .limit(limit)
        )
        if date:
            query = query.gte("logged_at", f"{date}T00:00:00+00:00").lte("logged_at", f"{date}T23:59:59+00:00")
        if meal_type:
            query = query.eq("occasion", meal_type)
        result = await query.execute()
        return result.data or []
    except Exception:
        logger.exception("Failed to fetch meal items for %s", account_id)
        return []


async def log_meal(
    account_id: str,
    conversation_id: str,
    meal_data: dict[str, Any],
) -> dict[str, Any]:
    """Insert a meal record and return the created row.

    Maps tool-facing field names to the existing Supabase schema:
      name        → food_name
      calories    → calories_kcal
      meal_type   → occasion
    """
    client = await get_client()
    if client is None:
        return {"error": "database not configured"}
    # Exact meal_items schema: food_name, portion, calories_kcal, occasion,
    # portion_multiplier (NOT NULL), protein_g, carbs_g, fat_g, logged_at (NOT NULL)
    normalised: dict[str, Any] = {
        "account_id": account_id,
        "conversation_id": conversation_id,
        "food_name": meal_data.get("name") or meal_data.get("food_name") or meal_data.get("description", ""),
        "portion": meal_data.get("portion", "1 serving"),
        "portion_multiplier": meal_data.get("portion_multiplier", 1),
        "occasion": meal_data.get("meal_type") or meal_data.get("occasion", "snack"),
        "calories_kcal": meal_data.get("calories") or meal_data.get("calories_kcal") or 0,
        "protein_g": meal_data.get("protein_g", 0),
        "carbs_g": meal_data.get("carbs_g", 0),
        "fat_g": meal_data.get("fat_g", 0),
        "logged_at": meal_data.get("logged_at") or datetime.now(timezone.utc).isoformat(),
    }
    if meal_data.get("description"):
        normalised["description"] = meal_data["description"]
    try:
        result = (
            await client.table("meal_items")
            .insert(normalised)
            .execute()
        )
        return result.data[0] if result.data else {}
    except Exception as exc:
        logger.exception("Failed to log meal for %s", account_id)
        return {"error": str(exc)}


async def log_workout(
    account_id: str,
    conversation_id: str,
    workout_data: dict[str, Any],
) -> dict[str, Any]:
    """Insert a workout record and return the created row."""
    client = await get_client()
    if client is None:
        return {"error": "database not configured"}
    try:
        result = (
            await client.table("workouts")
            .insert({"account_id": account_id, "conversation_id": conversation_id, **workout_data})
            .execute()
        )
        return result.data[0] if result.data else {}
    except Exception as exc:
        logger.exception("Failed to log workout for %s", account_id)
        return {"error": str(exc)}


async def log_medical_condition(
    account_id: str,
    condition_data: dict[str, Any],
) -> dict[str, Any]:
    """Insert a medical condition row.

    Exact schema: condition (NOT NULL), severity (NOT NULL), notes (NOT NULL ''),
    diagnosed_at (NOT NULL default now()).
    """
    client = await get_client()
    if client is None:
        return {"error": "database not configured"}
    record: dict[str, Any] = {
        "account_id": account_id,
        "condition": condition_data.get("condition", condition_data.get("condition_name", "")),
        "severity": condition_data.get("severity", "moderate"),
        "notes": condition_data.get("notes", ""),
    }
    if condition_data.get("diagnosed_at"):
        record["diagnosed_at"] = condition_data["diagnosed_at"]
    try:
        result = await client.table("medical_conditions").insert(record).execute()
        return result.data[0] if result.data else {}
    except Exception as exc:
        logger.exception("Failed to log condition for %s", account_id)
        return {"error": str(exc)}


async def log_medication(
    account_id: str,
    medication_data: dict[str, Any],
) -> dict[str, Any]:
    """Insert a medication row.

    Exact schema: name (NOT NULL), dosage (NOT NULL), frequency (NOT NULL),
    notes (NOT NULL ''), started_at (NOT NULL default now()), active (NOT NULL true).
    """
    client = await get_client()
    if client is None:
        return {"error": "database not configured"}
    record: dict[str, Any] = {
        "account_id": account_id,
        "name": medication_data.get("name", medication_data.get("medication_name", "")),
        "dosage": medication_data.get("dosage", ""),
        "frequency": medication_data.get("frequency", ""),
        "notes": medication_data.get("notes", ""),
        "active": medication_data.get("active", True),
    }
    if medication_data.get("started_at") or medication_data.get("start_date"):
        record["started_at"] = medication_data.get("started_at") or medication_data.get("start_date")
    try:
        result = await client.table("medications").insert(record).execute()
        return result.data[0] if result.data else {}
    except Exception as exc:
        logger.exception("Failed to log medication for %s", account_id)
        return {"error": str(exc)}
