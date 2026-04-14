"""Context Hydration Node — fetches profile, history, memories in parallel.

Identical logic to the multi-agent version; only the state import differs.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from src.infra import db
from src.infra.logger import setup_logger
from src.infra.redis_client import cache_delete, cache_get, cache_set
from src.agent.state import GraphState

logger = setup_logger(__name__)


# ── Constraint rule derivation ─────────────────────────────────────────────────

_CONDITION_RULES: dict[str, list[str]] = {
    "type 2 diabetes": [
        "No recommendations involving refined sugar or high-GI foods.",
        "Keep carbohydrate suggestions moderate and specify slow-digesting options.",
    ],
    "diabetes": [
        "No recommendations involving refined sugar or high-GI foods.",
    ],
    "hypertension": [
        "Avoid high-sodium food suggestions.",
        "Recommend low-sodium alternatives where relevant.",
    ],
    "celiac": [
        "All food suggestions must be strictly gluten-free.",
    ],
    "lactose intolerance": [
        "Avoid dairy products unless lactose-free alternatives are specified.",
    ],
}

_MEDICATION_RULES: dict[str, list[str]] = {
    "metformin": [
        "Do not recommend alcohol — it can cause lactic acidosis with metformin.",
    ],
    "warfarin": [
        "Avoid recommending high-Vitamin K foods (spinach, kale, broccoli) without caveats.",
    ],
    "ssri": [
        "Avoid recommending St. John's Wort supplements.",
    ],
}

_ALLERGY_RULE = "ALLERGY — never suggest {allergen} or foods containing it."


def derive_constraint_rules(profile: dict[str, Any]) -> list[str]:
    """Convert profile facts into a deterministic set of safety constraint strings."""
    rules: list[str] = []

    conditions: list[str] = profile.get("conditions", []) or []
    if isinstance(conditions, str):
        conditions = [conditions]
    for condition in conditions:
        lc = condition.lower()
        for key, condition_rules in _CONDITION_RULES.items():
            if key in lc:
                rules.extend(condition_rules)
                break

    medications: list[str] = profile.get("medications", []) or []
    if isinstance(medications, str):
        medications = [medications]
    for med in medications:
        lc = med.lower()
        for key, med_rules in _MEDICATION_RULES.items():
            if re.search(rf"\b{re.escape(key)}\b", lc):
                rules.extend(med_rules)
                break

    allergies: list[str] = profile.get("allergies", []) or []
    if isinstance(allergies, str):
        allergies = [s.strip() for s in allergies.split(",") if s.strip()]
    for allergen in allergies:
        rules.append(_ALLERGY_RULE.format(allergen=allergen))

    return list(dict.fromkeys(rules))  # deduplicate, preserve order


# ── Memory local filter (Opt 4) ───────────────────────────────────────────────

# Domains whose memories are always included regardless of what the current
# message is about. Medical context affects all wellness recommendations;
# general facts (occupation, lifestyle) are always relevant.
_ALWAYS_INCLUDE_DOMAINS: frozenset[str] = frozenset({"general", "medical"})


def _local_filter_memories(
    all_memories: list[dict],
    domains: list[str],
    limit: int = 10,
) -> list[str]:
    """Extract fact strings from a get_all_memories result and domain-filter them.

    Replaces the per-turn Qdrant semantic search when the memory cache is warm.
    - Always includes "general" and "medical" domain memories.
    - Includes other domains only when they appear in the inferred domain list.
    - When no domain restriction is inferred (empty list), all memories pass.
    """
    if not all_memories:
        return []

    facts: list[str] = []
    for mem in all_memories:
        text = mem.get("memory") or mem.get("fact") or mem.get("content", "")
        if not text:
            continue
        mem_domain = (mem.get("metadata") or {}).get("domain", "general")
        if not domains or mem_domain in domains or mem_domain in _ALWAYS_INCLUDE_DOMAINS:
            facts.append(text)

    return facts[:limit]


# ── Node ───────────────────────────────────────────────────────────────────────

async def context_hydration_node(state: GraphState) -> dict:
    """Populate user_profile, conversation_history, memory_context, constraint_rules."""
    user_id = state.get("user_id", "")
    session_id = state.get("session_id", "") or None
    conversation_id = state.get("conversation_id", "")

    profile: dict[str, Any] = {}
    cached_profile = None
    if user_id:
        cached_profile = await cache_get(f"profile:{user_id}")
        if cached_profile:
            profile = cached_profile
            logger.debug("Profile cache HIT — user=%s", user_id)

    async def _fetch_profile() -> dict[str, Any]:
        p = await db.get_user_profile(user_id)
        if p:
            await cache_set(f"profile:{user_id}", p, ttl_seconds=86400)
        return p

    tasks: list = []
    profile_future = None

    if not profile:
        profile_future = asyncio.create_task(_fetch_profile())
        tasks.append(profile_future)

    from datetime import date as date_cls
    today = date_cls.today().isoformat()

    user_message = state.get("user_message", "")

    from src.infra.mem0_client import get_all_memories, infer_domains
    # Opt 3: use history already in state (from MemorySaver checkpoint) if present.
    # Falls back to Supabase only on first turn or after pod restart.
    existing_history: list[dict] = state.get("conversation_history") or []
    if existing_history:
        history_future = None
        logger.debug("History from state — user=%s messages=%d", user_id, len(existing_history))
    else:
        history_future = asyncio.create_task(db.get_conversation_history(conversation_id))

    # Infer relevant domains from user message for local filtering (Opt 4).
    memory_query = user_message or "recent context"
    active_domains = infer_domains(memory_query)
    logger.debug("Memory domains inferred — user=%s domains=%s", user_id, active_domains)

    # Opt 4: check memory cache first; on miss, fetch all memories once and cache.
    # Local domain filter replaces the per-turn Qdrant semantic search (300–800ms).
    cached_all_memories = await cache_get(f"mem:all:{user_id}") if user_id else None
    if cached_all_memories is not None:
        memory_future = None
        logger.debug("Memory cache HIT — user=%s", user_id)
    else:
        async def _fetch_all_memories() -> list[dict]:
            mems = await get_all_memories(user_id)
            await cache_set(f"mem:all:{user_id}", mems if mems is not None else [], ttl_seconds=600)
            return mems or []
        memory_future = asyncio.create_task(_fetch_all_memories())
    # Opt 2: meals are cached per-user per-day; invalidated by write tools
    cached_meals = await cache_get(f"meals:{user_id}:{today}") if user_id else None
    if cached_meals is not None:
        meal_context_future = None
        logger.debug("Meals cache HIT — user=%s date=%s items=%d", user_id, today, len(cached_meals))
    else:
        meal_context_future = asyncio.create_task(db.get_meal_items(user_id, date=today, limit=30))

    if history_future is not None:
        tasks.append(history_future)
    if memory_future is not None:
        tasks.append(memory_future)
    if meal_context_future is not None:
        tasks.append(meal_context_future)

    await asyncio.gather(*tasks, return_exceptions=True)

    if profile_future is not None:
        exc = profile_future.exception()
        if exc:
            logger.warning("Profile fetch failed: %s", exc)
        else:
            profile = profile_future.result()

    if existing_history:
        history: list[dict] = existing_history
    elif history_future is not None:
        history_exc = history_future.exception()
        history = [] if history_exc else history_future.result()
    else:
        history = []

    if cached_all_memories is not None:
        # Cache hit: filter locally — no Qdrant call
        memories: list[str] = _local_filter_memories(cached_all_memories, active_domains)
        graph_relations: list[dict] = []
    elif memory_future is not None:
        memory_exc = memory_future.exception()
        if memory_exc:
            logger.warning("Memory fetch failed: %s", memory_exc)
            memories = []
        else:
            all_mems = memory_future.result()
            memories = _local_filter_memories(all_mems, active_domains)
        graph_relations = []
    else:
        memories = []
        graph_relations = []

    if cached_meals is not None:
        meal_items_context: list[dict] = cached_meals
    elif meal_context_future is not None:
        meal_ctx_exc = meal_context_future.exception()
        if meal_ctx_exc:
            meal_items_context = []
        else:
            meal_items_context = meal_context_future.result()
            # Cache result (even empty list) so subsequent turns skip Supabase
            asyncio.create_task(
                cache_set(f"meals:{user_id}:{today}", meal_items_context, ttl_seconds=86400)
            )
    else:
        meal_items_context = []

    constraint_rules = derive_constraint_rules(profile)

    logger.info(
        "Context hydrated — user=%s profile=%s history=%d memories=%d constraints=%d meals=%s",
        user_id,
        "cached" if cached_profile else "fetched",
        len(history),
        len(memories),
        len(constraint_rules),
        f"{len(meal_items_context)}(cached)" if cached_meals is not None else str(len(meal_items_context)),
    )

    return {
        "user_profile": profile,
        "conversation_history": history,
        "memory_context": memories,
        "graph_relations": graph_relations,
        "constraint_rules": constraint_rules,
        "meal_items_context": meal_items_context,
    }
