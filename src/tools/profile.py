"""Profile and memory tools — shared across all domain agents."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import src.infra.db as db
from src.infra.logger import setup_logger
from src.infra.redis_client import cache_delete

logger = setup_logger(__name__)


def _ctx(config: RunnableConfig) -> tuple[str, str, str]:
    c = config.get("configurable", {})
    return c.get("user_id", ""), c.get("conversation_id", ""), c.get("session_id", "")


class UpdateUserProfileInput(BaseModel):
    weight_kg: float | None = Field(None, description="Current body weight in kg.")
    height_cm: float | None = Field(None, description="Height in cm.")
    date_of_birth: str | None = Field(None, description="ISO-8601 date e.g. '1990-05-20'.")
    sex: str | None = Field(None, description="'male', 'female', or 'other'.")
    goals: list[str] | None = Field(None, description="User's health/fitness goals as a list.")
    allergies: list[str] | None = Field(None, description="Known food allergies.")
    conditions: list[str] | None = Field(None, description="Known medical conditions (summary).")
    medications: list[str] | None = Field(None, description="Current medications (summary).")


@tool("update_user_profile", args_schema=UpdateUserProfileInput)
async def update_user_profile(
    weight_kg: float | None,
    height_cm: float | None,
    date_of_birth: str | None,
    sex: str | None,
    goals: list[str] | None,
    allergies: list[str] | None,
    conditions: list[str] | None,
    medications: list[str] | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Save or update the user's personal profile (weight, height, goals, allergies, etc.)."""
    user_id, _, _ = _ctx(config)
    delta: dict[str, Any] = {}
    if weight_kg is not None:
        delta["weight_kg"] = weight_kg
    if height_cm is not None:
        delta["height_cm"] = height_cm
    if date_of_birth is not None:
        delta["date_of_birth"] = date_of_birth
    if sex is not None:
        delta["sex"] = sex
    if goals is not None:
        delta["goals"] = goals
    if allergies is not None:
        delta["allergies"] = allergies
    if conditions is not None:
        delta["conditions"] = conditions
    if medications is not None:
        delta["medications"] = medications
    if not delta:
        return {"status": "no_changes"}
    await db.upsert_user_profile(user_id, delta)
    await cache_delete(f"profile:{user_id}")
    return {
        "status": "saved",
        "updated_fields": list(delta.keys()),
        "updated_values": {k: (v if not isinstance(v, list) else ", ".join(v)) for k, v in delta.items()},
        "refresh": "user-profile",
    }


class SaveMemoryFactInput(BaseModel):
    fact: str = Field(
        ...,
        description=(
            "A concise factual statement about the user to remember long-term. "
            "Examples: 'User dislikes sweets.', "
            "'User works as a software engineer with limited lunch breaks.', "
            "'User prefers morning workouts.'"
        ),
    )
    domain: str = Field(
        "general",
        description=(
            "Domain category for this fact: "
            "'nutrition' (food preferences, dietary habits, allergies), "
            "'fitness' (workout preferences, activity level, fitness goals), "
            "'medical' (conditions, medications affecting recommendations), "
            "'general' (occupation, lifestyle, schedule, other)."
        ),
    )


@tool("save_memory_fact", args_schema=SaveMemoryFactInput)
async def save_memory_fact(fact: str, domain: str, config: RunnableConfig) -> dict[str, Any]:
    """Persist a notable fact or preference about the user for future conversations."""
    user_id, _, session_id = _ctx(config)
    from src.infra.mem0_client import add_memory
    return await add_memory(fact, user_id=user_id, domain=domain, run_id=session_id or None)


