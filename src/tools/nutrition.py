"""Nutrition domain tools.

Context (user_id, conversation_id) is passed via RunnableConfig.configurable
so tools work correctly inside create_react_agent's internal ToolNode.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import src.infra.db as db
from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.infra.redis_client import cache_delete

logger = setup_logger(__name__)


def _ctx(config: RunnableConfig) -> tuple[str, str]:
    """Extract user_id and conversation_id from config."""
    c = config.get("configurable", {})
    return c.get("user_id", ""), c.get("conversation_id", "")


async def _interpret_meal(description: str) -> dict[str, Any]:
    llm = ChatOpenAI(model=get_settings().llm_model_pipeline, temperature=0,
                     api_key=get_settings().openai_api_key)
    prompt = (
        "Extract nutritional info from this meal description as JSON with keys: "
        "name (str), calories (int), protein_g (float), carbs_g (float), fat_g (float). "
        "Use best estimates. Return ONLY the JSON object.\n\nMeal: " + description
    )
    response = await llm.ainvoke(prompt)
    raw = str(response.content).strip().strip("```json").strip("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"name": description, "calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0}


class LogMealInput(BaseModel):
    description: str = Field(..., description="Natural language description of the meal.")
    meal_type: str = Field("snack", description="One of: breakfast, lunch, dinner, snack.")
    logged_at: str | None = Field(None, description="ISO-8601 timestamp. Null = now.")


@tool("log_meal", args_schema=LogMealInput)
async def log_meal(description: str, meal_type: str, logged_at: str | None,
                   config: RunnableConfig) -> dict[str, Any]:
    """Log a meal for the user. Extracts macros automatically from the description."""
    from datetime import date as date_cls
    user_id, conversation_id = _ctx(config)
    macros = await _interpret_meal(description)
    meal_data: dict[str, Any] = {"meal_type": meal_type, "description": description, **macros}
    if logged_at:
        meal_data["logged_at"] = logged_at
    result = await db.log_meal(user_id, conversation_id, meal_data)
    # Invalidate today's meals cache so the next context_hydration fetches fresh data
    await cache_delete(f"meals:{user_id}:{date_cls.today().isoformat()}")
    return {"status": "logged", "meal": result, "refresh": "meal-items"}


class GetMealItemsInput(BaseModel):
    date: str | None = Field(None, description="Date in YYYY-MM-DD format. Null = today.")
    meal_type: str | None = Field(None, description="Filter by meal type: breakfast, lunch, dinner, snack. Null = all.")
    limit: int = Field(20, description="Max number of records to return.")


@tool("get_meal_items", args_schema=GetMealItemsInput)
async def get_meal_items(date: str | None, meal_type: str | None, limit: int,
                         config: RunnableConfig) -> dict[str, Any]:
    """Fetch the user's logged meal items from the database. Always call this when the user asks about their logged meals — never rely on conversation history for current meal data."""
    from datetime import date as date_cls, timezone
    user_id, _ = _ctx(config)
    resolved_date = date or date_cls.today().isoformat()
    rows = await db.get_meal_items(user_id, date=resolved_date, meal_type=meal_type, limit=limit)
    return {"status": "ok", "meals": rows, "date": resolved_date}


class EditMealItemInput(BaseModel):
    meal_item_id: str = Field(
        ...,
        description="UUID of the meal item to edit. Find it in the 'Today's Logged Meals' context section.",
    )
    food_name: str | None = Field(None, description="Corrected food name, e.g. 'Sourdough toast, 2 slices'.")
    description: str | None = Field(None, description="Full corrected meal description.")
    calories_kcal: int | None = Field(None, description="Re-estimated total calories for the corrected meal.")
    protein_g: float | None = Field(None, description="Re-estimated protein in grams.")
    carbs_g: float | None = Field(None, description="Re-estimated carbohydrates in grams.")
    fat_g: float | None = Field(None, description="Re-estimated fat in grams.")
    portion: str | None = Field(None, description="Corrected portion, e.g. '2 slices'.")
    occasion: str | None = Field(None, description="Corrected meal type: breakfast, lunch, dinner, or snack.")


@tool("edit_meal_item", args_schema=EditMealItemInput)
async def edit_meal_item(
    meal_item_id: str,
    food_name: str | None,
    description: str | None,
    calories_kcal: int | None,
    protein_g: float | None,
    carbs_g: float | None,
    fat_g: float | None,
    portion: str | None,
    occasion: str | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Edit an existing meal item by its ID. Use the id from the Today's Logged Meals section. Pass the corrected field values as individual arguments — always include all macro fields re-estimated for the corrected meal."""
    user_id, _ = _ctx(config)
    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "meal-items"}
    updates: dict[str, Any] = {}
    if food_name is not None:     updates["food_name"] = food_name
    if description is not None:   updates["description"] = description
    if calories_kcal is not None: updates["calories_kcal"] = calories_kcal
    if protein_g is not None:     updates["protein_g"] = protein_g
    if carbs_g is not None:       updates["carbs_g"] = carbs_g
    if fat_g is not None:         updates["fat_g"] = fat_g
    if portion is not None:       updates["portion"] = portion
    if occasion is not None:      updates["occasion"] = occasion
    if not updates:
        return {"status": "error", "message": "no fields provided to update", "refresh": "meal-items"}
    if not user_id:
        return {"status": "error", "message": "User not authenticated", "refresh": "meal-items"}
    try:
        from datetime import date as date_cls
        # Step 1: apply the update (supabase v2 update returns empty body by default)
        await client.table("meal_items").update(updates).eq("id", meal_item_id).eq("account_id", user_id).execute()
        # Step 2: fetch the updated row to confirm it exists and return full data
        verify = await client.table("meal_items").select("*").eq("id", meal_item_id).eq("account_id", user_id).execute()
        if not verify.data:
            return {"status": "error", "message": f"No meal item with id '{meal_item_id}' found for this user.", "refresh": "meal-items"}
        # Invalidate today's meals cache
        await cache_delete(f"meals:{user_id}:{date_cls.today().isoformat()}")
        return {"status": "updated", "meal": verify.data[0], "refresh": "meal-items"}
    except Exception as exc:
        logger.warning("edit_meal_item failed for user=%s id=%s: %s", user_id, meal_item_id, exc, exc_info=True)
        return {"status": "error", "message": str(exc), "refresh": "meal-items"}


class DeleteMealItemInput(BaseModel):
    meal_item_id: str = Field(..., description="UUID of the meal item to delete.")


@tool("delete_meal_item", args_schema=DeleteMealItemInput)
async def delete_meal_item(meal_item_id: str, config: RunnableConfig) -> dict[str, Any]:
    """Delete a meal item by its ID."""
    user_id, _ = _ctx(config)
    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "meal-items"}
    try:
        from datetime import date as date_cls
        await client.table("meal_items").delete().eq("id", meal_item_id).eq("account_id", user_id).execute()
        # Invalidate today's meals cache
        await cache_delete(f"meals:{user_id}:{date_cls.today().isoformat()}")
        return {"status": "deleted", "meal_item_id": meal_item_id, "refresh": "meal-items"}
    except Exception as exc:
        return {"status": "error", "message": str(exc), "refresh": "meal-items"}


