"""Medical domain tools.

Exact Supabase schema (verified via PostgREST OpenAPI):
  medical_conditions: id, account_id, condition (NOT NULL), severity (NOT NULL default 'moderate'),
                      notes (NOT NULL default ''), diagnosed_at (NOT NULL default now()),
                      active (NOT NULL default true), updated_at (TIMESTAMPTZ default now())
  medications:        id, account_id, name (NOT NULL), dosage (NOT NULL), frequency (NOT NULL),
                      notes (NOT NULL default ''), started_at (NOT NULL default now()),
                      active (NOT NULL default true), updated_at (TIMESTAMPTZ default now())

NOTE — if you see error code 42703 / "record new has no field updated_at", run this migration:
  ALTER TABLE medical_conditions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
  ALTER TABLE medications        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from pydantic import BaseModel, Field

import src.infra.db as db
from src.infra.logger import setup_logger

logger = setup_logger(__name__)


def _ctx(config: RunnableConfig) -> tuple[str, str]:
    c = config.get("configurable", {})
    return c.get("user_id", ""), c.get("conversation_id", "")


def _is_missing_updated_at(exc: Exception) -> bool:
    """Return True when the error is a trigger referencing a missing updated_at column.

    Root cause: a BEFORE UPDATE trigger on the table calls NEW.updated_at = now() but the
    updated_at column was never added.  Fix:
      ALTER TABLE medical_conditions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
      ALTER TABLE medications        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();
    """
    msg = str(exc)
    return "updated_at" in msg and ("42703" in msg or 'has no field "updated_at"' in msg or "has no field 'updated_at'" in msg)


# ── Medical conditions ─────────────────────────────────────────────────────────

class LogMedicalConditionInput(BaseModel):
    condition: str = Field(..., description="Name of the condition, e.g. 'Type 2 Diabetes'.")
    severity: Literal["mild", "moderate", "severe"] = Field("moderate")
    notes: str = Field("", description="Any extra context or management notes.")
    diagnosed_at: str | None = Field(None, description="ISO-8601 datetime. Null = now.")


@tool("log_medical_condition", args_schema=LogMedicalConditionInput)
async def log_medical_condition(
    condition: str, severity: str, notes: str, diagnosed_at: str | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Log a medical condition for the user."""
    user_id, _ = _ctx(config)
    data: dict[str, Any] = {"condition": condition, "severity": severity, "notes": notes}
    if diagnosed_at:
        data["diagnosed_at"] = diagnosed_at
    result = await db.log_medical_condition(user_id, data)
    return {"status": "logged", "condition": result, "refresh": "medical-conditions"}


class UpdateMedicalConditionInput(BaseModel):
    condition: str = Field(..., description="Exact condition name to update, e.g. 'Hypertension'.")
    severity: Literal["mild", "moderate", "severe"] | None = Field(
        None, description="New severity level. Omit if not changing."
    )
    active: bool | None = Field(None, description="Set false to mark condition as resolved/inactive. Omit if not changing.")
    notes: str | None = Field(None, description="New notes or management context. Omit if not changing.")
    diagnosed_at: str | None = Field(None, description="New ISO-8601 diagnosed date. Omit if not changing.")


@tool("update_medical_condition", args_schema=UpdateMedicalConditionInput)
async def update_medical_condition(
    condition: str,
    severity: str | None,
    active: bool | None,
    notes: str | None,
    diagnosed_at: str | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Update an existing medical condition record. Only pass the fields that need changing."""
    user_id, _ = _ctx(config)

    if not user_id:
        return {"status": "error", "message": "User not authenticated", "refresh": "medical-conditions"}

    updates: dict[str, Any] = {}
    if severity is not None:
        updates["severity"] = severity
    if active is not None:
        updates["active"] = active
    if notes is not None:
        updates["notes"] = notes
    if diagnosed_at is not None:
        updates["diagnosed_at"] = diagnosed_at

    if not updates:
        return {"status": "error", "message": "No fields to update — provide at least one of: severity, active, notes, diagnosed_at"}

    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "medical-conditions"}
    try:
        # Step 1: apply the update (supabase v2 update returns empty body by default)
        await (
            client.table("medical_conditions")
            .update(updates)
            .eq("account_id", user_id)
            .ilike("condition", condition)
            .execute()
        )
        # Step 2: fetch the updated row to confirm it exists and return full data
        verify = await (
            client.table("medical_conditions")
            .select("*")
            .eq("account_id", user_id)
            .ilike("condition", condition)
            .execute()
        )
        if not verify.data:
            return {
                "status": "error",
                "message": f"No condition matching '{condition}' found in your profile.",
                "refresh": "medical-conditions",
            }
        return {"status": "updated", "condition": verify.data[0], "refresh": "medical-conditions"}
    except Exception as exc:
        if _is_missing_updated_at(exc):
            logger.warning(
                "update_medical_condition: DB trigger references missing updated_at column on "
                "medical_conditions — run: ALTER TABLE medical_conditions ADD COLUMN IF NOT EXISTS "
                "updated_at TIMESTAMPTZ DEFAULT now();"
            )
            return {
                "status": "error",
                "message": "Database schema is missing the updated_at column on medical_conditions. "
                           "Please ask your administrator to run: "
                           "ALTER TABLE medical_conditions ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();",
                "refresh": "medical-conditions",
            }
        logger.warning("update_medical_condition failed for user=%s condition=%s: %s", user_id, condition, exc, exc_info=True)
        return {"status": "error", "message": str(exc), "refresh": "medical-conditions"}


# ── Medications ────────────────────────────────────────────────────────────────

class LogMedicationInput(BaseModel):
    name: str = Field(..., description="Drug name, e.g. 'Metformin'.")
    dosage: str = Field(..., description="Dose with units, e.g. '500mg'.")
    frequency: str = Field(..., description="How often taken, e.g. 'twice daily with meals'.")
    notes: str = Field("", description="Optional extra context.")
    started_at: str | None = Field(None, description="ISO-8601 datetime. Null = now.")


@tool("log_medication", args_schema=LogMedicationInput)
async def log_medication(
    name: str, dosage: str, frequency: str, notes: str, started_at: str | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Log a medication for the user."""
    user_id, _ = _ctx(config)
    data: dict[str, Any] = {"name": name, "dosage": dosage, "frequency": frequency, "notes": notes}
    if started_at:
        data["started_at"] = started_at
    result = await db.log_medication(user_id, data)
    return {"status": "logged", "medication": result, "refresh": "medications"}


class UpdateMedicationInput(BaseModel):
    name: str = Field(..., description="Exact medication name to update, e.g. 'Metformin'.")
    dosage: str | None = Field(None, description="New dose with units, e.g. '1000mg'. Omit if not changing.")
    frequency: str | None = Field(None, description="New frequency, e.g. 'once daily'. Omit if not changing.")
    active: bool | None = Field(None, description="Set false to mark medication as stopped. Omit if not changing.")
    notes: str | None = Field(None, description="New notes. Omit if not changing.")


@tool("update_medication", args_schema=UpdateMedicationInput)
async def update_medication(
    name: str,
    dosage: str | None,
    frequency: str | None,
    active: bool | None,
    notes: str | None,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Update an existing medication record. Only pass the fields that need changing."""
    user_id, _ = _ctx(config)

    if not user_id:
        return {"status": "error", "message": "User not authenticated", "refresh": "medications"}

    updates: dict[str, Any] = {}
    if dosage is not None:
        updates["dosage"] = dosage
    if frequency is not None:
        updates["frequency"] = frequency
    if active is not None:
        updates["active"] = active
    if notes is not None:
        updates["notes"] = notes

    if not updates:
        return {"status": "error", "message": "No fields to update — provide at least one of: dosage, frequency, active, notes"}

    client = await db.get_client()
    if client is None:
        return {"status": "error", "message": "database not configured", "refresh": "medications"}
    try:
        # Step 1: apply the update (supabase v2 update returns empty body by default)
        await (
            client.table("medications")
            .update(updates)
            .eq("account_id", user_id)
            .ilike("name", name)
            .execute()
        )
        # Step 2: fetch the updated row to confirm it exists and return full data
        verify = await (
            client.table("medications")
            .select("*")
            .eq("account_id", user_id)
            .ilike("name", name)
            .execute()
        )
        if not verify.data:
            return {
                "status": "error",
                "message": f"No medication matching '{name}' found in your profile.",
                "refresh": "medications",
            }
        return {"status": "updated", "medication": verify.data[0], "refresh": "medications"}
    except Exception as exc:
        if _is_missing_updated_at(exc):
            logger.warning(
                "update_medication: DB trigger references missing updated_at column on medications — "
                "run: ALTER TABLE medications ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();"
            )
            return {
                "status": "error",
                "message": "Database schema is missing the updated_at column on medications. "
                           "Please ask your administrator to run: "
                           "ALTER TABLE medications ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT now();",
                "refresh": "medications",
            }
        logger.warning("update_medication failed for user=%s name=%s: %s", user_id, name, exc, exc_info=True)
        return {"status": "error", "message": str(exc), "refresh": "medications"}


