"""AG-UI tool wrapper — intercepts LangChain tool calls for card emission.

``agui_wrap_tools(ALL_TOOLS, run_ctx)`` returns wrapped tools that emit
AG-UI events around execution:

- **WRITE tools** emit a 3-stage Shared State progression:
    1. Loading card (instant)
    2. LLM-generated preview card with interactive form
    3. Complete card after execution
  Between stages 2 and 3 the agent **suspends** until the client submits.

- **READ tools** execute normally, then emit an LLM-generated result card.

- **SILENT tools** pass through unchanged.

Card generation is fully LLM-driven — no hardcoded templates.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool

from src.agui import card_templates
from src.agui.events import CardEmit, ToolCallEnd, ToolCallStart
from src.agui.run_manager import RunContext
from src.infra.logger import setup_logger

logger = setup_logger(__name__)

# ── Tool classification ──────────────────────────────────────────────────────

WRITE_TOOLS = frozenset({
    "log_meal", "edit_meal_item", "delete_meal_item",
    "log_workout", "log_body_metrics",
    "update_user_profile", "update_fitness_goals",
    "log_medical_condition", "update_medical_condition",
    "log_medication", "update_medication",
    "log_from_image",
})

READ_TOOLS = frozenset({
    "get_meal_items", "calculate_macro_targets",
})

SILENT_TOOLS = frozenset({
    "save_memory_fact",
})


# ── Public API ───────────────────────────────────────────────────────────────

def agui_wrap_tools(
    tools: list[BaseTool],
    run_ctx: RunContext,
    profile_context: str = "",
) -> list[BaseTool]:
    """Return wrapped copies of *tools* with AG-UI event emission."""
    wrapped: list[BaseTool] = []
    for t in tools:
        name = t.name
        if name in WRITE_TOOLS:
            wrapped.append(_wrap_write(t, run_ctx, profile_context))
        elif name in READ_TOOLS:
            wrapped.append(_wrap_read(t, run_ctx))
        else:
            wrapped.append(t)
    return wrapped


# ── WRITE tool wrapper ───────────────────────────────────────────────────────

def _wrap_write(original: BaseTool, run_ctx: RunContext, profile_context: str) -> BaseTool:
    tool_name = original.name
    tool_desc = original.description or tool_name
    run_id = run_ctx.run_id
    _last_result: dict[str, Any] | None = None  # guards against duplicate calls in same run

    async def _invoke(**kwargs: Any) -> Any:
        nonlocal _last_result
        config = kwargs.pop("config", None)

        # Guard: if this WRITE tool already executed successfully in this run,
        # return the cached result instead of showing another form. The LLM
        # sometimes calls the same tool twice; the second call would show a
        # form with stale defaults, confusing the user.
        if _last_result is not None:
            logger.info("Tool %s already executed in this run — returning cached result", tool_name)
            return _last_result

        tc_id = f"tc_{uuid.uuid4().hex[:12]}"
        card_id = f"card_{tool_name}_{uuid.uuid4().hex[:8]}"

        # ── Stage 1: Loading card (instant feedback) ──────────────
        loading_json = card_templates.loading_card(tool_name)
        await run_ctx.emit(CardEmit(runId=run_id, cardId=card_id, cardJson=loading_json))

        # ── Stage 2: LLM-generated preview with form ─────────────
        preview_json = await card_templates.for_write_preview(
            tool_name=tool_name,
            tool_description=tool_desc,
            args=kwargs,
            profile_context=profile_context,
        )
        await run_ctx.emit(CardEmit(runId=run_id, cardId=card_id, cardJson=preview_json))

        # ── ToolCallStart — agent suspends ────────────────────────
        await run_ctx.emit(ToolCallStart(
            runId=run_id,
            toolCallId=tc_id,
            toolName=tool_name,
            args={**kwargs, "cardId": card_id},
        ))

        # Wait for client to POST /agui/tool-result
        try:
            user_result = await run_ctx.await_tool_result(tc_id)
        except Exception:
            user_result = {"skipped": True}
            logger.warning("Tool call %s timed out or failed", tc_id)

        # Execute real tool (or skip)
        if user_result.get("skipped"):
            result = {"status": "skipped", "message": "User skipped this action."}
        else:
            # Merge the user's form edits into the original LLM-proposed kwargs.
            # The client sends form field values keyed by element id (e.g.
            # "date_of_birth": "1990-05-15", "allergies": ["Peanuts"]).
            # These override the LLM defaults so the tool executes with the
            # user's actual edits — not the pre-filled values.
            merged_kwargs = {**kwargs}
            # Remove UI-only metadata that was injected for client tracking
            # but is not a real tool parameter.
            merged_kwargs.pop("cardId", None)
            for key, value in user_result.items():
                if key not in ("skipped",) and value is not None and value != "":
                    merged_kwargs[key] = value

            # Coerce form values to match the tool's Pydantic schema types.
            # The client sends text_input values as strings, but the tool may
            # expect list[str] (e.g. goals, allergies). Parse comma-separated
            # strings into lists where the schema declares a list type.
            schema = original.args_schema
            if schema is not None:
                for field_name, field_info in schema.model_fields.items():
                    if field_name not in merged_kwargs:
                        continue
                    val = merged_kwargs[field_name]
                    annotation = str(field_info.annotation or "")
                    # If schema expects list[str] but we got a plain string,
                    # split on comma and strip whitespace.
                    if "list" in annotation.lower() and isinstance(val, str):
                        # Split on comma or newline (LLM may use either as separator)
                        import re
                        merged_kwargs[field_name] = [
                            s.strip() for s in re.split(r'[,\n]', val) if s.strip()
                        ]

            logger.info(
                "Tool %s executing with merged kwargs: original=%s user_edits=%s",
                tool_name,
                list(kwargs.keys()),
                {k: v for k, v in user_result.items() if k != "skipped"},
            )
            if config is not None:
                merged_kwargs["config"] = config
            result = await original.ainvoke(merged_kwargs, config=config)

        # ── ToolCallEnd ──────────────────────────────────────────
        await run_ctx.emit(ToolCallEnd(runId=run_id, toolCallId=tc_id))

        # ── Stage 3: LLM-generated complete card ─────────────────
        complete_json = await card_templates.for_write_complete(
            tool_name=tool_name,
            tool_description=tool_desc,
            result=result,
        )
        await run_ctx.emit(CardEmit(runId=run_id, cardId=card_id, cardJson=complete_json))

        # Cache result to prevent duplicate form if agent retries
        _last_result = result
        return result

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        return_direct=original.return_direct,
    )


# ── READ tool wrapper ────────────────────────────────────────────────────────

def _wrap_read(original: BaseTool, run_ctx: RunContext) -> BaseTool:
    tool_name = original.name
    tool_desc = original.description or tool_name
    run_id = run_ctx.run_id

    async def _invoke(**kwargs: Any) -> Any:
        config = kwargs.pop("config", None)
        result = await original.ainvoke(kwargs, config=config)
        card_id = f"card_{tool_name}_{uuid.uuid4().hex[:8]}"
        card_json = await card_templates.for_read_result(
            tool_name=tool_name,
            tool_description=tool_desc,
            result=result,
        )
        await run_ctx.emit(CardEmit(runId=run_id, cardId=card_id, cardJson=card_json))
        return result

    return StructuredTool.from_function(
        coroutine=_invoke,
        name=original.name,
        description=original.description,
        args_schema=original.args_schema,
        return_direct=original.return_direct,
    )
