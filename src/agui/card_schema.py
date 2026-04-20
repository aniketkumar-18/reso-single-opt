"""LLM-driven ComposeCard JSON generator.

Instead of hardcoded per-tool templates, a lightweight LLM call (gpt-4o-mini)
generates the card structure dynamically from:
  - the tool name + description
  - the proposed args (or the tool result)
  - the user's profile context
  - the full ComposeCard element palette

This is the AG-UI "Shared State" approach from CopilotKit: the server drives
the UI structure via data, and the client renders whatever JSON it receives
with zero tool-specific code.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_openai import ChatOpenAI

from src.infra.config import get_settings
from src.infra.logger import setup_logger

logger = setup_logger(__name__)

# ── Element palette prompt ───────────────────────────────────────────────────

_ELEMENT_PALETTE = """
## ComposeCard JSON Schema

A ComposeCard has these top-level fields:
  { "title": str, "subtitle": str|null, "badge": str|null,
    "elements": [...],           // display-only summary shown in list mode
    "detailElements": [...],     // interactive form shown in hero/detail mode
    "actions": [...]             // CTA buttons at the bottom
  }

### Interactive elements (use in detailElements for forms):
Each interactive element MUST have a unique "id" field for form-data collection.
  {"type":"radio_group","id":"meal_type","label":"...","options":["A","B","C"],"selected":0}
  {"type":"chip_group","id":"allergies","chips":["X","Y","Z"],"selected":[]}
  {"type":"slider","id":"weight","label":"...","min":0.0,"max":100.0,"value":50.0,"unit":"kg"}
  {"type":"stepper","id":"servings","label":"...","value":5,"min":0,"max":20,"unit":"days"}
  {"type":"text_input","id":"description","label":"...","placeholder":"...","value":""}
  {"type":"segmented_control","id":"unit","options":["A","B","C"],"selected":0}
  {"type":"toggle","id":"fasting","label":"...","description":"...","checked":false}
  {"type":"star_rating","id":"rating","label":"...","rating":0,"maxStars":5}
  {"type":"checklist","id":"steps","items":[{"text":"...","checked":false}]}

### Display elements (use in elements for summaries):
  {"type":"metric_row","metrics":[{"label":"...","value":"...","unit":"..."}]}
  {"type":"progress","label":"...","value":50.0,"target":100.0,"unit":"...","status":"good"}
  {"type":"key_value_list","items":[{"key":"...","value":"..."}]}
  {"type":"section","title":"...","style":"default","elements":[...]}
  {"type":"status_banner","message":"...","style":"info"}  // info|success|warning|error
  {"type":"text","content":"...","style":"body"}  // body|subtitle|caption|bold
  {"type":"data_table","headers":["A","B"],"rows":[["1","2"]]}
  {"type":"divider"}

### Actions (server-driven behavior):
  {"label":"Confirm","style":"primary","handler":{"kind":"resolve_tool_call","collectForm":true}}
  {"label":"Skip","style":"secondary","handler":{"kind":"reject_tool_call"}}

Handler kinds: resolve_tool_call, reject_tool_call, dismiss, navigate_back, open_sheet,
  open_url (with "url"), send_message (with "message", "autoSend"),
  start_run (with "message", "context": {"tool_hint":"...", "intent":"re_edit|retry|delete|view"}).
Style values: primary, secondary, destructive, link.
"""

# ── Generation prompts ───────────────────────────────────────────────────────

_PREVIEW_SYSTEM = f"""You generate ComposeCard JSON for a mobile health & wellness app.
{_ELEMENT_PALETTE}

## Rules
- Return ONLY a valid JSON object (no markdown fences, no explanation).
- "elements" should be a SHORT summary of what the tool will do (1-3 display elements).
- "detailElements" should be an EXHAUSTIVE interactive form for ALL editable fields the tool accepts. Use the richest UI element that fits each field type. Pre-fill values from the proposed args or user profile when available.
- "actions" MUST include:
  - Primary CTA: {{"label":"Confirm & Log","style":"primary","handler":{{"kind":"resolve_tool_call","collectForm":true}}}}
  - Secondary:   {{"label":"Skip","style":"secondary","handler":{{"kind":"reject_tool_call"}}}}
- "badge" should be "Confirm" for preview cards.
- Every interactive element in "detailElements" MUST have an "id" field that EXACTLY MATCHES the corresponding tool parameter name (e.g. if the tool accepts "weight_kg", the slider's id must be "weight_kg"). This is critical — the id is used to map form values back to tool parameters on submission.
- Choose element types that match the data: radio_group for enums, slider for numeric ranges, chip_group for multi-select lists, text_input for free text, toggle for booleans, etc.
- For profile updates: include ALL updateable profile fields (weight, height, sex, DOB, goals, allergies, conditions, medications) as separate form elements with appropriate types.
"""

_COMPLETE_SYSTEM = f"""You generate ComposeCard JSON for a mobile health & wellness app.
{_ELEMENT_PALETTE}

## Rules
- Return ONLY a valid JSON object (no markdown fences, no explanation).
- This is a COMPLETION card shown after a tool executed successfully.
- "elements" should summarize what was saved/changed (key_value_list + status_banner with style "success").
- Do NOT include "detailElements" — this card is read-only.
- "badge" should be "Done".
- If the result indicates "skipped", show a status_banner with style "info" saying "No changes were made." and do NOT include actions.
- If the result indicates "saved" (not skipped), include ONE action for re-editing:
  {{"label":"Update","style":"secondary","handler":{{"kind":"start_run","message":"update my [context]","context":{{"tool_hint":"[tool_name]","intent":"re_edit"}}}}}}
  Replace [context] with a natural phrase (e.g. "profile", "meal", "workout") and [tool_name] with the actual tool name.
"""

_READ_SYSTEM = f"""You generate ComposeCard JSON for a mobile health & wellness app.
{_ELEMENT_PALETTE}

## Rules
- Return ONLY a valid JSON object (no markdown fences, no explanation).
- This is a READ-ONLY result card displaying data fetched by a tool.
- Use the richest display elements: metric_row for numbers, data_table for tabular data, progress for goals, key_value_list for general data.
- Do NOT include "detailElements" — this card is display-only.
- Do NOT include "actions" — no user interaction needed.
- "badge" should reflect the data status (e.g. "Logged", "Today", "Calculated").
"""


# ── Generator ────────────────────────────────────────────────────────────────

async def generate_card(
    mode: Literal["preview", "complete", "read_result"],
    tool_name: str,
    tool_description: str,
    args: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    profile_context: str = "",
) -> str:
    """Ask gpt-4o-mini to generate a ComposeCard JSON string.

    Falls back to a minimal generic card if the LLM call fails.
    """
    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.llm_model_pipeline,
        temperature=0,
        api_key=settings.openai_api_key,
    )

    if mode == "preview":
        system = _PREVIEW_SYSTEM
        user_msg = (
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Proposed args: {json.dumps(args or {}, default=str)}\n"
            f"User profile context: {profile_context or 'none'}\n\n"
            "Generate the preview ComposeCard JSON."
        )
    elif mode == "complete":
        system = _COMPLETE_SYSTEM
        user_msg = (
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Result: {json.dumps(result or {}, default=str)}\n\n"
            "Generate the completion ComposeCard JSON."
        )
    else:  # read_result
        system = _READ_SYSTEM
        user_msg = (
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Result: {json.dumps(result or {}, default=str)}\n\n"
            "Generate the read-result ComposeCard JSON."
        )

    try:
        response = await llm.ainvoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        raw = str(response.content).strip()
        # Strip markdown fences if the LLM wraps in ```json...```
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3].strip()
        # Validate JSON
        parsed = json.loads(raw)
        logger.info("LLM card generated for tool=%s mode=%s keys=%s", tool_name, mode, list(parsed.keys()))
        return json.dumps(parsed)
    except Exception as exc:
        logger.warning("LLM card generation failed for tool=%s mode=%s: %s", tool_name, mode, exc)
        return _fallback_card(mode, tool_name, args, result)


def _fallback_card(
    mode: str,
    tool_name: str,
    args: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> str:
    """Minimal generic card when LLM generation fails."""
    title = tool_name.replace("_", " ").title()
    if mode == "preview":
        kv = [{"key": k.replace("_", " ").title(), "value": str(v)}
              for k, v in (args or {}).items() if v is not None][:8]
        return json.dumps({
            "title": title,
            "subtitle": "Review before saving",
            "badge": "Confirm",
            "elements": [{"type": "key_value_list", "items": kv}] if kv else [],
            "actions": [
                {"label": "Confirm", "style": "primary", "handler": {"kind": "resolve_tool_call", "collectForm": True}},
                {"label": "Skip", "style": "secondary", "handler": {"kind": "reject_tool_call"}},
            ],
        })
    elif mode == "complete":
        status = (result or {}).get("status", "done")
        msg = "Saved successfully." if status != "skipped" else "No changes were made."
        style = "success" if status != "skipped" else "info"
        return json.dumps({
            "title": title,
            "badge": "Done",
            "elements": [{"type": "status_banner", "message": msg, "style": style}],
        })
    else:
        kv = [{"key": k, "value": str(v)}
              for k, v in (result or {}).items() if k not in ("status", "refresh")][:10]
        return json.dumps({
            "title": title,
            "badge": "Result",
            "elements": [{"type": "key_value_list", "items": kv}] if kv else [],
        })
