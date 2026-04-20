"""ComposeCard generation for AG-UI tool events — LLM-driven.

All card generation is delegated to ``card_schema.generate_card()`` which
calls gpt-4o-mini to produce the ComposeCard JSON dynamically. No hardcoded
per-tool templates exist — the LLM sees the tool schema, proposed args, and
user profile, then outputs the card structure.

This module exposes the same three-function API that ``tool_interceptor.py``
calls, but the implementations are now async.
"""

from __future__ import annotations

import json
from typing import Any

from src.agui.card_schema import generate_card


# ── Loading card (static — no LLM call needed) ──────────────────────────────

def loading_card(tool_name: str) -> str:
    """Stage 1: lightweight skeleton card emitted immediately while the LLM
    generates the real preview. Gives instant visual feedback."""
    title = tool_name.replace("_", " ").title()
    return json.dumps({
        "title": title,
        "subtitle": "Preparing…",
        "badge": "Loading",
        "elements": [
            {"type": "text", "content": "Generating form…", "style": "caption"},
            {"type": "progress", "label": "Building", "value": 0.5, "target": 1.0,
             "unit": "", "status": "default"},
        ],
    })


# ── WRITE tool cards ─────────────────────────────────────────────────────────

async def for_write_preview(
    tool_name: str,
    tool_description: str,
    args: dict[str, Any],
    profile_context: str = "",
) -> str:
    """Stage 2: LLM-generated preview card with interactive form + CTA."""
    return await generate_card(
        mode="preview",
        tool_name=tool_name,
        tool_description=tool_description,
        args=args,
        profile_context=profile_context,
    )


async def for_write_complete(
    tool_name: str,
    tool_description: str,
    result: dict[str, Any],
) -> str:
    """Stage 3: LLM-generated completion card showing what was saved."""
    return await generate_card(
        mode="complete",
        tool_name=tool_name,
        tool_description=tool_description,
        result=result,
    )


# ── READ tool cards ──────────────────────────────────────────────────────────

async def for_read_result(
    tool_name: str,
    tool_description: str,
    result: dict[str, Any],
) -> str:
    """Display-only card for read tool results."""
    return await generate_card(
        mode="read_result",
        tool_name=tool_name,
        tool_description=tool_description,
        result=result,
    )
