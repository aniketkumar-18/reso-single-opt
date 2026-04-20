"""Gateway request / response schemas.

Keeping Pydantic models here (separate from route handlers) lets other
modules import them without pulling in FastAPI routing machinery.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="The user's message.")
    conversation_id: str | None = Field(
        None,
        description="Existing conversation UUID. A new one is created if omitted.",
    )
    session_id: str | None = Field(
        None,
        description=(
            "Client session UUID used as LangGraph thread_id for checkpointing. "
            "Defaults to conversation_id."
        ),
    )
    image_url: str | None = Field(
        None,
        description=(
            "Optional image attached by the user. "
            "Accepts a public https:// URL or a data:image/<type>;base64,<data> string. "
            "Supported formats: JPEG, PNG, WebP, GIF. Max 15 MB."
        ),
    )
    stream: bool = Field(True, description="Set false to get a single JSON response.")
    workflow_mode: Literal["single"] = Field(
        "single",
        description='"single" — unified single wellness agent (all tools, one ReAct loop).',
    )


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    session_id: str
    refresh_entities: list[str]
    trace_id: str
    workflow_mode: str


# ── AG-UI protocol schemas ───────────────────────────────────────────────────

class AgUiRunRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096, description="The user's message.")
    conversation_id: str | None = Field(None, description="Existing conversation UUID.")
    session_id: str | None = Field(None, description="Client session UUID.")
    image_url: str | None = Field(None, description="Optional image attachment.")
    run_context: dict[str, str] | None = Field(
        None,
        description="Optional context from a start_run action (tool_hint, intent, entity_ref).",
    )


class AgUiToolResultRequest(BaseModel):
    run_id: str = Field(..., description="The AG-UI run that owns the pending tool call.")
    tool_call_id: str = Field(..., description="Tool call to resolve.")
    result: dict = Field(default_factory=dict, description="Form values submitted by the user.")
