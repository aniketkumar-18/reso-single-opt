"""AG-UI event types and SSE serialisation helpers.

Each event serialises to two W3C SSE fields:
  event: <EventType>
  data: <json payload>

The Android ``SseLineParser`` dispatches on the ``event:`` field before
parsing the JSON ``data:``, which means the client never needs to inspect
the JSON to determine the event type.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


# ── Lifecycle ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunStarted:
    runId: str
    threadId: str

@dataclass(frozen=True)
class RunFinished:
    runId: str

@dataclass(frozen=True)
class RunError:
    runId: str
    message: str

@dataclass(frozen=True)
class StepStarted:
    runId: str
    stepName: str

@dataclass(frozen=True)
class StepFinished:
    runId: str
    stepName: str


# ── Text streaming ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TextMessageStart:
    runId: str
    messageId: str

@dataclass(frozen=True)
class TextMessageContent:
    runId: str
    messageId: str
    delta: str

@dataclass(frozen=True)
class TextMessageEnd:
    runId: str
    messageId: str


# ── Card lifecycle ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CardEmit:
    runId: str
    cardId: str
    cardJson: str  # Escaped ComposeCard JSON — opaque to the protocol layer


# ── Tool calls ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCallStart:
    runId: str
    toolCallId: str
    toolName: str
    args: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ToolCallEnd:
    runId: str
    toolCallId: str


# ── Union type ───────────────────────────────────────────────────────────────

AgUiEvent = (
    RunStarted
    | RunFinished
    | RunError
    | StepStarted
    | StepFinished
    | TextMessageStart
    | TextMessageContent
    | TextMessageEnd
    | CardEmit
    | ToolCallStart
    | ToolCallEnd
)


# ── Serialisation ────────────────────────────────────────────────────────────

def to_sse(event: AgUiEvent) -> dict[str, str]:
    """Convert an AG-UI event to an ``sse_starlette``-compatible dict."""
    event_type = type(event).__name__
    payload = asdict(event)
    payload["type"] = event_type
    return {"data": json.dumps(payload, default=str)}


def to_sse_string(event: AgUiEvent) -> str:
    """Convert an AG-UI event to a JSON string for ``EventSourceResponse``.

    ``sse_starlette`` wraps plain strings with ``data:`` and ``\\n\\n``
    automatically, so we return just the JSON payload.
    """
    event_type = type(event).__name__
    payload = asdict(event)
    payload["type"] = event_type
    return json.dumps(payload, default=str)
