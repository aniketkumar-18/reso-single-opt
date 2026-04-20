"""AG-UI run lifecycle manager.

Each ``POST /api/v1/agui/run`` spawns a ``RunContext`` that coordinates
between the background agent task (which pushes events into an asyncio queue)
and the SSE generator (which reads from that queue and yields to the client).

Tool-call suspension is implemented via per-call ``asyncio.Future``s: the
WRITE-tool wrapper awaits its future, and the ``/agui/tool-result`` endpoint
completes it with the user's form submission.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from src.agui.events import AgUiEvent, to_sse_string
from src.infra.logger import setup_logger

logger = setup_logger(__name__)

# Timeout after which a pending tool call auto-cancels (seconds).
TOOL_CALL_TIMEOUT = 300  # 5 minutes


class RunContext:
    """Per-run state shared between the agent task and the SSE generator."""

    def __init__(self, run_id: str, thread_id: str) -> None:
        self.run_id = run_id
        self.thread_id = thread_id
        self._event_queue: asyncio.Queue[dict[str, str] | None] = asyncio.Queue()
        self._pending_tool_calls: dict[str, asyncio.Future[dict[str, Any]]] = {}

    # ── Event emission (called by agent task / tool wrappers) ─────────

    async def emit(self, event: AgUiEvent) -> None:
        """Push an event into the SSE queue."""
        sse_str = to_sse_string(event)
        logger.info("AG-UI emit run=%s type=%s", self.run_id, type(event).__name__)
        await self._event_queue.put(sse_str)

    async def close(self) -> None:
        """Signal end-of-stream to the SSE generator."""
        await self._event_queue.put(None)

    # ── SSE generator (consumed by the route handler) ─────────────────

    async def iter_events(self) -> AsyncIterator[str]:
        """Yield raw SSE strings until the sentinel ``None`` arrives."""
        while True:
            item = await self._event_queue.get()
            if item is None:
                logger.info("AG-UI SSE generator: sentinel received, closing run=%s", self.run_id)
                break
            logger.info("AG-UI SSE yield run=%s len=%d", self.run_id, len(item))
            yield item

    # ── Tool-call suspension (WRITE-tool wrappers) ────────────────────

    async def await_tool_result(self, tool_call_id: str) -> dict[str, Any]:
        """Block until the client POSTs to ``/agui/tool-result``.

        Raises ``asyncio.TimeoutError`` after ``TOOL_CALL_TIMEOUT`` seconds.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending_tool_calls[tool_call_id] = future
        try:
            return await asyncio.wait_for(future, timeout=TOOL_CALL_TIMEOUT)
        finally:
            self._pending_tool_calls.pop(tool_call_id, None)

    def resolve_tool_call(self, tool_call_id: str, result: dict[str, Any]) -> bool:
        """Complete a pending tool-call future with the client's result.

        Returns ``True`` if the tool call was found and resolved.
        """
        future = self._pending_tool_calls.get(tool_call_id)
        if future is not None and not future.done():
            logger.info("resolve_tool_call: tc=%s result_keys=%s payload=%s", tool_call_id, list(result.keys()), result)
            future.set_result(result)
            return True
        logger.warning("resolve_tool_call: tc=%s not found or already done", tool_call_id)
        return False


# ── Global run registry ──────────────────────────────────────────────────────

_active_runs: dict[str, RunContext] = {}


def create_run(run_id: str, thread_id: str) -> RunContext:
    ctx = RunContext(run_id, thread_id)
    _active_runs[run_id] = ctx
    return ctx


def get_run(run_id: str) -> RunContext | None:
    return _active_runs.get(run_id)


def remove_run(run_id: str) -> None:
    _active_runs.pop(run_id, None)
