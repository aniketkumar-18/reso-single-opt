"""AG-UI SSE endpoint — structured event streaming for the Android client.

POST /api/v1/agui/run
  Starts an agent run and returns an SSE stream of AG-UI events.
  Text streams as TextMessage events; tool calls emit CardEmit +
  ToolCallStart (WRITE) or CardEmit (READ) events.

POST /api/v1/agui/tool-result
  Resolves a pending tool call (Backend Tool Rendering pattern).
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from sse_starlette.sse import EventSourceResponse

from src.agui.events import (
    RunError,
    RunFinished,
    RunStarted,
    StepFinished,
    StepStarted,
    TextMessageContent,
    TextMessageEnd,
    TextMessageStart,
)
from src.agui.run_manager import create_run, get_run, remove_run, RunContext
from src.agui.tool_interceptor import agui_wrap_tools
from src.agent.nodes.context_hydration import context_hydration_node
from src.agent.nodes.wellness_agent import _build_messages
from src.agent.prompts import (
    WELLNESS_AGENT_SYSTEM,
    build_constraints_section,
    build_graph_relations_section,
    build_meal_context_section,
    build_memory_section,
    build_profile_section,
    build_response_format,
)
from src.gateway.middleware.auth import require_auth
from src.gateway.middleware.rate_limit import require_rate_limit
from src.gateway.schemas import AgUiRunRequest, AgUiToolResultRequest
from src.infra import db
from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.tools import ALL_TOOLS

logger = setup_logger(__name__)
router = APIRouter(prefix="/api/v1", tags=["agui"])


# ── SSE endpoint ─────────────────────────────────────────────────────────────

@router.post("/agui/run", response_model=None)
async def agui_run(
    request: Request,
    body: AgUiRunRequest,
    user_id: str = Depends(require_auth),
    _rate: None = Depends(require_rate_limit),
) -> EventSourceResponse:
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    conversation_id = body.conversation_id or str(uuid.uuid4())
    session_id = body.session_id or conversation_id

    await db.get_or_create_conversation(user_id, conversation_id)

    run_ctx = create_run(run_id, conversation_id)
    image_url = body.image_url or ""

    async def _run_agent() -> None:
        """Background task — runs hydration + agent with wrapped tools."""
        try:
            await run_ctx.emit(RunStarted(runId=run_id, threadId=conversation_id))

            # ── Step 1: Context hydration ─────────────────────────────
            await run_ctx.emit(StepStarted(runId=run_id, stepName="context_hydration"))
            hydration_state = {
                "user_id": user_id,
                "session_id": session_id,
                "conversation_id": conversation_id,
                "user_message": body.message,
                "image_url": image_url,
                "user_profile": {},
                "conversation_history": [],
                "memory_context": [],
                "graph_relations": [],
                "constraint_rules": [],
                "meal_items_context": [],
                "aggregated_response": "",
                "refresh_entities": [],
            }
            config = {"configurable": {
                "thread_id": session_id,
                "user_id": user_id,
                "session_id": session_id,
                "conversation_id": conversation_id,
            }}
            hydration_state = await context_hydration_node(hydration_state)
            await run_ctx.emit(StepFinished(runId=run_id, stepName="context_hydration"))

            # ── Step 2: Build agent with wrapped tools ────────────────
            await run_ctx.emit(StepStarted(runId=run_id, stepName="wellness_agent"))
            settings = get_settings()
            llm = ChatOpenAI(
                model=settings.llm_model_agent,
                temperature=0,
                api_key=settings.openai_api_key,
                streaming=True,
            ).with_config({"tags": ["final_response"]})

            system_prompt = WELLNESS_AGENT_SYSTEM.format(
                profile_section=build_profile_section(hydration_state.get("user_profile", {})),
                memory_section=build_memory_section(hydration_state.get("memory_context", [])),
                graph_relations_section=build_graph_relations_section(hydration_state.get("graph_relations", [])),
                constraints_section=build_constraints_section(hydration_state.get("constraint_rules", [])),
                meal_context_section=build_meal_context_section(hydration_state.get("meal_items_context", [])),
                response_format=build_response_format(),
            )

            # If a start_run action provided run_context with a tool_hint,
            # augment the system prompt so the agent reliably calls the right tool.
            if body.run_context:
                tool_hint = body.run_context.get("tool_hint", "")
                intent = body.run_context.get("intent", "")
                if tool_hint:
                    intent_desc = {
                        "re_edit": f"The user wants to re-edit their data. Call the {tool_hint} tool with their current data so they can review and update.",
                        "retry": f"The user wants to retry. Call the {tool_hint} tool again.",
                        "delete": f"The user wants to delete. Call the {tool_hint} tool to remove the item.",
                        "view": f"The user wants to view details. Call the {tool_hint} tool to fetch and display the data.",
                    }.get(intent, f"The user wants to use the {tool_hint} tool. Call it.")
                    system_prompt += f"\n\nACTION CONTEXT: {intent_desc}"
                    logger.info("run_context augmentation: tool_hint=%s intent=%s", tool_hint, intent)

            # Build a short profile summary for the LLM card generator
            profile = hydration_state.get("user_profile", {})
            profile_summary = ", ".join(
                f"{k}: {v}" for k, v in profile.items()
                if v is not None and v != "" and v != []
            ) or "no profile data"

            wrapped_tools = agui_wrap_tools(ALL_TOOLS, run_ctx, profile_context=profile_summary)
            agent = create_react_agent(model=llm, tools=wrapped_tools, prompt=system_prompt)

            messages = _build_messages(hydration_state)
            agent_config = {"configurable": {
                "user_id": user_id,
                "session_id": session_id,
                "conversation_id": conversation_id,
            }}

            # ── Step 3: Stream agent — emit text tokens ───────────────
            aggregated = ""
            msg_id = f"msg_{uuid.uuid4().hex[:8]}"
            text_started = False

            async for event in agent.astream_events(
                {"messages": messages}, config=agent_config, version="v2"
            ):
                kind = event.get("event", "")

                # Real token streaming
                if kind == "on_chat_model_stream" and "final_response" in event.get("tags", []):
                    chunk = event["data"]["chunk"]
                    content = chunk.content if hasattr(chunk, "content") else ""
                    tool_call_chunks = getattr(chunk, "tool_call_chunks", [])
                    if content and not tool_call_chunks:
                        if not text_started:
                            await run_ctx.emit(TextMessageStart(runId=run_id, messageId=msg_id))
                            text_started = True
                        await run_ctx.emit(TextMessageContent(runId=run_id, messageId=msg_id, delta=content))
                        aggregated += content

                # When the model finishes a text run and starts a new one (e.g.
                # between tool calls), close the current text message and mint a
                # fresh messageId so separate text blocks get distinct messages.
                if kind == "on_chat_model_end" and text_started:
                    await run_ctx.emit(TextMessageEnd(runId=run_id, messageId=msg_id))
                    text_started = False
                    msg_id = f"msg_{uuid.uuid4().hex[:8]}"

            # Close any trailing text stream
            if text_started:
                await run_ctx.emit(TextMessageEnd(runId=run_id, messageId=msg_id))

            # Fallback: chunk the aggregated text if no tokens were streamed
            if not aggregated:
                # Extract from agent messages
                try:
                    from src.agent.nodes.wellness_agent import _parse_agent_result
                    result_state = await agent.ainvoke(
                        {"messages": messages}, config=agent_config
                    )
                    final_content, _, _ = _parse_agent_result(result_state.get("messages", []))
                    aggregated = final_content
                except Exception:
                    pass

            if not text_started and aggregated:
                fb_id = f"msg_{uuid.uuid4().hex[:8]}"
                await run_ctx.emit(TextMessageStart(runId=run_id, messageId=fb_id))
                for i in range(0, len(aggregated), 30):
                    await run_ctx.emit(TextMessageContent(runId=run_id, messageId=fb_id, delta=aggregated[i:i+30]))
                await run_ctx.emit(TextMessageEnd(runId=run_id, messageId=fb_id))

            await run_ctx.emit(StepFinished(runId=run_id, stepName="wellness_agent"))
            await run_ctx.emit(RunFinished(runId=run_id))

            # Persist conversation — save the user's query and a concise
            # assistant summary. AG-UI tool interactions (form cards, tool
            # results) are NOT persisted as conversation history because they
            # flood the context and cause the agent to repeat tool-call
            # patterns on unrelated follow-up queries.
            await db.persist_message(conversation_id, "user", body.message)
            if aggregated:
                # Only persist the first ~500 chars of the text response to
                # keep history concise. The full card lifecycle is visible in
                # the hero view, not in history.
                summary = aggregated[:500]
                await db.persist_message(
                    conversation_id,
                    "assistant",
                    summary,
                    metadata={"refresh_entities": []},
                )
            # Trim old messages — keeps the last 10 user/assistant pairs so
            # stale profile-update exchanges don't bias future queries.
            await db.trim_conversation_history(conversation_id, keep_last=10)

        except Exception as exc:
            logger.exception("AG-UI run failed: user=%s run=%s", user_id, run_id)
            await run_ctx.emit(RunError(runId=run_id, message=str(exc)))
        finally:
            await run_ctx.close()
            remove_run(run_id)

    asyncio.create_task(_run_agent())

    return EventSourceResponse(
        run_ctx.iter_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


# ── Tool-result endpoint ─────────────────────────────────────────────────────

@router.post("/agui/tool-result")
async def agui_tool_result(
    body: AgUiToolResultRequest,
    user_id: str = Depends(require_auth),
) -> dict:
    """Resolve a pending tool call with the user's form submission."""
    run_ctx = get_run(body.run_id)
    if run_ctx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Run not found or already completed.",
        )
    resolved = run_ctx.resolve_tool_call(body.tool_call_id, body.result)
    if not resolved:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tool call not found or already resolved.",
        )
    return {"status": "ok"}
