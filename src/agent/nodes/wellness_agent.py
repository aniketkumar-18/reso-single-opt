"""Wellness Agent Node — single ReAct agent handling all three domains.

Replaces the classify_route + [nutrition_agent, fitness_agent, medical_agent]
+ aggregator pipeline from the multi-agent version.

Design decisions
────────────────
- ALL_TOOLS imported from src.tools (single registry, no duplication).
- Conversation history injected as Human/AI pairs for multi-turn context.
- LLM tagged "final_response" so the gateway SSE stream surfaces real tokens.
  Tool-calling steps carry tool_call_chunks, not plain content, so they are
  naturally filtered by the streaming code.
- _build_messages / _parse_agent_result are pure functions — easy to unit-test
  without running the full agent.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agent.prompts import (
    WELLNESS_AGENT_SYSTEM,
    build_constraints_section,
    build_graph_relations_section,
    build_meal_context_section,
    build_memory_section,
    build_profile_section,
    build_response_format,
)
from src.agent.state import GraphState
from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.infra.mem0_client import auto_save_conversation_memory
from src.tools import ALL_TOOLS

logger = setup_logger(__name__)


# ── Private helpers ────────────────────────────────────────────────────────────

def _build_messages(state: GraphState) -> list[BaseMessage]:
    """Build the agent message list from conversation history + current turn.

    Injects history as Human/AI pairs for multi-turn context, then appends
    the current user message — as a multimodal block when an image is present.
    """
    messages: list[BaseMessage] = []

    for entry in state.get("conversation_history", []):
        role = entry.get("role", "")
        content = entry.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    user_message: str = state.get("user_message", "")
    image_url: str = state.get("image_url", "")

    if image_url:
        messages.append(HumanMessage(content=[
            {"type": "text", "text": user_message or "Please analyse this image."},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]))
    else:
        messages.append(HumanMessage(content=user_message))

    return messages


def _parse_agent_result(
    result_messages: list[Any],
) -> tuple[str, list[str], list[str]]:
    """Extract structured output from a completed ReAct agent run.

    Returns:
        final_content:    Last AI message text that has no pending tool calls.
        tool_calls_made:  Names of every tool invoked during the loop.
        refresh_entities: UI refresh keys collected from ToolMessage JSON payloads.
    """
    tool_calls_made: list[str] = []
    final_content = ""
    refresh_entities: list[str] = []

    # Collect tool names from every AI message that carried tool calls.
    for msg in result_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls_made.extend(tc["name"] for tc in msg.tool_calls)

    # Final response = last AI message with no pending tool calls.
    for msg in result_messages:
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            final_content = str(msg.content)

    # Collect refresh keys from ToolMessage JSON payloads.
    for msg in result_messages:
        raw = getattr(msg, "content", "")
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
                if isinstance(payload, dict) and "refresh" in payload:
                    entity = payload["refresh"]
                    if entity not in refresh_entities:
                        refresh_entities.append(entity)
            except (json.JSONDecodeError, TypeError):
                pass

    return final_content, tool_calls_made, refresh_entities


# ── Node ───────────────────────────────────────────────────────────────────────

async def wellness_agent_node(state: GraphState, config: RunnableConfig) -> dict:
    """Single ReAct agent that handles nutrition, fitness, and medical domains."""
    user_id: str = state.get("user_id", "")
    session_id: str = state.get("session_id", "")
    conversation_id: str = state.get("conversation_id", "")
    user_message: str = state.get("user_message", "")

    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.llm_model_agent,
        temperature=0,
        api_key=settings.openai_api_key,
        streaming=True,
    ).with_config({"tags": ["final_response"]})

    system_prompt = WELLNESS_AGENT_SYSTEM.format(
        profile_section=build_profile_section(state.get("user_profile", {})),
        memory_section=build_memory_section(state.get("memory_context", [])),
        graph_relations_section=build_graph_relations_section(state.get("graph_relations", [])),
        constraints_section=build_constraints_section(state.get("constraint_rules", [])),
        meal_context_section=build_meal_context_section(state.get("meal_items_context", [])),
        response_format=build_response_format(),
    )

    agent = create_react_agent(model=llm, tools=ALL_TOOLS, prompt=system_prompt)

    agent_config: dict = {
        "configurable": {
            "user_id": user_id,
            "session_id": session_id,
            "conversation_id": conversation_id,
        }
    }

    try:
        result = await agent.ainvoke(
            {"messages": _build_messages(state)},
            config=agent_config,
        )

        final_content, tool_calls_made, refresh_entities = _parse_agent_result(
            result.get("messages", [])
        )

        logger.info(
            "Wellness agent done — tools=%s refresh=%s",
            tool_calls_made,
            refresh_entities,
        )

        # Feature 8 — auto-save wellness facts from conversation.
        # Non-blocking: capped at 15 s, never delays the streaming response.
        if final_content and user_id:
            asyncio.create_task(
                auto_save_conversation_memory(
                    user_message=user_message,
                    assistant_response=final_content,
                    user_id=user_id,
                    run_id=session_id or None,
                )
            )

        # Opt 3: append this turn to the state history so the MemorySaver checkpoint
        # carries it forward. context_hydration_node skips the Supabase fetch when
        # it finds a non-empty history here. Trimmed to CONTEXT_MESSAGE_LIMIT.
        updated_history: list[dict] = list(state.get("conversation_history") or [])
        updated_history.append({"role": "user", "content": user_message})
        if final_content:
            updated_history.append({"role": "assistant", "content": final_content})
        limit = settings.context_message_limit
        updated_history = updated_history[-limit:]

        return {
            "aggregated_response": final_content,
            "refresh_entities": refresh_entities,
            "conversation_history": updated_history,
        }

    except Exception:
        logger.exception("Wellness agent failed")
        return {
            "aggregated_response": "I encountered an error processing your request. Please try again.",
            "refresh_entities": [],
        }
