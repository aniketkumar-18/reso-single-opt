"""Single-agent LangGraph for Reso.

Graph topology (sequential, no fan-out):

    START
      ▼
  context_hydration   ← parallel fetch: profile, history, memories (same as multi-agent)
      ▼
  wellness_agent      ← single ReAct agent with all 13 tools + unified system prompt
      ▼
     END

Key differences from the multi-agent version:
- No classify_route node — the agent decides what to do from context.
- No _join / fan-out — sequential execution, lower latency for simple messages.
- No aggregator — the agent's final response IS the output.
- Conversation history is injected directly into the agent's message list so
  multi-turn context is preserved without a separate routing step.
- Redis checkpointer still enables pod-crash recovery (same pattern).
"""

from __future__ import annotations

from typing import AsyncGenerator

from langgraph.graph import END, START, StateGraph

from src.agent.nodes.context_hydration import context_hydration_node
from src.agent.nodes.wellness_agent import wellness_agent_node
from src.agent.state import GraphState
from src.infra.logger import setup_logger

logger = setup_logger(__name__)


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("context_hydration", context_hydration_node)
    graph.add_node("wellness_agent", wellness_agent_node)

    graph.add_edge(START, "context_hydration")
    graph.add_edge("context_hydration", "wellness_agent")
    graph.add_edge("wellness_agent", END)

    return graph


def compile_graph(checkpointer):
    """Compile the graph with the given checkpointer. Call after asetup()."""
    graph = build_graph()
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("LangGraph single-agent compiled (context_hydration → wellness_agent)")
    return compiled


# ── Graph invocation helpers ───────────────────────────────────────────────────

def _build_initial_state(
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    user_message: str,
    image_url: str = "",
) -> GraphState:
    return {
        "user_id": user_id,
        "session_id": session_id,
        "conversation_id": conversation_id,
        "user_message": user_message,
        "image_url": image_url,
        # conversation_history intentionally omitted — LangGraph MemorySaver preserves
        # it from the previous turn so context_hydration_node can skip the Supabase fetch.
        # Falls back to Supabase on first turn (cold) or after pod restart (MemorySaver cleared).
        "user_profile": {},
        "memory_context": [],
        "graph_relations": [],
        "constraint_rules": [],
        "meal_items_context": [],
        "aggregated_response": "",
        "refresh_entities": [],
    }


async def invoke_graph(
    compiled_graph,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    user_message: str,
    image_url: str = "",
) -> GraphState:
    """Run the full graph and return the final state."""
    config = {"configurable": {"thread_id": session_id}}
    initial_state = _build_initial_state(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        user_message=user_message,
        image_url=image_url,
    )
    final_state: GraphState = await compiled_graph.ainvoke(initial_state, config=config)
    return final_state


async def stream_graph_events(
    compiled_graph,
    *,
    user_id: str,
    session_id: str,
    conversation_id: str,
    user_message: str,
    image_url: str = "",
) -> AsyncGenerator[tuple[str, dict], None]:
    """Yield (event_kind, event) tuples from the graph via astream_events v2."""
    config = {"configurable": {"thread_id": session_id}}
    initial_state = _build_initial_state(
        user_id=user_id,
        session_id=session_id,
        conversation_id=conversation_id,
        user_message=user_message,
        image_url=image_url,
    )
    async for event in compiled_graph.astream_events(
        initial_state, config=config, version="v2"
    ):
        yield event["event"], event
