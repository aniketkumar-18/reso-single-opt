"""AG-UI protocol layer — agent-to-UI event streaming.

Adds structured card events, tool-call suspension, and backend tool rendering
on top of the existing LangGraph wellness agent. The core agent graph and tools
remain unmodified; this package wraps them with an event-emitting adapter.
"""
