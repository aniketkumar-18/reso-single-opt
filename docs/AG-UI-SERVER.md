# AG-UI Server Architecture — Backend-Driven UI for Resonic

## Overview

The AG-UI server module (`src/agui/`) turns any LangChain tool call into a structured, interactive card experience on the Android client. The server drives **all** UI decisions — card layout, form fields, button behavior — through SSE events and LLM-generated `ComposeCard` JSON. The client is a generic renderer with no tool-specific code.

This document covers the server-side integration only. For the Android client and the full AG-UI protocol, see `docs/AG-UI-ARCHITECTURE.md` in the monorepo root.

---

## System Flow

```
User sends message
  │
  ▼
POST /api/v1/agui/run
  │
  ├─ 1. RunStarted + StepStarted (instant)
  │
  ├─ 2. Context hydration (profile, memory, constraints)
  │     StepFinished + StepStarted
  │
  ├─ 3. LangGraph agent runs with wrapped tools
  │     │
  │     ├─ WRITE tool called (e.g. update_user_profile)
  │     │   ├─ CardEmit: Loading card (instant skeleton)
  │     │   ├─ CardEmit: LLM-generated preview form (same cardId → replaces)
  │     │   ├─ ToolCallStart: agent SUSPENDS
  │     │   │     │
  │     │   │     ▼
  │     │   │   Client renders form, user edits, taps "Confirm"
  │     │   │     │
  │     │   │     ▼
  │     │   │   POST /api/v1/agui/tool-result (form data)
  │     │   │     │
  │     │   │     ▼
  │     │   ├─ Server merges form data → executes tool
  │     │   ├─ ToolCallEnd
  │     │   └─ CardEmit: LLM-generated complete card (same cardId → replaces)
  │     │
  │     ├─ READ tool called (e.g. get_meal_items)
  │     │   ├─ Tool executes immediately
  │     │   └─ CardEmit: LLM-generated result card
  │     │
  │     └─ SILENT tool (e.g. save_memory_fact)
  │         └─ Pass-through, no events
  │
  ├─ 4. Text streaming (final LLM response)
  │     TextMessageStart → TextMessageContent × N → TextMessageEnd
  │
  └─ 5. StepFinished + RunFinished
        SSE stream closes
```

---

## Module Map

```
src/agui/
├── __init__.py
├── events.py            # AG-UI event dataclasses + SSE serialization
├── run_manager.py       # RunContext: event queue + tool-call suspension futures
├── tool_interceptor.py  # Wraps tools with card emission + form merge + duplicate guard
├── card_schema.py       # LLM-driven ComposeCard generator (gpt-4o-mini)
└── card_templates.py    # Async wrappers for card generation modes

src/gateway/routes/
└── agui.py              # SSE endpoint + tool-result endpoint + history trimming

static/
└── agui.html            # Web client — served at /agui (vanilla JS, marked.js)
```

---

## Events (`events.py`)

All events are Python dataclasses serialized to JSON via `to_sse_string()`. The `type` field is injected automatically from the class name.

| Event               | Fields                                        | Purpose                                      |
| ------------------- | --------------------------------------------- | -------------------------------------------- |
| `RunStarted`        | `runId`, `threadId`                            | Agent run begins                             |
| `RunFinished`       | `runId`                                        | Run complete                                 |
| `RunError`          | `runId`, `message`                             | Run failed                                   |
| `StepStarted`       | `runId`, `stepName`                            | Workflow step begins                         |
| `StepFinished`      | `runId`, `stepName`                            | Step complete                                |
| `TextMessageStart`  | `runId`, `messageId`                           | Text streaming begins                        |
| `TextMessageContent`| `runId`, `messageId`, `delta`                  | Text chunk                                   |
| `TextMessageEnd`    | `runId`, `messageId`                           | Text complete                                |
| `CardEmit`          | `runId`, `cardId`, `cardJson`                  | Card created or replaced (same cardId = update) |
| `ToolCallStart`     | `runId`, `toolCallId`, `toolName`, `args`      | Agent suspends, client shows form            |
| `ToolCallEnd`       | `runId`, `toolCallId`                          | Tool call resolved                           |

### Wire Format

```
data: {"type":"CardEmit","runId":"run_abc","cardId":"card_meal_1","cardJson":"{...}"}

data: {"type":"ToolCallStart","runId":"run_abc","toolCallId":"tc_1","toolName":"log_meal","args":{...}}
```

`sse_starlette` wraps each string with `data:` and `\n\n` automatically.

---

## Run Manager (`run_manager.py`)

### `RunContext`

Each `POST /api/v1/agui/run` creates a `RunContext` that coordinates between:
- **Agent task** (background `asyncio.create_task`) — pushes events into a queue
- **SSE generator** (`iter_events()`) — reads from the queue and yields to the client

```python
class RunContext:
    _event_queue: asyncio.Queue[str | None]      # None = sentinel to close SSE
    _pending_tool_calls: dict[str, asyncio.Future] # tool_call_id → Future

    async def emit(event) → None         # Push event into SSE queue
    async def close() → None             # Send sentinel to close stream
    async def iter_events() → AsyncIterator[str]  # SSE generator
    async def await_tool_result(tool_call_id) → dict  # Block until client POSTs
    def resolve_tool_call(tool_call_id, result) → bool  # Complete the future
```

### Tool Call Suspension

WRITE tools suspend the agent using `asyncio.Future`:

1. `_wrap_write` calls `await run_ctx.await_tool_result(tc_id)` — creates a Future and waits
2. Client sees the form card, user edits, taps "Confirm"
3. Client POSTs to `/api/v1/agui/tool-result` with `{run_id, tool_call_id, result}`
4. Route handler calls `run_ctx.resolve_tool_call(tool_call_id, result)` — sets the Future result
5. `await_tool_result` returns the form data — agent resumes

**Timeout:** `TOOL_CALL_TIMEOUT = 300` seconds (5 minutes). If the client doesn't respond, the Future raises `TimeoutError` and the tool is marked as skipped. The Android client's SSE read timeout is set to 6 minutes to exceed this.

### Global Registry

```python
_active_runs: dict[str, RunContext]

create_run(run_id, thread_id) → RunContext   # Creates and registers
get_run(run_id) → RunContext | None           # Lookup for tool-result endpoint
remove_run(run_id) → None                     # Cleanup after run completes
```

---

## Tool Interceptor (`tool_interceptor.py`)

### Tool Classification

Tools are classified into three categories that determine their AG-UI behavior:

| Category | Behavior | Examples |
| -------- | -------- | -------- |
| **WRITE** | 3-stage card lifecycle (loading → form → complete) + agent suspension | `log_meal`, `update_user_profile`, `log_workout` |
| **READ** | Execute immediately, emit result card | `get_meal_items`, `calculate_macro_targets` |
| **SILENT** | Pass-through, no card events | `save_memory_fact` |

> **Known gap:** Classification is currently via hardcoded `frozenset`s. A future improvement would read classification from tool metadata.

### `agui_wrap_tools(tools, run_ctx, profile_context)`

Returns wrapped copies of all tools. WRITE and READ tools get AG-UI wrappers; SILENT and unclassified tools pass through unchanged.

### WRITE Tool Lifecycle

```python
async def _invoke(**kwargs):
    # Duplicate guard: if this tool already executed in this run,
    # return the cached result. Prevents the LLM from showing a
    # second form with stale defaults.
    if _last_result is not None:
        return _last_result

    # Stage 1: Loading card (instant, hardcoded skeleton)
    emit(CardEmit(cardId, loading_card()))

    # Stage 2: LLM-generated preview with interactive form
    preview_json = await card_templates.for_write_preview(tool_name, args, profile_context)
    emit(CardEmit(cardId, preview_json))  # Same cardId → replaces Stage 1

    # Agent suspends: ToolCallStart
    emit(ToolCallStart(toolCallId, toolName, args))
    user_result = await run_ctx.await_tool_result(toolCallId)

    # Merge user form edits into tool kwargs
    merged_kwargs = {**kwargs}
    merged_kwargs.pop("cardId", None)  # Remove UI-only metadata
    for key, value in user_result.items():
        if key not in ("skipped",) and value is not None and value != "":
            merged_kwargs[key] = value

    # Type coercion: if tool schema expects list[str] but form sent a
    # comma/newline-separated string, split it into a list.
    for field_name, field_info in original.args_schema.model_fields.items():
        if field_name in merged_kwargs and "list" in str(field_info.annotation).lower():
            val = merged_kwargs[field_name]
            if isinstance(val, str):
                merged_kwargs[field_name] = [s.strip() for s in re.split(r'[,\n]', val) if s.strip()]

    # Execute actual tool with merged kwargs
    result = await original.ainvoke(merged_kwargs)

    # Cache result for duplicate guard
    _last_result = result

    # ToolCallEnd + Stage 3: Complete card
    emit(ToolCallEnd(toolCallId))
    complete_json = await card_templates.for_write_complete(tool_name, result)
    emit(CardEmit(cardId, complete_json))  # Same cardId → replaces Stage 2
```

### Duplicate Call Guard

The LLM agent sometimes calls the same WRITE tool twice in one run. The second call would show a new form with stale defaults (from context hydration, before the first call updated the DB), confusing the user into thinking nothing was saved.

The guard: `_last_result` (scoped per tool per run) caches the first successful result. Subsequent invocations return the cache immediately — no loading card, no form, no suspension. The agent sees the same `{"status": "saved", ...}` result and moves on to its text summary.

### Form Data Merge

The critical step where client form edits become tool parameters:

1. Start with LLM's original proposed args (`kwargs`)
2. Remove `cardId` (UI tracking metadata, not a tool param)
3. Overlay each key from `user_result` where value is non-null and non-empty
4. **Type coercion**: for each field, check the tool's Pydantic `args_schema`. If the schema expects `list[str]` but the form sent a comma/newline-separated string, split it: `re.split(r'[,\n]', val)`. This handles `text_input` fields used for list data (goals, allergies, conditions, etc.)
5. Pass merged dict to `original.ainvoke()`

**Contract:** Form element `id` fields in the ComposeCard JSON MUST match tool parameter names exactly. This is enforced by the LLM prompt instruction in `card_schema.py`, not by runtime validation.

### Tool Result with Updated Values

WRITE tools return `updated_values` alongside `updated_fields` so the LLM's summary text references the actual saved data — not the stale context hydration snapshot from run start:

```python
return {
    "status": "saved",
    "updated_fields": ["weight_kg", "date_of_birth", "allergies"],
    "updated_values": {"weight_kg": 93.16, "date_of_birth": "1986-05-05", "allergies": "peanuts, gluten"},
    "refresh": "user-profile",
}
```

### READ Tool Lifecycle

```python
async def _invoke(**kwargs):
    result = await original.ainvoke(kwargs)
    card_json = await card_templates.for_read_result(tool_name, result)
    emit(CardEmit(cardId, card_json))
    return result
```

No suspension, no form, no user interaction — just a display card.

---

## Card Schema (`card_schema.py`)

### LLM-Driven Card Generation

All cards (except the loading skeleton) are generated by calling `gpt-4o-mini` with a structured prompt. The LLM sees:
- The **element palette** (all supported UI element types + their JSON schemas)
- The **tool name, description, and proposed args** (or result)
- **Mode-specific rules** (preview/complete/read_result)

### Three Generation Modes

| Mode | Prompt | Purpose | Has `detailElements`? | Has `actions`? |
| ---- | ------ | ------- | --------------------- | -------------- |
| `preview` | `_PREVIEW_SYSTEM` | Interactive form for WRITE tool | Yes (full form) | Yes (Confirm + Skip) |
| `complete` | `_COMPLETE_SYSTEM` | Read-only result after execution | No | No |
| `read_result` | `_READ_SYSTEM` | Display-only card for READ tool | No | No |

### Element Palette

The `_ELEMENT_PALETTE` prompt teaches the LLM all 17 available element types:

**Interactive (for `detailElements`):** `radio_group`, `chip_group`, `slider`, `stepper`, `text_input`, `segmented_control`, `toggle`, `star_rating`, `checklist`

**Display (for `elements`):** `metric_row`, `progress`, `key_value_list`, `section`, `status_banner`, `text`, `data_table`, `divider`

### Action Handler Schema

The palette also teaches the LLM the server-driven action format:

```json
{"label": "Confirm", "style": "primary", "handler": {"kind": "resolve_tool_call", "collectForm": true}}
{"label": "Skip", "style": "secondary", "handler": {"kind": "reject_tool_call"}}
```

Available handler kinds: `resolve_tool_call`, `reject_tool_call`, `dismiss`, `navigate_back`, `open_sheet`, `open_url`, `send_message`

Style values: `primary`, `secondary`, `destructive`, `link`

### Form ID → Tool Parameter Contract

The `_PREVIEW_SYSTEM` prompt includes this critical instruction:

> Every interactive element in "detailElements" MUST have an "id" field that EXACTLY MATCHES the corresponding tool parameter name (e.g. if the tool accepts "weight_kg", the slider's id must be "weight_kg").

This ensures the client's form snapshot keys align with the tool's Pydantic schema, so the server-side merge produces valid tool kwargs.

### Fallback

If LLM generation fails, `_fallback_card()` produces a minimal generic card:
- Preview: key-value list of proposed args + Confirm/Skip buttons
- Complete: status_banner (success/skipped)
- Read result: key-value list of result fields

---

## Card Templates (`card_templates.py`)

Thin async wrappers around `card_schema.generate_card()`:

| Function | Mode | When Called |
| -------- | ---- | ---------- |
| `loading_card(tool_name)` | — (static) | Immediately when WRITE tool starts |
| `for_write_preview(tool_name, args, profile_context)` | `preview` | After loading card, before suspension |
| `for_write_complete(tool_name, result)` | `complete` | After tool execution |
| `for_read_result(tool_name, result)` | `read_result` | After READ tool execution |

`loading_card` is the only non-LLM card — a hardcoded skeleton with a progress bar for instant feedback while the LLM generates the real form.

---

## Gateway Routes (`gateway/routes/agui.py`)

### `POST /api/v1/agui/run`

1. Creates a `RunContext` and registers it in the global registry
2. Spawns a background task (`asyncio.create_task(_run_agent())`)
3. Returns an `EventSourceResponse` that reads from the run's event queue

The background task:
1. Hydrates context (profile, memory, conversation history, constraints)
2. Builds a `create_react_agent` with all tools wrapped via `agui_wrap_tools`
3. Streams agent events — text tokens emit `TextMessage*`, tool calls handled by wrappers
4. On completion: persists messages to DB, emits `RunFinished`, closes the stream

### `POST /api/v1/agui/tool-result`

```python
@router.post("/agui/tool-result")
async def agui_tool_result(body: AgUiToolResultRequest, user_id: str = Depends(require_auth)):
    run_ctx = get_run(body.run_id)
    if run_ctx is None:
        raise HTTPException(404, "Run not found or already completed.")
    resolved = run_ctx.resolve_tool_call(body.tool_call_id, body.result)
    if not resolved:
        raise HTTPException(404, "Tool call not found or already resolved.")
    return {"status": "ok"}
```

**Request body:**
```json
{
  "run_id": "run_abc123",
  "tool_call_id": "tc_def456",
  "result": {
    "weight_kg": 80.0,
    "date_of_birth": "1990-05-15",
    "allergies": ["Peanuts", "Tree nuts"],
    "goals": "Lose weight, Workout 4x/week"
  }
}
```

---

## End-to-End Data Flow: WRITE Tool

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Android Client                              │
│                                                                     │
│  1. User types "update my profile"                                  │
│  2. POST /api/v1/agui/run → SSE stream opens                       │
│  3. Receives RunStarted → opens Hero surface                       │
│  4. Receives CardEmit (loading) → shows skeleton                   │
│  5. Receives CardEmit (preview) → shows form with sliders,         │
│     text inputs, toggles — all with id fields                      │
│  6. Receives ToolCallStart → agent is suspended                    │
│  7. User edits form fields (DOB, allergies, etc.)                  │
│     CardFormState captures every edit via onValueChange             │
│  8. User taps "Confirm & Log"                                      │
│     → handler.kind = "resolve_tool_call", collectForm = true       │
│     → formState.snapshot() captures all current values              │
│     → toJsonElement() preserves Float/Int/Boolean/List types        │
│     → POST /api/v1/agui/tool-result with typed JSON payload        │
│  9. Receives ToolCallEnd → form collapses                          │
│ 10. Receives CardEmit (complete) → shows "Saved successfully"      │
│ 11. Receives RunFinished → "Done" button appears                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           Server                                    │
│                                                                     │
│  1. /agui/run: create RunContext, spawn background agent task       │
│  2. Hydrate context (profile, memory, constraints)                 │
│  3. Agent decides to call update_user_profile(weight_kg=75, ...)   │
│  4. _wrap_write intercepts:                                        │
│     a. emit CardEmit(loading) — instant feedback                   │
│     b. LLM generates preview form → emit CardEmit(preview)        │
│     c. emit ToolCallStart → create Future, await                   │
│  5. /agui/tool-result: resolve Future with client's form data      │
│  6. _wrap_write resumes:                                           │
│     a. Merge user_result into kwargs (pop cardId, overlay edits)   │
│     b. Execute update_user_profile with merged kwargs              │
│     c. emit ToolCallEnd                                            │
│     d. LLM generates complete card → emit CardEmit(complete)       │
│  7. Agent generates final text response                            │
│  8. emit RunFinished, close SSE stream                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Conversation Context Management

### History Windowing

The agent's context is built from conversation history + the current user message. To prevent stale tool-call exchanges from biasing the agent:

- **`_MAX_HISTORY_MESSAGES = 6`** (`wellness_agent.py`) — `_build_messages` only includes the last 6 messages from the DB as history. The current user message is always appended as the final `HumanMessage`.
- **`trim_conversation_history(keep_last=10)`** (`db.py`) — after each run, old messages beyond the last 10 are deleted from the DB. Prevents unbounded growth.
- **Message truncation** — persisted assistant messages are capped at 500 characters. Full card lifecycles (Loading → Form → Complete) are visible in the hero view, not in conversation history.

### Why Cards Don't Appear for All Queries

Cards (`CardEmit` events) only appear when the agent calls a tool:
- **WRITE tool** (e.g. `log_meal`, `update_user_profile`) → 3-stage card lifecycle
- **READ tool** (e.g. `get_meal_items`, `calculate_macro_targets`) → result card

Informational queries ("Compare chicken breast and tofu", "What's in a banana?") produce **text-only** responses — the agent answers directly with markdown via `TextMessage*` events. No tool call = no card.

## Web Client

A single-file web client at `static/agui.html` serves the full AG-UI experience in a browser at `/agui`. It replicates the Android flow with:

- SSE streaming via `fetch` + `ReadableStream`
- Markdown rendering via `marked.js` for text messages
- ComposeCard JSON rendering for cards (all 19 element types)
- Form state capture + tool result POST
- All 8 `ActionHandler` kinds
- Confirmation dialogs, toast feedback, dark theme

## Configuration

| Config | Location | Value | Purpose |
| ------ | -------- | ----- | ------- |
| `TOOL_CALL_TIMEOUT` | `run_manager.py` | 300s (5 min) | Auto-cancel tool calls if client doesn't respond |
| `_MAX_HISTORY_MESSAGES` | `wellness_agent.py` | 6 | Max conversation history messages sent to agent |
| `keep_last` | `db.trim_conversation_history` | 10 | Max messages retained in DB per conversation |
| `llm_model_pipeline` | `config.py` / `.env` | gpt-4o-mini | LLM model for card generation |
| `llm_model_agent` | `config.py` / `.env` | (configurable) | LLM model for the main agent |
| `WRITE_TOOLS` | `tool_interceptor.py` | 12 tools | Tools that get form cards + suspension |
| `READ_TOOLS` | `tool_interceptor.py` | 2 tools | Tools that get result cards |
| `SILENT_TOOLS` | `tool_interceptor.py` | 1 tool | Tools with no card events |

---

## Extension Guide

### Adding a new tool

1. Create the tool in `src/tools/`
2. Add its name to `WRITE_TOOLS`, `READ_TOOLS`, or `SILENT_TOOLS` in `tool_interceptor.py`
3. Done — the LLM generates form/result cards automatically based on the tool's name, description, and args

### Adding a new card element type

1. Add the type to `_ELEMENT_PALETTE` in `card_schema.py`
2. Add the corresponding `CardElement` subtype + renderer on Android
3. The LLM will start using it in generated cards

### Customizing card generation

Modify the system prompts in `card_schema.py`:
- `_PREVIEW_SYSTEM` — rules for interactive form cards
- `_COMPLETE_SYSTEM` — rules for post-execution result cards
- `_READ_SYSTEM` — rules for read-only display cards

### Adding a new action handler

1. Add the `@Serializable` `ActionHandler` subclass on Android (`ComposeCardModels.kt`)
2. Add dispatch in `executeHandler()` (`ChatFullScreenViewModel.kt`)
3. Document the `kind` value in `_ELEMENT_PALETTE` actions section
4. The LLM can start emitting it in card actions

---

## Implementation Status

All handler kinds are fully implemented end-to-end:

| Handler | Server | Android | Notes |
| ------- | ------ | ------- | ----- |
| `resolve_tool_call` | Form merge + type coercion + duplicate guard | `executeHandler` → `agentRepository.resolveToolCall()` | Live edit capture, `toJsonElement()` type preservation, `list[str]` auto-split |
| `start_run` | Receives `run_context` → augments agent prompt with `tool_hint` + `intent` | `executeHandler` → `runAgentWithContext()` → new SSE run with context | Re-edit, retry, delete flows from complete cards |
| `reject_tool_call` | Tool skipped, agent resumes | `executeHandler` → `agentRepository.rejectToolCall()` | |
| `api_call` | N/A (client-direct) | `executeApiCall()` with JSON body, full HTTP method dispatch, confirmation dialog, toast feedback | `onSuccess`: dismiss, navigate_back, toast |
| `dismiss` | N/A | Clears action sheet stack | |
| `navigate_back` | N/A | Pops sheet stack or closes hero | |
| `open_sheet` | N/A | Pushes `action.sheet` onto stack | |
| `open_url` | N/A | `_pendingUrl` → UI launches Intent | |
| `send_message` | N/A | `viewModel.send()` or populate input | `autoSend` flag |

## Remaining Improvements

| Gap | Severity | Description |
| --- | -------- | ----------- |
| Hardcoded tool classification | Medium | `WRITE_TOOLS`/`READ_TOOLS` frozensets should be tool metadata |
| No form → tool schema validation | Medium | Element IDs must match tool params by LLM convention, not runtime check |
| LLM schema introspection | Medium | `_PREVIEW_SYSTEM` should extract field info from tool Pydantic schemas dynamically |
| Wellness-specific prompt guidance | Low | `_PREVIEW_SYSTEM` mentions "weight, height, goals" specifically |
| Empty-string filtering in merge | Low | `value != ""` filter may discard intentionally empty values |
| Loading card is static | Low | Hardcoded skeleton, not LLM-generated |
