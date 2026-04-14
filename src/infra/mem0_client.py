"""Memory module — Mem0 AsyncMemory with Qdrant vector store + Neo4j graph memory.

Architecture:
- Vector store : Qdrant Cloud — semantic similarity search with enhanced metadata filtering
- Graph store  : Neo4j Aura — entity/relationship extraction for multi-hop recall
- LLM          : OpenAI (pipeline model) — fact extraction, memory updates, reranking
- Embedder     : OpenAI text-embedding-ada-002 — 1536-dim vectors
- Reranker     : OpenAI gpt-4o-mini (LLM reranker) — second-pass relevance scoring

Features implemented:
  Feature 1 — Graph Memory             : Neo4j entity/relationship extraction on every add
  Feature 2 — Metadata Filtering       : Domain-aware Qdrant filters reduce noise in context
  Feature 3 — Reranker-Enhanced Search : LLM reranker reorders vector hits by true relevance
  Feature 4 — Async Memory Operations  : Retry+timeout, timed logging, run_id scoping, full CRUD
  Feature 5 — Multimodal Support       : Extract and store memories from food/workout/medical images
  Feature 6 — Custom Fact Extraction   : Wellness-specific prompt filters out greetings/requests;
                                         only persistent health facts (nutrition/fitness/medical/
                                         general) reach the vector store. Applied via Mem0 config
                                         AND as a standalone LLM validator for the Supabase fallback.
  Feature 7 — Custom Update Memory     : Wellness-aware ADD/UPDATE/DELETE/NONE reconciliation prompt
                                         replaces Mem0's generic updater. Handles body-metric
                                         supersession, medication/condition lifecycle (stop → DELETE),
                                         preference flips (UPDATE), and semantic dedup (NONE).
                                         Applied via Mem0 config AND resolve_memory_actions() helper
                                         that gives the Supabase fallback the same reconciliation logic.
  Feature 8 — OpenAI Compatibility     : Mem0 proxy client (`mem0.proxy.main.Mem0`) used as a
                                         drop-in OpenAI replacement for auto-save of wellness facts
                                         from every conversation turn. Fills the gap when the agent
                                         doesn't call save_memory_fact explicitly. Proxy uses Qdrant
                                         only (lighter than full AsyncMemory) + Feature 6/7 prompts.
                                         Wired into wellness_agent_node as a non-critical post-agent
                                         auto-save step. Falls back gracefully if unconfigured.
  Feature 9 — Configure OSS Stack      : Production-grade config tuning: configurable collection
                                         name (tenant isolation + retention), reranker top_k from
                                         settings (default 10, docs recommend 10-20). Startup
                                         validate_mem0_config() probes each component (Qdrant, LLM,
                                         embedder) and logs per-component status. get_mem0_config_
                                         summary() surfaces config info in health checks without
                                         exposing secrets. Wired into app lifespan + /health endpoint.
  Feature 13 — Configurable Reranker    : Reranker provider is now selectable via
                                         MEM0_RERANKER_PROVIDER. _build_reranker_config()
                                         produces the correct Mem0 reranker block for:
                                         llm_reranker, cohere, sentence_transformer,
                                         huggingface, zero_entropy, none (disables).
                                         Includes WELLNESS_RERANKER_SCORING_PROMPT — a
                                         domain-specific LLM scoring prompt that weighs
                                         health facts, recency, actionability, and domain
                                         relevance for wellness queries. Provider-specific
                                         settings honour docs-recommended candidate sizes
                                         (cohere≤100, sentence_transformer≤50, etc.) and
                                         batch_size / device / top_k tuning params from
                                         MEM0_RERANKER_MODEL / MEM0_RERANKER_TOP_K.
  Feature 12 — Configurable Embedder    : Embedding provider is now selectable via
                                         MEM0_EMBEDDER_PROVIDER. _build_embedder_config()
                                         produces the correct Mem0 {"provider", "config"}
                                         block for 9 providers: openai, azure_openai, ollama,
                                         huggingface, google_ai, vertexai, together, lmstudio,
                                         aws_bedrock. MEM0_EMBEDDER_MODEL overrides the
                                         provider default. MEM0_EMBEDDER_DIMS must match the
                                         model's output dimensions (default 1536). Changing
                                         provider/dims requires re-indexing. Both AsyncMemory
                                         and proxy share the same configured embedder.
  Feature 11 — Configurable Vector DB   : Vector store is now selectable via MEM0_VECTOR_STORE_PROVIDER.
                                         _build_vector_store_config() produces the correct Mem0
                                         {"provider", "config"} block for: qdrant, pgvector/supabase,
                                         chroma, pinecone. Chroma runs embedded (CHROMA_PATH) or in
                                         client-server mode (CHROMA_HOST + CHROMA_PORT). Pinecone
                                         requires PINECONE_API_KEY + PINECONE_INDEX_NAME. All share
                                         the same collection name and embedding dims (1536). Both
                                         AsyncMemory and proxy use the same configured store.
                                         is_mem0_configured / is_proxy_configured updated to be
                                         provider-aware. validate_mem0_config() probes the configured
                                         vector store instead of always Qdrant. Config summary
                                         reflects actual provider.
  Feature 10 — Configurable LLM        : Mem0 memory operations (extraction, update, graph) can use
                                         a dedicated LLM provider independently from the wellness
                                         agent LLM. _build_llm_config() returns the correct Mem0
                                         {"provider", "config"} block for any of 8 supported
                                         providers: openai, openai_structured, azure_openai,
                                         anthropic, groq, ollama, litellm, aws_bedrock.
                                         Controlled via MEM0_LLM_PROVIDER / MEM0_LLM_MODEL /
                                         MEM0_LLM_TEMPERATURE / MEM0_LLM_MAX_TOKENS /
                                         MEM0_LLM_BASE_URL env vars. AWS Bedrock additionally
                                         reads AWS_REGION / AWS_ACCESS_KEY_ID /
                                         AWS_SECRET_ACCESS_KEY. Defaults preserve existing
                                         behaviour (openai + llm_model_pipeline). Config precedence:
                                         explicit config > env vars > defaults (per Mem0 docs).

Async operation standards (Feature 4):
  - All ops use _async_retry: 3 attempts, exponential backoff (1s → 2s → 4s), 10s timeout each
  - All ops are timed and logged with duration for production observability
  - run_id scoping: session_id passed as run_id for per-session memory grouping and audit
  - Singleton AsyncMemory reused per process — no reconnects per request
  - Full CRUD API: add, search, get_all, get, update, delete, delete_all, history

Fallback: if Qdrant/Neo4j not configured → plain Supabase storage with SQL domain filter.

Public API:
    add_memory(fact, user_id, domain, run_id)
    search_memories(query, user_id, limit, domains, run_id)
    search_memories_with_graph(query, user_id, limit, domains, filters, rerank, run_id)
    get_all_memories(user_id, run_id)
    get_memory(memory_id)
    update_memory(memory_id, data)
    delete_memory(memory_id)
    delete_all_memories(user_id, run_id)
    get_memory_history(memory_id)
    should_rerank(query)
    infer_domains(query)
    build_domain_filter(domains)
    extract_wellness_facts(text)          ← Feature 6
    resolve_memory_actions(new_facts, existing_memories)  ← Feature 7
    create_memory_aware_chat(messages, user_id, model, run_id, filters, limit)  ← Feature 8
    auto_save_conversation_memory(user_message, assistant_response, user_id, run_id)  ← Feature 8
    validate_mem0_config()          ← Feature 9 / 11 / 12
    get_mem0_config_summary()       ← Feature 9 / 11 / 12
    get_supported_vector_stores()   ← Feature 11
    get_supported_embedders()       ← Feature 12
    get_supported_rerankers()       ← Feature 13
    add_memory_from_image(image_url, context_text, user_id, domain, run_id)
    get_mem0()
    get_mem0_proxy()  ← Feature 8
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from src.infra.logger import setup_logger

logger = setup_logger(__name__)

# ── Feature 6 — Custom Fact Extraction Prompt ─────────────────────────────────
#
# Instructs Mem0's internal LLM extractor to save ONLY persistent wellness facts.
# Applied via config["custom_fact_extraction_prompt"] + config["version"] = "v1.1".
# Also used standalone by extract_wellness_facts() for the Supabase fallback path,
# keeping storage behaviour identical whether or not Mem0 is configured.
#
# Prompt design:
#   • Four target categories: NUTRITION, FITNESS, MEDICAL, GENERAL
#   • Positive examples: preferences, conditions, medications, personal stats
#   • Negative examples: greetings, questions, transient states, assistant replies
#   • Output: JSON with a "facts" key only — no extra text

WELLNESS_FACT_EXTRACTION_PROMPT = """\
Extract only persistent health and wellness facts about the user. Ignore everything else.

Target categories:
  NUTRITION : food preferences, dietary restrictions, allergies, eating habits, macro goals
  FITNESS   : workout preferences, activity level, exercise habits, fitness goals, body metrics
  MEDICAL   : diagnosed conditions, medications (name + dosage + frequency), symptoms, history
  GENERAL   : personal stats (age, weight, height, sex), occupation, lifestyle, sleep, schedule

Rules:
  - Write each fact as a concise third-person statement about the user.
  - Store only persistent facts — not one-off requests, questions, or greetings.
  - Do NOT store: questions to the assistant, advice requests, transient hunger/mood states.
  - Do NOT store: assistant responses, logging confirmations, or filler phrases.
  - If nothing meets the criteria, return {"facts": []}.

Examples:

Input: Hi, how are you?
Output: {"facts": []}

Input: What should I eat for breakfast?
Output: {"facts": []}

Input: Can you give me a high-protein meal plan?
Output: {"facts": []}

Input: I had oatmeal for breakfast today.
Output: {"facts": []}

Input: I'm vegetarian and allergic to tree nuts.
Output: {"facts": ["User is vegetarian.", "User is allergic to tree nuts."]}

Input: I prefer morning workouts, usually 5 days a week.
Output: {"facts": ["User prefers morning workouts.", "User exercises approximately 5 days per week."]}

Input: I was diagnosed with type 2 diabetes. I take metformin 500mg twice daily.
Output: {"facts": ["User has type 2 diabetes.", "User takes metformin 500mg twice daily."]}

Input: I work night shifts as a nurse and sleep from 8am to 4pm.
Output: {"facts": ["User works night shifts as a nurse.", "User sleeps from approximately 8am to 4pm."]}

Input: I weigh 78 kg and want to lose 5 kg.
Output: {"facts": ["User weighs 78 kg.", "User's fitness goal is to lose 5 kg."]}

Input: I dislike spicy food and rarely drink alcohol.
Output: {"facts": ["User dislikes spicy food.", "User rarely drinks alcohol."]}

Input: Thanks, that sounds great!
Output: {"facts": []}

Return only JSON with a "facts" key. No explanation or extra text.
"""

# ── Feature 7 — Custom Update Memory Prompt ───────────────────────────────────
#
# Controls how Mem0 reconciles newly extracted facts against existing memories.
# Applied via config["custom_update_memory_prompt"].
# Also used standalone by resolve_memory_actions() for the Supabase fallback,
# keeping UPDATE/DELETE behaviour consistent on both storage paths.
#
# Wellness-specific decision rules:
#   ADD    — fact is genuinely new (e.g. new condition, first weight log)
#   UPDATE — fact supersedes an older value on the same topic:
#              • body metrics (weight, body fat %) — newer measurement replaces old
#              • medication dose/frequency change
#              • condition severity or status change
#              • preference reversal (morning → evening workouts)
#   DELETE — fact contradicts or invalidates existing memory:
#              • "stopped taking <medication>" → DELETE that medication entry
#              • "no longer has <condition>" / "resolved" → DELETE condition entry
#              • explicit corrections ("I'm not vegetarian any more") → DELETE
#   NONE   — incoming fact is already captured or is semantically equivalent

WELLNESS_UPDATE_MEMORY_PROMPT = """You are a smart memory manager for a health and wellness assistant.
You control the memory of users and must decide how new facts change existing memories.

You can perform four operations:
  ADD    — new fact not present in memory at all
  UPDATE — fact updates/supersedes an existing memory on the same topic
  DELETE — fact contradicts or explicitly removes an existing memory
  NONE   — fact is already captured or conveys the same meaning (no change needed)

## Wellness-specific decision rules

Body metrics (weight, body fat %, waist):
  - A new measurement ALWAYS supersedes the previous one → UPDATE (keep same ID).
  - Never ADD a duplicate metric; replace the old value.

Medications:
  - Dose or frequency change → UPDATE the existing medication entry (keep same ID).
  - User stops the medication ("stopped taking X", "no longer on X") → DELETE that entry.
  - New medication not in memory → ADD.

Medical conditions:
  - Severity or status change ("my diabetes is mild now", "in remission") → UPDATE.
  - Condition explicitly resolved ("no longer have X", "cured", "resolved") → DELETE.
  - New condition not in memory → ADD.

Dietary preferences & restrictions:
  - Preference reversal ("I'm no longer vegetarian") → DELETE old preference, ADD new one.
  - More specific version of existing pref ("loves cheese pizza" → "loves cheese and chicken pizza") → UPDATE.
  - Semantically equivalent ("loves cheese pizza" vs "likes cheese pizza") → NONE.

Fitness preferences & goals:
  - Workout time change ("now I prefer evenings") → UPDATE existing preference.
  - New target weight or goal type → UPDATE existing goal entry.
  - Minor elaboration of same goal → NONE.

Personal stats (age, height, sex, occupation):
  - Correction or change → UPDATE.
  - Same value already stored → NONE.

## Output format

Return a JSON object with a single "memory" key. Each entry must have:
  - "id"        : string — reuse the existing ID for UPDATE/DELETE/NONE; generate new for ADD
  - "text"      : string — the final memory text (updated or original)
  - "event"     : "ADD" | "UPDATE" | "DELETE" | "NONE"
  - "old_memory": string — REQUIRED for UPDATE only; the exact text being replaced

## Examples

### Example 1 — Body metric supersession (UPDATE)
Old Memory:
  [{"id": "m1", "text": "User weighs 75 kg."}]
Retrieved facts: ["User weighs 80 kg."]
Output:
  {"memory": [{"id": "m1", "text": "User weighs 80 kg.", "event": "UPDATE", "old_memory": "User weighs 75 kg."}]}

### Example 2 — Medication stopped (DELETE)
Old Memory:
  [{"id": "m2", "text": "User takes metformin 500mg twice daily."}, {"id": "m3", "text": "User has type 2 diabetes."}]
Retrieved facts: ["User stopped taking metformin."]
Output:
  {"memory": [{"id": "m2", "text": "User takes metformin 500mg twice daily.", "event": "DELETE"}, {"id": "m3", "text": "User has type 2 diabetes.", "event": "NONE"}]}

### Example 3 — Dose change (UPDATE)
Old Memory:
  [{"id": "m4", "text": "User takes lisinopril 5mg once daily."}]
Retrieved facts: ["User's lisinopril dose increased to 10mg once daily."]
Output:
  {"memory": [{"id": "m4", "text": "User takes lisinopril 10mg once daily.", "event": "UPDATE", "old_memory": "User takes lisinopril 5mg once daily."}]}

### Example 4 — Condition resolved (DELETE)
Old Memory:
  [{"id": "m5", "text": "User has hypertension."}, {"id": "m6", "text": "User is vegetarian."}]
Retrieved facts: ["User's hypertension is resolved."]
Output:
  {"memory": [{"id": "m5", "text": "User has hypertension.", "event": "DELETE"}, {"id": "m6", "text": "User is vegetarian.", "event": "NONE"}]}

### Example 5 — Workout preference flip (UPDATE)
Old Memory:
  [{"id": "m7", "text": "User prefers morning workouts."}]
Retrieved facts: ["User now prefers evening workouts."]
Output:
  {"memory": [{"id": "m7", "text": "User prefers evening workouts.", "event": "UPDATE", "old_memory": "User prefers morning workouts."}]}

### Example 6 — New allergy not in memory (ADD)
Old Memory:
  [{"id": "m8", "text": "User is vegetarian."}]
Retrieved facts: ["User is allergic to shellfish."]
Output:
  {"memory": [{"id": "m8", "text": "User is vegetarian.", "event": "NONE"}, {"id": "m9", "text": "User is allergic to shellfish.", "event": "ADD"}]}

### Example 7 — Semantic duplicate (NONE)
Old Memory:
  [{"id": "m10", "text": "User is vegetarian."}]
Retrieved facts: ["User follows a vegetarian diet."]
Output:
  {"memory": [{"id": "m10", "text": "User is vegetarian.", "event": "NONE"}]}

Return only JSON with a "memory" key. No explanation or extra text.
"""

# ── Singleton state ────────────────────────────────────────────────────────────

_mem0: Any = None
_mem0_initialized = False
_init_lock = asyncio.Lock()

# Feature 8 — proxy client singleton (separate from AsyncMemory singleton)
_mem0_proxy: Any = None
_mem0_proxy_initialized = False
_proxy_init_lock = asyncio.Lock()

# ── Async operation config ─────────────────────────────────────────────────────

_RETRY_ATTEMPTS = 3
_RETRY_TIMEOUT_SECONDS = 10.0
_RETRY_BASE_DELAY = 1.0   # seconds; doubles each attempt (1 → 2 → 4)

# ── Domain keywords for inference ─────────────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "nutrition": [
        "eat", "food", "meal", "diet", "calorie", "protein", "carb", "fat",
        "breakfast", "lunch", "dinner", "snack", "drink", "cook", "recipe",
        "vegetarian", "vegan", "allergy", "nutrition", "macro", "weight",
        "portion", "hunger", "appetite", "sugar", "sodium", "vitamin",
    ],
    "fitness": [
        "workout", "exercise", "gym", "run", "walk", "swim", "bike", "cycle",
        "muscle", "cardio", "strength", "training", "steps", "body", "stretch",
        "yoga", "jog", "lift", "push", "pull", "squat", "hiit", "rest day",
        "active", "sedentary", "morning workout", "evening workout",
    ],
    "medical": [
        "doctor", "medication", "medicine", "condition", "disease", "symptom",
        "diagnosis", "prescription", "health issue", "pain", "blood pressure",
        "blood sugar", "diabetes", "hypertension", "cholesterol", "allergy",
        "supplement", "treatment", "therapy", "clinic", "hospital",
        "metformin", "insulin", "statin", "warfarin", "ssri", "antidepressant",
        "chronic", "disorder", "syndrome", "anxiety", "depression",
    ],
}


# ── Feature 4 helpers ──────────────────────────────────────────────────────────

async def _async_retry(operation_name: str, coro_factory: Any) -> Any:
    """Run a coroutine with retry + exponential backoff + per-attempt timeout.

    Args:
        operation_name: Human-readable name for logging.
        coro_factory: Zero-argument callable that returns a fresh coroutine each call.
    """
    last_exc: Exception | None = None
    for attempt in range(_RETRY_ATTEMPTS):
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(coro_factory(), timeout=_RETRY_TIMEOUT_SECONDS)
            duration = time.monotonic() - start
            logger.debug(
                "Mem0 %s completed — attempt=%d duration=%.2fs",
                operation_name, attempt + 1, duration,
            )
            return result
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            logger.warning(
                "Mem0 %s timeout — attempt=%d/%d duration=%.2fs",
                operation_name, attempt + 1, _RETRY_ATTEMPTS, duration,
            )
            last_exc = asyncio.TimeoutError(
                f"{operation_name} timed out after {_RETRY_TIMEOUT_SECONDS}s"
            )
        except Exception as exc:
            duration = time.monotonic() - start
            logger.warning(
                "Mem0 %s error — attempt=%d/%d duration=%.2fs error=%s",
                operation_name, attempt + 1, _RETRY_ATTEMPTS, duration, exc,
            )
            last_exc = exc

        if attempt < _RETRY_ATTEMPTS - 1:
            await asyncio.sleep(_RETRY_BASE_DELAY * (2 ** attempt))

    raise last_exc  # type: ignore[misc]


def infer_domains(query: str) -> list[str]:
    """Detect relevant memory domains from query text.

    "general" is always included — lifestyle/occupation facts apply everywhere.
    If no specific domain detected, returns all domains (broad search).
    """
    if not query:
        return ["nutrition", "fitness", "medical", "general"]

    q = query.lower()
    detected: list[str] = []

    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            detected.append(domain)

    if "general" not in detected:
        detected.append("general")

    if detected == ["general"]:
        return ["nutrition", "fitness", "medical", "general"]

    return detected


def should_rerank(query: str) -> bool:
    """Rerank only when the query is nuanced enough to benefit (>3 words)."""
    return len(query.split()) > 3


def build_domain_filter(domains: list[str]) -> dict | None:
    """Build Mem0 enhanced metadata filter for domain-scoped retrieval.

    Returns None when all domains included (no filter = search everything).
    """
    all_domains = {"nutrition", "fitness", "medical", "general"}
    if set(domains) >= all_domains:
        return None

    return {
        "AND": [
            {"domain": {"in": domains}},
            {"domain": {"ne": "archived"}},
        ]
    }


# ── Mem0 config ────────────────────────────────────────────────────────────────

# ── Feature 10 — Configurable LLM provider ────────────────────────────────────

#: Providers with known Mem0 support + their default models
_SUPPORTED_PROVIDERS: dict[str, str] = {
    "openai":            "gpt-4o-mini",
    "openai_structured": "gpt-4o-mini",          # OpenAI structured-outputs variant
    "azure_openai":      "gpt-4o-mini",
    "anthropic":         "claude-sonnet-4-20250514",
    "groq":              "llama-3.1-8b-instant",
    "ollama":            "llama3.1",
    "litellm":           "gpt-4o-mini",
    "aws_bedrock":       "anthropic.claude-3-5-haiku-20241022-v1:0",
}


def get_supported_llm_providers() -> list[str]:
    """Return the list of Mem0-supported LLM provider names."""
    return list(_SUPPORTED_PROVIDERS)


def _build_llm_config(settings: Any) -> dict:
    """Build Mem0 LLM config block for the configured provider.

    Implements the Mem0 config schema:
      {"provider": "<name>", "config": {<provider-specific settings>}}

    Config values precedence (per Mem0 docs):
      1. Explicit config values (from settings fields)
      2. Environment variables (read by Pydantic settings)
      3. Provider defaults (_SUPPORTED_PROVIDERS default models)

    Supported providers: openai, openai_structured, azure_openai, anthropic,
    groq, ollama, litellm, aws_bedrock.
    Falls back to openai on unknown provider (logs a warning).

    Args:
        settings: Application settings instance.

    Returns:
        Dict with "provider" and "config" keys ready for Mem0 config.
    """
    provider = (settings.mem0_llm_provider or "openai").lower().strip()
    # Explicit model > empty string → fallback to pipeline model > provider default
    model = (
        settings.mem0_llm_model
        or settings.llm_model_pipeline
        or _SUPPORTED_PROVIDERS.get(provider, "gpt-4o-mini")
    )
    temperature = settings.mem0_llm_temperature
    max_tokens = settings.mem0_llm_max_tokens

    if provider == "openai":
        cfg: dict = {
            "model": model,
            "temperature": temperature,
            "api_key": settings.openai_api_key,
        }
        if max_tokens:
            cfg["max_tokens"] = max_tokens
        if settings.mem0_llm_base_url:
            cfg["openai_base_url"] = settings.mem0_llm_base_url
        return {"provider": "openai", "config": cfg}

    elif provider == "openai_structured":
        # OpenAI structured-outputs variant — uses response_format=json_schema internally.
        # Note: reasoning models do not support temperature (per Mem0/OpenAI docs).
        cfg = {
            "model": model,
            "temperature": temperature,
            "api_key": settings.openai_api_key,
        }
        if settings.mem0_llm_base_url:
            cfg["openai_base_url"] = settings.mem0_llm_base_url
        return {"provider": "openai_structured", "config": cfg}

    elif provider == "azure_openai":
        return {
            "provider": "azure_openai",
            "config": {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": settings.azure_openai_api_key or settings.openai_api_key,
                "azure_kwargs": {
                    "azure_deployment": settings.azure_deployment_name or model,
                    "api_base": settings.azure_openai_endpoint,
                },
            },
        }

    elif provider == "anthropic":
        return {
            "provider": "anthropic",
            "config": {
                "model": model if model != settings.llm_model_pipeline else _SUPPORTED_PROVIDERS["anthropic"],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": settings.anthropic_api_key,
            },
        }

    elif provider == "groq":
        return {
            "provider": "groq",
            "config": {
                "model": model if model != settings.llm_model_pipeline else _SUPPORTED_PROVIDERS["groq"],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": settings.groq_api_key,
            },
        }

    elif provider == "ollama":
        base_url = settings.mem0_llm_base_url or "http://localhost:11434"
        return {
            "provider": "ollama",
            "config": {
                "model": model if model != settings.llm_model_pipeline else _SUPPORTED_PROVIDERS["ollama"],
                "temperature": temperature,
                "ollama_base_url": base_url,
            },
        }

    elif provider == "litellm":
        cfg = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_key": settings.openai_api_key,
        }
        if settings.mem0_llm_base_url:
            cfg["openai_base_url"] = settings.mem0_llm_base_url
        return {"provider": "litellm", "config": cfg}

    elif provider == "aws_bedrock":
        # Requires boto3. Auth via env vars (AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY)
        # or IAM role — follows standard boto3 credential chain (per AWS docs).
        bedrock_model = (
            settings.mem0_llm_model
            or _SUPPORTED_PROVIDERS["aws_bedrock"]
        )
        cfg = {
            "model": bedrock_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Include explicit credentials only if set — otherwise boto3 uses its chain
        if settings.aws_access_key_id:
            cfg["aws_access_key"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            cfg["aws_secret_key"] = settings.aws_secret_access_key
        if settings.aws_region:
            cfg["aws_region"] = settings.aws_region
        return {"provider": "aws_bedrock", "config": cfg}

    else:
        logger.warning(
            "Unknown Mem0 LLM provider %r (supported: %s) — falling back to openai",
            provider, ", ".join(_SUPPORTED_PROVIDERS),
        )
        return {
            "provider": "openai",
            "config": {
                "model": settings.llm_model_pipeline,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "api_key": settings.openai_api_key,
            },
        }


# ── Feature 11 — Configurable Vector Database ─────────────────────────────────

#: Vector store providers with known Mem0 support
_SUPPORTED_VECTOR_STORES: dict[str, str] = {
    "qdrant":    "Qdrant Cloud / self-hosted (default)",
    "pgvector":  "PostgreSQL + pgvector (maps to Mem0 supabase provider)",
    "supabase":  "Supabase (PostgreSQL + pgvector via connection string)",
    "chroma":    "ChromaDB — embedded (path) or client-server (host + port)",
    "pinecone":  "Pinecone managed cloud vector database",
}

#: Embedding dimensions for text-embedding-3-small (fixed across all providers)
_EMBEDDING_DIMS = 1536


def get_supported_vector_stores() -> list[str]:
    """Return the list of Mem0-supported vector store provider names."""
    return list(_SUPPORTED_VECTOR_STORES)


def _build_vector_store_config(settings: Any) -> dict:
    """Build Mem0 vector_store config block for the configured provider.

    All providers share the same collection name (mem0_collection_name) and
    embedding dimensions (1536 for text-embedding-3-small).

    Supported providers: qdrant, pgvector, supabase, chroma, pinecone.
    Falls back to qdrant on unknown provider (logs a warning).

    Args:
        settings: Application settings instance.

    Returns:
        Dict with "provider" and "config" keys ready for Mem0 config.
    """
    provider = (settings.mem0_vector_store_provider or "qdrant").lower().strip()
    collection = settings.mem0_collection_name
    dims = settings.mem0_embedder_dims

    if provider == "qdrant":
        cfg: dict = {
            "collection_name": collection,
            "embedding_model_dims": dims,
            "url": settings.qdrant_url,
        }
        if settings.qdrant_api_key:
            cfg["api_key"] = settings.qdrant_api_key
        return {"provider": "qdrant", "config": cfg}

    elif provider in ("pgvector", "supabase"):
        # Both map to Mem0's "supabase" provider (PostgreSQL + pgvector).
        # Uses supabase_connection_string which is already a required Reso setting.
        return {
            "provider": "supabase",
            "config": {
                "collection_name": collection,
                "embedding_model_dims": dims,
                "connection_string": settings.supabase_connection_string,
            },
        }

    elif provider == "chroma":
        cfg = {
            "collection_name": collection,
            "embedding_model_dims": dims,
        }
        if settings.chroma_host:
            # Client-server mode: connect to running ChromaDB instance
            cfg["host"] = settings.chroma_host
            cfg["port"] = settings.chroma_port
        else:
            # Embedded mode: persist to local path — useful for dev/staging
            cfg["path"] = settings.chroma_path
        return {"provider": "chroma", "config": cfg}

    elif provider == "pinecone":
        return {
            "provider": "pinecone",
            "config": {
                "collection_name": settings.pinecone_index_name or collection,
                "embedding_model_dims": dims,
                "api_key": settings.pinecone_api_key,
            },
        }

    else:
        logger.warning(
            "Unknown Mem0 vector store provider %r (supported: %s) — falling back to qdrant",
            provider, ", ".join(_SUPPORTED_VECTOR_STORES),
        )
        cfg = {
            "collection_name": collection,
            "embedding_model_dims": dims,
            "url": settings.qdrant_url,
        }
        if settings.qdrant_api_key:
            cfg["api_key"] = settings.qdrant_api_key
        return {"provider": "qdrant", "config": cfg}


# ── Feature 12 — Configurable Embedder ────────────────────────────────────────

#: Supported Mem0 embedding providers + their default models
_SUPPORTED_EMBEDDERS: dict[str, str] = {
    "openai":       "text-embedding-ada-002",  # 1536 dims, universally available
    "azure_openai": "text-embedding-ada-002",
    "ollama":       "nomic-embed-text",         # 768 dims typical
    "huggingface":  "multi-qa-MiniLM-L6-cos-v1",  # 384 dims typical
    "google_ai":    "models/text-embedding-004",   # 768 dims
    "vertexai":     "text-embedding-004",
    "together":     "togethercomputer/m2-bert-80M-32k-retrieval",
    "lmstudio":     "text-embedding-nomic-embed-text-v1.5",  # local
    "aws_bedrock":  "amazon.titan-embed-text-v2:0",           # 1024 dims
}

#: Default embedding dimensions per provider (for the default model)
_DEFAULT_EMBEDDER_DIMS: dict[str, int] = {
    "openai":       1536,
    "azure_openai": 1536,
    "ollama":       768,
    "huggingface":  384,
    "google_ai":    768,
    "vertexai":     768,
    "together":     768,
    "lmstudio":     768,
    "aws_bedrock":  1024,
}


def get_supported_embedders() -> list[str]:
    """Return the list of Mem0-supported embedding provider names."""
    return list(_SUPPORTED_EMBEDDERS)


def _build_embedder_config(settings: Any) -> dict:
    """Build Mem0 embedder config block for the configured provider.

    Implements the Mem0 embedder config schema:
      {"provider": "<name>", "config": {<provider-specific settings>}}

    IMPORTANT: embedding_dims in the config must match the model's actual output
    dimensions. Mismatches cause vector store errors on first index operation.
    Always set MEM0_EMBEDDER_DIMS to match the chosen model.

    Supported providers: openai, azure_openai, ollama, huggingface, google_ai,
    vertexai, together, lmstudio, aws_bedrock.
    Falls back to openai on unknown provider (logs a warning).

    Args:
        settings: Application settings instance.

    Returns:
        Dict with "provider" and "config" keys ready for Mem0 config.
    """
    provider = (settings.mem0_embedder_provider or "openai").lower().strip()
    model = settings.mem0_embedder_model or _SUPPORTED_EMBEDDERS.get(provider, "text-embedding-3-small")
    dims = settings.mem0_embedder_dims

    if provider == "openai":
        cfg: dict = {
            "model": model,
            "embedding_dims": dims,
            "api_key": settings.openai_api_key,
        }
        if settings.mem0_embedder_base_url:
            cfg["openai_base_url"] = settings.mem0_embedder_base_url
        return {"provider": "openai", "config": cfg}

    elif provider == "azure_openai":
        return {
            "provider": "azure_openai",
            "config": {
                "model": model,
                "embedding_dims": dims,
                "azure_kwargs": {
                    "api_key": settings.azure_openai_api_key or settings.openai_api_key,
                    "api_base": settings.azure_openai_endpoint,
                    "azure_deployment": settings.azure_deployment_name or model,
                },
            },
        }

    elif provider == "ollama":
        base_url = settings.mem0_embedder_base_url or "http://localhost:11434"
        return {
            "provider": "ollama",
            "config": {
                "model": model,
                "embedding_dims": dims,
                "ollama_base_url": base_url,
            },
        }

    elif provider in ("huggingface", "hugging_face"):
        return {
            "provider": "huggingface",
            "config": {
                "model": model,
                "embedding_dims": dims,
                "model_kwargs": {"trust_remote_code": True},
            },
        }

    elif provider == "google_ai":
        return {
            "provider": "google_ai",
            "config": {
                "model": model,
                "embedding_dims": dims,
                "api_key": settings.openai_api_key,  # reuse or set GOOGLE_AI_API_KEY separately
            },
        }

    elif provider == "vertexai":
        return {
            "provider": "vertexai",
            "config": {
                "model": model,
                "embedding_dims": dims,
            },
        }

    elif provider == "together":
        return {
            "provider": "together",
            "config": {
                "model": model,
                "embedding_dims": dims,
                "api_key": settings.openai_api_key,
            },
        }

    elif provider == "lmstudio":
        base_url = settings.mem0_embedder_base_url or "http://localhost:1234"
        return {
            "provider": "lmstudio",
            "config": {
                "model": model,
                "embedding_dims": dims,
                "lmstudio_base_url": base_url,
            },
        }

    elif provider == "aws_bedrock":
        cfg = {
            "model": model,
            "embedding_dims": dims,
        }
        if settings.aws_access_key_id:
            cfg["aws_access_key"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            cfg["aws_secret_key"] = settings.aws_secret_access_key
        if settings.aws_region:
            cfg["aws_region"] = settings.aws_region
        return {"provider": "aws_bedrock", "config": cfg}

    else:
        logger.warning(
            "Unknown Mem0 embedder provider %r (supported: %s) — falling back to openai",
            provider, ", ".join(_SUPPORTED_EMBEDDERS),
        )
        return {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002",
                "embedding_dims": 1536,
                "api_key": settings.openai_api_key,
            },
        }


# ── Feature 13 — Configurable Reranker ────────────────────────────────────────

#: Supported Mem0 reranker providers + their default models
_SUPPORTED_RERANKERS: dict[str, str] = {
    "llm_reranker":         "gpt-4o-mini",               # LLM-based scoring (default)
    "cohere":               "rerank-english-v3.0",        # API-first, best quality
    "zero_entropy":         "zerank-1",                   # Managed neural reranker
    "sentence_transformer": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # On-device
    "huggingface":          "BAAI/bge-reranker-large",    # On-device, privacy
    "none":                 "",                           # Disabled
}

# Wellness-domain LLM scoring prompt for the llm_reranker provider.
# Scores each memory against the user's query on dimensions that matter
# for a personal wellness assistant: health fact relevance, recency of
# body metrics, actionability, and domain specificity.
WELLNESS_RERANKER_SCORING_PROMPT = """You are a relevance scoring assistant for a personal wellness AI.
Given a user query and a stored wellness memory, score how relevant the memory is.

Scoring criteria:
- HEALTH RELEVANCE (50%): Does the memory directly relate to the health/wellness topic in the query?
- RECENCY WEIGHT (20%): Body metrics (weight, blood pressure, etc.) should be scored higher if recent.
- ACTIONABILITY (20%): Can this memory directly inform a wellness recommendation or action?
- DOMAIN MATCH (10%): Does the memory domain (nutrition/fitness/medical/general) match the query intent?

Score the relevance on a scale from 0.0 to 1.0, where:
- 1.0 = Perfectly relevant wellness fact that directly answers the query
- 0.8-0.9 = Highly relevant with strong actionable health context
- 0.6-0.7 = Moderately relevant, useful background wellness information
- 0.4-0.5 = Tangentially relevant wellness data
- 0.0-0.3 = Not relevant to this wellness query

Query: "{query}"
Memory: "{document}"

Provide only a single numerical score between 0.0 and 1.0."""


def get_supported_rerankers() -> list[str]:
    """Return the list of Mem0-supported reranker provider names."""
    return list(_SUPPORTED_RERANKERS)


def _build_reranker_config(settings: Any) -> dict | None:
    """Build Mem0 reranker config block for the configured provider.

    Returns None when provider is "none" — Mem0 interprets a missing/None
    reranker key as disabled, skipping the second-pass scoring step.

    Provider selection guidance (per Mem0 docs + performance guide):
      llm_reranker    — default; bespoke scoring via wellness prompt (Feature 13)
      cohere          — API-first, best quality; use ≤100 candidates
      zero_entropy    — managed neural reranker; top quality, API cost
      sentence_transformer — on-device cross-encoder; use ≤50 candidates
      huggingface     — on-device; privacy-sensitive deployments; use ≤30
      none            — disables reranking (low-latency paths)

    Args:
        settings: Application settings instance.

    Returns:
        Mem0 reranker config dict, or None if provider is "none".
    """
    provider = (settings.mem0_reranker_provider or "llm_reranker").lower().strip()
    top_k = settings.mem0_reranker_top_k
    model = settings.mem0_reranker_model or _SUPPORTED_RERANKERS.get(provider, "")

    if provider == "none":
        return None  # Reranking disabled

    if provider == "llm_reranker":
        cfg: dict = {
            "provider": settings.mem0_llm_provider or "openai",
            "model": model or settings.llm_model_pipeline,
            "api_key": settings.openai_api_key,
            "temperature": 0.0,   # Deterministic scoring (per docs)
            "max_tokens": 100,    # Limit — score is a single float
            "top_k": top_k,
        }
        # Feature 13 — wellness-domain scoring prompt for the LLM reranker
        if getattr(settings, "mem0_reranker_use_wellness_prompt", True):
            cfg["scoring_prompt"] = WELLNESS_RERANKER_SCORING_PROMPT
        return {"provider": "llm_reranker", "config": cfg}

    if provider == "cohere":
        if not settings.cohere_api_key:
            logger.warning(
                "Mem0 reranker: provider=cohere but COHERE_API_KEY not set — "
                "falling back to llm_reranker"
            )
            return _build_reranker_config_llm_fallback(settings, top_k)
        cfg = {
            "model": model or _SUPPORTED_RERANKERS["cohere"],
            "top_k": top_k,
            "api_key": settings.cohere_api_key,
            "return_documents": False,  # Reduce response size (per docs)
        }
        return {"provider": "cohere", "config": cfg}

    if provider == "zero_entropy":
        if not settings.zero_entropy_api_key:
            logger.warning(
                "Mem0 reranker: provider=zero_entropy but ZERO_ENTROPY_API_KEY not set — "
                "falling back to llm_reranker"
            )
            return _build_reranker_config_llm_fallback(settings, top_k)
        cfg = {
            "model": model or _SUPPORTED_RERANKERS["zero_entropy"],
            "top_k": top_k,
            "api_key": settings.zero_entropy_api_key,
        }
        return {"provider": "zero_entropy", "config": cfg}

    if provider == "sentence_transformer":
        # On-device cross-encoder — no API key needed
        cfg = {
            "model": model or _SUPPORTED_RERANKERS["sentence_transformer"],
            "top_k": top_k,
            "batch_size": 32,  # Docs-recommended default
        }
        return {"provider": "sentence_transformer", "config": cfg}

    if provider == "huggingface":
        cfg = {
            "model": model or _SUPPORTED_RERANKERS["huggingface"],
            "top_k": top_k,
        }
        if settings.huggingface_api_key:
            cfg["api_key"] = settings.huggingface_api_key
        return {"provider": "huggingface", "config": cfg}

    # Unknown provider — warn and fall back to llm_reranker
    logger.warning(
        "Unknown Mem0 reranker provider %r (supported: %s) — falling back to llm_reranker",
        provider, ", ".join(_SUPPORTED_RERANKERS),
    )
    return _build_reranker_config_llm_fallback(settings, top_k)


def _build_reranker_config_llm_fallback(settings: Any, top_k: int) -> dict:
    """Build a minimal llm_reranker config as fallback."""
    return {
        "provider": "llm_reranker",
        "config": {
            "provider": "openai",
            "model": settings.llm_model_pipeline,
            "api_key": settings.openai_api_key,
            "temperature": 0.0,
            "max_tokens": 100,
            "top_k": top_k,
        },
    }


def _build_mem0_config(settings: Any) -> dict:
    return {
        # Feature 6 — Custom Fact Extraction: wellness-domain filter applied during every add().
        # Feature 7 — Custom Update Memory: wellness-aware reconciliation (UPDATE/DELETE lifecycle).
        # version "v1.1" is required by Mem0 when either custom prompt is set.
        "version": "v1.1",
        "custom_fact_extraction_prompt": WELLNESS_FACT_EXTRACTION_PROMPT,
        "custom_update_memory_prompt": WELLNESS_UPDATE_MEMORY_PROMPT,
        # Feature 10 — configurable LLM provider
        "llm": _build_llm_config(settings),
        # Feature 12 — configurable embedder provider
        "embedder": _build_embedder_config(settings),
        # Feature 11 — configurable vector store provider
        "vector_store": _build_vector_store_config(settings),
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": settings.neo4j_url,
                "username": settings.neo4j_username,
                "password": settings.neo4j_password,
                **({"database": settings.neo4j_database} if settings.neo4j_database else {}),
            },
        },
        # Feature 13 — configurable reranker provider (None disables reranking)
        **({} if (reranker_cfg := _build_reranker_config(settings)) is None
           else {"reranker": reranker_cfg}),
    }


async def get_mem0() -> Any:
    """Return the AsyncMemory singleton. Returns None if not configured or init failed."""
    global _mem0, _mem0_initialized
    if _mem0_initialized:
        return _mem0

    async with _init_lock:
        if _mem0_initialized:
            return _mem0

        from src.infra.config import get_settings
        settings = get_settings()

        if not settings.is_mem0_configured:
            logger.warning(
                "Mem0 not fully configured (missing QDRANT_URL / NEO4J_URL / NEO4J_PASSWORD) "
                "— falling back to Supabase memory storage"
            )
            _mem0_initialized = True
            return None

        try:
            from mem0 import AsyncMemory

            config = _build_mem0_config(settings)
            result = AsyncMemory.from_config(config_dict=config)
            if asyncio.iscoroutine(result):
                result = await result
            _mem0 = result

            # Ensure domain payload index exists — required for metadata filtering.
            # Mem0 creates user_id/agent_id/run_id/actor_id indexes but not custom fields.
            try:
                from qdrant_client import AsyncQdrantClient
                qc = AsyncQdrantClient(
                    url=settings.qdrant_url,
                    api_key=settings.qdrant_api_key or None,
                )
                await qc.create_payload_index(
                    collection_name=settings.mem0_collection_name,
                    field_name="domain",
                    field_schema="keyword",
                )
                await qc.close()
            except Exception:
                pass  # index already exists or collection not ready yet — non-fatal

            logger.info(
                "Mem0 AsyncMemory initialized — vector_store=qdrant graph_store=neo4j "
                "reranker=llm(gpt-4o-mini) version=v1.1 "
                "features=[graph_memory, metadata_filtering, reranker_search, "
                "async_ops, multimodal, custom_fact_extraction, custom_update_memory]"
            )
        except Exception as exc:
            logger.warning(
                "Mem0 initialization failed: %s — falling back to Supabase memory",
                exc,
                exc_info=True,
            )
            _mem0 = None

        _mem0_initialized = True
        return _mem0


# ── Feature 8 — OpenAI Compatibility: Mem0 proxy client ──────────────────────
#
# Mem0's `mem0.proxy.main.Mem0` mirrors the OpenAI client interface.
# `client.chat.completions.create(messages, model, user_id, run_id, ...)` works
# identically to OpenAI's API but automatically extracts and stores memory facts
# from the conversation. This is the "drop-in" replacement feature.
#
# In our workflow it serves a specific role: auto-save any wellness facts from
# every conversation turn that the ReAct agent may not have explicitly saved via
# save_memory_fact (e.g. the user mentions an allergy in passing without it being
# a primary intent). The proxy is wired into wellness_agent_node as a non-critical
# post-agent background step.
#
# Proxy config vs AsyncMemory config:
#   - No graph_store  (graph memory handled by full AsyncMemory in Feature 1)
#   - No reranker     (proxy retrieval doesn't need ranking; it's for auto-save)
#   - Includes Feature 6 + 7 prompts for consistent extraction & update rules
#   - Only needs QDRANT_URL (not Neo4j) → `is_proxy_configured` guard

def _build_mem0_proxy_config(settings: Any) -> dict:
    """Build OSS config for the Mem0 proxy (lighter than AsyncMemory — Qdrant only)."""
    return {
        "version": "v1.1",
        "custom_fact_extraction_prompt": WELLNESS_FACT_EXTRACTION_PROMPT,
        "custom_update_memory_prompt": WELLNESS_UPDATE_MEMORY_PROMPT,
        # Feature 10 — proxy uses the same configurable LLM as AsyncMemory
        "llm": _build_llm_config(settings),
        # Feature 12 — proxy uses the same configured embedder as AsyncMemory
        "embedder": _build_embedder_config(settings),
        # Feature 11 — proxy shares the same configured vector store as AsyncMemory
        "vector_store": _build_vector_store_config(settings),
    }


async def get_mem0_proxy() -> Any:
    """Return the Mem0 proxy client singleton (mem0.proxy.main.Mem0).

    The proxy client is the Feature 8 OpenAI-compatible interface — it exposes
    `client.chat.completions.create(messages, model, user_id, run_id, ...)` and
    auto-saves memory facts from every call.

    Returns None if:
    - Qdrant URL not configured (proxy requires a vector store)
    - mem0.proxy.main.Mem0 import fails (older mem0ai version)
    """
    global _mem0_proxy, _mem0_proxy_initialized

    if _mem0_proxy_initialized:
        return _mem0_proxy

    async with _proxy_init_lock:
        if _mem0_proxy_initialized:
            return _mem0_proxy

        from src.infra.config import get_settings
        settings = get_settings()

        if not settings.is_proxy_configured:
            logger.info(
                "Mem0 proxy client not configured (QDRANT_URL missing) "
                "— auto_save_conversation_memory will be skipped"
            )
            _mem0_proxy_initialized = True
            return None

        try:
            from mem0.proxy.main import Mem0

            config = _build_mem0_proxy_config(settings)
            _mem0_proxy = Mem0(config=config)
            logger.info(
                "Mem0 proxy client initialized — vector_store=qdrant version=v1.1 "
                "features=[openai_compat, custom_fact_extraction, custom_update_memory]"
            )
        except ImportError:
            logger.warning(
                "mem0.proxy.main.Mem0 not available in this mem0ai version "
                "— upgrade mem0ai>=0.1.44 for OpenAI compatibility feature"
            )
            _mem0_proxy = None
        except Exception as exc:
            logger.warning("Mem0 proxy client init failed: %s", exc)
            _mem0_proxy = None

        _mem0_proxy_initialized = True
        return _mem0_proxy


async def create_memory_aware_chat(
    messages: list[dict[str, Any]],
    user_id: str,
    model: str | None = None,
    run_id: str | None = None,
    filters: dict | None = None,
    limit: int = 10,
    **kwargs: Any,
) -> Any:
    """Drop-in async wrapper for Mem0's OpenAI-compatible chat completions.

    Mirrors the OpenAI `client.chat.completions.create()` interface but routes
    through the Mem0 proxy, which:
      1. Retrieves relevant memories and injects them into context automatically.
      2. Extracts and stores new wellness facts from the response.

    Falls back to a plain OpenAI call if the Mem0 proxy is unavailable.

    Args:
        messages  : List of chat messages (same shape as OpenAI API).
        user_id   : User identifier — required for memory scoping.
        model     : LLM model ID. Defaults to settings.llm_model_pipeline.
        run_id    : Session identifier for per-session memory grouping.
        filters   : Mem0 retrieval filters (e.g. domain metadata filters).
        limit     : Max memories retrieved per call (default 10).
        **kwargs  : Extra params forwarded to the underlying LLM call.

    Returns:
        OpenAI-compatible completion object (choices[0].message.content, usage, etc.).
    """
    from src.infra.config import get_settings
    settings = get_settings()
    resolved_model = model or settings.llm_model_pipeline

    proxy = await get_mem0_proxy()

    if proxy is not None:
        try:
            call_kwargs: dict[str, Any] = {
                "messages": messages,
                "model": resolved_model,
                "user_id": user_id,
            }
            if run_id:
                call_kwargs["run_id"] = run_id
            if filters:
                call_kwargs["filters"] = filters
            if limit != 10:
                call_kwargs["limit"] = limit
            call_kwargs.update(kwargs)

            # The Mem0 proxy client is synchronous — run in thread pool to avoid
            # blocking the async event loop.
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: proxy.chat.completions.create(**call_kwargs),
                ),
                timeout=30.0,
            )
            logger.debug(
                "Mem0 proxy chat completion — user=%s run_id=%s model=%s",
                user_id, run_id, resolved_model,
            )
            return result
        except Exception as exc:
            logger.warning("Mem0 proxy chat completion failed: %s — falling back to OpenAI", exc)

    # Fallback: plain OpenAI call (no memory injection or auto-save)
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    return await asyncio.wait_for(
        client.chat.completions.create(
            messages=messages,
            model=resolved_model,
            **kwargs,
        ),
        timeout=30.0,
    )


async def auto_save_conversation_memory(
    user_message: str,
    assistant_response: str,
    user_id: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Auto-extract and persist wellness facts from a completed conversation turn.

    Sends the user message + assistant response through the Mem0 proxy so that
    any wellness facts mentioned in either side of the exchange are automatically
    extracted (Feature 6) and reconciled against existing memories (Feature 7),
    without requiring the ReAct agent to explicitly call save_memory_fact.

    This fills the gap where the agent might not have called save_memory_fact for
    facts mentioned incidentally (e.g. "by the way I'm also allergic to soy").

    Called from wellness_agent_node after every successful response with a short
    timeout (15 s) so it never adds latency to the user-facing stream.

    Args:
        user_message       : The user's original message for this turn.
        assistant_response : The agent's final text response.
        user_id            : User identifier for memory scoping.
        run_id             : Session ID for per-session memory grouping.

    Returns:
        Status dict {"status": "saved"|"skipped"|"error", ...}
    """
    if not user_id or not (user_message or assistant_response):
        return {"status": "skipped", "reason": "missing user_id or content"}

    proxy = await get_mem0_proxy()
    if proxy is None:
        return {"status": "skipped", "reason": "proxy not configured"}

    try:
        from src.infra.config import get_settings
        settings = get_settings()

        messages: list[dict[str, Any]] = []
        if user_message:
            messages.append({"role": "user", "content": user_message})
        if assistant_response:
            messages.append({"role": "assistant", "content": assistant_response})

        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "model": settings.llm_model_pipeline,
            "user_id": user_id,
        }
        if run_id:
            call_kwargs["run_id"] = run_id

        loop = asyncio.get_running_loop()
        await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: proxy.chat.completions.create(**call_kwargs),
            ),
            timeout=15.0,
        )
        logger.info(
            "Mem0 proxy auto-save completed — user=%s run_id=%s", user_id, run_id
        )
        # Opt 4: do NOT invalidate the memory cache here. auto_save runs on every
        # turn and clearing the cache here defeats the optimization (every turn becomes
        # a cold miss). Invalidation is handled by save_memory_fact (explicit agent save)
        # + the 10-minute TTL as a safety net.
        return {"status": "saved", "user_id": user_id, "run_id": run_id}

    except asyncio.TimeoutError:
        logger.warning("auto_save_conversation_memory timed out — user=%s", user_id)
        return {"status": "timeout"}
    except Exception as exc:
        logger.warning("auto_save_conversation_memory failed — user=%s error=%s", user_id, exc)
        return {"status": "error", "error": str(exc)}


# ── Feature 6 — Standalone fact extractor (Supabase fallback parity) ──────────

async def extract_wellness_facts(text: str) -> list[str]:
    """Run WELLNESS_FACT_EXTRACTION_PROMPT against `text` and return fact strings.

    Used by the Supabase fallback path so that _supabase_add_memory applies the
    same wellness filter as Mem0's custom_fact_extraction_prompt, keeping storage
    behaviour consistent whether Mem0 is configured or not.

    Returns an empty list on any LLM failure (fail-open: caller decides whether
    to save the original text as-is or skip it).

    Args:
        text: Raw user message or agent-written fact string.
    """
    if not text or not text.strip():
        return []
    try:
        from openai import AsyncOpenAI
        from src.infra.config import get_settings
        import json as _json

        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.llm_model_pipeline,
                temperature=0,
                max_completion_tokens=300,
                messages=[
                    {"role": "system", "content": WELLNESS_FACT_EXTRACTION_PROMPT},
                    {"role": "user", "content": text},
                ],
            ),
            timeout=8.0,
        )
        raw_content = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences if the LLM wraps the JSON
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`").lstrip("json").strip()
        parsed = _json.loads(raw_content)
        facts = parsed.get("facts", [])
        return [f for f in facts if isinstance(f, str) and f.strip()]
    except Exception as exc:
        logger.debug("extract_wellness_facts failed (non-critical): %s", exc)
        return []


# ── Feature 7 — Standalone update reconciler (Supabase fallback parity) ───────

async def resolve_memory_actions(
    new_facts: list[str],
    existing_memories: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply WELLNESS_UPDATE_MEMORY_PROMPT to reconcile new facts against existing memories.

    Returns a list of action dicts that mirrors Mem0's internal memory-update output:
      {"id": "...", "text": "...", "event": "ADD"|"UPDATE"|"DELETE"|"NONE",
       "old_memory": "..." (UPDATE only)}

    Used by the Supabase fallback path so that _supabase_add_memory applies the
    same ADD/UPDATE/DELETE/NONE lifecycle logic as Mem0's custom_update_memory_prompt,
    keeping behaviour identical whether Mem0 is configured or not.

    Falls back to returning each new fact as ADD on any LLM failure (fail-open).

    Args:
        new_facts          : List of newly extracted fact strings (from Feature 6 extractor).
        existing_memories  : List of dicts with at minimum {"id": ..., "text": ...} — the
                             user's current memories from the DB.
    """
    if not new_facts:
        return []

    if not existing_memories:
        # Nothing to reconcile — all facts are new
        return [
            {"id": f"new_{i}", "text": f, "event": "ADD"}
            for i, f in enumerate(new_facts)
        ]

    try:
        import json as _json
        from openai import AsyncOpenAI
        from src.infra.config import get_settings

        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        # Build the prompt payload mirroring Mem0's internal format
        old_memory_payload = _json.dumps(
            [{"id": str(m.get("id", i)), "text": m.get("fact", m.get("text", ""))}
             for i, m in enumerate(existing_memories)],
            ensure_ascii=False,
        )
        retrieved_facts_payload = _json.dumps(new_facts, ensure_ascii=False)

        user_content = (
            f"Old Memory:\n{old_memory_payload}\n\n"
            f"Retrieved facts: {retrieved_facts_payload}"
        )

        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.llm_model_pipeline,
                temperature=0,
                max_completion_tokens=600,
                messages=[
                    {"role": "system", "content": WELLNESS_UPDATE_MEMORY_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            ),
            timeout=10.0,
        )
        raw_content = (response.choices[0].message.content or "").strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("`").lstrip("json").strip()

        parsed = _json.loads(raw_content)
        actions: list[dict[str, Any]] = parsed.get("memory", [])
        logger.debug(
            "resolve_memory_actions — %d new facts, %d existing, %d actions returned",
            len(new_facts), len(existing_memories), len(actions),
        )
        return actions

    except Exception as exc:
        logger.debug("resolve_memory_actions failed (non-critical): %s — using ADD fallback", exc)
        # Fail-open: treat all new facts as ADD
        return [{"id": f"new_{i}", "text": f, "event": "ADD"} for i, f in enumerate(new_facts)]


# ── Core write/read API ────────────────────────────────────────────────────────

async def add_memory(
    fact: str,
    user_id: str,
    domain: str = "general",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Persist a memory fact with domain metadata and optional session scoping.

    run_id (= session_id) groups memories by conversation session, enabling
    per-session retrieval and audit via get_memory_history.

    With Mem0: vector + graph, semantic dedup, retry+timeout.
    Fallback: Supabase exact-string dedup.
    """
    if not fact or not user_id:
        return {"status": "skipped", "reason": "missing fact or user_id"}

    m = await get_mem0()

    if m is not None:
        try:
            messages = [{"role": "user", "content": fact}]
            kwargs: dict[str, Any] = {
                "user_id": user_id,
                "metadata": {"domain": domain},
            }
            if run_id:
                kwargs["run_id"] = run_id

            result = await _async_retry(
                "add_memory",
                lambda: m.add(messages, **kwargs),
            )
            logger.info(
                "Mem0 memory saved — user=%s domain=%s run_id=%s fact=%r",
                user_id, domain, run_id, fact,
            )
            return {"status": "saved", "fact": fact, "domain": domain, "result": result}
        except Exception as exc:
            logger.warning("Mem0 add_memory failed after retries: %s — falling back", exc)

    return await _supabase_add_memory(fact, user_id, domain)


async def search_memories(
    query: str,
    user_id: str,
    limit: int = 10,
    domains: list[str] | None = None,
    run_id: str | None = None,
) -> list[str]:
    """Return relevant memory facts as a flat list of strings."""
    result = await search_memories_with_graph(
        query, user_id=user_id, limit=limit, domains=domains, run_id=run_id
    )
    return result["facts"]


async def search_memories_with_graph(
    query: str,
    user_id: str,
    limit: int = 10,
    domains: list[str] | None = None,
    filters: dict | None = None,
    rerank: bool | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return memory facts + graph relations with filtering, reranking, and retry.

    Feature 2 — Domain filtering: inferred from query unless overridden.
    Feature 3 — Reranking: smart toggle (>3 words) unless overridden.
    Feature 4 — Retry + timeout on every attempt; run_id scopes to session.
    """
    if not user_id:
        return {"facts": [], "relations": []}

    m = await get_mem0()

    active_filters = filters
    if active_filters is None:
        resolved_domains = domains if domains is not None else infer_domains(query)
        active_filters = build_domain_filter(resolved_domains)
        logger.debug(
            "Mem0 search filters — user=%s domains=%s run_id=%s",
            user_id, resolved_domains, run_id,
        )

    use_rerank = should_rerank(query) if rerank is None else rerank

    if m is not None:
        search_kwargs: dict[str, Any] = {"user_id": user_id, "limit": limit}
        if active_filters is not None:
            search_kwargs["filters"] = active_filters
        if run_id:
            search_kwargs["run_id"] = run_id

        # Attempt 1: with reranking
        if use_rerank:
            try:
                raw = await _async_retry(
                    "search_memories(reranked)",
                    lambda: m.search(query, **search_kwargs, rerank=True),
                )
                result = _parse_mem0_results(raw)
                logger.debug(
                    "Mem0 search(reranked) — user=%s facts=%d relations=%d",
                    user_id, len(result["facts"]), len(result["relations"]),
                )
                return result
            except Exception as exc:
                logger.warning(
                    "Mem0 reranked search failed: %s — retrying vector-only", exc
                )

        # Attempt 2: vector-only
        try:
            raw = await _async_retry(
                "search_memories(vector)",
                lambda: m.search(query, **search_kwargs),
            )
            result = _parse_mem0_results(raw)
            logger.debug(
                "Mem0 search(vector) — user=%s facts=%d relations=%d",
                user_id, len(result["facts"]), len(result["relations"]),
            )
            return result
        except Exception as exc:
            logger.warning(
                "Mem0 search_memories failed after retries: %s — falling back to Supabase", exc
            )

    resolved_domains = domains if domains is not None else infer_domains(query)
    facts = await _supabase_search_memories(user_id, limit=limit, domains=resolved_domains)
    return {"facts": facts, "relations": []}


# ── Feature 4 — Full CRUD API ─────────────────────────────────────────────────

async def get_all_memories(
    user_id: str,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return all stored memories for a user (optionally scoped to a session).

    With Mem0: calls AsyncMemory.get_all() with retry + timeout.
    Fallback: fetches from Supabase user_memories table.
    """
    m = await get_mem0()

    if m is not None:
        try:
            kwargs: dict[str, Any] = {"user_id": user_id}
            if run_id:
                kwargs["run_id"] = run_id
            raw = await _async_retry("get_all_memories", lambda: m.get_all(**kwargs))
            # Mem0 returns {"results": [...]} or a plain list depending on version
            if isinstance(raw, dict):
                return raw.get("results", [])
            return raw if isinstance(raw, list) else []
        except Exception as exc:
            logger.warning("Mem0 get_all_memories failed after retries: %s — falling back", exc)

    # Supabase fallback
    from src.infra import db
    client = await db.get_client()
    if client is None:
        return []
    try:
        res = (
            await client.table("user_memories")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return res.data or []
    except Exception as exc:
        logger.warning("Supabase get_all_memories failed for user %s: %s", user_id, exc)
        return []


async def get_memory(memory_id: str) -> dict[str, Any] | None:
    """Fetch a single memory by its Mem0 ID.

    Returns None if Mem0 is not configured or the memory is not found.
    """
    m = await get_mem0()
    if m is None:
        return None
    try:
        return await _async_retry("get_memory", lambda: m.get(memory_id))
    except Exception as exc:
        logger.warning("Mem0 get_memory failed for id=%s: %s", memory_id, exc)
        return None


async def update_memory(memory_id: str, data: str) -> dict[str, Any]:
    """Replace the text of an existing memory.

    Args:
        memory_id: Mem0 memory UUID.
        data:      New text content for the memory.
    """
    m = await get_mem0()
    if m is None:
        return {"status": "error", "reason": "Mem0 not configured"}
    try:
        result = await _async_retry("update_memory", lambda: m.update(memory_id, data))
        logger.info("Mem0 memory updated — id=%s", memory_id)
        return {"status": "updated", "memory_id": memory_id, "result": result}
    except Exception as exc:
        logger.warning("Mem0 update_memory failed for id=%s: %s", memory_id, exc)
        return {"status": "error", "memory_id": memory_id, "error": str(exc)}


async def delete_memory(memory_id: str) -> dict[str, Any]:
    """Delete a single memory by its Mem0 ID."""
    m = await get_mem0()
    if m is None:
        return {"status": "error", "reason": "Mem0 not configured"}
    try:
        result = await _async_retry("delete_memory", lambda: m.delete(memory_id))
        logger.info("Mem0 memory deleted — id=%s", memory_id)
        return {"status": "deleted", "memory_id": memory_id, "result": result}
    except Exception as exc:
        logger.warning("Mem0 delete_memory failed for id=%s: %s", memory_id, exc)
        return {"status": "error", "memory_id": memory_id, "error": str(exc)}


async def delete_all_memories(
    user_id: str,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Delete all memories for a user, optionally scoped to a session.

    With Mem0: calls AsyncMemory.delete_all() with retry + timeout.
    No Supabase fallback (destructive operation — only executes if Mem0 configured).
    """
    m = await get_mem0()
    if m is None:
        return {"status": "error", "reason": "Mem0 not configured"}
    try:
        kwargs: dict[str, Any] = {"user_id": user_id}
        if run_id:
            kwargs["run_id"] = run_id
        result = await _async_retry("delete_all_memories", lambda: m.delete_all(**kwargs))
        logger.info("Mem0 all memories deleted — user=%s run_id=%s", user_id, run_id)
        return {"status": "deleted_all", "user_id": user_id, "run_id": run_id, "result": result}
    except Exception as exc:
        logger.warning("Mem0 delete_all_memories failed for user=%s: %s", user_id, exc)
        return {"status": "error", "user_id": user_id, "error": str(exc)}


async def get_memory_history(memory_id: str) -> list[dict[str, Any]]:
    """Return the change history for a specific memory (ADD → UPDATE → DELETE chain).

    Useful for auditing how a memory evolved over time.
    Returns an empty list if Mem0 is not configured or history is unavailable.
    """
    m = await get_mem0()
    if m is None:
        return []
    try:
        result = await _async_retry("get_memory_history", lambda: m.history(memory_id))
        return result if isinstance(result, list) else []
    except Exception as exc:
        logger.warning("Mem0 get_memory_history failed for id=%s: %s", memory_id, exc)
        return []


# ── Feature 5 — Multimodal Support ────────────────────────────────────────────

# Supported image MIME types per Mem0 docs
_SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"}
# Max base64 payload ~15 MB (20 MB limit with encoding overhead)
_MAX_BASE64_BYTES = 15 * 1024 * 1024


def _validate_image_input(image_url: str) -> None:
    """Raise ValueError for obviously invalid inputs before hitting Mem0."""
    if not image_url:
        raise ValueError("image_url is required")

    # Base64 data URI — validate prefix and size
    if image_url.startswith("data:"):
        if ";base64," not in image_url:
            raise ValueError(
                "Invalid base64 data URI — expected format: data:image/<type>;base64,<data>"
            )
        mime = image_url.split(";")[0].replace("data:", "").strip()
        if mime not in _SUPPORTED_IMAGE_TYPES:
            raise ValueError(
                f"Unsupported image type '{mime}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_IMAGE_TYPES))}"
            )
        # Rough size check: base64 is ~4/3 of binary size
        b64_data = image_url.split(";base64,", 1)[1]
        approx_bytes = len(b64_data) * 3 // 4
        if approx_bytes > _MAX_BASE64_BYTES:
            raise ValueError(
                f"Image too large (~{approx_bytes // (1024*1024):.1f} MB). "
                "Maximum is 15 MB. Compress or resize before sending."
            )
        return

    # URL — basic format check
    if not (image_url.startswith("http://") or image_url.startswith("https://")):
        raise ValueError(
            "image_url must be a public https:// URL or a data:image/<type>;base64,<data> string"
        )


async def add_memory_from_image(
    image_url: str,
    context_text: str,
    user_id: str,
    domain: str = "nutrition",
    run_id: str | None = None,
) -> dict[str, Any]:
    """Extract and persist memories from an image (food photo, nutrition label, etc.).

    Builds a Mem0 multimodal message: context text + image_url content block.
    Mem0 runs the image through the configured vision LLM, extracts key facts,
    and stores them as searchable memories alongside text memories.

    Wellness use cases:
      - Food photo       → extracted meal items, portion estimates, ingredients
      - Nutrition label  → exact macros, serving size, ingredients list
      - Workout screenshot → activity, duration, calories burned
      - Medical document → condition name, medication, dosage

    Falls back to vision LLM extraction + plain Supabase if Mem0 not configured.

    Args:
        image_url   : Public https:// URL or data:image/<type>;base64,<data> string
        context_text: User's description ("This is my lunch", "Nutrition label for oats")
        user_id     : User identifier
        domain      : Memory domain — "nutrition", "fitness", "medical", "general"
        run_id      : Session ID for scoped grouping (Feature 4)
    """
    if not user_id:
        return {"status": "skipped", "reason": "missing user_id"}

    # Validate before hitting any external service
    try:
        _validate_image_input(image_url)
    except ValueError as exc:
        logger.warning("Multimodal add rejected — %s", exc)
        return {"status": "error", "error": str(exc)}

    m = await get_mem0()

    if m is not None:
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context_text or "Extract health-relevant details from this image."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ]
            kwargs: dict[str, Any] = {
                "user_id": user_id,
                "metadata": {"domain": domain, "source": "image"},
            }
            if run_id:
                kwargs["run_id"] = run_id

            result = await _async_retry(
                "add_memory_from_image",
                lambda: m.add(messages, **kwargs),
            )
            logger.info(
                "Mem0 image memory saved — user=%s domain=%s run_id=%s",
                user_id, domain, run_id,
            )
            return {
                "status": "saved",
                "source": "image",
                "domain": domain,
                "result": result,
            }
        except Exception as exc:
            logger.warning(
                "Mem0 add_memory_from_image failed: %s — falling back to vision extraction",
                exc,
            )

    # ── Fallback: vision LLM extraction → Supabase ────────────────────────────
    return await _vision_extract_and_save(image_url, context_text, user_id, domain)


async def _vision_extract_and_save(
    image_url: str,
    context_text: str,
    user_id: str,
    domain: str,
) -> dict[str, Any]:
    """Fallback: use OpenAI vision to extract facts, then save to Supabase."""
    from src.infra.config import get_settings
    try:
        from openai import AsyncOpenAI

        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)

        domain_instruction = {
            "nutrition": "Extract food items, portion sizes, calories, and macronutrients.",
            "fitness":   "Extract workout type, duration, intensity, and calories burned.",
            "medical":   "Extract condition names, medication names, dosages, and frequencies.",
            "general":   "Extract any health or lifestyle relevant facts.",
        }.get(domain, "Extract any health-relevant facts.")

        response = await client.chat.completions.create(
            model=settings.llm_model_pipeline,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Context: {context_text}\n\n"
                                f"{domain_instruction}\n"
                                "Return a concise third-person factual statement about the user "
                                "based on what you see. One sentence only."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            max_completion_tokens=150,
        )
        extracted = response.choices[0].message.content or ""
        if extracted:
            result = await _supabase_add_memory(extracted.strip(), user_id, domain)
            logger.info(
                "Vision fallback — extracted and saved to Supabase: user=%s fact=%r",
                user_id, extracted,
            )
            return {
                "status": result.get("status", "saved"),
                "source": "image_vision_fallback",
                "extracted_fact": extracted,
                "domain": domain,
            }
    except Exception as exc:
        logger.warning("Vision fallback extraction failed: %s", exc)

    return {"status": "error", "error": "Image processing failed — both Mem0 and vision fallback unavailable"}


# ── Feature 9 — Configure OSS Stack: validation + config summary ──────────────

def _mask_secret(value: str, keep_chars: int = 4) -> str:
    """Return value with all but the first `keep_chars` characters masked."""
    if not value:
        return ""
    visible = value[:keep_chars]
    return visible + "***"


def _mask_url(url: str) -> str:
    """Return URL with credentials/tokens stripped for safe logging."""
    if not url:
        return ""
    # For Qdrant cloud URLs: show scheme + host, mask rest
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.hostname}"
    except Exception:
        return url[:20] + "***"


def _summarise_vector_store(settings: Any) -> dict:
    """Return a safe, secret-free summary dict for the configured vector store."""
    provider = (settings.mem0_vector_store_provider or "qdrant").lower()
    collection = settings.mem0_collection_name
    dims = settings.mem0_embedder_dims

    if provider == "qdrant":
        return {
            "provider": "qdrant",
            "url": _mask_url(settings.qdrant_url),
            "api_key": _mask_secret(settings.qdrant_api_key),
            "collection": collection,
            "embedding_dims": dims,
        }
    if provider in ("pgvector", "supabase"):
        # Mask the connection string — it contains credentials
        conn = settings.supabase_connection_string
        masked_conn = _mask_url(conn) if conn else ""
        return {
            "provider": "supabase",
            "connection": masked_conn,
            "collection": collection,
            "embedding_dims": dims,
        }
    if provider == "chroma":
        return {
            "provider": "chroma",
            "mode": "server" if settings.chroma_host else "embedded",
            "host": settings.chroma_host or None,
            "port": settings.chroma_port if settings.chroma_host else None,
            "path": settings.chroma_path if not settings.chroma_host else None,
            "collection": collection,
            "embedding_dims": dims,
        }
    if provider == "pinecone":
        return {
            "provider": "pinecone",
            "index": settings.pinecone_index_name,
            "api_key": _mask_secret(settings.pinecone_api_key),
            "collection": collection,
            "embedding_dims": dims,
        }
    # Unknown — show minimal safe info
    return {
        "provider": provider,
        "collection": collection,
        "embedding_dims": dims,
    }


def get_mem0_config_summary() -> dict[str, Any]:
    """Return a human-readable, secret-free summary of the Mem0 OSS configuration.

    Safe to log and surface in the /health endpoint. API keys, passwords, and
    full URLs are masked. Shows per-component provider, model, and key settings.

    Returns:
        Dict with keys: mem0_configured, proxy_configured, vector_store, llm,
        embedder, graph_store, reranker, custom_prompts.
    """
    from src.infra.config import get_settings
    settings = get_settings()

    return {
        "mem0_configured": settings.is_mem0_configured,
        "proxy_configured": settings.is_proxy_configured,
        # Feature 11 — reflect actual configured vector store provider
        "vector_store": _summarise_vector_store(settings),
        "llm": {
            # Feature 10 — reflect actual configured provider
            "provider": settings.mem0_llm_provider,
            "model": settings.mem0_llm_model or settings.llm_model_pipeline,
            "temperature": settings.mem0_llm_temperature,
            "max_tokens": settings.mem0_llm_max_tokens,
            "base_url": _mask_url(settings.mem0_llm_base_url) if settings.mem0_llm_base_url else None,
        },
        # Feature 12 — reflect actual configured embedder
        "embedder": {
            "provider": settings.mem0_embedder_provider,
            "model": settings.mem0_embedder_model or _SUPPORTED_EMBEDDERS.get(
                (settings.mem0_embedder_provider or "openai").lower(), "text-embedding-3-small"
            ),
            "dims": settings.mem0_embedder_dims,
        },
        "graph_store": {
            "provider": "neo4j" if settings.neo4j_url else "none",
            "url": _mask_url(settings.neo4j_url) if settings.neo4j_url else None,
            "username": settings.neo4j_username if settings.neo4j_url else None,
        },
        # Feature 13 — reflect actual configured reranker
        "reranker": {
            "provider": settings.mem0_reranker_provider,
            "model": settings.mem0_reranker_model or _SUPPORTED_RERANKERS.get(
                (settings.mem0_reranker_provider or "llm_reranker").lower(), ""
            ),
            "top_k": settings.mem0_reranker_top_k,
            "wellness_prompt": getattr(settings, "mem0_reranker_use_wellness_prompt", True),
        },
        "custom_prompts": {
            "fact_extraction": True,   # Feature 6
            "update_memory": True,     # Feature 7
        },
        "version": "v1.1",
    }


async def validate_mem0_config() -> dict[str, str]:
    """Probe each Mem0 OSS component and return per-component status.

    Runs at startup to catch misconfigurations early (wrong Qdrant URL, model
    dimension mismatch, invalid API key) before the first user request.

    Each component is tested independently — a failure in one does not block
    others. All exceptions are caught so this function never raises.

    Returns:
        Dict mapping component name → "ok" | "unconfigured" | "error: <msg>"
        Keys: qdrant, embedder, llm, graph_store, overall
    """
    from src.infra.config import get_settings
    settings = get_settings()

    vs_provider = (settings.mem0_vector_store_provider or "qdrant").lower()
    emb_provider = (settings.mem0_embedder_provider or "openai").lower()

    status: dict[str, str] = {
        "vector_store": "unconfigured",
        "embedder": "unconfigured",
        "llm": "unconfigured",
        "graph_store": "unconfigured",
        "overall": "unconfigured",
    }

    if not settings.is_mem0_configured and not settings.is_proxy_configured:
        logger.info("Mem0 config validation: vector store not configured — skipping all checks")
        return status

    # ── Vector store connectivity (Feature 11 — provider-aware) ───────────────
    try:
        if vs_provider == "qdrant":
            from qdrant_client import QdrantClient
            qc = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, qc.get_collections),
                timeout=8.0,
            )
            status["vector_store"] = "ok"
            logger.info("Mem0 config validation: vector_store ✓ (qdrant %s)", _mask_url(settings.qdrant_url))

        elif vs_provider in ("pgvector", "supabase"):
            # Lightweight check — confirm connection string is present
            if not settings.supabase_connection_string:
                status["vector_store"] = "error: SUPABASE_CONNECTION_STRING not set"
            else:
                status["vector_store"] = "configured"
                logger.info("Mem0 config validation: vector_store ✓ (supabase/pgvector connection string present)")

        elif vs_provider == "chroma":
            status["vector_store"] = "configured"
            logger.info("Mem0 config validation: vector_store ✓ (chroma — no remote probe needed)")

        elif vs_provider == "pinecone":
            if not settings.pinecone_api_key:
                status["vector_store"] = "error: PINECONE_API_KEY not set"
            else:
                status["vector_store"] = "configured"
                logger.info("Mem0 config validation: vector_store ✓ (pinecone api key present)")

        else:
            status["vector_store"] = f"error: unknown vector store provider {vs_provider!r}"

    except ImportError as exc:
        status["vector_store"] = f"error: missing dependency — {exc}"
    except asyncio.TimeoutError:
        status["vector_store"] = f"error: {vs_provider} connection timeout after 8s"
    except Exception as exc:
        status["vector_store"] = f"error: {exc!s}"
        logger.warning("Mem0 config validation: vector_store failed (%s) — %s", vs_provider, exc)

    # ── Embedder (Feature 12 — provider-aware) ─────────────────────────────────
    emb_model = settings.mem0_embedder_model or _SUPPORTED_EMBEDDERS.get(emb_provider, "text-embedding-ada-002")
    emb_dims = settings.mem0_embedder_dims
    try:
        if emb_provider == "openai":
            from openai import AsyncOpenAI
            oc = AsyncOpenAI(api_key=settings.openai_api_key)
            resp = await asyncio.wait_for(
                oc.embeddings.create(input=["ping"], model=emb_model),
                timeout=8.0,
            )
            actual_dims = len(resp.data[0].embedding)
            if actual_dims != emb_dims:
                status["embedder"] = f"error: expected {emb_dims} dims, got {actual_dims} — check MEM0_EMBEDDER_DIMS"
                logger.warning("Mem0 config validation: embedder dim mismatch (expected=%d got=%d)", emb_dims, actual_dims)
            else:
                status["embedder"] = "ok"
                logger.info("Mem0 config validation: embedder ✓ (openai %s, %d dims)", emb_model, emb_dims)

        elif emb_provider == "azure_openai":
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                status["embedder"] = "error: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set"
            else:
                status["embedder"] = "configured"
                logger.info("Mem0 config validation: embedder ✓ (azure_openai credentials present)")

        elif emb_provider == "ollama":
            base_url = settings.mem0_embedder_base_url or "http://localhost:11434"
            status["embedder"] = "configured"
            logger.info("Mem0 config validation: embedder ✓ (ollama %s at %s)", emb_model, base_url)

        elif emb_provider in ("huggingface", "hugging_face"):
            status["embedder"] = "configured"
            logger.info("Mem0 config validation: embedder ✓ (huggingface %s)", emb_model)

        elif emb_provider == "aws_bedrock":
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                status["embedder"] = "configured"
                logger.info("Mem0 config validation: embedder ✓ (aws_bedrock credentials present)")
            else:
                try:
                    import boto3
                    session = boto3.Session()
                    if session.get_credentials() is not None:
                        status["embedder"] = "configured"
                    else:
                        status["embedder"] = "error: AWS credentials not found"
                except ImportError:
                    status["embedder"] = "error: boto3 not installed (pip install boto3)"

        else:
            # google_ai, vertexai, together, lmstudio — check configured
            status["embedder"] = "configured"
            logger.info("Mem0 config validation: embedder configured (%s %s)", emb_provider, emb_model)

    except asyncio.TimeoutError:
        status["embedder"] = f"error: {emb_provider} embedder timeout after 8s"
    except Exception as exc:
        status["embedder"] = f"error: {exc!s}"
        logger.warning("Mem0 config validation: embedder failed (%s) — %s", emb_provider, exc)

    # ── LLM (validate the configured provider, not always OpenAI) ────────────
    provider = settings.mem0_llm_provider or "openai"
    model = settings.mem0_llm_model or settings.llm_model_pipeline
    try:
        if provider in ("openai", "openai_structured", "litellm"):
            from openai import AsyncOpenAI
            kwargs: dict = {"api_key": settings.openai_api_key}
            if settings.mem0_llm_base_url:
                kwargs["base_url"] = settings.mem0_llm_base_url
            oc = AsyncOpenAI(**kwargs)
            await asyncio.wait_for(
                oc.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_completion_tokens=10,
                ),
                timeout=10.0,
            )
            status["llm"] = "ok"
            logger.info("Mem0 config validation: LLM ✓ (provider=%s model=%s)", provider, model)

        elif provider == "anthropic":
            if not settings.anthropic_api_key:
                status["llm"] = "error: ANTHROPIC_API_KEY not set"
            else:
                import anthropic as anthropic_sdk
                ac = anthropic_sdk.AsyncAnthropic(api_key=settings.anthropic_api_key)
                await asyncio.wait_for(
                    ac.messages.create(
                        model=model,
                        max_tokens=1,
                        messages=[{"role": "user", "content": "ping"}],
                    ),
                    timeout=10.0,
                )
                status["llm"] = "ok"
                logger.info("Mem0 config validation: LLM ✓ (provider=anthropic model=%s)", model)

        elif provider == "groq":
            if not settings.groq_api_key:
                status["llm"] = "error: GROQ_API_KEY not set"
            else:
                from openai import AsyncOpenAI
                gc = AsyncOpenAI(api_key=settings.groq_api_key, base_url="https://api.groq.com/openai/v1")
                await asyncio.wait_for(
                    gc.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                    ),
                    timeout=10.0,
                )
                status["llm"] = "ok"
                logger.info("Mem0 config validation: LLM ✓ (provider=groq model=%s)", model)

        elif provider == "azure_openai":
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                status["llm"] = "error: AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set"
            else:
                status["llm"] = "configured"  # skip live probe (Azure needs deployment name)
                logger.info("Mem0 config validation: LLM Azure OpenAI credentials present")

        elif provider == "ollama":
            base_url = settings.mem0_llm_base_url or "http://localhost:11434"
            from openai import AsyncOpenAI
            oc2 = AsyncOpenAI(api_key="ollama", base_url=f"{base_url}/v1")
            await asyncio.wait_for(
                oc2.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                ),
                timeout=8.0,
            )
            status["llm"] = "ok"
            logger.info("Mem0 config validation: LLM ✓ (provider=ollama base_url=%s)", base_url)

        elif provider == "aws_bedrock":
            # Skip live probe — boto3 invocation requires model access grant in Bedrock console.
            # Validate that credentials are present via env vars or IAM role.
            if settings.aws_access_key_id and settings.aws_secret_access_key:
                status["llm"] = "configured"
                logger.info(
                    "Mem0 config validation: LLM AWS Bedrock credentials present (region=%s)",
                    settings.aws_region or "us-east-1",
                )
            else:
                try:
                    import boto3
                    session = boto3.Session()
                    creds = session.get_credentials()
                    if creds is not None:
                        status["llm"] = "configured"
                        logger.info("Mem0 config validation: LLM AWS Bedrock IAM credentials found")
                    else:
                        status["llm"] = "error: AWS credentials not found (set AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY or configure IAM role)"
                except ImportError:
                    status["llm"] = "error: boto3 not installed (pip install boto3)"
                except Exception as exc:
                    status["llm"] = f"error: AWS credential check failed — {exc}"

        else:
            status["llm"] = f"error: unknown provider {provider!r}"

    except asyncio.TimeoutError:
        status["llm"] = f"error: {provider} LLM timeout after 10s"
    except Exception as exc:
        status["llm"] = f"error: {exc!s}"
        logger.warning("Mem0 config validation: LLM failed (provider=%s) — %s", provider, exc)

    # ── Neo4j graph store — test actual connectivity ──────────────────────────
    if settings.neo4j_url and settings.neo4j_password:
        try:
            from langchain_neo4j import Neo4jGraph

            neo4j_kwargs: dict = {
                "url": settings.neo4j_url,
                "username": settings.neo4j_username,
                "password": settings.neo4j_password,
                "refresh_schema": False,
            }
            if settings.neo4j_database:
                neo4j_kwargs["database"] = settings.neo4j_database

            def _test_neo4j_connection() -> None:
                Neo4jGraph(**neo4j_kwargs)

            await asyncio.wait_for(
                asyncio.to_thread(_test_neo4j_connection),
                timeout=10.0,
            )
            status["graph_store"] = "ok"
            logger.info(
                "Mem0 config validation: Neo4j connection OK (url=%s user=%s db=%s)",
                _mask_url(settings.neo4j_url),
                settings.neo4j_username,
                settings.neo4j_database or "neo4j",
            )
        except asyncio.TimeoutError:
            status["graph_store"] = "error: Neo4j connection timed out after 10s"
            logger.warning(
                "Mem0 config validation: Neo4j timed out (%s)", _mask_url(settings.neo4j_url)
            )
        except Exception as exc:
            status["graph_store"] = f"error: {exc}"
            logger.warning(
                "Mem0 config validation: Neo4j connection failed (url=%s user=%s db=%s) — %s",
                _mask_url(settings.neo4j_url),
                settings.neo4j_username,
                settings.neo4j_database or "neo4j",
                exc,
                exc_info=True,
            )
    elif settings.neo4j_url:
        status["graph_store"] = "error: NEO4J_PASSWORD missing"
    else:
        status["graph_store"] = "unconfigured"

    # ── Overall ────────────────────────────────────────────────────────────────
    component_statuses = [status["vector_store"], status["embedder"], status["llm"]]
    if all(s == "ok" for s in component_statuses):
        status["overall"] = "ok"
    elif any(s.startswith("error") for s in component_statuses):
        status["overall"] = "degraded"
    else:
        status["overall"] = "partial"

    logger.info("Mem0 config validation complete — overall=%s", status["overall"])
    return status


# ── Result normalisation ───────────────────────────────────────────────────────

def _parse_mem0_results(raw: Any) -> dict[str, Any]:
    """Normalise Mem0 search response into {"facts": [...], "relations": [...]}."""
    facts: list[str] = []
    relations: list[dict] = []

    if isinstance(raw, dict):
        for r in raw.get("results", []):
            memory_text = r.get("memory", "")
            if memory_text:
                facts.append(memory_text)
        relations = raw.get("relations", [])
    elif isinstance(raw, list):
        for r in raw:
            memory_text = r.get("memory", "") if isinstance(r, dict) else str(r)
            if memory_text:
                facts.append(memory_text)

    return {"facts": facts, "relations": relations}


# ── Supabase fallback helpers ──────────────────────────────────────────────────

async def _supabase_add_memory(
    fact: str, user_id: str, domain: str = "general"
) -> dict[str, Any]:
    """Save a memory fact to Supabase with Feature 6 extraction + Feature 7 reconciliation.

    Pipeline:
      1. Feature 6 — extract_wellness_facts(): filter noise, split compound inputs
      2. Feature 7 — resolve_memory_actions(): compare extracted facts against existing
         memories and decide ADD / UPDATE / DELETE / NONE for each
      3. Apply each action to Supabase (insert / update / delete / skip)

    This mirrors the behaviour of Mem0's custom_fact_extraction_prompt +
    custom_update_memory_prompt pipeline on the primary Mem0 path.
    Falls back to simple insert on any LLM failure (fail-open).
    """
    from src.infra import db

    # ── Step 1: Feature 6 — wellness fact extraction ──────────────────────────
    extracted = await extract_wellness_facts(fact)
    facts_to_reconcile: list[str] = extracted if extracted else [fact]

    if not facts_to_reconcile:
        return {"status": "skipped", "reason": "no wellness facts extracted"}

    # ── Step 2: Feature 7 — fetch existing memories for reconciliation ────────
    existing: list[dict[str, Any]] = []
    db_client = await db.get_client()
    if db_client is not None:
        try:
            res = (
                await db_client.table("user_memories")
                .select("id, fact, domain")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(50)
                .execute()
            )
            # Normalise to {id, text} for the reconciler
            existing = [
                {"id": row["id"], "text": row["fact"], "domain": row.get("domain", "general")}
                for row in (res.data or [])
            ]
        except Exception as exc:
            logger.debug("Could not fetch existing memories for reconciliation: %s", exc)

    actions = await resolve_memory_actions(facts_to_reconcile, existing)

    # ── Step 3: Apply each action ─────────────────────────────────────────────
    results = []
    if db_client is None:
        return {"status": "skipped", "reason": "db not configured"}

    for action in actions:
        event = action.get("event", "ADD")
        text = action.get("text", "")
        action_id = action.get("id", "")

        if event == "NONE":
            results.append({"status": "unchanged", "fact": text})
            continue

        if event == "ADD":
            try:
                await db_client.table("user_memories").insert(
                    {"user_id": user_id, "fact": text, "domain": domain}
                ).execute()
                logger.info(
                    "Supabase memory ADD — user=%s domain=%s fact=%r", user_id, domain, text
                )
                results.append({"status": "saved", "event": "ADD", "fact": text, "domain": domain})
            except Exception as exc:
                logger.warning("Supabase ADD failed: %s", exc)
                results.append({"status": "error", "event": "ADD", "error": str(exc)})

        elif event == "UPDATE":
            # action_id is the Supabase row UUID of the memory being replaced
            try:
                await db_client.table("user_memories").update(
                    {"fact": text, "domain": domain}
                ).eq("id", action_id).eq("user_id", user_id).execute()
                old = action.get("old_memory", "")
                logger.info(
                    "Supabase memory UPDATE — user=%s id=%s old=%r new=%r",
                    user_id, action_id, old, text,
                )
                results.append({"status": "updated", "event": "UPDATE", "fact": text, "old_memory": old})
            except Exception as exc:
                logger.warning("Supabase UPDATE failed (id=%s): %s — falling back to ADD", action_id, exc)
                try:
                    await db_client.table("user_memories").insert(
                        {"user_id": user_id, "fact": text, "domain": domain}
                    ).execute()
                    results.append({"status": "saved", "event": "ADD_FALLBACK", "fact": text})
                except Exception:
                    results.append({"status": "error", "event": "UPDATE", "error": str(exc)})

        elif event == "DELETE":
            try:
                await db_client.table("user_memories").delete().eq("id", action_id).eq("user_id", user_id).execute()
                logger.info(
                    "Supabase memory DELETE — user=%s id=%s fact=%r", user_id, action_id, text
                )
                results.append({"status": "deleted", "event": "DELETE", "fact": text})
            except Exception as exc:
                logger.warning("Supabase DELETE failed (id=%s): %s", action_id, exc)
                results.append({"status": "error", "event": "DELETE", "error": str(exc)})

    return results[0] if results else {"status": "skipped", "reason": "no actions applied"}


async def _supabase_search_memories(
    user_id: str,
    limit: int = 50,
    domains: list[str] | None = None,
) -> list[str]:
    from src.infra import db
    try:
        client = await db.get_client()
        if client is None:
            return []

        query = (
            client.table("user_memories")
            .select("fact, domain")
            .eq("user_id", user_id)
        )

        all_domains = {"nutrition", "fitness", "medical", "general"}
        if domains and set(domains) < all_domains:
            query = query.in_("domain", domains)

        result = await query.order("created_at", desc=True).limit(limit).execute()
        return [row["fact"] for row in (result.data or [])]

    except Exception as exc:
        logger.warning("Supabase search_memories failed for user %s: %s", user_id, exc)
        return await db.get_recent_memories(user_id)


