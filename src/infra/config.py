"""Application configuration via pydantic-settings.

All values are loaded from environment variables / .env file.
Provides a lazily-instantiated singleton via `get_settings()`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    llm_model_agent: str = Field("gpt-4o", alias="LLM_MODEL_AGENT")
    llm_model_pipeline: str = Field("gpt-4o-mini", alias="LLM_MODEL_PIPELINE")

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = Field("", alias="SUPABASE_URL")
    supabase_service_role_key: str = Field("", alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_connection_string: str = Field("", alias="SUPABASE_CONNECTION_STRING")

    # ── Qdrant (vector store for Mem0) ────────────────────────────────────────
    qdrant_url: str = Field("", alias="QDRANT_URL")
    qdrant_api_key: str = Field("", alias="QDRANT_API_KEY")

    # ── Neo4j (graph memory for Mem0) ─────────────────────────────────────────
    neo4j_url: str = Field("", alias="NEO4J_URL")
    neo4j_username: str = Field("neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field("", alias="NEO4J_PASSWORD")
    # Aura instances use the instance ID as the database name (not 'neo4j').
    # Leave blank to use the driver default ('neo4j'), which works on self-hosted.
    neo4j_database: str = Field("", alias="NEO4J_DATABASE")

    # ── Mem0 OSS tuning (Feature 9 — Configure OSS Stack) ────────────────────
    # Collection name isolates tenants and enables per-tenant retention policies.
    # Reranker top_k: docs recommend 10–20; values >20 add latency without gains.
    mem0_collection_name: str = Field("reso_memories", alias="MEM0_COLLECTION_NAME")
    mem0_reranker_top_k: int = Field(10, alias="MEM0_RERANKER_TOP_K")

    # ── Mem0 Reranker (Feature 13 — Configurable Reranker) ───────────────────
    # Selects the reranker that rescores vector search hits for precision.
    # Supported: llm_reranker | cohere | sentence_transformer | huggingface
    #            | zero_entropy | none (disables reranking)
    # Defaults to "llm_reranker" (existing behaviour — OpenAI gpt-4o-mini).
    # API-first (cohere, zero_entropy) → best quality, adds network latency.
    # Self-hosted (sentence_transformer, huggingface) → privacy, no API cost.
    # "none" → disables second-pass reranking (use for low-latency paths).
    mem0_reranker_provider: str = Field("llm_reranker", alias="MEM0_RERANKER_PROVIDER")
    # Model override for the chosen provider (empty = provider default)
    mem0_reranker_model: str = Field("", alias="MEM0_RERANKER_MODEL")
    # Whether to use the wellness-specific LLM scoring prompt (Feature 13)
    mem0_reranker_use_wellness_prompt: bool = Field(True, alias="MEM0_RERANKER_USE_WELLNESS_PROMPT")
    # API keys for external reranker services
    cohere_api_key: str = Field("", alias="COHERE_API_KEY")
    zero_entropy_api_key: str = Field("", alias="ZERO_ENTROPY_API_KEY")
    huggingface_api_key: str = Field("", alias="HUGGINGFACE_API_KEY")

    # ── Mem0 Vector Store (Feature 11 — Configurable Vector Database) ────────
    # Selects which vector database backs Mem0 memory operations. Supported:
    #   qdrant | pgvector | supabase | chroma | pinecone
    # Defaults to "qdrant" (existing behaviour). "pgvector"/"supabase" both map
    # to Mem0's "supabase" provider using supabase_connection_string.
    # "chroma" runs embedded (path) or client-server (host + port).
    # "pinecone" requires PINECONE_API_KEY.
    mem0_vector_store_provider: str = Field("qdrant", alias="MEM0_VECTOR_STORE_PROVIDER")
    # Chroma settings (embedded or server mode)
    chroma_path: str = Field("./chroma_db", alias="CHROMA_PATH")
    chroma_host: str = Field("", alias="CHROMA_HOST")
    chroma_port: int = Field(8000, alias="CHROMA_PORT")
    # Pinecone settings
    pinecone_api_key: str = Field("", alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field("reso-memories", alias="PINECONE_INDEX_NAME")

    # ── Mem0 Embedder (Feature 12 — Configurable Embedder) ───────────────────
    # Selects which embedding model backs Mem0 memory indexing + search. Supported:
    #   openai | azure_openai | ollama | huggingface | google_ai | vertexai
    #   together | lmstudio | aws_bedrock
    # Defaults to "openai" + text-embedding-3-small (preserves current behaviour).
    # IMPORTANT: if you change provider/model, also set MEM0_EMBEDDER_DIMS to
    # match the new model's output dimensions, and re-index all collections.
    mem0_embedder_provider: str = Field("openai", alias="MEM0_EMBEDDER_PROVIDER")
    mem0_embedder_model: str = Field("", alias="MEM0_EMBEDDER_MODEL")
    # Embedding dimensions — must match the model; 1536 for text-embedding-3-small,
    # 768 for many HuggingFace/Ollama models. Changing this requires re-indexing.
    mem0_embedder_dims: int = Field(1536, alias="MEM0_EMBEDDER_DIMS")
    # Optional: custom base URL for OpenAI-compatible embedder proxies / Ollama / LM Studio
    mem0_embedder_base_url: str = Field("", alias="MEM0_EMBEDDER_BASE_URL")

    # ── Mem0 LLM provider (Feature 10 — Configurable LLM Providers) ──────────
    # Mem0 memory operations (extraction, update, reranking) can use a different
    # LLM provider/model than the main wellness agent. Supported providers:
    #   openai | openai_structured | azure_openai | anthropic | groq | ollama
    #   litellm | aws_bedrock
    # Defaults to "openai" + llm_model_pipeline (preserves current behaviour).
    # Docs precedence: explicit config > env vars > defaults.
    mem0_llm_provider: str = Field("openai", alias="MEM0_LLM_PROVIDER")
    # Empty string → falls back to llm_model_pipeline
    mem0_llm_model: str = Field("", alias="MEM0_LLM_MODEL")
    # Docs recommend ≤ 0.2 for deterministic fact extraction
    mem0_llm_temperature: float = Field(0.0, alias="MEM0_LLM_TEMPERATURE")
    mem0_llm_max_tokens: int = Field(2000, alias="MEM0_LLM_MAX_TOKENS")
    # Optional: custom OpenAI-compatible base URL (LiteLLM, local proxy, Ollama)
    mem0_llm_base_url: str = Field("", alias="MEM0_LLM_BASE_URL")

    # ── Alternative provider API keys (only set the one you're using) ─────────
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    groq_api_key: str = Field("", alias="GROQ_API_KEY")
    azure_openai_api_key: str = Field("", alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str = Field("", alias="AZURE_OPENAI_ENDPOINT")
    azure_deployment_name: str = Field("", alias="AZURE_DEPLOYMENT_NAME")

    # ── AWS Bedrock (Feature 10 — Configurable LLM Providers) ────────────────
    # Requires boto3 authentication — set credentials via env vars or IAM role.
    # Docs: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    aws_region: str = Field("us-east-1", alias="AWS_REGION")
    aws_access_key_id: str = Field("", alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field("", alias="AWS_SECRET_ACCESS_KEY")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins for staging/production.
    # Example: "https://app.example.com,https://admin.example.com"
    # Ignored in development mode (wildcard is used there without credentials).
    cors_origins: str = Field("", alias="CORS_ORIGINS")

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret: str = Field(..., alias="JWT_SECRET")
    jwt_algorithm: str = Field("HS256", alias="JWT_ALGORITHM")

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_requests: int = Field(60, alias="RATE_LIMIT_REQUESTS")
    rate_limit_window_seconds: int = Field(60, alias="RATE_LIMIT_WINDOW_SECONDS")

    # ── Observability ─────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = Field(
        "http://localhost:4317", alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    otel_service_name: str = Field("reso-orchestrator", alias="OTEL_SERVICE_NAME")

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: Literal["development", "staging", "production"] = Field(
        "development", alias="APP_ENV"
    )
    log_level: str = Field("info", alias="LOG_LEVEL")
    context_message_limit: int = Field(30, alias="CONTEXT_MESSAGE_LIMIT")
    memory_lookback_days: int = Field(7, alias="MEMORY_LOOKBACK_DAYS")

    @property
    def is_mem0_configured(self) -> bool:
        """Feature 11 — true when the configured vector store has its required credentials."""
        provider = (self.mem0_vector_store_provider or "qdrant").lower()
        if provider == "qdrant":
            return bool(self.qdrant_url and self.neo4j_url and self.neo4j_password)
        if provider in ("pgvector", "supabase"):
            return bool(self.supabase_connection_string)
        if provider == "chroma":
            return True  # embedded mode needs no external credentials
        if provider == "pinecone":
            return bool(self.pinecone_api_key)
        return False

    @property
    def is_proxy_configured(self) -> bool:
        """Feature 8 — proxy needs only the vector store (no graph store / reranker)."""
        provider = (self.mem0_vector_store_provider or "qdrant").lower()
        if provider == "qdrant":
            return bool(self.qdrant_url)
        if provider in ("pgvector", "supabase"):
            return bool(self.supabase_connection_string)
        if provider == "chroma":
            return True
        if provider == "pinecone":
            return bool(self.pinecone_api_key)
        return False

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_must_be_set(cls, v: str) -> str:
        if v == "change-me":
            raise ValueError(
                "JWT_SECRET is still set to the insecure default 'change-me'. "
                "Generate a secure value with: "
                "python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return v

    @field_validator("supabase_url", "supabase_service_role_key", "supabase_connection_string", mode="before")
    @classmethod
    def allow_empty_for_local_dev(cls, v: str) -> str:
        return v or ""

    @property
    def cors_origins_list(self) -> list[str]:
        """Parsed list of allowed CORS origins from the CORS_ORIGINS env var."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
