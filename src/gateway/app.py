"""FastAPI application factory — single-agent wellness assistant.

Responsibilities:
  - Create the app with lifespan (startup/shutdown hooks).
  - Compile the single-agent graph and attach to app.state.
  - Register middleware (CORS, structured logging).
  - Mount routers.
  - Instrument with OTEL.
"""

from __future__ import annotations

import logging
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from langgraph.checkpoint.memory import MemorySaver

from src.gateway.middleware.tracing import setup_tracing
from src.gateway.routes.agui import router as agui_router
from src.gateway.routes.chat import router as chat_router
from src.infra.config import get_settings
from src.infra.logger import setup_logger
from src.infra.redis_client import close_redis


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level, force=True)
    for noisy in ("httpx", "httpcore", "openai", "langgraph", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # transformers 5.x prints a LOAD REPORT warning for position_ids mismatch on
    # multi-qa-MiniLM-L6-cos-v1 — benign architectural difference, safe to suppress.
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    # mem0 internal INFO logs (index creation, etc.) are too verbose for production.
    logging.getLogger("mem0").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Pydantic serializer warnings",
    )
    # Supabase library passes deprecated kwargs to httpx — not our code, safe to suppress.
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="supabase")


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: compile single-agent graph. Shutdown: close connections."""
    logger = setup_logger(__name__)
    logger.info("Starting up Reso single-agent gateway…")

    checkpointer = MemorySaver()

    # ── Single-agent graph (unified wellness agent) ────────────────────────────
    try:
        from src.agent.graph import compile_graph as compile_single
        app.state.single_graph = compile_single(checkpointer)
        logger.info("Single-agent graph compiled and ready")
    except Exception:
        logger.exception("Failed to compile single-agent graph — will be None")
        app.state.single_graph = None

    # ── Mem0 OSS stack — config validation + singleton pre-warm ───────────────
    # Feature 9: validate each component at startup and log per-component status.
    # Non-blocking — failures degrade gracefully to Supabase fallback.
    try:
        from src.infra.mem0_client import (
            get_mem0,
            get_mem0_proxy,
            validate_mem0_config,
            get_mem0_config_summary,
        )
        summary = get_mem0_config_summary()
        logger.info(
            "Mem0 OSS config — collection=%s reranker_top_k=%s "
            "qdrant=%s graph=%s",
            summary["vector_store"]["collection"],
            summary["reranker"]["top_k"],
            summary["vector_store"]["url"] or "unconfigured",
            summary["graph_store"]["provider"],
        )
        validation = await validate_mem0_config()
        for component, comp_status in validation.items():
            if comp_status.startswith("error"):
                logger.warning("Mem0 component %s: %s", component, comp_status)
            elif comp_status != "unconfigured":
                logger.info("Mem0 component %s: %s", component, comp_status)
        await get_mem0()
        await get_mem0_proxy()
    except Exception:
        logger.warning("Mem0 startup initialisation failed — will fall back to Supabase")

    yield

    logger.info("Shutting down…")
    await close_redis()


# ── Factory ────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    _configure_logging()
    settings = get_settings()

    app = FastAPI(
        title="Reso — Single-Agent Wellness Assistant",
        description="Unified single wellness agent: all tools in one ReAct loop",
        version="0.2.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    setup_tracing(app)

    if settings.app_env == "development":
        # Wildcard origin and allow_credentials=True is an invalid CORS combination
        # (browsers reject it). In dev we allow all origins without credentials.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        # Staging / production: explicit origins with credentials allowed.
        # Set CORS_ORIGINS to a comma-separated list of allowed origins.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(chat_router)
    app.include_router(agui_router)

    static_dir = Path(__file__).parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_ui() -> FileResponse:
            return FileResponse(static_dir / "index.html")

        @app.get("/agui", include_in_schema=False)
        async def serve_agui() -> FileResponse:
            return FileResponse(static_dir / "agui.html")

        @app.head("/", include_in_schema=False)
        async def head_ui() -> dict:
            return {}

    return app
