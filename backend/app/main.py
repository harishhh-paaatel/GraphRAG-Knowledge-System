"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import documents, graph, health, hybrid, search, vector
from app.api.handlers import generation_error_handler
from app.core.config import Settings, get_settings
from app.core.database import make_engine, make_session_factory
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.services.embeddings.factory import build_embedder
from app.services.embeddings.service import VectorService
from app.services.extraction.factory import build_extraction_provider
from app.services.generation.errors import GenerationError
from app.services.generation.history import QueryHistoryStore
from app.services.generation.registry import LLMRuntime
from app.services.graph.service import GraphService
from app.services.retrieval.graph_search import GraphRetriever
from app.services.retrieval.hybrid import HybridRetriever
from app.workers.ingestion_worker import IngestionWorker

API_PREFIX = "/api/v1"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the app. Passing ``settings`` makes tests fully isolated."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.graph.initialize()
        application.state.vectors.initialize()
        application.state.worker.recover_interrupted_jobs()
        yield
        application.state.worker.shutdown()
        application.state.hybrid.close()
        application.state.llm.close()

    app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = make_engine(settings.resolved_database_url)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.graph = GraphService(app.state.session_factory, settings.graph_file)
    app.state.graph_retriever = GraphRetriever(app.state.graph, app.state.session_factory, settings)
    app.state.vectors = VectorService(
        app.state.session_factory,
        settings.vector_index_file,
        build_embedder(settings),
        sync_on_startup=settings.vector_sync_on_startup,
    )
    app.state.hybrid = HybridRetriever(
        app.state.vectors,
        app.state.graph_retriever,
        app.state.session_factory,
        path_timeout_seconds=settings.hybrid_path_timeout_seconds,
    )
    app.state.llm = LLMRuntime(settings)
    app.state.query_history = QueryHistoryStore(
        app.state.session_factory, enabled=settings.query_history
    )
    app.state.worker = IngestionWorker(
        app.state.session_factory,
        settings,
        build_extraction_provider(settings),
        app.state.graph,
        app.state.vectors,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    register_exception_handlers(app)
    app.add_exception_handler(GenerationError, generation_error_handler)
    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(graph.router, prefix=API_PREFIX)
    app.include_router(search.router, prefix=API_PREFIX)
    app.include_router(hybrid.router, prefix=API_PREFIX)
    app.include_router(vector.router, prefix=API_PREFIX)
    return app


app = create_app()
