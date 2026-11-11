"""FastAPI dependencies."""

from collections.abc import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.services.embeddings.service import VectorService
from app.services.generation.history import QueryHistoryStore
from app.services.generation.providers.base import LLMProvider
from app.services.generation.registry import LLMRuntime
from app.services.graph.service import GraphService
from app.services.retrieval.graph_search import GraphRetriever
from app.services.retrieval.hybrid import HybridRetriever
from app.workers.ingestion_worker import IngestionWorker


def get_session(request: Request) -> Iterator[Session]:
    """One session per request; endpoints commit explicitly, errors roll back."""
    session = request.app.state.session_factory()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_worker(request: Request) -> IngestionWorker:
    return request.app.state.worker


def get_graph(request: Request) -> GraphService:
    return request.app.state.graph


def get_vectors(request: Request) -> VectorService:
    return request.app.state.vectors


def get_graph_retriever(request: Request) -> GraphRetriever:
    return request.app.state.graph_retriever


def get_hybrid(request: Request) -> HybridRetriever:
    return request.app.state.hybrid


def get_llm_runtime(request: Request) -> LLMRuntime:
    return request.app.state.llm


def get_llm_provider(request: Request) -> LLMProvider:
    """The configured provider (built on first use; raises ``ProviderNotConfigured`` if it is
    not available, which the API layer maps to 503 ``llm_not_configured``)."""
    return request.app.state.llm.provider()


def get_query_history(request: Request) -> QueryHistoryStore:
    return request.app.state.query_history
