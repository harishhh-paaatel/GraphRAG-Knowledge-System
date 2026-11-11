"""Health endpoint."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Request
from sqlalchemy import text

from app.schemas.health import ComponentStatus, HealthResponse

router = APIRouter(tags=["health"])


def _check_database(request: Request) -> ComponentStatus:
    try:
        with request.app.state.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return ComponentStatus(status="ok")
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(status="error", detail=type(exc).__name__)


def _check_data_dir(data_dir: Path) -> ComponentStatus:
    try:
        with tempfile.TemporaryFile(dir=data_dir):
            pass
        return ComponentStatus(status="ok")
    except OSError as exc:
        return ComponentStatus(status="error", detail=type(exc).__name__)


def _check_graph(request: Request) -> ComponentStatus:
    info = request.app.state.graph.status()
    detail = f"revision={info['revision']} nodes={info['node_count']} edges={info['edge_count']}"
    if info["state"] == "ok":
        return ComponentStatus(status="ok", detail=detail)
    if info["state"] == "stale":  # queries rebuild automatically; not a failure
        return ComponentStatus(status="ok", detail=detail + " (stale; rebuilds on next access)")
    return ComponentStatus(status="error", detail=info["persist_error"] or "graph unavailable")


def _check_vectors(request: Request) -> ComponentStatus:
    info = request.app.state.vectors.status()
    detail = (
        f"revision={info['index_revision']} vectors={info['vector_count']} model={info['model_id']}"
    )
    if info["state"] == "ok":
        return ComponentStatus(status="ok", detail=detail)
    if info["state"] == "building":
        return ComponentStatus(status="ok", detail=detail + " (initial build in progress)")
    if info["state"] == "stale":  # reads re-synchronize automatically
        return ComponentStatus(status="ok", detail=detail + " (stale; syncs on next access)")
    return ComponentStatus(
        status="error",
        detail=info["last_error"] or info["persist_error"] or "vector index unavailable",
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Liveness plus dependency checks. Graph and vector indexes report their own state."""
    settings = request.app.state.settings
    components = {
        "database": _check_database(request),
        "data_dir": _check_data_dir(settings.data_dir),
        "graph_index": _check_graph(request),
        "vector_index": _check_vectors(request),
    }
    degraded = any(c.status == "error" for c in components.values())
    return HealthResponse(
        status="degraded" if degraded else "ok",
        version=settings.app_version,
        environment=settings.environment,
        components=components,
    )
