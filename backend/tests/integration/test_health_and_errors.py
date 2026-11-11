from fastapi import APIRouter

from app.core.database import make_engine
from app.core.errors import AppError


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["environment"] == "test"
    assert body["components"]["database"]["status"] == "ok"
    assert body["components"]["data_dir"]["status"] == "ok"
    # Phase 5: the vector index is real and reports its own state (empty, but healthy).
    assert body["components"]["vector_index"]["status"] == "ok"
    assert "vectors=0" in body["components"]["vector_index"]["detail"]


def test_health_degraded_when_database_unreachable(app, client, tmp_path):
    app.state.engine.dispose()
    app.state.engine = make_engine(f"sqlite:///{(tmp_path / 'missing' / 'x.db').as_posix()}")
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert r.json()["components"]["database"]["status"] == "error"


def test_unknown_route_uses_error_envelope(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "http_404"


def test_app_error_and_unhandled_error_are_safe(app, client):
    router = APIRouter()

    @router.get("/boom")
    def boom():
        raise RuntimeError("secret internal detail")

    @router.get("/dup")
    def dup():
        raise AppError(409, "duplicate", "Already exists")

    app.include_router(router)
    r = client.get("/boom")
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
    assert r.json()["error"]["code"] == "internal_error"

    r = client.get("/dup")
    assert r.status_code == 409
    assert r.json() == {"error": {"code": "duplicate", "message": "Already exists"}}


def test_cors_allows_configured_origin_only(client):
    ok = client.options(
        "/api/v1/health",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
    )
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options(
        "/api/v1/health",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"},
    )
    assert "access-control-allow-origin" not in bad.headers
