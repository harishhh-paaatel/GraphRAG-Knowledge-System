"""Dependency injection of the provider runtime and the history store into the application."""

from typing import Annotated

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from app.api.deps import get_llm_provider, get_llm_runtime, get_query_history
from app.core.config import Settings
from app.main import create_app
from app.models import Base
from app.services.generation.errors import ProviderTimeout
from app.services.generation.history import QueryHistoryStore
from app.services.generation.providers.base import LLMProvider
from app.services.generation.registry import LLMRuntime
from tests.llm_helpers import ScriptedProvider

ProviderDep = Annotated[LLMProvider, Depends(get_llm_provider)]


def build_app(tmp_path, **kw):
    app = create_app(Settings(data_dir=tmp_path, environment="test", _env_file=None, **kw))
    Base.metadata.create_all(app.state.engine)

    @app.get("/_probe/provider")
    def provider_probe(provider: ProviderDep):
        return {"name": provider.name, "model": provider.model, "mode": provider.mode.value}

    @app.get("/_probe/fail")
    def failing_probe(provider: ProviderDep):
        raise ProviderTimeout(
            "the model did not answer; key sk-abc123 was rejected", secrets=("sk-abc123",)
        )

    return app


@pytest.fixture
def app(tmp_path):
    application = build_app(tmp_path)
    yield application
    application.state.engine.dispose()


# ------------------------------------------------------------------------- start-up


def test_the_application_starts_with_the_default_extractive_provider(app):
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
    state = app.state.llm.describe()
    assert state["configured_provider"] == "extractive"  # the default provider
    assert state["registered"] is True  # extractive is registered after M1
    assert state["built"] is False  # still not built at start-up (lazy)


def test_the_runtime_and_the_history_store_are_on_the_application(app):
    assert isinstance(app.state.llm, LLMRuntime)
    assert isinstance(app.state.query_history, QueryHistoryStore)
    assert app.state.query_history.enabled is True


def test_the_history_store_follows_the_setting(tmp_path):
    off = build_app(tmp_path, query_history=False)
    try:
        assert off.state.query_history.enabled is False
    finally:
        off.state.engine.dispose()


def test_existing_endpoints_are_unaffected(app):
    with TestClient(app) as client:
        assert client.get("/api/v1/graph/overview").status_code == 200
        assert client.get("/api/v1/vector/index").status_code == 200
        assert client.get("/api/v1/documents").status_code == 200


# ------------------------------------------------------------------ dependency injection


def test_the_extractive_provider_is_now_registered_and_returns_200(app):
    """After M1 the extractive provider is built on demand and answers successfully."""
    with TestClient(app) as client:
        r = client.get("/_probe/provider")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "extractive" and body["mode"] == "extractive"


def test_an_unregistered_provider_name_still_gives_503_llm_not_configured(tmp_path):
    """An unregistered provider name still raises ProviderNotConfigured → 503."""

    app = build_app(tmp_path, llm_provider="mock")
    # Remove mock from the registry so it's unknown.
    app.state.llm.registry._factories.pop("mock", None)
    try:
        with TestClient(app) as client:
            r = client.get("/_probe/provider")
        assert r.status_code == 503
        error = r.json()["error"]
        assert error["code"] == "llm_not_configured" and "mock" in error["message"]
        assert "Traceback" not in r.text
    finally:
        app.state.engine.dispose()


def test_an_injected_provider_is_resolved_once_and_shared(app):
    provider = ScriptedProvider(name="injected", model="test-model")
    app.state.llm.registry.register("extractive", lambda settings: provider)
    with TestClient(app) as client:
        first = client.get("/_probe/provider").json()
        second = client.get("/_probe/provider").json()
    assert first == second == {"name": "injected", "model": "test-model", "mode": "mock"}
    assert app.state.llm.provider() is provider


def test_dependencies_can_be_overridden_in_tests(app):
    fake = ScriptedProvider(name="overridden")
    app.dependency_overrides[get_llm_provider] = lambda: fake
    with TestClient(app) as client:
        assert client.get("/_probe/provider").json()["name"] == "overridden"
    app.dependency_overrides.clear()


def test_a_provider_error_that_escapes_is_mapped_and_scrubbed(app):
    app.state.llm.registry.register("extractive", lambda settings: ScriptedProvider())
    with TestClient(app) as client:
        r = client.get("/_probe/fail")
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "llm_timeout"
    assert set(r.json()["error"]) == {"code", "message"}
    message = r.json()["error"]["message"]
    assert "sk-abc123" not in r.text
    assert len(message.splitlines()) == 1


def test_shutdown_closes_the_provider(tmp_path):
    application = build_app(tmp_path)
    provider = ScriptedProvider()
    application.state.llm.registry.register("extractive", lambda settings: provider)
    try:
        with TestClient(application) as client:
            client.get("/_probe/provider")
            assert provider.closed is False
        assert provider.closed is True
    finally:
        application.state.engine.dispose()


def test_the_dependency_helpers_return_the_application_objects(app):
    class Req:
        pass

    req = Req()
    req.app = app  # type: ignore[attr-defined]
    assert get_llm_runtime(req) is app.state.llm  # type: ignore[arg-type]
    assert get_query_history(req) is app.state.query_history  # type: ignore[arg-type]


# ------------------------------------------------------------- history on the app's own DB


def test_history_recorded_through_the_application_is_stored_in_its_database(app):
    from tests.integration.test_query_history_store import record

    hid = app.state.query_history.record(record())
    assert hid
    with app.state.session_factory() as s:
        from app.models import QueryHistory

        row = s.get(QueryHistory, hid)
        assert row.question == "Who works at Acme Inc?" and row.generation_mode == "extractive"
