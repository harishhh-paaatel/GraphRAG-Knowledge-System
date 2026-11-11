"""Shared fixtures: every test gets an isolated data dir and SQLite database."""

import os

# The default suite must never download or load a real model: use the deterministic stub.
# Real-model tests opt in explicitly (see ``tests/slow``).
os.environ.setdefault("GRAPHRAG_EMBEDDING_PROVIDER", "hashing")

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.main import create_app
from app.models import Base


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(data_dir=tmp_path, environment="test", _env_file=None)


@pytest.fixture
def app(settings):
    application = create_app(settings)
    Base.metadata.create_all(application.state.engine)
    yield application
    application.state.engine.dispose()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture
def session(app) -> Iterator[Session]:
    s = app.state.session_factory()
    try:
        yield s
    finally:
        s.close()


def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``slow`` (real model / network) unless explicitly requested."""
    if os.environ.get("GRAPHRAG_RUN_SLOW") == "1" or "slow" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="slow test: set GRAPHRAG_RUN_SLOW=1 or run with -m slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
