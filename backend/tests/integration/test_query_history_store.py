"""Query history on the existing ``query_history`` table.

The first group of tests VALIDATES that the existing columns are sufficient (and records the
constraints that follow from them) before anything relies on them: they run against a database
created by the real Alembic migrations, not only the ORM metadata.
"""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

from app.core.database import make_engine, make_session_factory
from app.models import Base, Document, QueryHistory
from app.schemas.query import EXCERPT_CHARS
from app.services.generation.history import (
    MAX_LIST_LIMIT,
    MAX_METADATA_BYTES,
    QueryHistoryStore,
    QueryRecord,
    record_from_response,
)
from app.services.generation.types import GenerationMode
from tests.llm_helpers import make_response, make_source

BACKEND = Path(__file__).resolve().parents[2]


@pytest.fixture
def migrated(tmp_path):
    """A database built by the real migrations (what a deployed system has)."""
    db = tmp_path / "migrated.db"
    url = f"sqlite:///{db.as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url  # migrations/env.py reads the URL from here, not from the ini
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    assert Path(engine.url.database) == db  # the temporary file, never the project's database
    yield engine
    engine.dispose()


@pytest.fixture
def store(migrated):
    return QueryHistoryStore(make_session_factory(migrated))


def record(**kw) -> QueryRecord:
    base = dict(
        question="Who works at Acme Inc?",
        strategy="hybrid",
        status="answered",
        generation_mode="extractive",
        execution_time_ms=12.5,
        answer="Alice works at Acme Inc. [S1]",
    )
    base.update(kw)
    return QueryRecord(**base)


# ============================================================ are the existing fields enough?


def test_the_migrated_table_has_exactly_the_columns_the_history_relies_on(migrated):
    cols = {c["name"]: c for c in inspect(migrated).get_columns("query_history")}
    assert set(cols) == {
        "id",
        "question",
        "retrieval_strategy",
        "answer",
        "execution_time_ms",
        "generation_mode",
        "created_at",
        "metadata",
    }
    assert all(not c["nullable"] for c in cols.values())  # every column is NOT NULL
    assert cols["retrieval_strategy"]["type"].length == 16
    assert cols["generation_mode"]["type"].length == 16
    assert cols["id"]["type"].length == 32


def test_the_table_can_be_listed_newest_first_because_created_at_is_indexed(migrated):
    indexes = inspect(migrated).get_indexes("query_history")
    assert any(i["column_names"] == ["created_at"] for i in indexes)


def test_there_is_no_status_provider_or_model_column_so_those_must_live_in_the_metadata(migrated):
    names = {c["name"] for c in inspect(migrated).get_columns("query_history")}
    assert not names & {"status", "provider", "model", "reason", "citations", "usage"}


def test_every_value_the_history_will_store_fits_the_declared_column_lengths():
    assert max(len(m.value) for m in GenerationMode) <= 16
    assert len("hybrid") <= 16
    # the longest values this phase will ever write
    for mode in GenerationMode:
        record(generation_mode=mode.value).row_values()


def test_the_answer_column_is_not_null_so_unanswered_questions_store_an_empty_string(store):
    hid = store.record(record(status="insufficient_evidence", reason="model_declined", answer=None))
    entry = store.get(hid)
    assert entry.answer is None  # presented as "no answer"...
    with store._factory() as s:
        assert s.get(QueryHistory, hid).answer == ""  # ...stored as "" because of NOT NULL
    assert entry.status == "insufficient_evidence" and entry.reason == "model_declined"


def test_a_maximal_record_survives_a_real_migrated_database(store):
    big_question = ("Wer arbeitet bei Acme? \U0001f600 " * 80)[:2000]
    citations = [
        {"marker": f"S{i}", "chunk_id": f"chunk-{i}", "document_id": "d", "excerpt": "é" * 500}
        for i in range(1, 9)
    ]
    hid = store.record(
        record(
            question=big_question,
            answer="a" * 5000,
            citations=citations,
            usage={"prompt_tokens": 900, "completion_tokens": 120},
            grounding={"passed": True},
            retrieval={"mode": "hybrid"},
            timings_ms={"total": 1.0},
            provider="ollama",
            model="qwen2.5:1.5b-instruct",
            prompt_version="v1",
        )
    )
    e = store.get(hid)
    assert e.question == big_question and len(e.answer) == 5000
    assert len(e.metadata["citations"]) == 8
    assert all(len(c["excerpt"]) == EXCERPT_CHARS for c in e.metadata["citations"])  # trimmed
    assert (e.metadata["provider"], e.metadata["model"]) == ("ollama", "qwen2.5:1.5b-instruct")
    assert e.metadata["usage"]["prompt_tokens"] == 900


def test_the_orm_model_and_the_migrated_schema_agree(migrated):
    """The same drift check the migration tests use, restricted to this table."""
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    with migrated.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert not [d for d in diff if "query_history" in repr(d)]


# ================================================================== store behaviour


def test_record_get_roundtrip_and_what_is_and_is_not_stored(store):
    hid = store.record(
        record(
            citations=[
                {
                    "marker": "S1",
                    "chunk_id": "c1",
                    "document_id": "d1",
                    "filename": "notes.txt",
                    "page_number": None,
                    "section": "Team",
                    "excerpt": "Alice works at Acme Inc.",
                    "cited_in_sentences": [0],
                }
            ],
            provider="extractive",
            model="extractive",
            prompt_version="v1",
        )
    )
    e = store.get(hid)
    assert (e.question, e.answer, e.status) == (
        "Who works at Acme Inc?",
        "Alice works at Acme Inc. [S1]",
        "answered",
    )
    assert (e.generation_mode, e.retrieval_strategy, e.execution_time_ms) == (
        "extractive",
        "hybrid",
        12.5,
    )
    assert e.created_at is not None and e.metadata["citations"][0]["chunk_id"] == "c1"
    stored_keys = set(e.metadata)
    assert not stored_keys & {"prompt", "messages", "api_key", "base_url", "url", "key"}


def test_nothing_secret_or_bulky_is_stored_even_if_a_caller_tries(store):
    rec = record(citations=[{"marker": "S1", "excerpt": "ok"}])
    hid = store.record(rec)
    flat = str(store.get(hid).metadata)
    assert "sk-" not in flat and "http" not in flat


def test_listing_is_newest_first_paginated_and_counts_everything(store):
    ids = [store.record(record(question=f"question {i}")) for i in range(7)]
    items, total = store.list(limit=3, offset=0)
    assert total == 7 and [i.question for i in items] == ["question 6", "question 5", "question 4"]
    items, _ = store.list(limit=3, offset=3)
    assert [i.question for i in items] == ["question 3", "question 2", "question 1"]
    items, _ = store.list(limit=3, offset=6)
    assert [i.id for i in items] == [ids[0]]
    assert store.list(limit=3, offset=50) == ([], 7)


def test_listing_limits_are_clamped(store):
    for i in range(3):
        store.record(record(question=f"q{i}"))
    assert len(store.list(limit=0)[0]) == 1
    assert len(store.list(limit=10**6)[0]) == 3
    assert MAX_LIST_LIMIT == 200
    assert store.list(limit=2, offset=-5)[0][0].question == "q2"


def test_get_and_delete(store):
    hid = store.record(record())
    assert store.get("nope") is None
    assert store.delete("nope") is False
    assert store.delete(hid) is True
    assert store.get(hid) is None and store.delete(hid) is False
    assert store.list()[1] == 0


def test_a_disabled_store_records_nothing_and_never_touches_the_database(migrated):
    real = make_session_factory(migrated)
    opened = []

    def counting_factory():
        opened.append(1)
        return real()

    disabled = QueryHistoryStore(counting_factory, enabled=False)  # type: ignore[arg-type]
    assert disabled.record(record()) is None
    assert opened == []  # no session was even opened
    assert QueryHistoryStore(real).list()[1] == 0  # and nothing reached the table
    enabled = QueryHistoryStore(counting_factory, enabled=True)  # type: ignore[arg-type]
    assert enabled.record(record()) is not None and opened  # the counter does count


def test_a_failing_database_never_fails_the_caller_and_the_question_is_not_logged(caplog):
    def broken():
        raise RuntimeError("database is locked")

    store = QueryHistoryStore(broken, enabled=True)  # type: ignore[arg-type]
    with caplog.at_level(logging.ERROR):
        assert store.record(record(question="my very private question")) is None
    assert "Could not record query history" in caplog.text
    assert "my very private question" not in caplog.text


@pytest.mark.parametrize(
    "bad",
    [
        {"strategy": "x" * 17},
        {"generation_mode": "x" * 17},
        {"execution_time_ms": -1.0},
    ],
)
def test_values_that_would_not_fit_the_columns_are_refused_and_logged_not_raised(
    store, bad, caplog
):
    with caplog.at_level(logging.ERROR):
        assert store.record(record(**bad)) is None
    assert store.list()[1] == 0
    assert "Could not record query history" in caplog.text


def test_oversized_metadata_is_trimmed_to_a_bounded_size(store):
    heavy = [
        {"marker": f"S{i}", "chunk_id": "c" * 64, "excerpt": "x" * 300, "filename": "f" * 255}
        for i in range(1, 400)
    ]
    hid = store.record(
        record(citations=heavy, warnings=["w" * 500] * 50, grounding={"p": "g" * 90000})
    )
    e = store.get(hid)
    import json

    assert len(json.dumps(e.metadata).encode()) <= MAX_METADATA_BYTES
    assert e.metadata["truncated"] is True and len(e.metadata["citations"]) < len(heavy)
    assert e.metadata["status"] == "answered"  # the essentials survive


def test_an_entry_stays_readable_after_its_document_is_deleted(store, migrated):
    hid = store.record(
        record(
            citations=[
                {"marker": "S1", "chunk_id": "gone", "document_id": "gone", "excerpt": "kept"}
            ]
        )
    )
    with make_session_factory(migrated)() as s:  # deleting documents does not touch history
        s.execute(text("DELETE FROM documents"))
        s.commit()
        assert s.query(Document).count() == 0
    assert store.get(hid).metadata["citations"][0]["excerpt"] == "kept"


def test_concurrent_recording_from_many_threads(store):
    barrier = threading.Barrier(8)

    def write(i):
        barrier.wait(timeout=10)
        return store.record(record(question=f"concurrent {i}"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(write, range(8)))
    assert None not in ids and len(set(ids)) == 8
    assert store.list()[1] == 8


# ============================================================ built from the API response


def test_a_history_record_is_built_from_a_response_without_losing_provenance(store):
    resp = make_response()
    rec = record_from_response(resp, 42.0)
    assert (rec.status, rec.generation_mode, rec.strategy, rec.execution_time_ms) == (
        "answered",
        "mock",
        "hybrid",
        42.0,
    )
    hid = store.record(rec)
    e = store.get(hid)
    cite = e.metadata["citations"][0]
    assert (cite["marker"], cite["chunk_id"], cite["document_id"]) == ("S1", "c1", "d1")
    assert (cite["filename"], cite["section"], cite["page_number"]) == ("notes.txt", "Team", None)
    assert e.metadata["grounding"]["passed"] is True
    assert e.metadata["retrieval"] == {
        "mode": "hybrid",
        "complete": True,
        "contributing": ["vector", "graph"],
    }
    assert e.metadata["usage"]["estimated_prompt_tokens"] == 12
    assert e.metadata["provider"] == "scripted" and e.metadata["prompt_version"] == "v1"


def test_unanswered_responses_are_recorded_with_an_empty_answer_and_their_reason(store):
    resp = make_response(
        status="insufficient_evidence", answer=None, citations=[], reason="no_context_after_budget"
    )
    e = store.get(store.record(record_from_response(resp, 3.0)))
    assert e.answer is None and e.status == "insufficient_evidence"
    assert e.reason == "no_context_after_budget"


def test_long_excerpts_in_a_response_are_stored_trimmed(store):
    src = make_source(excerpt="y" * EXCERPT_CHARS)
    resp = make_response(citations=[src], sources=[src])
    e = store.get(store.record(record_from_response(resp, 1.0)))
    assert len(e.metadata["citations"][0]["excerpt"]) == EXCERPT_CHARS
