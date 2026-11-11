from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from app.core.database import make_engine
from app.models import (
    Document,
    DocumentChunk,
    Entity,
    EntityMention,
    ProcessingJob,
    Relationship,
)

BACKEND = Path(__file__).resolve().parents[2]


def _doc(checksum="a" * 64) -> Document:
    return Document(
        filename="a.txt", stored_path="x", file_type="txt", size_bytes=1, checksum=checksum
    )


def test_migration_creates_all_tables(tmp_path):
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")
    tables = set(inspect(make_engine(url)).get_table_names())
    expected = {
        "documents",
        "document_chunks",
        "entities",
        "entity_mentions",
        "relationships",
        "processing_jobs",
        "query_history",
        "evaluation_runs",
        "entity_candidates",
    }
    assert expected <= tables


def test_duplicate_checksum_rejected(session):
    session.add(_doc())
    session.commit()
    session.add(_doc())
    with pytest.raises(IntegrityError):
        session.commit()


def test_foreign_keys_enforced(session):
    session.add(DocumentChunk(id="c1", document_id="missing", chunk_index=0, content="x"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_document_delete_cascades_everything(session):
    doc = _doc()
    session.add(doc)
    session.flush()
    chunk = DocumentChunk(id="c1", document_id=doc.id, chunk_index=0, content="Python uses FastAPI")
    a, b = (
        Entity(canonical_name="Python", normalized_key="python", entity_type="TECHNOLOGY"),
        Entity(canonical_name="FastAPI", normalized_key="fastapi", entity_type="TECHNOLOGY"),
    )
    session.add_all([chunk, a, b, ProcessingJob(document_id=doc.id)])
    session.flush()
    session.add_all(
        [
            EntityMention(entity_id=a.id, document_id=doc.id, chunk_id="c1", surface_form="Python"),
            Relationship(
                source_entity_id=a.id,
                target_entity_id=b.id,
                relationship_type="USED_WITH",
                supporting_text="Python uses FastAPI",
                source_document_id=doc.id,
                source_chunk_id="c1",
            ),
        ]
    )
    session.commit()

    session.delete(doc)
    session.commit()

    for model in (DocumentChunk, EntityMention, Relationship, ProcessingJob):
        assert session.query(model).count() == 0, model.__name__
    # Entities survive document deletion at the DB level; orphan cleanup is a service concern.
    assert session.query(Entity).count() == 2


def test_entity_unique_per_key_and_type(session):
    session.add(Entity(canonical_name="Mercury", normalized_key="mercury", entity_type="PLANET"))
    session.add(Entity(canonical_name="Mercury", normalized_key="mercury", entity_type="ELEMENT"))
    session.commit()  # same name, different type: allowed (ambiguity preserved)
    session.add(Entity(canonical_name="Mercury", normalized_key="mercury", entity_type="PLANET"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_migrations_match_orm_models_with_no_drift(tmp_path):
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from app.models import Base

    url = f"sqlite:///{(tmp_path / 'd.db').as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")
    with make_engine(url).connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []


def test_migration_0002_downgrades_and_upgrades_cleanly(tmp_path):
    url = f"sqlite:///{(tmp_path / 'r.db').as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "0001")
    assert "entity_candidates" not in inspect(make_engine(url)).get_table_names()
    command.upgrade(cfg, "head")
    assert "entity_candidates" in inspect(make_engine(url)).get_table_names()


def test_migration_0003_backfills_legacy_evidence(tmp_path):
    """Rows written before 0003 get a name and (for the earliest mention) the description."""
    from sqlalchemy import text

    url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "0002")

    engine = make_engine(url)
    with engine.begin() as c:
        for i, ts in ((1, "2026-01-01 10:00:00"), (2, "2026-01-02 10:00:00")):
            c.execute(
                text(
                    "INSERT INTO documents (id, filename, stored_path, file_type, size_bytes, "
                    "checksum, upload_timestamp, processing_status, metadata) VALUES "
                    "(:id, 'f', 'x', 'txt', 1, :cs, :ts, 'completed', '{}')"
                ),
                {"id": f"d{i}", "cs": f"{i:064d}", "ts": ts},
            )
            c.execute(
                text(
                    "INSERT INTO document_chunks (id, document_id, chunk_index, content, metadata) "
                    "VALUES (:id, :d, 0, 'x', '{}')"
                ),
                {"id": f"c{i}", "d": f"d{i}"},
            )
        c.execute(
            text(
                "INSERT INTO entities (id, canonical_name, normalized_key, entity_type, "
                "description, metadata) VALUES ('e1', 'OpenAI', 'openai', 'ORGANIZATION', "
                "'AI lab', '{}')"
            )
        )
        for i in (2, 1):  # inserted out of order on purpose
            c.execute(
                text(
                    "INSERT INTO entity_mentions (id, entity_id, document_id, chunk_id, "
                    "surface_form, provenance) VALUES (:id, 'e1', :d, :c, 'OpenAI', 'old')"
                ),
                {"id": f"m{i}", "d": f"d{i}", "c": f"c{i}"},
            )
    engine.dispose()

    command.upgrade(cfg, "head")
    engine = make_engine(url)
    with engine.connect() as c:
        rows = c.execute(
            text("SELECT id, extracted_name, description FROM entity_mentions ORDER BY id")
        ).all()
    engine.dispose()
    assert rows == [("m1", "OpenAI", "AI lab"), ("m2", "OpenAI", None)]


def test_migration_0004_creates_graph_state_with_a_starting_revision(tmp_path):
    from sqlalchemy import text

    url = f"sqlite:///{(tmp_path / 'g.db').as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    with engine.connect() as c:
        assert c.execute(text("SELECT id, revision FROM graph_state")).all() == [(1, 1)]
    engine.dispose()
    command.downgrade(cfg, "0003")
    assert "graph_state" not in inspect(make_engine(url)).get_table_names()
    command.upgrade(cfg, "head")
    assert "graph_state" in inspect(make_engine(url)).get_table_names()


def test_migration_0005_creates_vector_tables_with_cascades(tmp_path):
    from sqlalchemy import text

    url = f"sqlite:///{(tmp_path / 'v.db').as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")
    engine = make_engine(url)
    tables = set(inspect(engine).get_table_names())
    assert {"chunk_embeddings", "vector_state"} <= tables
    with engine.begin() as c:
        assert c.execute(text("SELECT id, revision FROM vector_state")).all() == [(1, 1)]
        c.execute(
            text(
                "INSERT INTO documents (id, filename, stored_path, file_type, size_bytes, "
                "checksum, upload_timestamp, processing_status, metadata) VALUES "
                "('d', 'f', 'x', 'txt', 1, 'c', '2026-01-01', 'completed', '{}')"
            )
        )
        c.execute(
            text(
                "INSERT INTO document_chunks (id, document_id, chunk_index, content, metadata) "
                "VALUES ('k', 'd', 0, 'text', '{}')"
            )
        )
        c.execute(
            text(
                "INSERT INTO chunk_embeddings (chunk_id, document_id, content_hash, model_id, "
                "dimension, config_fingerprint, status, created_at, updated_at) VALUES "
                "('k', 'd', 'h', 'm', 8, 'f', 'indexed', '2026-01-01', '2026-01-01')"
            )
        )
        assert c.execute(text("SELECT count(*) FROM chunk_embeddings")).scalar() == 1
        c.execute(text("DELETE FROM document_chunks WHERE id = 'k'"))  # FK cascade
        assert c.execute(text("SELECT count(*) FROM chunk_embeddings")).scalar() == 0
    engine.dispose()
    command.downgrade(cfg, "0004")
    assert "chunk_embeddings" not in inspect(make_engine(url)).get_table_names()
    command.upgrade(cfg, "head")
    assert "vector_state" in inspect(make_engine(url)).get_table_names()


def test_migration_0006_adds_token_columns_and_round_trips(tmp_path):
    url = f"sqlite:///{(tmp_path / 't.db').as_posix()}"
    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    cfg.attributes["url"] = url
    command.upgrade(cfg, "head")
    columns = {c["name"]: c for c in inspect(make_engine(url)).get_columns("chunk_embeddings")}
    assert {"token_count", "window_count"} <= set(columns)
    assert columns["token_count"]["nullable"] and columns["window_count"]["nullable"]
    command.downgrade(cfg, "0005")
    after = {c["name"] for c in inspect(make_engine(url)).get_columns("chunk_embeddings")}
    assert not ({"token_count", "window_count"} & after)
    command.upgrade(cfg, "head")
    again = {c["name"] for c in inspect(make_engine(url)).get_columns("chunk_embeddings")}
    assert {"token_count", "window_count"} <= again
