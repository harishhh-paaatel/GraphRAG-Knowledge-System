"""Entity name/description/alias are derived from remaining evidence, never left stale."""

from datetime import datetime, timedelta

import pytest
from sqlalchemy import select

from app.models import (
    Document,
    DocumentChunk,
    Entity,
    EntityCandidate,
    EntityMention,
    Relationship,
)
from app.services.extraction.mock import MockExtractionProvider
from app.services.extraction.persistence import ChunkResult, replace_document_knowledge
from app.services.extraction.types import ChunkExtraction, ExtractedEntity, ExtractedRelationship
from app.services.ingestion import service

TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PRODUCT", "CONCEPT", "EVENT"}
T0 = datetime(2026, 1, 1, 12, 0, 0)
_n = iter(range(10_000))


def E(name, etype="ORGANIZATION", surface=None, desc=None):  # noqa: N802
    return ExtractedEntity(name, etype, surface_form=surface, description=desc)


def R(src, tgt, rtype, support):  # noqa: N802
    return ExtractedRelationship(src, tgt, rtype, support)


def add_doc(session, text, minute):
    """A document with an explicit upload time, so 'earliest evidence' is unambiguous."""
    n = next(_n)
    doc = Document(
        filename=f"d{n}.txt",
        stored_path="x",
        file_type="txt",
        size_bytes=len(text),
        checksum=f"{n + 5000:064d}",
        processing_status="completed",
        upload_timestamp=T0 + timedelta(minutes=minute),
    )
    session.add(doc)
    session.flush()
    chunk = DocumentChunk(id=f"c{n}", document_id=doc.id, chunk_index=0, content=text)
    session.add(chunk)
    session.commit()
    return doc, chunk


def store(session, doc, chunk, extraction):
    replace_document_knowledge(
        session,
        doc.id,
        [ChunkResult(chunk.id, chunk.content, extraction)],
        allowed_types=TYPES,
        fuzzy_threshold=0.88,
        provider_name="test",
    )
    session.commit()


def entity(session, key, etype="ORGANIZATION"):
    session.expire_all()
    return session.scalars(
        select(Entity).where(Entity.normalized_key == key, Entity.entity_type == etype)
    ).first()


def two_doc_openai(session):
    """OpenAI supported by two documents that disagree on name, description and spelling."""
    d1, c1 = add_doc(session, "OpenAI is an AI lab.", minute=0)
    d2, c2 = add_doc(session, "Open AI is a research company.", minute=1)
    store(session, d1, c1, ChunkExtraction([E("OpenAI", desc="AI lab (doc one)")], []))
    store(session, d2, c2, ChunkExtraction([E("Open AI", desc="Research company (doc two)")], []))
    return d1, d2


# ---- two documents supporting one entity -------------------------------------------------


def test_initial_state_records_provenance_for_name_description_and_aliases(session):
    d1, d2 = two_doc_openai(session)
    e = entity(session, "openai")
    assert (e.canonical_name, e.description) == ("OpenAI", "AI lab (doc one)")
    meta = e.entity_metadata
    assert meta["canonical_name_source"]["document_id"] == d1.id
    assert meta["description_source"]["document_id"] == d1.id
    assert meta["aliases"] == ["Open AI"]
    assert meta["alias_sources"] == {"Open AI": [d2.id]}
    assert meta["canonical_name_source"]["chunk_id"].startswith("c")


# ---- deleting the first document ---------------------------------------------------------


def test_deleting_first_document_recomputes_everything_from_the_survivor(session, settings):
    d1, d2 = two_doc_openai(session)
    before_id = entity(session, "openai").id

    service.delete_document(session, settings, d1.id)

    e = entity(session, "openai")
    assert e is not None and e.id == before_id  # entity and its deterministic ID survive
    assert e.canonical_name == "Open AI"  # recomputed: only doc two remains
    assert e.description == "Research company (doc two)"  # doc one's description is gone
    assert e.entity_metadata["canonical_name_source"]["document_id"] == d2.id
    assert e.entity_metadata["description_source"]["document_id"] == d2.id
    assert e.entity_metadata["aliases"] == [] and e.entity_metadata["alias_sources"] == {}
    assert e.normalized_key == "openai"  # identity key never rewritten


def test_deleting_a_later_document_keeps_the_established_values(session, settings):
    d1, d2 = two_doc_openai(session)
    service.delete_document(session, settings, d2.id)
    e = entity(session, "openai")
    assert (e.canonical_name, e.description) == ("OpenAI", "AI lab (doc one)")
    assert e.entity_metadata["aliases"] == []  # "Open AI" lived only in the deleted document


# ---- deleting the last supporting document ----------------------------------------------


def test_deleting_the_last_supporting_document_removes_entity_and_dependents(session, settings):
    d1, d2 = two_doc_openai(session)
    service.delete_document(session, settings, d1.id)
    service.delete_document(session, settings, d2.id)
    for model in (Entity, EntityMention, Relationship, EntityCandidate):
        assert session.query(model).count() == 0, model.__name__


# ---- canonical name recomputation --------------------------------------------------------


def test_canonical_name_follows_upload_time_not_processing_order(session):
    d_early, c_early = add_doc(session, "OpenAI exists.", minute=0)
    d_late, c_late = add_doc(session, "Open AI exists.", minute=5)
    # The later document is processed FIRST (e.g. the earlier one was retried).
    store(session, d_late, c_late, ChunkExtraction([E("Open AI")], []))
    assert entity(session, "open ai").canonical_name == "Open AI"
    store(session, d_early, c_early, ChunkExtraction([E("OpenAI")], []))
    e = entity(session, "open ai")
    assert e.canonical_name == "OpenAI"
    assert e.entity_metadata["canonical_name_source"]["document_id"] == d_early.id


def test_reprocessing_a_document_does_not_change_which_evidence_is_earliest(session):
    d1, d2 = two_doc_openai(session)
    d1_chunk = session.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == d1.id)
    ).one()
    for _ in range(2):
        store(session, d1, d1_chunk, ChunkExtraction([E("OpenAI", desc="AI lab (doc one)")], []))
    e = entity(session, "openai")
    assert (e.canonical_name, e.description) == ("OpenAI", "AI lab (doc one)")


# ---- description recomputation -----------------------------------------------------------


def test_description_that_depends_only_on_a_deleted_document_is_dropped(session, settings):
    d1, c1 = add_doc(session, "Berlin is large.", minute=0)
    d2, c2 = add_doc(session, "Berlin hosts events.", minute=1)
    store(
        session, d1, c1, ChunkExtraction([E("Berlin", "LOCATION", desc="Capital of Germany")], [])
    )
    store(session, d2, c2, ChunkExtraction([E("Berlin", "LOCATION")], []))  # no description
    assert entity(session, "berlin", "LOCATION").description == "Capital of Germany"

    service.delete_document(session, settings, d1.id)

    e = entity(session, "berlin", "LOCATION")
    assert e is not None  # still supported by document two
    assert e.description is None
    assert e.entity_metadata["description_source"] is None


def test_description_from_a_later_document_is_kept_while_it_exists(session, settings):
    d1, c1 = add_doc(session, "Berlin is large.", minute=0)
    d2, c2 = add_doc(session, "Berlin hosts events.", minute=1)
    store(session, d1, c1, ChunkExtraction([E("Berlin", "LOCATION")], []))
    store(session, d2, c2, ChunkExtraction([E("Berlin", "LOCATION", desc="German capital")], []))
    assert entity(session, "berlin", "LOCATION").description == "German capital"
    service.delete_document(session, settings, d1.id)
    assert entity(session, "berlin", "LOCATION").description == "German capital"
    service.delete_document(session, settings, d2.id)
    assert entity(session, "berlin", "LOCATION") is None


def test_reprocessing_with_a_changed_description_updates_the_entity(session):
    d1, c1 = add_doc(session, "Berlin is large.", minute=0)
    store(session, d1, c1, ChunkExtraction([E("Berlin", "LOCATION", desc="old")], []))
    store(session, d1, c1, ChunkExtraction([E("Berlin", "LOCATION", desc="new")], []))
    assert entity(session, "berlin", "LOCATION").description == "new"
    store(session, d1, c1, ChunkExtraction([E("Berlin", "LOCATION")], []))
    assert entity(session, "berlin", "LOCATION").description is None


# ---- alias cleanup -----------------------------------------------------------------------


def test_alias_sources_shrink_and_aliases_disappear_with_their_documents(session, settings):
    d1, c1 = add_doc(session, "OpenAI one.", minute=0)
    d2, c2 = add_doc(session, "Open AI two.", minute=1)
    d3, c3 = add_doc(session, "Open AI three.", minute=2)
    store(session, d1, c1, ChunkExtraction([E("OpenAI")], []))
    store(session, d2, c2, ChunkExtraction([E("Open AI")], []))
    store(session, d3, c3, ChunkExtraction([E("Open AI")], []))
    meta = entity(session, "openai").entity_metadata
    assert meta["alias_sources"] == {"Open AI": sorted([d2.id, d3.id])}

    service.delete_document(session, settings, d2.id)
    meta = entity(session, "openai").entity_metadata
    assert meta["aliases"] == ["Open AI"] and meta["alias_sources"] == {"Open AI": [d3.id]}

    service.delete_document(session, settings, d3.id)
    meta = entity(session, "openai").entity_metadata
    assert meta["aliases"] == [] and meta["alias_sources"] == {}


def test_a_surface_that_becomes_canonical_is_no_longer_listed_as_an_alias(session, settings):
    d1, d2 = two_doc_openai(session)
    service.delete_document(session, settings, d1.id)
    assert entity(session, "openai").entity_metadata["aliases"] == []


# ---- relationship evidence cleanup -------------------------------------------------------


def test_relationship_evidence_is_removed_per_document(session, settings):
    d1, c1 = add_doc(session, "Alice works at Acme Corp.", minute=0)
    d2, c2 = add_doc(session, "Alice is employed by Acme Corp.", minute=1)
    people = [E("Alice", "PERSON"), E("Acme Corp")]
    store(
        session,
        d1,
        c1,
        ChunkExtraction(people, [R("Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp")]),
    )
    store(
        session,
        d2,
        c2,
        ChunkExtraction(
            people, [R("Alice", "Acme Corp", "WORKS_AT", "Alice is employed by Acme Corp")]
        ),
    )
    assert session.query(Relationship).count() == 2

    service.delete_document(session, settings, d1.id)
    rel = session.scalars(select(Relationship)).one()
    assert (
        rel.source_document_id == d2.id and rel.supporting_text == "Alice is employed by Acme Corp"
    )
    assert session.query(Entity).count() == 2  # both endpoints still evidenced by doc two

    service.delete_document(session, settings, d2.id)
    assert session.query(Relationship).count() == 0 and session.query(Entity).count() == 0


def test_no_relationship_ever_points_at_a_missing_entity_or_document(session, settings):
    d1, c1 = add_doc(session, "Alice works at Acme Corp in Berlin.", minute=0)
    d2, c2 = add_doc(session, "Acme Corp is located in Berlin.", minute=1)
    store(
        session,
        d1,
        c1,
        ChunkExtraction(
            [E("Alice", "PERSON"), E("Acme Corp"), E("Berlin", "LOCATION")],
            [R("Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp")],
        ),
    )
    store(
        session,
        d2,
        c2,
        ChunkExtraction(
            [E("Acme Corp"), E("Berlin", "LOCATION")],
            [R("Acme Corp", "Berlin", "LOCATED_IN", "Acme Corp is located in Berlin")],
        ),
    )
    service.delete_document(session, settings, d1.id)

    session.expire_all()
    entity_ids = set(session.scalars(select(Entity.id)))
    doc_ids = set(session.scalars(select(Document.id)))
    chunk_ids = set(session.scalars(select(DocumentChunk.id)))
    for r in session.scalars(select(Relationship)):
        assert {r.source_entity_id, r.target_entity_id} <= entity_ids
        assert r.source_document_id in doc_ids and r.source_chunk_id in chunk_ids
    for m in session.scalars(select(EntityMention)):
        assert m.entity_id in entity_ids and m.document_id in doc_ids and m.chunk_id in chunk_ids
    assert {e.canonical_name for e in session.scalars(select(Entity))} == {"Acme Corp", "Berlin"}


# ---- identity is never touched ------------------------------------------------------------


def test_distinct_entities_stay_distinct_through_recomputation(session, settings):
    d1, c1 = add_doc(session, "Globex Systems builds tools.", minute=0)
    d2, c2 = add_doc(session, "Globex Systemz builds toys.", minute=1)
    d3, c3 = add_doc(session, "Globex Systems is large.", minute=2)
    store(session, d1, c1, ChunkExtraction([E("Globex Systems", desc="Toolmaker")], []))
    store(session, d2, c2, ChunkExtraction([E("Globex Systemz")], []))
    store(session, d3, c3, ChunkExtraction([E("Globex Systems", desc="Big firm")], []))
    ids = {e.normalized_key: e.id for e in session.scalars(select(Entity))}
    assert set(ids) == {"globex systems", "globex systemz"}

    service.delete_document(session, settings, d1.id)  # triggers a description recompute

    session.expire_all()
    after = {e.normalized_key: e.id for e in session.scalars(select(Entity))}
    assert after == ids  # nothing merged, split, renamed or re-IDed
    assert entity(session, "globex systems").description == "Big firm"
    assert session.query(EntityCandidate).count() == 1  # the review flag is still there


def test_recomputed_name_equal_to_another_entitys_name_does_not_merge(session, settings):
    d1, c1 = add_doc(session, "OpenAI one.", minute=0)
    d2, c2 = add_doc(session, "Open AI two. Open AI Labs three.", minute=1)
    store(session, d1, c1, ChunkExtraction([E("OpenAI")], []))
    store(session, d2, c2, ChunkExtraction([E("Open AI"), E("Open AI Labs")], []))
    service.delete_document(session, settings, d1.id)  # canonical of entity "openai" -> "Open AI"
    keys = sorted(e.normalized_key for e in session.scalars(select(Entity)))
    assert keys == ["open ai labs", "openai"]
    assert entity(session, "openai").canonical_name == "Open AI"


def test_entity_ids_are_deterministic_across_delete_and_reingest(session, settings):
    d1, c1 = add_doc(session, "Alice works at Acme Corp.", minute=0)
    ex = ChunkExtraction([E("Alice", "PERSON"), E("Acme Corp")], [])
    store(session, d1, c1, ex)
    ids = {e.normalized_key: e.id for e in session.scalars(select(Entity))}
    service.delete_document(session, settings, d1.id)
    assert session.query(Entity).count() == 0
    d2, c2 = add_doc(session, "Alice works at Acme Corp.", minute=3)
    store(session, d2, c2, ex)
    assert {e.normalized_key: e.id for e in session.scalars(select(Entity))} == ids


# ---- end to end through the API ----------------------------------------------------------


def test_api_delete_recomputes_from_remaining_documents(client, app, session):
    def provider(desc, name):
        return MockExtractionProvider(
            {
                "covers": ChunkExtraction(
                    [ExtractedEntity(name, "ORGANIZATION", description=desc)], []
                )
            }
        )

    def up(filename, body, prov):
        app.state.worker.provider = prov
        r = client.post("/api/v1/documents/upload", files={"file": (filename, body, "text/plain")})
        assert app.state.worker.drain(30)
        return r.json()["document"]["id"]

    first = up(
        "a.txt", b"Report one covers OpenAI in depth.", provider("From report one", "OpenAI")
    )
    second = up(
        "b.txt", b"Report two covers Open AI briefly.", provider("From report two", "Open AI")
    )
    e = entity(session, "openai")
    assert (e.canonical_name, e.description) == ("OpenAI", "From report one")

    assert client.delete(f"/api/v1/documents/{first}").status_code == 204
    e = entity(session, "openai")
    assert (e.canonical_name, e.description) == ("Open AI", "From report two")
    assert e.entity_metadata["description_source"]["document_id"] == second

    assert client.delete(f"/api/v1/documents/{second}").status_code == 204
    assert entity(session, "openai") is None


@pytest.mark.parametrize("which", ["first", "second"])
def test_reprocessing_through_the_pipeline_keeps_derived_values(client, app, session, which):
    prov = MockExtractionProvider(
        {
            "covers": ChunkExtraction(
                [ExtractedEntity("OpenAI", "ORGANIZATION", description="Lab")], []
            )
        }
    )
    app.state.worker.provider = prov
    ids = []
    for name, body in (
        ("a.txt", b"Report one covers OpenAI."),
        ("b.txt", b"Report two covers OpenAI too."),
    ):
        r = client.post("/api/v1/documents/upload", files={"file": (name, body, "text/plain")})
        app.state.worker.drain(30)
        ids.append(r.json()["document"]["id"])
    before = entity(session, "openai")
    snapshot = (before.id, before.canonical_name, before.description)

    # Force a re-run of one document's job; evidence is replaced, derived values must not move.
    from app.models import ProcessingJob
    from app.services.ingestion.pipeline import run_ingestion

    doc_id = ids[0] if which == "first" else ids[1]
    job = ProcessingJob(document_id=doc_id, status="pending", attempt=2)
    session.add(job)
    session.commit()
    run_ingestion(app.state.session_factory, app.state.settings, job.id, prov)

    after = entity(session, "openai")
    assert (after.id, after.canonical_name, after.description) == snapshot
    assert service  # imported for deletion helpers used above
