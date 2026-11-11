"""Resolution, provenance, idempotency and deletion consistency against a real database."""

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
from app.services.extraction.persistence import (
    ChunkResult,
    purge_document_knowledge,
    refresh_entities,
    replace_document_knowledge,
)
from app.services.extraction.types import ChunkExtraction, ExtractedEntity, ExtractedRelationship
from app.services.ingestion import service

TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PRODUCT", "CONCEPT", "EVENT"}
_counter = iter(range(10_000))


def E(name, etype, surface=None, desc=None):  # noqa: N802
    return ExtractedEntity(name, etype, surface_form=surface, description=desc)


def R(src, tgt, rtype, support):  # noqa: N802
    return ExtractedRelationship(src, tgt, rtype, support)


def add_doc(session, text):
    """Insert a document with one chunk directly (no pipeline)."""
    n = next(_counter)
    doc = Document(
        filename=f"d{n}.txt",
        stored_path="x",
        file_type="txt",
        size_bytes=len(text),
        checksum=f"{n:064d}",
        processing_status="completed",
    )
    session.add(doc)
    session.flush()
    chunk = DocumentChunk(id=f"chunk{n}", document_id=doc.id, chunk_index=0, content=text)
    session.add(chunk)
    session.commit()
    return doc, chunk


def store(session, doc, chunk, extraction, provider="test"):
    stats = replace_document_knowledge(
        session,
        doc.id,
        [ChunkResult(chunk.id, chunk.content, extraction)],
        allowed_types=TYPES,
        fuzzy_threshold=0.88,
        provider_name=provider,
    )
    session.commit()
    return stats


def entities(session):
    session.expire_all()
    return {(e.canonical_name, e.entity_type): e for e in session.scalars(select(Entity))}


def candidates(session):
    session.expire_all()
    names = {e.id: e.canonical_name for e in session.scalars(select(Entity))}
    return {
        (frozenset((names[c.entity_a_id], names[c.entity_b_id])), c.reason)
        for c in session.scalars(select(EntityCandidate))
    }


def snapshot(session):
    """Everything the knowledge tables hold, minus random IDs, for before/after comparison."""
    session.expire_all()
    names = {e.id: e.canonical_name for e in session.scalars(select(Entity))}
    return (
        sorted(
            (e.canonical_name, e.entity_type, e.normalized_key)
            for e in session.scalars(select(Entity))
        ),
        sorted(
            (names[m.entity_id], m.chunk_id, m.surface_form, m.provenance)
            for m in session.scalars(select(EntityMention))
        ),
        sorted(
            (
                names[r.source_entity_id],
                r.relationship_type,
                names[r.target_entity_id],
                r.supporting_text,
                r.source_chunk_id,
            )
            for r in session.scalars(select(Relationship))
        ),
    )


ALICE = "Alice works at Acme Corp in Berlin."
ALICE_EXTRACTION = ChunkExtraction(
    [
        E("Alice", "PERSON", desc="An employee"),
        E("Acme Corp", "ORGANIZATION"),
        E("Berlin", "LOCATION"),
    ],
    [R("Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp")],
)


# ---- provenance -------------------------------------------------------------------------


def test_mentions_and_relationships_keep_provenance(session):
    doc, chunk = add_doc(session, ALICE)
    stats = store(session, doc, chunk, ALICE_EXTRACTION, provider="rule_based")
    assert (stats.mentions, stats.relationships, stats.entities_created) == (3, 1, 3)

    mention = session.scalars(
        select(EntityMention).join(Entity).where(Entity.canonical_name == "Alice")
    ).one()
    assert (mention.document_id, mention.chunk_id) == (doc.id, chunk.id)
    assert mention.surface_form == "Alice" and mention.provenance == "rule_based"
    assert mention.confidence is None  # never fabricated

    rel = session.scalars(select(Relationship)).one()
    assert (rel.source_document_id, rel.source_chunk_id) == (doc.id, chunk.id)
    assert rel.provenance == "rule_based" and rel.confidence is None
    assert rel.supporting_text in chunk.content  # traceable to the stored chunk text
    assert entities(session)[("Alice", "PERSON")].description == "An employee"


def test_invalid_relationships_never_reach_the_database(session):
    doc, chunk = add_doc(session, ALICE)
    bad = ChunkExtraction(
        ALICE_EXTRACTION.entities,
        [
            R("Alice", "Acme Corp", "WORKS_AT", "Alice is the CEO of Acme Corp"),  # invented
            R("Alice", "Berlin", "WORKS_AT", "Alice works at Acme Corp in Berlin"),  # wrong types
            R("Alice", "Globex", "WORKS_AT", "Alice works at Acme Corp"),  # unknown endpoint
        ],
    )
    stats = store(session, doc, chunk, bad)
    assert session.scalars(select(Relationship)).all() == []
    assert stats.rejected_relationships == 3 and stats.relationships == 0
    assert len(entities(session)) == 3  # the valid entities are still kept


def test_entities_not_in_the_chunk_are_not_stored(session):
    doc, chunk = add_doc(session, ALICE)
    store(session, doc, chunk, ChunkExtraction([E("Alice", "PERSON"), E("Bob", "PERSON")], []))
    assert set(entities(session)) == {("Alice", "PERSON")}


def test_failed_chunks_are_counted_not_hidden(session):
    doc, chunk = add_doc(session, ALICE)
    stats = replace_document_knowledge(
        session,
        doc.id,
        [ChunkResult(chunk.id, chunk.content, None)],
        allowed_types=TYPES,
        fuzzy_threshold=0.88,
        provider_name="t",
    )
    assert stats.failed_chunks == 1 and stats.mentions == 0


# ---- exact matching & aliases -----------------------------------------------------------


def test_same_entity_in_two_documents_is_one_entity_with_two_mentions(session):
    d1, c1 = add_doc(session, "OpenAI released ChatGPT.")
    d2, c2 = add_doc(session, "Investors funded OpenAI last year.")
    store(session, d1, c1, ChunkExtraction([E("OpenAI", "ORGANIZATION")], []))
    store(session, d2, c2, ChunkExtraction([E("openai", "ORGANIZATION", surface="OpenAI")], []))
    ents = entities(session)
    assert len(ents) == 1
    mentions = session.scalars(select(EntityMention)).all()
    assert {m.document_id for m in mentions} == {d1.id, d2.id}


def test_alias_tracking_and_compact_merge(session):
    d1, c1 = add_doc(session, "OpenAI released ChatGPT.")
    d2, c2 = add_doc(session, "Open AI is based in California.")
    store(session, d1, c1, ChunkExtraction([E("OpenAI", "ORGANIZATION")], []))
    store(session, d2, c2, ChunkExtraction([E("Open AI", "ORGANIZATION")], []))
    ents = entities(session)
    assert list(ents) == [("OpenAI", "ORGANIZATION")]  # first canonical name wins
    assert ents[("OpenAI", "ORGANIZATION")].entity_metadata["aliases"] == ["Open AI"]
    assert candidates(session) == set()  # a safe merge needs no review


def test_corporate_suffix_variants_merge_for_organizations_only(session):
    d1, c1 = add_doc(session, "Acme Inc. hired Alice. Acme is big.")
    store(
        session,
        d1,
        c1,
        ChunkExtraction([E("Acme Inc.", "ORGANIZATION"), E("Acme", "ORGANIZATION")], []),
    )
    assert len(entities(session)) == 1


# ---- ambiguity: never merge on similarity ----------------------------------------------


def test_same_name_different_type_stays_separate_and_is_flagged(session):
    d1, c1 = add_doc(session, "Mercury is the closest planet.")
    d2, c2 = add_doc(session, "Mercury is a liquid metal.")
    store(session, d1, c1, ChunkExtraction([E("Mercury", "LOCATION")], []))
    store(session, d2, c2, ChunkExtraction([E("Mercury", "CONCEPT")], []))
    assert set(entities(session)) == {("Mercury", "LOCATION"), ("Mercury", "CONCEPT")}
    assert candidates(session) == {(frozenset({"Mercury"}), "same_name_different_type")}


def test_fuzzy_lookalikes_are_candidates_not_merges(session):
    d1, c1 = add_doc(session, "Globex Systems builds tools.")
    d2, c2 = add_doc(session, "Globex Systemz builds toys.")
    store(session, d1, c1, ChunkExtraction([E("Globex Systems", "ORGANIZATION")], []))
    store(session, d2, c2, ChunkExtraction([E("Globex Systemz", "ORGANIZATION")], []))
    assert len(entities(session)) == 2
    assert (frozenset({"Globex Systems", "Globex Systemz"}), "fuzzy_name") in candidates(session)
    score = session.scalars(select(EntityCandidate.score)).first()
    assert score is not None and score >= 0.88


def test_versions_differing_in_digits_are_neither_merged_nor_flagged(session):
    d1, c1 = add_doc(session, "Python 2 is old.")
    d2, c2 = add_doc(session, "Python 3 is current.")
    store(session, d1, c1, ChunkExtraction([E("Python 2", "TECHNOLOGY")], []))
    store(session, d2, c2, ChunkExtraction([E("Python 3", "TECHNOLOGY")], []))
    assert len(entities(session)) == 2 and candidates(session) == set()


def test_person_surname_is_a_candidate_for_the_full_name_not_a_merge(session):
    d1, c1 = add_doc(session, "Sebastian Ramirez created FastAPI.")
    d2, c2 = add_doc(session, "Ramirez gave a talk.")
    store(session, d1, c1, ChunkExtraction([E("Sebastian Ramirez", "PERSON")], []))
    store(session, d2, c2, ChunkExtraction([E("Ramirez", "PERSON")], []))
    assert len(entities(session)) == 2
    assert (frozenset({"Sebastian Ramirez", "Ramirez"}), "token_subset") in candidates(session)


def test_compact_match_with_several_existing_entities_is_ambiguous(session):
    # Two distinct entities already share the compact key "email" (possible via older data or
    # separate ingestion paths). A new "email" must NOT pick one arbitrarily.
    session.add_all(
        [
            Entity(canonical_name="e-mail", normalized_key="e-mail", entity_type="CONCEPT"),
            Entity(canonical_name="e mail", normalized_key="e mail", entity_type="CONCEPT"),
        ]
    )
    session.commit()
    doc, chunk = add_doc(session, "We discussed email today.")
    store(session, doc, chunk, ChunkExtraction([E("email", "CONCEPT")], []))
    assert len(entities(session)) == 3  # nothing merged
    flagged = {r for _, r in candidates(session)}
    assert flagged == {"compact_ambiguous"}
    assert session.query(EntityCandidate).count() == 2  # one per existing look-alike


def test_candidates_are_not_duplicated_on_reprocessing(session):
    d1, c1 = add_doc(session, "Globex Systems builds tools.")
    d2, c2 = add_doc(session, "Globex Systemz builds toys.")
    ex1 = ChunkExtraction([E("Globex Systems", "ORGANIZATION")], [])
    ex2 = ChunkExtraction([E("Globex Systemz", "ORGANIZATION")], [])
    store(session, d1, c1, ex1)
    store(session, d2, c2, ex2)
    for _ in range(3):
        store(session, d2, c2, ex2)
    assert session.query(EntityCandidate).count() == 1


# ---- duplicates and idempotency ---------------------------------------------------------


def test_repeated_mentions_in_one_chunk_create_one_record(session):
    doc, chunk = add_doc(session, "Python is great. Python is fast. python rocks.")
    ex = ChunkExtraction([E("Python", "TECHNOLOGY")] * 3 + [E("python", "TECHNOLOGY")], [])
    store(session, doc, chunk, ex)
    mentions = session.scalars(select(EntityMention)).all()
    assert sorted(m.surface_form for m in mentions) == ["Python", "python"]  # distinct surfaces
    assert len(entities(session)) == 1


def test_reprocessing_is_idempotent(session):
    doc, chunk = add_doc(session, ALICE)
    store(session, doc, chunk, ALICE_EXTRACTION)
    first = snapshot(session)
    ids_before = {k: e.id for k, e in entities(session).items()}
    for _ in range(3):
        store(session, doc, chunk, ALICE_EXTRACTION)
    assert snapshot(session) == first
    assert {k: e.id for k, e in entities(session).items()} == ids_before  # entities stay stable


def test_reprocessing_with_changed_output_replaces_old_evidence(session):
    doc, chunk = add_doc(session, ALICE)
    store(session, doc, chunk, ALICE_EXTRACTION)
    store(session, doc, chunk, ChunkExtraction([E("Alice", "PERSON")], []))
    assert set(entities(session)) == {("Alice", "PERSON")}  # Acme/Berlin no longer evidenced
    assert session.query(Relationship).count() == 0


def test_overlap_repeated_sentence_stored_once_per_document(session):
    doc, c1 = add_doc(session, ALICE)
    c2 = DocumentChunk(id="overlap", document_id=doc.id, chunk_index=1, content="More. " + ALICE)
    session.add(c2)
    session.commit()
    results = [ChunkResult(c.id, c.content, ALICE_EXTRACTION) for c in (c1, c2)]
    replace_document_knowledge(
        session, doc.id, results, allowed_types=TYPES, fuzzy_threshold=0.88, provider_name="t"
    )
    session.commit()
    assert session.query(Relationship).count() == 1  # same evidence text, one record
    assert session.query(EntityMention).count() == 6  # but a mention per chunk, for retrieval


# ---- deletion consistency ---------------------------------------------------------------


def test_deleting_a_document_keeps_entities_other_documents_still_reference(session, settings):
    d1, c1 = add_doc(session, "Alice works at Acme Corp in Berlin.")
    d2, c2 = add_doc(session, "Acme Corp is located in Berlin.")
    store(session, d1, c1, ALICE_EXTRACTION)
    store(
        session,
        d2,
        c2,
        ChunkExtraction(
            [E("Acme Corp", "ORGANIZATION"), E("Berlin", "LOCATION")],
            [R("Acme Corp", "Berlin", "LOCATED_IN", "Acme Corp is located in Berlin")],
        ),
    )
    assert len(entities(session)) == 3

    service.delete_document(session, settings, d1.id)

    remaining = entities(session)
    assert set(remaining) == {("Acme Corp", "ORGANIZATION"), ("Berlin", "LOCATION")}  # Alice gone
    session.expire_all()
    assert {m.document_id for m in session.scalars(select(EntityMention))} == {d2.id}
    rels = session.scalars(select(Relationship)).all()
    assert [r.relationship_type for r in rels] == ["LOCATED_IN"]
    assert rels[0].source_document_id == d2.id


def test_deleting_removes_stale_aliases_and_candidates(session, settings):
    d1, c1 = add_doc(session, "OpenAI released ChatGPT.")
    d2, c2 = add_doc(session, "Open AI is based in California.")
    store(session, d1, c1, ChunkExtraction([E("OpenAI", "ORGANIZATION")], []))
    store(session, d2, c2, ChunkExtraction([E("Open AI", "ORGANIZATION")], []))
    assert entities(session)[("OpenAI", "ORGANIZATION")].entity_metadata["aliases"] == ["Open AI"]
    service.delete_document(session, settings, d2.id)
    assert entities(session)[("OpenAI", "ORGANIZATION")].entity_metadata["aliases"] == []

    g1, gc1 = add_doc(session, "Globex Systems builds tools.")
    g2, gc2 = add_doc(session, "Globex Systemz builds toys.")
    store(session, g1, gc1, ChunkExtraction([E("Globex Systems", "ORGANIZATION")], []))
    store(session, g2, gc2, ChunkExtraction([E("Globex Systemz", "ORGANIZATION")], []))
    assert session.query(EntityCandidate).count() == 1
    service.delete_document(session, settings, g2.id)
    assert session.query(EntityCandidate).count() == 0  # cascaded with the removed entity


def test_deleting_all_documents_leaves_no_knowledge_behind(session, settings):
    docs = []
    for text in (ALICE, "Acme Corp is located in Berlin."):
        d, c = add_doc(session, text)
        store(
            session,
            d,
            c,
            ALICE_EXTRACTION
            if text == ALICE
            else ChunkExtraction([E("Acme Corp", "ORGANIZATION"), E("Berlin", "LOCATION")], []),
        )
        docs.append(d)
    for d in docs:
        service.delete_document(session, settings, d.id)
    for model in (Entity, EntityMention, Relationship, EntityCandidate, DocumentChunk):
        assert session.query(model).count() == 0, model.__name__


def test_purge_and_refresh_are_safe_on_unknown_ids(session):
    assert purge_document_knowledge(session, "nope") == set()
    assert refresh_entities(session, {"missing"}) == 0


@pytest.mark.parametrize("provider", ["rule_based", "mock"])
def test_provider_name_is_recorded(session, provider):
    doc, chunk = add_doc(session, ALICE)
    store(session, doc, chunk, ALICE_EXTRACTION, provider=provider)
    assert {m.provenance for m in session.scalars(select(EntityMention))} == {provider}


def test_same_fact_stated_twice_in_one_chunk_is_stored_once(session):
    text = "Ramirez created FastAPI. FastAPI was created by Ramirez."
    doc, chunk = add_doc(session, text)
    ex = ChunkExtraction(
        [E("Ramirez", "PERSON"), E("FastAPI", "TECHNOLOGY")],
        [
            R("Ramirez", "FastAPI", "DEVELOPED", "Ramirez created FastAPI"),
            R("Ramirez", "FastAPI", "DEVELOPED", "FastAPI was created by Ramirez"),
        ],
    )
    stats = store(session, doc, chunk, ex)
    rel = session.scalars(select(Relationship)).one()
    assert rel.supporting_text == "Ramirez created FastAPI"  # first evidence wins
    assert stats.relationships == 1
