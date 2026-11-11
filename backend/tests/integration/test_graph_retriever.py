"""GraphRetriever used directly: caps, caching, mention limits and ordering invariants."""

import pytest

from app.services.extraction.types import ChunkExtraction
from app.services.retrieval.entity_linker import UnknownEntities
from app.services.retrieval.graph_search import GraphRetriever, GraphSearchLimitError
from tests.graph_search_helpers import seed_corpus
from tests.integration.test_evidence_recompute import E, R, add_doc, store


@pytest.fixture
def retriever(app, settings):
    return GraphRetriever(app.state.graph, app.state.session_factory, settings)


@pytest.fixture
def corpus(session):
    return seed_corpus(session)


def test_hits_are_contiguously_ranked_and_decided_by_chains_from_none(retriever, corpus):
    outcome = retriever.search("Acme Corp", hops=2)
    assert [h.rank for h in outcome.hits] == list(range(1, len(outcome.hits) + 1))
    assert outcome.hits[0].decided_by is None
    assert all(h.decided_by is not None for h in outcome.hits[1:])
    assert outcome.ranked_total == len(outcome.hits) == 2


def test_the_outcome_keeps_the_pool_size_after_truncation(retriever, corpus):
    outcome = retriever.search("Acme Corp", max_chunks=1)
    assert len(outcome.hits) == 1 and outcome.ranked_total == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"hops": 0}, "hops must be between 1 and 2"),
        ({"hops": 3}, "hops must be between 1 and 2"),
        ({"max_chunks": 0}, "max_chunks must be between 1 and 50"),
        ({"max_chunks": 51}, "max_chunks must be between 1 and 50"),
        ({"direction": "sideways"}, "direction must be one of"),
    ],
)
def test_parameters_beyond_the_configured_caps_are_refused(retriever, corpus, kwargs, message):
    with pytest.raises(GraphSearchLimitError, match=message):
        retriever.search("Acme Corp", **kwargs)


def test_the_hop_cap_follows_the_graph_api_cap(app, settings, corpus):
    from app.core.config import Settings

    tight = Settings(data_dir=settings.data_dir, graph_max_hops=1, _env_file=None)
    retriever = GraphRetriever(app.state.graph, app.state.session_factory, tight)
    with pytest.raises(GraphSearchLimitError, match="between 1 and 1"):
        retriever.search("Acme Corp", hops=2)
    assert retriever.search("Acme Corp", hops=1).hits


def test_unknown_entity_ids_raise_before_any_work(retriever, corpus):
    with pytest.raises(UnknownEntities):
        retriever.search("Acme Corp", entity_ids=["nope"])


def test_the_alias_index_is_reused_until_the_graph_changes(retriever, session, corpus):
    retriever.search("Acme Corp")
    retriever.search("Alice")
    assert retriever.linker.builds == 1
    doc, chunk = add_doc(session, "Cara works at Acme Corp.", minute=20)
    store(
        session,
        doc,
        chunk,
        ChunkExtraction(
            [E("Cara", "PERSON"), E("Acme Corp")],
            [R("Cara", "Acme Corp", "WORKS_AT", "Cara works at Acme Corp")],
        ),
    )
    outcome = retriever.search("Who is Cara?")
    assert retriever.linker.builds == 2
    assert [e.name for e in outcome.link.linked] == ["Cara"] and outcome.hits


def test_mentions_per_entity_is_capped_and_deterministic(retriever, session):
    for minute in range(4):
        doc, chunk = add_doc(session, f"Zed appears here number {minute}.", minute=minute)
        store(session, doc, chunk, ChunkExtraction([E("Zed", "PERSON")], []))
    outcome = retriever.search("Zed", include_mentions=True, mentions_per_entity=2)
    assert len(outcome.hits) == 2
    again = retriever.search("Zed", include_mentions=True, mentions_per_entity=2)
    assert [h.chunk.chunk_id for h in again.hits] == [h.chunk.chunk_id for h in outcome.hits]
    assert all(e.kind == "mention" for h in outcome.hits for e in h.evidence)
    assert retriever.search("Zed").hits == []  # mentions are opt-in


def test_the_graph_revision_is_reported_and_follows_the_database(retriever, session, corpus):
    first = retriever.search("Acme Corp")
    assert first.revision == first.database_revision
    doc, chunk = add_doc(session, "Dan works at Acme Corp.", minute=30)
    store(
        session,
        doc,
        chunk,
        ChunkExtraction(
            [E("Dan", "PERSON"), E("Acme Corp")],
            [R("Dan", "Acme Corp", "WORKS_AT", "Dan works at Acme Corp")],
        ),
    )
    second = retriever.search("Acme Corp")
    assert second.revision > first.revision == first.database_revision
    assert second.revision == second.database_revision
    assert len(second.hits) == 3


def test_a_stale_graph_is_never_served(retriever, session, corpus):
    before = {h.chunk.chunk_id for h in retriever.search("Acme Corp").hits}
    doc, chunk = add_doc(session, "Eve works at Acme Corp.", minute=40)
    store(
        session,
        doc,
        chunk,
        ChunkExtraction(
            [E("Eve", "PERSON"), E("Acme Corp")],
            [R("Eve", "Acme Corp", "WORKS_AT", "Eve works at Acme Corp")],
        ),
    )
    after = {h.chunk.chunk_id for h in retriever.search("Acme Corp").hits}
    assert after - before == {chunk.id}
