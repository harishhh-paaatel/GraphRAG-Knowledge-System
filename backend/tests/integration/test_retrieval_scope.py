"""Shared SQLite helpers of the retrieval paths: document scope and chunk lookup."""

import pytest

from app.models import Document, DocumentChunk
from app.services.retrieval.scope import DocumentScope, lookup_chunks, resolve_document_scope
from tests.integration.test_evidence_recompute import add_doc


def add_doc_with_status(session, status, name):
    doc = Document(
        filename=name,
        stored_path="x",
        file_type="txt",
        size_bytes=1,
        checksum=f"{abs(hash(name)) % 10**12:064d}",
        processing_status=status,
    )
    session.add(doc)
    session.flush()
    session.add(DocumentChunk(id=f"chunk-{name}", document_id=doc.id, chunk_index=0, content=name))
    session.commit()
    return doc


@pytest.fixture
def corpus(session):
    done, done_chunk = add_doc(session, "A completed document.", minute=0)
    other, other_chunk = add_doc(session, "Another completed one.", minute=1)
    pending = add_doc_with_status(session, "pending", "pending.txt")
    failed = add_doc_with_status(session, "failed", "failed.txt")
    return done, done_chunk, other, other_chunk, pending, failed


# ------------------------------------------------------------------------ document scope


def test_without_a_filter_everything_except_unfinished_documents_is_searchable(session, corpus):
    done, _, other, _, pending, failed = corpus
    scope = resolve_document_scope(session, None)
    assert scope.allowed is None and scope.unknown == ()
    assert scope.not_ready == {pending.id, failed.id}
    assert scope.permits(done.id) and scope.permits(other.id)
    assert not scope.permits(pending.id) and not scope.permits(failed.id)


def test_a_filter_keeps_only_requested_documents_that_exist_and_are_completed(session, corpus):
    done, _, _, _, pending, _ = corpus
    scope = resolve_document_scope(session, [done.id, pending.id, "no-such-document"])
    assert scope.allowed == {done.id}
    assert scope.unknown == ("no-such-document",)
    assert scope.permits(done.id)
    assert not scope.permits(pending.id)  # exists but is not ready
    assert not scope.permits("someone-else")  # completed documents outside the filter


def test_unknown_ids_are_reported_sorted_and_not_an_error(session, corpus):
    scope = resolve_document_scope(session, ["zzz", "aaa", "mmm"])
    assert scope.allowed == frozenset()
    assert scope.unknown == ("aaa", "mmm", "zzz")


def test_an_empty_filter_allows_nothing_rather_than_everything(session, corpus):
    scope = resolve_document_scope(session, [])
    assert scope.allowed == frozenset() and scope.unknown == ()
    assert not scope.permits(corpus[0].id)


def test_duplicate_ids_in_the_filter_are_collapsed(session, corpus):
    done = corpus[0]
    scope = resolve_document_scope(session, [done.id, done.id, "x", "x"])
    assert scope.allowed == {done.id} and scope.unknown == ("x",)


def test_the_scope_value_is_immutable():
    scope = DocumentScope(allowed=None, not_ready=frozenset())
    with pytest.raises(AttributeError):
        scope.allowed = frozenset()  # type: ignore[misc]


# ------------------------------------------------------------------------- chunk lookup


def test_lookup_returns_the_stored_chunk_with_its_filename_and_location(session, corpus):
    done, done_chunk, *_ = corpus
    done_chunk.page_number, done_chunk.section = 3, "Intro"
    session.commit()
    records = lookup_chunks(session, [done_chunk.id])
    rec = records[done_chunk.id]
    assert (rec.chunk_id, rec.document_id, rec.filename) == (done_chunk.id, done.id, done.filename)
    assert (rec.content, rec.page_number, rec.section, rec.chunk_index) == (
        "A completed document.",
        3,
        "Intro",
        0,
    )


def test_vanished_chunks_are_absent_and_nothing_is_invented_for_them(session, corpus):
    _, done_chunk, *_ = corpus
    records = lookup_chunks(session, [done_chunk.id, "deleted-chunk", "never-existed"])
    assert set(records) == {done_chunk.id}


def test_a_chunk_whose_document_was_deleted_disappears(session, corpus):
    done, done_chunk, *_ = corpus
    session.delete(session.get(Document, done.id))
    session.commit()
    assert lookup_chunks(session, [done_chunk.id]) == {}


def test_a_scope_removes_chunks_of_documents_it_does_not_permit(session, corpus):
    done, done_chunk, other, other_chunk, pending, _ = corpus
    scope = resolve_document_scope(session, [done.id])
    ids = [done_chunk.id, other_chunk.id, f"chunk-{pending.filename}"]
    assert set(lookup_chunks(session, ids, scope)) == {done_chunk.id}
    # without a filter only the unfinished document's chunk is excluded
    unfiltered = resolve_document_scope(session, None)
    assert set(lookup_chunks(session, ids, unfiltered)) == {done_chunk.id, other_chunk.id}
    # no scope at all: the raw stored rows
    assert set(lookup_chunks(session, ids)) == set(ids)


def test_duplicate_requested_ids_return_one_record_and_an_empty_request_returns_nothing(
    session, corpus
):
    _, done_chunk, *_ = corpus
    assert list(lookup_chunks(session, [done_chunk.id] * 3)) == [done_chunk.id]
    assert lookup_chunks(session, []) == {}


def test_very_large_requests_are_batched(session, corpus):
    _, done_chunk, _, other_chunk, *_ = corpus
    ids = [f"missing-{i}" for i in range(1700)] + [done_chunk.id, other_chunk.id]
    assert set(lookup_chunks(session, ids)) == {done_chunk.id, other_chunk.id}


def test_lookup_reports_the_stored_processing_status(session, corpus):
    done, done_chunk, _, _, pending, _ = corpus
    records = lookup_chunks(session, [done_chunk.id, f"chunk-{pending.filename}"])
    assert records[done_chunk.id].processing_status == "completed"
    assert records[f"chunk-{pending.filename}"].processing_status == "pending"


def test_a_shared_scope_gives_exactly_the_same_vector_search_as_resolving_it_inside(
    app, session, corpus
):
    from app.services.retrieval.vector_search import search_vectors

    vectors, factory = app.state.vectors, app.state.session_factory
    vectors.sync()
    for ids in (None, [corpus[0].id], [corpus[0].id, "nope"], []):
        with factory() as s:
            scope = resolve_document_scope(s, ids)
        inside = search_vectors(vectors, factory, "completed document", top_k=5, document_ids=ids)
        shared = search_vectors(
            vectors, factory, "completed document", top_k=5, document_ids=ids, scope=scope
        )
        assert [h.chunk_id for h in shared.results] == [h.chunk_id for h in inside.results]
        assert shared.unknown_document_ids == inside.unknown_document_ids
