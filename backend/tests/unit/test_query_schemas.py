"""The shared /query contract: request caps, response invariants, no confidence field."""

import pytest
from pydantic import BaseModel, ValidationError

from app.schemas.graph_search import EvidenceOut
from app.schemas.query import (
    ANSWER_NOTE,
    EXCERPT_CHARS,
    HistoryDetailOut,
    HistoryEntryOut,
    HistoryListOut,
    QueryRequest,
    QueryResponse,
    SourceOut,
)
from tests.llm_helpers import make_response, make_source

# --------------------------------------------------------------------------- request


def test_a_question_alone_is_a_valid_request_with_safe_defaults():
    r = QueryRequest(question="Who works at Acme Inc?")
    assert r.retrieval.limit == 8 and r.retrieval.require_complete_retrieval is False
    assert r.retrieval.vector.top_k == 20 and r.retrieval.graph.direction == "both"
    assert r.retrieval.fusion.method == "rrf"
    assert r.generation.include_graph_facts is True and r.generation.context_tokens is None
    assert r.include_prompt is False


@pytest.mark.parametrize("field", ["provider", "model", "base_url", "api_key", "llm", "url"])
def test_the_caller_cannot_choose_the_provider_model_or_url(field):
    with pytest.raises(ValidationError):
        QueryRequest(question="q", **{field: "http://evil.example"})
    with pytest.raises(ValidationError):
        QueryRequest(question="q", generation={field: "x"})


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "q", "surprise": 1},
        {"question": "q", "retrieval": {"limit": 0}},
        {"question": "q", "retrieval": {"vector": {"top_k": 0}}},
        {"question": "q", "retrieval": {"graph": {"direction": "sideways"}}},
        {"question": "q", "retrieval": {"fusion": {"vector_weight": 0, "graph_weight": 0}}},
        {"question": "q", "retrieval": {"document_ids": [str(i) for i in range(101)]}},
        {"question": "q", "retrieval": {"surprise": True}},
        {"question": "q", "generation": {"context_tokens": 10}},
        {"question": "q", "generation": {"max_answer_tokens": 1}},
        {"question": "q", "generation": {"grounding_policy": "ignore"}},
    ],
)
def test_invalid_requests_are_rejected(body):
    with pytest.raises(ValidationError):
        QueryRequest(**body)


def test_retrieval_options_reuse_the_hybrid_search_blocks():
    r = QueryRequest(
        question="q",
        retrieval={
            "limit": 5,
            "vector": {"top_k": 7, "similarity_threshold": 0.3},
            "graph": {"direction": "out", "hops": 2, "entity_ids": ["e1"]},
            "fusion": {"rrf_k": 10, "vector_weight": 2.0},
            "require_complete_retrieval": True,
        },
        generation={"context_tokens": 512, "grounding_policy": "annotate"},
    )
    assert r.retrieval.graph.direction == "outgoing"  # alias normalised by the shared block
    assert (r.retrieval.vector.top_k, r.retrieval.fusion.rrf_k) == (7, 10)
    assert r.generation.grounding_policy == "annotate"


# -------------------------------------------------------------------------- response


def test_a_valid_answered_response_round_trips_through_json():
    resp = make_response()
    again = QueryResponse.model_validate_json(resp.model_dump_json())
    assert again == resp
    assert resp.answer_note == ANSWER_NOTE and "not verified" in resp.answer_note


def test_an_answered_response_needs_an_answer_and_no_reason():
    with pytest.raises(ValidationError, match="non-empty answer"):
        make_response(answer=None)
    with pytest.raises(ValidationError, match="non-empty answer"):
        make_response(answer="   ")
    with pytest.raises(ValidationError, match="no reason"):
        make_response(reason="model_declined")


@pytest.mark.parametrize("status", ["insufficient_evidence", "ungrounded"])
def test_a_response_without_an_answer_must_not_carry_one(status):
    kwargs = {"reason": "no_retrieval_results"} if status == "insufficient_evidence" else {}
    with pytest.raises(ValidationError, match="must not carry an answer"):
        make_response(status=status, citations=[], **kwargs)  # answer text still set
    ok = make_response(status=status, answer=None, citations=[], **kwargs)
    assert ok.answer is None and ok.status == status


def test_insufficient_evidence_needs_a_reason_and_other_statuses_must_not_have_one():
    with pytest.raises(ValidationError, match="needs a reason"):
        make_response(status="insufficient_evidence", answer=None, citations=[])
    with pytest.raises(ValidationError, match="only valid for insufficient_evidence"):
        make_response(status="ungrounded", answer=None, citations=[], reason="model_declined")
    for reason in ("no_retrieval_results", "no_context_after_budget", "model_declined"):
        make_response(status="insufficient_evidence", answer=None, citations=[], reason=reason)
    with pytest.raises(ValidationError):
        make_response(status="insufficient_evidence", answer=None, citations=[], reason="because")


def test_only_an_answered_response_has_citations_and_each_is_also_a_source():
    cited = make_source("S2", "c2")
    with pytest.raises(ValidationError, match="also be listed in sources"):
        make_response(citations=[cited])  # S2 is not among the sources
    with pytest.raises(ValidationError, match="only an answered response has citations"):
        make_response(status="ungrounded", answer=None)  # citations left in place
    ok = make_response(citations=[cited], sources=[make_source(), cited])
    assert [c.marker for c in ok.citations] == ["S2"]


@pytest.mark.parametrize("marker", ["S0", "s1", "S", "1", "S01", "S1a", "[S1]", "S-1", ""])
def test_citation_markers_must_look_like_s_followed_by_a_positive_number(marker):
    with pytest.raises(ValidationError):
        make_source(marker=marker)


def test_excerpts_are_short():
    make_source(excerpt="x" * EXCERPT_CHARS)
    with pytest.raises(ValidationError):
        make_source(excerpt="x" * (EXCERPT_CHARS + 2))


def test_statuses_modes_and_reasons_are_closed_sets():
    with pytest.raises(ValidationError):
        make_response(status="failed")
    with pytest.raises(ValidationError):
        make_response(generation={**make_response().generation.model_dump(), "mode": "gpt"})


def _field_names(model: type[BaseModel], seen: set[type]) -> set[str]:
    names: set[str] = set()
    if model in seen or model is EvidenceOut:  # evidence rows are checked separately below
        return names
    seen.add(model)
    for name, field in model.model_fields.items():
        names.add(name)
        stack = [field.annotation]
        while stack:
            ann = stack.pop()
            if isinstance(ann, type) and issubclass(ann, BaseModel):
                names |= _field_names(ann, seen)
            stack.extend(getattr(ann, "__args__", ()) or ())
    return names


def test_no_answer_level_field_claims_confidence_or_correctness():
    names = _field_names(QueryResponse, set()) | _field_names(QueryRequest, set())
    forbidden = {"confidence", "probability", "certainty", "relevance_score", "accuracy", "correct"}
    assert not {n for n in names if any(f in n.lower() for f in forbidden)}, sorted(names)
    assert "score" not in names and "answer_score" not in names


def test_the_only_confidence_in_the_contract_is_the_extractors_own_on_evidence_rows():
    """Relationship evidence carries whatever confidence an extractor reported (None today);
    it describes an extraction, never an answer, and is never computed here."""
    field = EvidenceOut.model_fields["confidence"]
    assert field.is_required() or field.default is None
    sources = SourceOut.model_fields
    assert "confidence" not in sources and "confidence" not in QueryResponse.model_fields


def test_the_prompt_is_a_debug_only_optional_field():
    assert make_response().prompt is None
    assert "Only in debug mode" in QueryResponse.model_fields["prompt"].description


def test_sources_keep_their_provenance_fields():
    s = SourceOut.model_fields
    for field in ("chunk_id", "document_id", "source", "retrieved_by", "fused_rank", "evidence"):
        assert field in s
    assert s["still_exists"].default is True


# ------------------------------------------------------------------------- history


def test_history_schemas_describe_entries_lists_and_details():
    entry = HistoryEntryOut(
        id="h1",
        question="q",
        answer=None,
        status="insufficient_evidence",
        reason="model_declined",
        generation_mode="llm",
        retrieval_strategy="hybrid",
        execution_time_ms=12.5,
        created_at="2026-11-06T10:00:00",
    )
    detail = HistoryDetailOut(**entry.model_dump(), metadata={"citations": []})
    page = HistoryListOut(items=[entry], total=1, limit=50, offset=0)
    assert detail.metadata == {"citations": []} and page.items[0].answer is None
