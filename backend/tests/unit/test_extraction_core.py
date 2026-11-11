"""Normalization, validation (including invalid-relationship rejection) and the LLM provider."""

import json

import pytest

from app.core.config import Settings
from app.services.extraction.factory import ConfigurationError, build_extraction_provider
from app.services.extraction.llm import LLMExtractionProvider
from app.services.extraction.normalization import compact_key, normalize_key, squash
from app.services.extraction.types import (
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionError,
)
from app.services.extraction.validation import validate_entities, validate_relationships

TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "TECHNOLOGY", "PRODUCT", "CONCEPT", "EVENT"}
TEXT = "Alice works at Acme Corp in Berlin.\nFastAPI uses Pydantic."


def E(name, etype, **kw):  # noqa: N802 - terse test helper
    return ExtractedEntity(name, etype, **kw)


def R(src, tgt, rtype, support, **kw):  # noqa: N802
    return ExtractedRelationship(src, tgt, rtype, support, **kw)


# ---- normalization ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("OpenAI", "  openai  "),
        ("Open-AI", "Open AI"),
        ("The Linux Foundation", "linux foundation"),
        ("U.S.A.", "u.s.a"),
        ("Café  Noir", "café noir"),
    ],
)
def test_keys_equal_for_meaningless_differences(a, b):
    assert normalize_key(a) == normalize_key(b) or compact_key(normalize_key(a)) == compact_key(
        normalize_key(b)
    )


def test_normalize_key_exact_cases():
    assert normalize_key("  OpenAI,  ") == "openai"
    assert normalize_key("Node.js") == "node.js"
    assert normalize_key("C++") == "c++"
    assert normalize_key("The Beatles") == "beatles"
    assert normalize_key("The") == "the"  # never reduce a name to nothing


def test_corporate_suffix_dropped_only_for_organizations():
    assert normalize_key("Acme Inc.", "ORGANIZATION") == "acme"
    assert normalize_key("Acme Corp", "ORGANIZATION") == normalize_key("ACME", "ORGANIZATION")
    assert normalize_key("Acme Inc", "PRODUCT") == "acme inc"
    assert normalize_key("Inc", "ORGANIZATION") == "inc"


def test_different_words_are_never_normalized_together():
    assert normalize_key("Python 2") != normalize_key("Python 3")
    assert normalize_key("Apple") != normalize_key("Apples")
    assert compact_key(normalize_key("Open AI")) == compact_key(normalize_key("OpenAI"))
    assert squash("  Hello   WORLD ") == "hello world"


# ---- entity validation ------------------------------------------------------------------


def test_entities_valid_and_deduplicated():
    valid, rejected = validate_entities(
        TEXT,
        [E("Alice", "PERSON"), E("alice", "person", surface_form="Alice"), E("Berlin", "LOCATION")],
        TYPES,
    )
    assert [(v.name, v.entity_type, v.surface_form) for v in valid][0] == (
        "Alice",
        "PERSON",
        "Alice",
    )
    assert len(valid) == 2 and not rejected


def test_entity_rejections():
    valid, rejected = validate_entities(
        TEXT,
        [
            E("Bob", "PERSON"),  # not in text
            E("Alice", "SPACESHIP"),  # type not enabled
            E("", "PERSON"),  # empty
            E("Alice", "PERSON", confidence=1.5),  # bad confidence
            E("Acme", "ORGANIZATION", surface_form="Acme Corp"),  # ok: surface found
        ],
        TYPES,
    )
    assert [v.name for v in valid] == ["Acme"]
    assert valid[0].surface_form == "Acme Corp"
    reasons = sorted(r.reason for r in rejected)
    assert len(reasons) == 4
    assert "text not found in chunk" in reasons
    assert any("not enabled" in r for r in reasons)


def test_entity_found_via_name_when_surface_form_is_wrong():
    valid, _ = validate_entities(TEXT, [E("Berlin", "LOCATION", surface_form="Munich")], TYPES)
    assert valid[0].surface_form == "Berlin"


def _ents():
    valid, _ = validate_entities(
        TEXT,
        [
            E("Alice", "PERSON"),
            E("Acme Corp", "ORGANIZATION"),
            E("Berlin", "LOCATION"),
            E("FastAPI", "TECHNOLOGY"),
            E("Pydantic", "TECHNOLOGY"),
        ],
        TYPES,
    )
    return valid


def test_valid_relationship_accepted_with_evidence():
    rels, rejected = validate_relationships(
        TEXT,
        _ents(),
        [R("Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp", confidence=0.9)],
    )
    assert not rejected
    assert rels[0].relationship_type == "WORKS_AT"
    assert rels[0].supporting_text == "Alice works at Acme Corp"
    assert rels[0].confidence == 0.9


@pytest.mark.parametrize(
    ("rel", "reason"),
    [
        (
            R("Alice", "Acme Corp", "WORKS_AT", "Alice founded Acme Corp"),
            "supporting text not found",
        ),
        (R("Alice", "Acme Corp", "WORKS_AT", ""), "missing supporting text"),
        (R("Alice", "Globex", "WORKS_AT", "Alice works at Acme Corp"), "not an entity extracted"),
        (
            R("Alice", "Acme Corp", "MARRIED_TO", "Alice works at Acme Corp"),
            "unknown relationship type",
        ),
        (R("Alice", "Alice", "RELATED_TO", "Alice works at Acme Corp"), "self-relationship"),
        (
            R("Alice", "Berlin", "WORKS_AT", "Alice works at Acme Corp in Berlin"),
            "not valid for WORKS_AT",
        ),
        (R("Acme Corp", "Alice", "WORKS_AT", "Alice works at Acme Corp"), "not valid for WORKS_AT"),
        (R("Alice", "Berlin", "LOCATED_IN", "Alice works at Acme"), "does not mention both"),
        (
            R("Alice", "Acme Corp", "WORKS_AT", "Alice works at Acme Corp", confidence=7),
            "confidence",
        ),
    ],
)
def test_invalid_relationships_are_rejected(rel, reason):
    rels, rejected = validate_relationships(TEXT, _ents(), [rel])
    assert rels == []
    assert len(rejected) == 1 and reason in rejected[0].reason


def test_evidence_match_ignores_case_and_whitespace_but_not_words():
    ok, _ = validate_relationships(
        TEXT, _ents(), [R("Alice", "Acme Corp", "WORKS_AT", "alice  WORKS at\nacme corp")]
    )
    assert len(ok) == 1


def test_ambiguous_endpoint_is_rejected():
    text = "Mercury orbits the sun. Mercury is a metal. Mercury relates to Python."
    valid, _ = validate_entities(
        text, [E("Mercury", "LOCATION"), E("Mercury", "CONCEPT"), E("Python", "TECHNOLOGY")], TYPES
    )
    rels, rejected = validate_relationships(
        text, valid, [R("Mercury", "Python", "RELATED_TO", "Mercury relates to Python")]
    )
    assert rels == [] and "ambiguous" in rejected[0].reason


# ---- LLM provider -----------------------------------------------------------------------


class FakeClient:
    def __init__(self, reply):
        self.reply, self.prompts = reply, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


GOOD = {
    "entities": [
        {"name": "Alice", "type": "PERSON", "confidence": 0.99},
        {"name": "Acme Corp", "type": "ORGANIZATION", "description": "A company"},
    ],
    "relationships": [
        {
            "source": "Alice",
            "target": "Acme Corp",
            "type": "WORKS_AT",
            "supporting_text": "Alice works at Acme Corp",
        }
    ],
}


def test_llm_provider_parses_json_even_with_prose_and_fences():
    client = FakeClient("Sure! ```json\n" + json.dumps(GOOD) + "\n``` hope that helps")
    out = LLMExtractionProvider(client, ["PERSON", "ORGANIZATION"]).extract(TEXT)
    assert [e.name for e in out.entities] == ["Alice", "Acme Corp"]
    assert out.entities[1].description == "A company"
    assert out.entities[0].confidence is None  # self-reported scores are deliberately ignored
    assert out.relationships[0].relationship_type == "WORKS_AT"
    assert "PERSON, ORGANIZATION" in client.prompts[0] and "WORKS_AT" in client.prompts[0]
    assert TEXT in client.prompts[0]


@pytest.mark.parametrize(
    "reply",
    ["no json here", '{"entities": [', '{"entities": "oops"}', '{"entities": [{"name": "x"}]}'],
)
def test_llm_provider_bad_output_raises_extraction_error(reply):
    with pytest.raises(ExtractionError):
        LLMExtractionProvider(FakeClient(reply), ["PERSON"]).extract(TEXT)


def test_llm_provider_wraps_client_errors_without_leaking_details():
    with pytest.raises(ExtractionError) as exc:
        LLMExtractionProvider(FakeClient(RuntimeError("api_key=sk-secret")), ["PERSON"]).extract(
            TEXT
        )
    assert "sk-secret" not in str(exc.value)


def test_hallucinated_llm_relationship_is_caught_by_validation():
    hallucinated = {
        "entities": GOOD["entities"],
        "relationships": [
            {
                "source": "Alice",
                "target": "Acme Corp",
                "type": "WORKS_AT",
                "supporting_text": "Alice is the CEO of Acme Corp",  # not in the chunk
            }
        ],
    }
    out = LLMExtractionProvider(FakeClient(json.dumps(hallucinated)), ["PERSON"]).extract(TEXT)
    valid_entities, _ = validate_entities(TEXT, out.entities, TYPES)
    rels, rejected = validate_relationships(TEXT, valid_entities, out.relationships)
    assert rels == [] and "not found in chunk" in rejected[0].reason


# ---- settings / factory -----------------------------------------------------------------


def test_factory_builds_known_providers_and_refuses_unwired_llm(tmp_path):
    assert (
        build_extraction_provider(Settings(data_dir=tmp_path, _env_file=None)).name == "rule_based"
    )
    mock = Settings(data_dir=tmp_path, extraction_provider="mock", _env_file=None)
    assert build_extraction_provider(mock).name == "mock"
    with pytest.raises(ConfigurationError):
        build_extraction_provider(
            Settings(data_dir=tmp_path, extraction_provider="llm", _env_file=None)
        )


def test_entity_types_are_configurable_and_validated(tmp_path):
    s = Settings(data_dir=tmp_path, entity_types=["PERSON", "DATE"], _env_file=None)
    assert s.entity_types == ["PERSON", "DATE"]
    for bad in ([], ["person"], ["PERSON", "bad type"]):
        with pytest.raises(ValueError):
            Settings(data_dir=tmp_path, entity_types=bad, _env_file=None)


def test_empty_extraction_is_valid():
    assert ChunkExtraction().entities == []
