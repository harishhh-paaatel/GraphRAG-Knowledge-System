import pytest

from app.core.config import Settings
from app.services.extraction.rule_based import RuleBasedExtractor
from app.services.extraction.validation import validate_entities, validate_relationships

TYPES = Settings(_env_file=None).entity_types


@pytest.fixture
def extractor():
    return RuleBasedExtractor(TYPES)


def _triples(out):
    return {(r.source, r.relationship_type, r.target) for r in out.relationships}


def _entities(out):
    return {(e.name, e.entity_type) for e in out.entities}


def test_extracts_all_entity_categories(extractor):
    text = (
        "Sebastian Ramirez presented at PyCon Conference in Berlin. OpenAI released ChatGPT. "
        "The Python Software Foundation supports Python and machine learning research."
    )
    got = _entities(extractor.extract(text))
    assert ("Sebastian Ramirez", "PERSON") in got
    assert ("PyCon Conference", "EVENT") in got
    assert ("Berlin", "LOCATION") in got
    assert ("OpenAI", "ORGANIZATION") in got
    assert ("ChatGPT", "PRODUCT") in got
    assert ("Python Software Foundation", "ORGANIZATION") in got
    assert ("Python", "TECHNOLOGY") in got
    assert ("machine learning", "CONCEPT") in got


def test_extracts_each_relationship_type(extractor):
    cases = {
        "Alice works at Acme Corp.": ("Alice", "WORKS_AT", "Acme Corp"),
        "OpenAI developed ChatGPT.": ("OpenAI", "DEVELOPED", "ChatGPT"),
        "FastAPI uses Pydantic.": ("FastAPI", "USES", "Pydantic"),
        "Acme Corp is located in Berlin.": ("Acme Corp", "LOCATED_IN", "Berlin"),
        "SQLAlchemy is part of Python.": ("SQLAlchemy", "PART_OF", "Python"),
        "Linus Torvalds wrote Linux.": ("Linus Torvalds", "AUTHORED", "Linux"),
        "FastAPI depends on Starlette.": ("FastAPI", "DEPENDS_ON", "Starlette"),
        "Neo4j is related to graph database.": ("Neo4j", "RELATED_TO", "graph database"),
    }
    for text, triple in cases.items():
        assert triple in _triples(extractor.extract(text)), text


def test_passive_voice_reverses_direction(extractor):
    out = extractor.extract("FastAPI was created by Sebastian Ramirez.")
    assert _triples(out) == {("Sebastian Ramirez", "DEVELOPED", "FastAPI")}
    out = extractor.extract("Starlette is used by FastAPI.")
    assert _triples(out) == {("FastAPI", "USES", "Starlette")}


def test_types_inferred_only_when_unambiguous(extractor):
    # "Zorblax Corp" is typed by its keyword; "Qwenta" is only knowable from WORKS_AT's target.
    out = extractor.extract("Alice works at Qwenta.")
    assert ("Qwenta", "ORGANIZATION") in _entities(out)
    # DEVELOPED's source could be a person or an organisation, so an unknown subject is dropped.
    out = extractor.extract("Zxqv developed FastAPI.")
    assert _triples(out) == set() and ("Zxqv", "ORGANIZATION") not in _entities(out)


def test_supporting_text_is_an_exact_slice_and_validates(extractor):
    text = "Intro line.\nAlice works at Acme Corp in Berlin. Acme Corp is located in Berlin."
    out = extractor.extract(text)
    for r in out.relationships:
        assert r.supporting_text in text
    valid_entities, _ = validate_entities(text, out.entities, set(TYPES))
    rels, rejected = validate_relationships(text, valid_entities, out.relationships)
    assert len(rels) == len(out.relationships) == 2 and not rejected


def test_conservative_by_design_no_guessing(extractor):
    for text in [
        "Alice, who works at Acme Corp, likes Python.",  # trigger not directly between entities
        "Alice might work at Acme Corp.",
        "Python and FastAPI are popular.",  # co-occurrence is not a relationship
        "Tests use pytest.",  # unknown subject
        "alice works at acme corp.",  # lowercase names are not recognised
    ]:
        assert extractor.extract(text).relationships == [], text


def test_no_confidence_is_reported(extractor):
    out = extractor.extract("Alice works at Acme Corp. FastAPI uses Pydantic.")
    assert all(e.confidence is None for e in out.entities)
    assert all(r.confidence is None for r in out.relationships)


def test_deterministic(extractor):
    text = "Alice works at Acme Corp. FastAPI was created by Sebastian Ramirez."
    assert extractor.extract(text) == extractor.extract(text)


def test_respects_enabled_entity_types():
    only_tech = RuleBasedExtractor({"TECHNOLOGY", "PERSON"})
    out = only_tech.extract("Alice works at Acme Corp. FastAPI uses Pydantic. OpenAI is in Paris.")
    assert {t for _, t in _entities(out)} <= {"TECHNOLOGY", "PERSON"}
    assert _triples(out) == {("FastAPI", "USES", "Pydantic")}  # WORKS_AT needs ORGANIZATION


def test_custom_gazetteer_replaces_default():
    custom = RuleBasedExtractor({"TECHNOLOGY"}, gazetteer={"Rustlang": "TECHNOLOGY"})
    assert _entities(custom.extract("Rustlang and Python.")) == {("Rustlang", "TECHNOLOGY")}


def test_empty_and_symbol_only_text(extractor):
    for text in ("", "   ", "!!! ???", "12345"):
        out = extractor.extract(text)
        assert out.entities == [] and out.relationships == []


def test_stop_words_are_not_entities(extractor):
    assert _entities(extractor.extract("The results were good. However, It rained.")) == set()
