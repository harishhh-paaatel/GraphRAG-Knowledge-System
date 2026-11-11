"""Normalization collisions: what the entity keys do and, as important, what they do NOT do.

These pin down the behaviour of the key function that both ingestion and the Phase 6 query
linker build on. Query-time behaviour that does not exist yet (stripping a possessive from a
*query* token, the capitalisation guard for short names, ambiguity reporting) is specified in
docs/phase-6-proposal.md section 4.2 and is tested when the linker is implemented (M2).
"""

import pytest

from app.services.extraction.normalization import compact_key, normalize_key
from app.services.extraction.resolver import make_entity_id

# --------------------------------------------------- identical keys for different entities


@pytest.mark.parametrize("spelling", ["Mercury", "MERCURY", "mercury", "  Mercury  "])
def test_spellings_that_differ_only_in_case_or_spacing_share_one_key(spelling):
    assert normalize_key(spelling, "LOCATION") == "mercury"


def test_the_same_key_for_two_types_still_gives_two_distinct_entity_identities():
    assert normalize_key("Mercury", "LOCATION") == normalize_key("Mercury", "PRODUCT") == "mercury"
    planet = make_entity_id("LOCATION", "mercury")
    brand = make_entity_id("PRODUCT", "mercury")
    assert planet != brand
    assert make_entity_id("LOCATION", "mercury") == planet  # deterministic


def test_an_organization_and_a_non_organization_can_collide_after_suffix_removal():
    org = normalize_key("Acme Inc", "ORGANIZATION")
    thing = normalize_key("Acme", "PRODUCT")
    assert org == thing == "acme"  # a lookup by key alone would see two candidates
    assert make_entity_id("ORGANIZATION", org) != make_entity_id("PRODUCT", thing)


# ------------------------------------------------------------------- corporate suffixes


@pytest.mark.parametrize(
    "name",
    [
        "Acme",
        "Acme Inc",
        "Acme Inc.",
        "ACME  Corp",
        "The Acme Corporation",
        "Acme Co Ltd",
        "acme, llc",
        "Acme Incorporated",
        "Acme GmbH",
    ],
)
def test_corporate_suffixes_are_removed_for_organizations_only(name):
    assert normalize_key(name, "ORGANIZATION") == "acme"


@pytest.mark.parametrize("etype", ["PERSON", "PRODUCT", "LOCATION", None])
def test_other_types_and_untyped_names_keep_their_suffix(etype):
    assert normalize_key("Acme Inc", etype) == "acme inc"
    assert normalize_key("Acme Corp", etype) == "acme corp"


def test_only_a_trailing_suffix_is_removed_and_words_are_never_rewritten():
    assert normalize_key("Inc Acme", "ORGANIZATION") == "inc acme"
    assert normalize_key("Acmeinc", "ORGANIZATION") == "acmeinc"
    holdings = normalize_key("Acme Incorporated Holdings", "ORGANIZATION")
    assert holdings == "acme incorporated holdings"


def test_a_name_that_is_only_a_suffix_is_not_emptied():
    assert normalize_key("Corp", "ORGANIZATION") == "corp"
    assert normalize_key("Inc", "ORGANIZATION") == "inc"


# ---------------------------------------------------------------------------- possessives


def test_ingestion_normalization_does_not_strip_a_possessive():
    """So Alice's and Alice are different keys. The query linker (M2) strips a trailing 's
    from *query* tokens before lookup instead of changing how entities are keyed."""
    assert normalize_key("Alice's", "PERSON") == "alice's"
    assert normalize_key("Alice's", "PERSON") != normalize_key("Alice", "PERSON")


def test_straight_and_curly_apostrophes_give_the_same_key():
    assert normalize_key("Alice’s", "PERSON") == normalize_key("Alice's", "PERSON")
    assert normalize_key("O’Brien", "PERSON") == normalize_key("O'Brien", "PERSON")


def test_an_apostrophe_inside_a_word_is_kept_but_one_at_a_word_edge_is_dropped():
    assert normalize_key("O'Brien", "PERSON") == "o'brien"
    # a trailing apostrophe (plural possessive) is stripped at the token edge
    assert normalize_key("Dunkin' Donuts", "ORGANIZATION") == "dunkin donuts"


# ------------------------------------------------------------------------ short names


@pytest.mark.parametrize("spelling", ["Go", "GO", "go", " go "])
def test_short_names_lose_their_capitalisation_in_the_key(spelling):
    """Go (the language), GO and the verb go are indistinguishable by key alone: that is why
    the M2 linker links names shorter than 4 characters only on an exact-case match."""
    assert normalize_key(spelling, "TECHNOLOGY") == "go"


@pytest.mark.parametrize(
    "name, key",
    [("C++", "c++"), ("C#", "c#"), ("AT&T", "at&t"), ("Node.js", "node.js"), ("AI", "ai")],
)
def test_symbols_inside_short_names_survive(name, key):
    assert normalize_key(name) == key


# ------------------------------------------------------------------------ compact keys


def test_compact_keys_collapse_spacing_but_the_normalized_keys_stay_different():
    assert normalize_key("Open AI") != normalize_key("OpenAI")
    assert compact_key(normalize_key("Open AI")) == compact_key(normalize_key("OpenAI"))
