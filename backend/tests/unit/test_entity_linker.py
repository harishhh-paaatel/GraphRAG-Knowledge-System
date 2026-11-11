"""Exact entity linking inside natural-language queries (proposal section 4.2)."""

import pytest

from app.services.retrieval.entity_linker import (
    AliasIndex,
    EntityLinker,
    UnknownEntities,
    tokenize_query,
)
from tests.graph_search_helpers import make_snapshot, node

NODES = [
    node("acme", "Acme Inc", "ORGANIZATION", aliases=["Acme Corporation"], docs=("d1", "d2")),
    node("alice", "Alice", "PERSON"),
    node("bobsmith", "Bob Smith", "PERSON", aliases=["Bobby"]),
    node("smith", "Smith", "PERSON"),
    node("go", "Go", "TECHNOLOGY"),
    node("ai", "AI", "CONCEPT"),
    node("ibm", "IBM Corp", "ORGANIZATION"),
    node("c", "C", "TECHNOLOGY"),
    node("rust", "Rust", "TECHNOLOGY"),
    node("nodejs", "Node.js", "TECHNOLOGY"),
    node("stlouis", "St. Louis", "LOCATION"),
    node("zeta", "Zeta Inc", "PRODUCT"),
    node("zorro", "Zorro", "PRODUCT"),
    node("mercury_p", "Mercury", "LOCATION", docs=("d1",)),
    node("mercury_m", "Mercury", "PRODUCT", docs=("d1", "d2", "d3")),
    node("alphabeta", "Alpha Beta", "CONCEPT"),
    node("betagamma", "Beta Gamma", "CONCEPT"),
    node("newyork", "New York", "LOCATION"),
    node("york", "York", "LOCATION"),
    node("berlin", "Berlin", "LOCATION"),
]


@pytest.fixture(scope="module")
def snap():
    return make_snapshot(NODES)


@pytest.fixture
def linker():
    return EntityLinker(max_name_tokens=8)


def ids(result):
    return [e.entity_id for e in result.linked]


def link(linker, snap, query, **kw):
    return linker.link(snap, query, **kw)


# ----------------------------------------------------------------- names and aliases


@pytest.mark.parametrize(
    "query",
    [
        "Alice",
        "Who is Alice?",
        "alice works here",
        "Tell me about ALICE please",
        "What did the team say to alice",
        "  Alice  ",
    ],
)
def test_a_name_is_found_anywhere_in_a_natural_language_query(linker, snap, query):
    result = link(linker, snap, query)
    assert ids(result) == ["alice"] and result.seeds == ("alice",)
    assert result.linked[0].match_type == "exact_name" and result.linked[0].resolved_by == "query"


def test_whitespace_of_any_kind_between_name_words_is_irrelevant(linker, snap):
    assert ids(link(linker, snap, "Did Bob\n   Smith\tjoin?")) == ["bobsmith"]


def test_an_alias_links_the_entity_and_says_it_was_an_alias(linker, snap):
    result = link(linker, snap, "What did Bobby say?")
    assert ids(result) == ["bobsmith"]
    assert result.linked[0].match_type == "exact_alias" and result.linked[0].span == "Bobby"


def test_the_reported_span_is_the_query_text_that_matched(linker, snap):
    result = link(linker, snap, "Who works at ACME Inc?")
    assert result.linked[0].span == "ACME Inc"
    # a comma ends a name, so only the part before it is the span
    assert link(linker, snap, "Who works at ACME, inc?").linked[0].span == "ACME"
    assert result.linked[0].name == "Acme Inc" and result.linked[0].entity_type == "ORGANIZATION"


def test_several_different_entities_in_one_query_are_all_linked(linker, snap):
    result = link(linker, snap, "How is Alice connected to Berlin and Acme Inc?")
    assert sorted(ids(result)) == ["acme", "alice", "berlin"]
    assert result.seeds == ("acme", "alice", "berlin")


def test_the_same_entity_named_twice_is_one_seed(linker, snap):
    result = link(linker, snap, "Alice met Alice")
    assert ids(result) == ["alice", "alice"] and result.seeds == ("alice",)


# ----------------------------------------------------------------- corporate suffixes


@pytest.mark.parametrize(
    "query",
    [
        "acme",
        "Acme Inc",
        "ACME, Inc.",
        "the Acme Corporation",
        "Acme Corp",
        "Acme Co Ltd",
        "Acme LLC announced",
    ],
)
def test_corporate_suffixes_are_ignored_when_looking_up_organizations(linker, snap, query):
    result = link(linker, snap, query)
    assert ids(result) == ["acme"], query


def test_the_suffix_is_not_stripped_for_non_organizations(linker, snap):
    assert ids(link(linker, snap, "Zeta")) == []
    assert ids(link(linker, snap, "Zeta Inc")) == ["zeta"]
    assert ids(link(linker, snap, "Zeta Corp")) == []


def test_a_suffix_is_never_stripped_to_reach_a_non_organization(linker, snap):
    """'Zorro Inc' must not link the PRODUCT 'Zorro': stripping is for ORGANIZATION keys only."""
    assert ids(link(linker, snap, "Zorro")) == ["zorro"]
    result = link(linker, snap, "Zorro Inc")
    assert ids(result) == ["zorro"] and result.linked[0].span == "Zorro"  # only the bare name
    assert ids(link(linker, snap, "Zorro Corporation")) == ["zorro"]


def test_a_suffix_only_counts_at_the_end_of_the_name(linker, snap):
    assert ids(link(linker, snap, "Inc Acme")) == ["acme"]  # 'Acme' itself still matches
    result = link(linker, snap, "Inc Acme")
    assert result.linked[0].span == "Acme"


# ---------------------------------------------------------------------------- possessives


@pytest.mark.parametrize(
    "query, expected",
    [
        ("Where is Alice's manager?", "alice"),
        ("Alice’s manager", "alice"),
        ("Acme Inc's headquarters", "acme"),
        ("Acme's CEO", "acme"),
        ("Bob Smith's laptop", "bobsmith"),
        ("What is Go's tooling like?", "go"),
    ],
)
def test_a_trailing_possessive_is_accepted(linker, snap, query, expected):
    assert ids(link(linker, snap, query)) == [expected]


def test_the_possessive_is_the_only_morphological_allowance(linker, snap):
    assert ids(link(linker, snap, "Alices manager")) == []
    assert ids(link(linker, snap, "the Alices")) == []
    assert ids(link(linker, snap, "Berlins")) == []


def test_a_possessive_in_the_middle_of_a_name_does_not_match_the_name(linker, snap):
    assert ids(link(linker, snap, "Bob's Smith")) == ["smith"]  # only 'Smith' is a real match


# ------------------------------------------------------------------------- short names


def test_a_short_name_links_only_with_its_stored_capitalisation(linker, snap):
    assert ids(link(linker, snap, "Is Go fast?")) == ["go"]
    assert ids(link(linker, snap, "Where do I go next?")) == []
    assert ids(link(linker, snap, "GO")) == []
    assert ids(link(linker, snap, "go")) == []


def test_short_names_with_a_capitalisation_of_their_own(linker, snap):
    assert ids(link(linker, snap, "AI is everywhere")) == ["ai"]
    assert ids(link(linker, snap, "ai is everywhere")) == []
    assert ids(link(linker, snap, "Ai is everywhere")) == []


def test_a_short_organization_keeps_its_capitalisation_guard_even_with_a_suffix(linker, snap):
    assert ids(link(linker, snap, "I work at IBM Corp")) == ["ibm"]
    assert ids(link(linker, snap, "I work at IBM")) == ["ibm"]
    assert ids(link(linker, snap, "I work at ibm corp")) == []
    assert ids(link(linker, snap, "IBM.")) == ["ibm"]


def test_single_character_names_are_not_linkable(linker, snap):
    assert ids(link(linker, snap, "Vitamin C helps")) == []
    assert "c" not in linker.index_for(snap).by_key


def test_longer_names_are_case_insensitive(linker, snap):
    assert ids(link(linker, snap, "rust")) == ["rust"]
    assert ids(link(linker, snap, "RUST")) == ["rust"]


# ----------------------------------------------------------------------- word boundaries


@pytest.mark.parametrize(
    "query",
    ["Acmeville", "Alicia", "trust me", "Berlinn", "Alic", "unalice", "Rustic", "Smithson"],
)
def test_partial_words_never_match(linker, snap, query):
    assert ids(link(linker, snap, query)) == [], query


@pytest.mark.parametrize("query", ["Acm Inc", "Alise", "Bob Smyth", "Berln", "Acme-ish"])
def test_no_fuzzy_matching(linker, snap, query):
    assert ids(link(linker, snap, query)) == [], query


def test_words_joined_by_a_hyphen_or_slash_are_one_token(linker, snap):
    assert ids(link(linker, snap, "Berlin-based")) == []
    assert ids(link(linker, snap, "Go/Rust")) == []


def test_internal_punctuation_of_a_name_is_kept(linker, snap):
    assert ids(link(linker, snap, "Is Node.js fast?")) == ["nodejs"]
    assert ids(link(linker, snap, "Is node.js fast?")) == ["nodejs"]
    assert ids(link(linker, snap, "Is nodejs fast?")) == []


def test_sentence_punctuation_ends_a_name(linker, snap):
    assert ids(link(linker, snap, "I like Alpha. Beta is big")) == []
    assert ids(link(linker, snap, "I like Alpha, Beta is big")) == []
    assert ids(link(linker, snap, "I like Alpha Beta, and more")) == ["alphabeta"]
    assert ids(link(linker, snap, "Is it Alpha - Beta?")) == []


def test_a_stored_name_with_an_internal_period_and_space_is_a_known_limitation(linker, snap):
    assert ids(link(linker, snap, "I visited St. Louis")) == []


def test_punctuation_only_and_empty_queries_link_nothing(linker, snap):
    for query in ["", "   ", "?!", "- - -", "..."]:
        result = link(linker, snap, query)
        assert result.linked == () and result.seeds == ()


# ------------------------------------------------------------------------ longest match


def test_the_longest_match_wins_and_the_contained_match_is_reported(linker, snap):
    result = link(linker, snap, "Did Bob Smith join?")
    assert ids(result) == ["bobsmith"]
    assert [(s.span, s.contained_in) for s in result.suppressed] == [("Smith", "Bob Smith")]
    assert result.ambiguous == ()


def test_a_shorter_name_alone_still_matches(linker, snap):
    assert ids(link(linker, snap, "Did Smith join?")) == ["smith"]


def test_new_york_contains_york(linker, snap):
    result = link(linker, snap, "Flights to New York")
    assert ids(result) == ["newyork"]
    assert result.suppressed[0].span == "York"
    assert ids(link(linker, snap, "Flights to York")) == ["york"]


def test_partially_overlapping_matches_are_ambiguous_and_nothing_is_linked(linker, snap):
    result = link(linker, snap, "Is Alpha Beta Gamma real?")
    assert result.linked == () and result.seeds == ()
    [amb] = result.ambiguous
    assert amb.reason == "overlapping_spans" and amb.action == "skipped"
    assert sorted(amb.spans) == ["Alpha Beta", "Beta Gamma"]
    assert [c.entity_id for c in amb.candidates] == ["alphabeta", "betagamma"]


def test_an_overlap_can_be_resolved_with_an_explicit_entity_id(linker, snap):
    result = link(linker, snap, "Is Alpha Beta Gamma real?", entity_ids=["betagamma"])
    assert ids(result) == ["betagamma"] and result.seeds == ("betagamma",)
    [amb] = result.ambiguous
    assert amb.action == "resolved" and amb.resolved_to == ("betagamma",)
    assert result.linked[0].resolved_by == "entity_ids" and result.linked[0].span == "Beta Gamma"


# --------------------------------------------------------------------------- ambiguity


def test_one_name_for_two_entities_is_reported_with_every_candidate_and_never_chosen(linker, snap):
    result = link(linker, snap, "Tell me about Mercury")
    assert result.linked == () and result.seeds == ()
    [amb] = result.ambiguous
    assert amb.reason == "multiple_entities" and amb.action == "skipped" and amb.resolved_to == ()
    assert [(c.entity_id, c.entity_type, c.document_count) for c in amb.candidates] == [
        ("mercury_m", "PRODUCT", 3),
        ("mercury_p", "LOCATION", 1),
    ]
    assert amb.span == "Mercury"


def test_an_explicit_entity_id_resolves_the_ambiguity_to_that_entity_only(linker, snap):
    result = link(linker, snap, "Tell me about Mercury", entity_ids=["mercury_p"])
    assert ids(result) == ["mercury_p"] and result.seeds == ("mercury_p",)
    [amb] = result.ambiguous
    assert amb.action == "resolved" and amb.resolved_to == ("mercury_p",)
    assert result.linked[0].resolved_by == "entity_ids" and result.linked[0].span == "Mercury"


def test_pinning_both_candidates_links_both_and_nothing_else(linker, snap):
    result = link(linker, snap, "Mercury", entity_ids=["mercury_m", "mercury_p"])
    assert sorted(ids(result)) == ["mercury_m", "mercury_p"]
    assert result.ambiguous[0].resolved_to == ("mercury_m", "mercury_p")


def test_ambiguous_mentions_do_not_block_other_unambiguous_entities(linker, snap):
    result = link(linker, snap, "Did Alice visit Mercury?")
    assert ids(result) == ["alice"] and len(result.ambiguous) == 1


def test_an_explicit_id_that_the_query_does_not_mention_is_added_as_a_seed(linker, snap):
    result = link(linker, snap, "Who is Alice?", entity_ids=["berlin"])
    assert result.seeds == ("alice", "berlin")
    by_id = {e.entity_id: e for e in result.linked}
    assert by_id["berlin"].resolved_by == "entity_ids" and by_id["berlin"].span is None
    assert by_id["alice"].resolved_by == "query"


def test_automatic_linking_can_be_switched_off(linker, snap):
    result = link(linker, snap, "Who is Alice?", auto_link=False)
    assert result.linked == () and result.seeds == ()
    result = link(linker, snap, "Who is Alice?", entity_ids=["berlin"], auto_link=False)
    assert result.seeds == ("berlin",) and linker.builds == 0  # the index was never needed


def test_unknown_entity_ids_are_rejected_and_listed(linker, snap):
    with pytest.raises(UnknownEntities) as caught:
        link(linker, snap, "Alice", entity_ids=["zzz", "alice", "aaa"])
    assert caught.value.entity_ids == ["aaa", "zzz"]


def test_entities_are_never_merged_while_linking(linker, snap):
    """Two entities with one name stay two candidates; two names stay two entities."""
    result = link(linker, snap, "Mercury and Alice")
    assert ids(result) == ["alice"]
    assert {c.entity_id for c in result.ambiguous[0].candidates} == {"mercury_m", "mercury_p"}
    assert len(snap.graph) == len(NODES)  # linking never touches the graph


# ------------------------------------------------------------------- index and caching


def test_the_index_is_cached_per_graph_revision(linker, snap):
    link(linker, snap, "Alice")
    link(linker, snap, "Bob Smith")
    assert linker.builds == 1
    newer = make_snapshot(NODES, revision=2)
    link(linker, newer, "Alice")
    assert linker.builds == 2
    link(linker, newer, "Alice")
    assert linker.builds == 2


def test_a_changed_graph_is_seen_after_the_revision_changes(linker):
    first = make_snapshot([node("alice", "Alice", "PERSON")], revision=1)
    assert ids(link(linker, first, "Alice and Zed")) == ["alice"]
    second = make_snapshot(
        [node("alice", "Alice", "PERSON"), node("zed", "Zed", "PERSON")], revision=2
    )
    assert sorted(ids(link(linker, second, "Alice and Zed"))) == ["alice", "zed"]


def test_names_longer_than_the_configured_token_limit_are_not_indexed():
    short = EntityLinker(max_name_tokens=2)
    snapshot = make_snapshot(
        [node("long", "Alpha Beta Gamma Delta", "CONCEPT"), node("ok", "Alpha Beta", "CONCEPT")]
    )
    assert ids(short.link(snapshot, "Alpha Beta Gamma Delta")) == ["ok"]
    index = short.index_for(snapshot)
    assert index.skipped_too_long == 1 and index.max_key_tokens == 2


def test_the_index_holds_names_and_aliases_with_their_stored_spellings(snap):
    index = AliasIndex.build(snap.graph, snap.revision, 8)
    [entry] = index.by_key["bobby"]
    assert entry.entity_id == "bobsmith" and entry.match_type == "exact_alias"
    assert entry.forms == {"Bobby"}
    [acme] = index.by_key["acme"]
    assert acme.match_type == "exact_name" and acme.forms == {
        "Acme Inc",
        "Acme Corporation",
        "Acme",
    }


# ------------------------------------------------------------------------- determinism


def test_the_result_does_not_depend_on_node_insertion_order(linker):
    forward = make_snapshot(NODES)
    backward = make_snapshot(list(reversed(NODES)))
    query = "Did Alice, Bob Smith and Acme Inc visit Mercury or Alpha Beta Gamma?"
    a = EntityLinker().link(forward, query)
    b = EntityLinker().link(backward, query)
    assert a == b


def test_repeated_calls_give_identical_results(linker, snap):
    query = "Is Go's Node.js faster than Rust at Acme Corp?"
    assert link(linker, snap, query) == link(linker, snap, query)
    assert sorted(ids(link(linker, snap, query))) == ["acme", "go", "nodejs", "rust"]


# --------------------------------------------------------------------------- tokenising


def test_tokenisation_keeps_the_original_spelling_for_the_capitalisation_guard():
    tokens = tokenize_query("Is GO fast, Alice's friend?")
    assert [t.norm for t in tokens] == ["is", "go", "fast", "alice's", "friend"]
    assert [t.raw for t in tokens][1] == "GO"
    assert tokens[3].alt == "alice" and tokens[3].alt_raw == "Alice"
    assert tokens[2].boundary_after and not tokens[1].boundary_after
