"""Deterministic rule-based extractor.

This exists so the whole pipeline is testable and demoable without any model. It is
deliberately precise-but-low-recall and says so: it only knows (a) the entities in its
gazetteer, (b) capitalised phrases it can classify from explicit cues (organisation keywords,
a small first-name list, titles, event keywords), and (c) relationships expressed by a fixed set
of verb phrases that sit *directly* between two entities. Real extraction quality is the job
of the LLM provider (Phase 7); do not read the rule-based output as a quality baseline.

No confidence is reported: pattern matches are not probabilities.
"""

import re
from dataclasses import dataclass

from app.services.extraction.normalization import clean_display_name
from app.services.extraction.types import (
    RELATIONSHIP_RULES,
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
)

DEFAULT_GAZETTEER: dict[str, str] = {
    # canonical name -> entity type
    **{
        n: "TECHNOLOGY"
        for n in (
            "Python",
            "FastAPI",
            "Django",
            "Flask",
            "SQLAlchemy",
            "SQLite",
            "PostgreSQL",
            "Pydantic",
            "NetworkX",
            "FAISS",
            "PyTorch",
            "TensorFlow",
            "React",
            "TypeScript",
            "JavaScript",
            "Docker",
            "Kubernetes",
            "Ollama",
            "Alembic",
            "Uvicorn",
            "Linux",
            "Neo4j",
            "Redis",
            "NumPy",
            "pytest",
            "Starlette",
            "Node.js",
        )
    },
    **{n: "PRODUCT" for n in ("ChatGPT", "GitHub Copilot")},
    **{
        n: "ORGANIZATION"
        for n in (
            "OpenAI",
            "Anthropic",
            "Google",
            "Microsoft",
            "Meta",
            "Amazon",
            "Apple",
            "Hugging Face",
            "GitHub",
            "Mozilla",
        )
    },
    **{
        n: "LOCATION"
        for n in (
            "Paris",
            "London",
            "Berlin",
            "Tokyo",
            "India",
            "Germany",
            "France",
            "Europe",
            "California",
            "San Francisco",
            "New York",
            "United States",
            "Mumbai",
        )
    },
    **{
        n: "CONCEPT"
        for n in (
            "machine learning",
            "knowledge graph",
            "retrieval-augmented generation",
            "vector search",
            "graph database",
            "semantic search",
            "deep learning",
            "natural language processing",
        )
    },
}

FIRST_NAMES = frozenset(
    """Alice Bob Carol Dave David Eve Frank Grace Heidi Ivan Judy Michael Sarah John Jane
    Mary James Robert Linda Susan Thomas Charles Daniel Matthew Anna Emma Olivia Sophia Liam
    Noah Lucas Sebastian Guido Linus Tim Elon Satya Sundar Sam Dario Ada Alan Marie Albert
    Priya Rahul Amit Wei Li Chen Yuki Hiroshi Maria Carlos Jose Ahmed Fatima Omar Elena""".split()
)
_STOP_INITIAL = frozenset(
    """The A An This That These Those It Its In On At For With By From To And But Or However
    Also He She They We You I His Her Their Our After Before While When If As Some Many Most
    Each Every One Two There Here What Which Who How Why Tests Today Yesterday Meanwhile""".split()
)
_ORG_KEYWORDS = frozenset(
    """Inc Corp Corporation Ltd LLC GmbH University Institute Foundation Labs Laboratories
    Technologies Systems Company Group Association Society College Academy""".split()
)
_EVENT_KEYWORDS = frozenset({"Conference", "Summit", "Hackathon", "Festival", "Symposium"})
_TITLE_BEFORE = re.compile(r"(?:Dr|Prof|Mr|Ms|Mrs)\.?\s+$")

_TOKEN = r"[A-Z][A-Za-z0-9'’\-]*"
_CAP_SEQ = re.compile(rf"{_TOKEN}(?:(?:\s+(?:of|de|van|von)\s+|\s+){_TOKEN})*")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])|\n+")

_ADV = r"(?:(?:also|originally|primarily|mainly|currently|now|still)\s+)?"
# (pattern that must fully match the text between two entities, relationship, reversed?)
_TRIGGERS: list[tuple[re.Pattern[str], str, bool]] = [
    (re.compile(p, re.I), rel, rev)
    for p, rel, rev in [
        (
            rf"(?:was|is|were|are)\s+{_ADV}(?:developed|created|built|designed|released|maintained)\s+by",
            "DEVELOPED",
            True,
        ),
        (rf"(?:was|is|were|are)\s+{_ADV}(?:written|authored)\s+by", "AUTHORED", True),
        (rf"(?:was|is|were|are)\s+{_ADV}used\s+by", "USES", True),
        (rf"(?:is|are)\s+{_ADV}(?:required|needed)\s+by", "DEPENDS_ON", True),
        (
            rf"{_ADV}(?:works?|worked)\s+(?:at|for)|is\s+employed\s+(?:at|by)|joined",
            "WORKS_AT",
            False,
        ),
        (
            rf"{_ADV}(?:developed|created|built|designed|released|maintains|maintained|launched|introduced)",
            "DEVELOPED",
            False,
        ),
        (
            rf"{_ADV}(?:uses?|used|utili[sz]es|leverages|relies\s+on|is\s+built\s+(?:on|with|using)"
            r"|is\s+written\s+in|is\s+implemented\s+in|is\s+powered\s+by)",
            "USES",
            False,
        ),
        (r"(?:depends?|depended)\s+on|requires?|is\s+dependent\s+on", "DEPENDS_ON", False),
        (
            r"(?:is|are|was|were)\s+(?:located|based|headquartered|situated)\s+in",
            "LOCATED_IN",
            False,
        ),
        (
            r"(?:is|are)\s+(?:an?\s+)?(?:part|component|module|member|subset)\s+of|belongs?\s+to|is\s+included\s+in",
            "PART_OF",
            False,
        ),
        (r"wrote|authored|co-authored|is\s+the\s+author\s+of", "AUTHORED", False),
        (
            r"(?:is|are)\s+(?:closely\s+)?(?:related|connected|linked|similar)\s+to|(?:is|are)\s+associated\s+with",
            "RELATED_TO",
            False,
        ),
    ]
]


@dataclass
class _Mention:
    start: int
    end: int
    surface: str
    name: str
    entity_type: str | None


class RuleBasedExtractor:
    """Gazetteer + cue + verb-phrase extractor. Deterministic: same text, same output."""

    name = "rule_based"

    def __init__(
        self,
        entity_types: set[str] | frozenset[str] | list[str],
        gazetteer: dict[str, str] | None = None,
    ) -> None:
        self._types = set(entity_types)
        entries = DEFAULT_GAZETTEER if gazetteer is None else gazetteer
        self._gaz = {k.casefold(): (k, t) for k, t in entries.items() if t in self._types}
        terms = sorted(self._gaz, key=len, reverse=True)
        self._gaz_re = (
            re.compile(r"(?<![\w])(?:" + "|".join(re.escape(t) for t in terms) + r")(?![\w])", re.I)
            if terms
            else None
        )

    def extract(self, text: str) -> ChunkExtraction:
        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelationship] = []
        pos = 0
        spans = []
        for m in _SENTENCE_SPLIT.finditer(text):
            spans.append((pos, m.start()))
            pos = m.end()
        spans.append((pos, len(text)))

        for start, end in spans:
            if end <= start:
                continue
            mentions = self._mentions(text, start, end)
            for i in range(len(mentions) - 1):
                first, second = mentions[i], mentions[i + 1]
                found = self._relation(text, first, second)
                if found is None:
                    continue
                rel, reversed_, src_type, tgt_type = found
                src, tgt = (second, first) if reversed_ else (first, second)
                src.entity_type, tgt.entity_type = src_type, tgt_type
                relationships.append(
                    ExtractedRelationship(
                        source=src.name,
                        target=tgt.name,
                        relationship_type=rel,
                        supporting_text=text[first.start : second.end],
                    )
                )
            # Untyped mentions that no relation could type are dropped: capitalised words
            # alone are not evidence of an entity.
            entities.extend(
                ExtractedEntity(m.name, m.entity_type, surface_form=m.surface)
                for m in mentions
                if m.entity_type is not None
            )
        return ChunkExtraction(entities, relationships)

    # -- mention detection ---------------------------------------------------------------

    def _mentions(self, text: str, lo: int, hi: int) -> list[_Mention]:
        found: list[_Mention] = []
        if self._gaz_re is not None:
            for m in self._gaz_re.finditer(text, lo, hi):
                canonical, etype = self._gaz[m.group(0).casefold()]
                found.append(_Mention(m.start(), m.end(), m.group(0), canonical, etype))
        for m in _CAP_SEQ.finditer(text, lo, hi):
            mention = self._classify(text, m.start(), m.end())
            if mention is not None:
                found.append(mention)
        # Typed beats untyped, then longer beats shorter; keep a non-overlapping set.
        ranked = sorted(found, key=lambda x: (x.entity_type is None, -(x.end - x.start), x.start))
        chosen: list[_Mention] = []
        for cand in ranked:
            if all(cand.end <= c.start or cand.start >= c.end for c in chosen):
                chosen.append(cand)
        chosen.sort(key=lambda x: x.start)
        return chosen

    def _classify(self, text: str, start: int, end: int) -> _Mention | None:
        words = list(re.finditer(r"\S+", text[start:end]))
        skip = 0
        while skip < len(words) and words[skip].group(0) in _STOP_INITIAL:
            skip += 1
        if skip == len(words):
            return None
        real_start = start + words[skip].start()
        tokens = [w.group(0) for w in words[skip:]]
        surface = clean_display_name(text[real_start:end])
        if not surface:
            return None
        etype: str | None = None
        if any(t in _ORG_KEYWORDS for t in tokens) and "ORGANIZATION" in self._types:
            etype = "ORGANIZATION"
        elif tokens[-1] in _EVENT_KEYWORDS and "EVENT" in self._types:
            etype = "EVENT"
        elif (
            "PERSON" in self._types
            and 2 <= len(tokens) <= 3
            and (
                tokens[0] in FIRST_NAMES
                or _TITLE_BEFORE.search(text[max(0, real_start - 6) : real_start])
            )
        ):
            etype = "PERSON"
        return _Mention(real_start, end, surface, surface, etype)

    # -- relationship detection --------------------------------------------------------

    def _relation(self, text: str, a: _Mention, b: _Mention) -> tuple[str, bool, str, str] | None:
        between = text[a.end : b.start].strip()
        if not between:
            return None
        for pattern, rel, reverse in _TRIGGERS:
            if not pattern.fullmatch(between):
                continue
            src, tgt = (b, a) if reverse else (a, b)
            types = self._assign_types(rel, src, tgt)
            return None if types is None else (rel, reverse, types[0], types[1])
        return None

    def _assign_types(self, rel: str, src: _Mention, tgt: _Mention) -> tuple[str, str] | None:
        """Use known types; infer an unknown one only when the relation allows exactly one."""
        allowed_src, allowed_tgt = RELATIONSHIP_RULES[rel]
        resolved = []
        for mention, allowed in ((src, allowed_src), (tgt, allowed_tgt)):
            if mention.entity_type is not None:
                if allowed is not None and mention.entity_type not in allowed:
                    return None
                resolved.append(mention.entity_type)
            elif allowed is not None and len(allowed) == 1:
                inferred = next(iter(allowed))
                if inferred not in self._types:
                    return None
                resolved.append(inferred)
            else:
                return None
        return resolved[0], resolved[1]
