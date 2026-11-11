"""Entity-name normalization used for exact matching.

Normalization only removes differences that carry no meaning (case, spacing, punctuation,
leading "the", corporate suffixes on organizations). It never rewrites words.
"""

import re
import unicodedata

_CORP_SUFFIXES = frozenset(
    {"inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc", "co", "gmbh", "plc"}
)
_DISALLOWED = re.compile(r"[^\w\s+#.&'/-]")
_SPACE = re.compile(r"\s+")
_COMPACT_DROP = re.compile(r"[\s'.\-_/]")


def clean_display_name(name: str) -> str:
    """Whitespace-collapsed, control-char-free name suitable for display."""
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C" or ch in "\t\n ")
    return _SPACE.sub(" ", name).strip()


def normalize_key(name: str, entity_type: str | None = None) -> str:
    """Case/space/punctuation-insensitive key; equal keys + equal type => same entity."""
    s = clean_display_name(name).casefold().replace("’", "'")
    s = _DISALLOWED.sub(" ", s)
    tokens = [t.strip(".,;:'-/") for t in s.split()]
    tokens = [t for t in tokens if t]
    if tokens and tokens[0] == "the" and len(tokens) > 1:
        tokens = tokens[1:]
    if entity_type == "ORGANIZATION":
        while len(tokens) > 1 and tokens[-1] in _CORP_SUFFIXES:
            tokens.pop()
    return " ".join(tokens)


def compact_key(key: str) -> str:
    """Key with spaces and joiners removed: "open ai" and "openai" collide here."""
    return _COMPACT_DROP.sub("", key)


def squash(text: str) -> str:
    """Whitespace-collapsed, case-folded text for evidence containment checks."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip().casefold()
