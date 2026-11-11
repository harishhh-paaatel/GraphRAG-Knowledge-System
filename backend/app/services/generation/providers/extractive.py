"""Extractive provider — the default, model-free, CPU-only answer generator.

Determinism contract: given identical ``GenerationRequest`` input, the output is always the
same. This makes it the control group for Phase 8 faithfulness metrics.

Algorithm (documented; not claimed to be optimal):
1. The user message is split into sentences and each is scored by the fraction of the
   question's content words that appear in it (case-insensitive, after simple stop-word
   removal).  Content words are defined as alphabetic tokens longer than two characters
   that are not in the stop-word list; a minimum of one token is always kept.
2. All sentences of all sources are ranked by their score (ties broken by source rank, then
   sentence order within the source).
3. The top-scored sentences are taken in rank order until the answer budget (characters,
   not tokens) is reached; the sentence that would overflow is skipped, and smaller later
   ones may still fit — the same greedy-with-skip policy as the context builder.
4. Each selected sentence is emitted with its ``[S#]`` marker; the final text is the
   concatenation, stripped of leading/trailing whitespace.
5. If no sentence scores above zero (the question has no content words in the passages, or
   all sources are empty) the special sentinel ``INSUFFICIENT_EVIDENCE`` is returned, which
   the parsing step will convert to ``status: insufficient_evidence``.

The sentences are verbatim copies of source text, so the grounding checks always pass:
every cited content word is literally present in the cited source, and all capitalised
terms are present by construction.  This is by design; it is also why the mode is
``"extractive"`` and never ``"llm"``.

What this provider is NOT:
- It does not understand language or intent.
- The lexical overlap score is not a semantic similarity measure.
- It cannot synthesise information across sentences.
- Sentence splitting is English-centric and heuristic (documented limitation).
"""

from __future__ import annotations

import logging
import re

from app.core.config import Settings
from app.services.generation.providers.base import LLMProvider
from app.services.generation.types import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
    ProviderInfo,
)

logger = logging.getLogger(__name__)

_MODEL_NAME = "extractive-v1"

# Simple English stop words for content-word extraction.  Deliberately short: false
# positives (keeping a stop word) reduce recall modestly; false negatives (dropping a
# content word) reduce precision.  Accuracy beyond this list is not worth a dependency.
_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "not",
        "that",
        "this",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "which",
        "who",
        "what",
        "when",
        "where",
        "how",
        "why",
        "can",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "than",
        "then",
        "so",
        "just",
        "also",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "both",
        "each",
        "some",
        "any",
        "all",
        "no",
        "nor",
        "up",
        "out",
        "off",
        "own",
        "same",
        "too",
        "very",
        "s",
        "t",
        "re",
        "ll",
        "ve",
        "d",
        "m",
    }
)

# Sentence boundary: end of sentence punctuation followed by space/newline and an
# uppercase letter, or the end of the string.  Handles common abbreviations (Mr., Dr.,
# etc.) heuristically by not splitting after a single uppercase letter + period.
_SENTENCE_RE = re.compile(
    r"(?<!\b[A-Z])(?<!\b[A-Z][a-z])(?<=[.!?])\s+(?=[A-Z\"'\(])|(?<=[.!?])\s*$"
)

# Default answer character budget for the extractive provider (generous; the context
# builder already respected a token budget when selecting sources).
_DEFAULT_ANSWER_CHARS = 1200


def _split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences. Returns the original as a single item when
    the text is too short to contain a sentence boundary."""
    if not text.strip():
        return []
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _content_words(text: str) -> frozenset[str]:
    """Lower-case alphabetic tokens longer than two characters, not in the stop list."""
    tokens = frozenset(
        w.lower() for w in re.findall(r"[A-Za-z]{3,}", text) if w.lower() not in _STOP_WORDS
    )
    if not tokens:
        # If the question has only stop words / short tokens, keep ALL tokens so we
        # never produce a completely empty query vocabulary.
        tokens = frozenset(w.lower() for w in re.findall(r"[A-Za-z]+", text))
    return tokens


def _score(sentence: str, query_words: frozenset[str]) -> float:
    """Fraction of ``query_words`` that appear in ``sentence`` (0 when query is empty)."""
    if not query_words:
        return 0.0
    sentence_lower = sentence.lower()
    matched = sum(1 for w in query_words if re.search(r"\b" + re.escape(w) + r"\b", sentence_lower))
    return matched / len(query_words)


def _extract_answer(
    user_message: str,
    answer_chars: int,
) -> str:
    """Core algorithm: parse source blocks from the prompt, score and select sentences.

    The user message follows the Phase 7 prompt template: source blocks are delimited by
    ``[Source S# — …]`` … ``[End S#]`` markers.  We extract each block's text and marker,
    score its sentences, and select greedily.
    """
    # Extract the question from the last line(s) of the user message.
    # The question follows the last occurrence of a blank line before the final non-empty line,
    # or we fall back to the entire user message.
    # This is intentionally loose — the exact format is defined in M2.
    question_match = re.search(r"Question:\s*(.+)$", user_message, re.MULTILINE)
    question_text = question_match.group(1).strip() if question_match else user_message
    query_words = _content_words(question_text)

    # Find source blocks.  Format (from M2 context builder, anticipated):
    #   [Source S1 — filename.txt, section "Team", page 2]
    #   ... text ...
    #   [End S1]
    source_block_re = re.compile(
        r"\[Source (S\d+)[^\]]*\]\s*\n(.*?)\n?\[End \1\]",
        re.DOTALL,
    )
    blocks = source_block_re.findall(user_message)

    if not blocks:
        # Fallback: if the prompt does not contain source blocks yet (e.g. in tests that
        # pass plain text), treat the whole user message as a single unlabelled source.
        # This keeps the provider usable during development before M2 is complete.
        sentences = _split_sentences(user_message)
        candidates = [("S1", _score(s, query_words), 0, i, s) for i, s in enumerate(sentences)]
    else:
        candidates = []
        for src_rank, (marker, text) in enumerate(blocks):
            for sent_idx, sentence in enumerate(_split_sentences(text)):
                score = _score(sentence, query_words)
                candidates.append((marker, score, src_rank, sent_idx, sentence))

    # Sort by score descending, then source rank ascending, then sentence position ascending.
    candidates.sort(key=lambda t: (-t[1], t[2], t[3]))

    if not candidates or candidates[0][1] == 0.0:
        return "INSUFFICIENT_EVIDENCE"

    # Greedy budget-aware selection.
    selected: list[tuple[str, str]] = []  # (marker, sentence)
    used_chars = 0
    for marker, score, _src_rank, _sent_idx, sentence in candidates:
        if score == 0.0:
            break
        needed = len(sentence) + (len(marker) + 4)  # " [Sx]" appended
        if used_chars + needed > answer_chars:
            continue  # skip; keep trying smaller ones
        selected.append((marker, sentence))
        used_chars += needed

    if not selected:
        return "INSUFFICIENT_EVIDENCE"

    # Emit in marker-order (not score-order) so the answer reads naturally.
    # We sort by the first digit of the marker number to preserve source order.
    def marker_key(item: tuple[str, str]) -> int:
        m = re.match(r"S(\d+)", item[0])
        return int(m.group(1)) if m else 0

    selected.sort(key=marker_key)

    parts = [f"{sentence} [{marker}]" for marker, sentence in selected]
    return "  ".join(parts)


class ExtractiveProvider:
    """Model-free provider. Selects and cites verbatim sentences from the prompt sources.

    ``generation_mode`` is always ``"extractive"``. Answers always pass the grounding
    checks because the selected text is literally copied from the cited sources.
    """

    name = "extractive"
    mode = GenerationMode.EXTRACTIVE
    model = _MODEL_NAME

    def __init__(self, answer_chars: int = _DEFAULT_ANSWER_CHARS) -> None:
        self._answer_chars = answer_chars

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # Find the user message (last message, guaranteed by GenerationRequest validation).
        user_message = request.messages[-1].content
        answer_text = _extract_answer(user_message, self._answer_chars)
        return GenerationResult(
            text=answer_text,
            model=self.model,
            finish_reason="stop",
            # No real usage to report; we record the estimated prompt size from the
            # context builder.  The extractive provider is CPU-only, no model is loaded.
            usage=None,
        )

    def info(self) -> ProviderInfo:
        return ProviderInfo(name=self.name, mode=self.mode, model=self.model)

    def close(self) -> None:
        pass  # no resources to release


def build(settings: Settings) -> LLMProvider:
    """Factory: create an ``ExtractiveProvider`` from settings."""
    # The answer character budget is derived from the token budget using the same
    # chars-per-token estimate the context builder uses.  This keeps the extractive
    # provider's output consistent with the configured budget.
    answer_chars = int(settings.llm_max_answer_tokens * settings.answer_chars_per_token)
    return ExtractiveProvider(answer_chars=max(answer_chars, 200))
