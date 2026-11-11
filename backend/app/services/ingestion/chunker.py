"""Deterministic text chunking on paragraph/sentence boundaries with character overlap."""

import hashlib
import re
from dataclasses import dataclass

from app.services.ingestion.parsers import Segment

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class ChunkData:
    """A chunk ready for persistence. ``char_start/end`` index the normalized segment text."""

    id: str
    chunk_index: int
    content: str
    page_number: int | None
    section: str | None
    char_start: int
    char_end: int


def make_chunk_id(checksum: str, chunk_index: int) -> str:
    """Stable ID: identical file content always yields identical chunk IDs."""
    return hashlib.sha1(f"{checksum}:{chunk_index}".encode()).hexdigest()


def _spans(text: str, pattern: re.Pattern[str], lo: int, hi: int) -> list[tuple[int, int]]:
    """Split text[lo:hi] on ``pattern``; return non-empty, whitespace-trimmed spans."""
    out: list[tuple[int, int]] = []
    pos = lo
    for m in pattern.finditer(text, lo, hi):
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, hi))
    trimmed = []
    for s, e in out:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            trimmed.append((s, e))
    return trimmed


def _atomic_spans(text: str, size: int) -> list[tuple[int, int]]:
    """Break text into spans no longer than ``size``, preferring paragraph then sentence."""
    atoms: list[tuple[int, int]] = []
    for ps, pe in _spans(text, _PARAGRAPH, 0, len(text)):
        if pe - ps <= size:
            atoms.append((ps, pe))
            continue
        for ss, se in _spans(text, _SENTENCE, ps, pe):
            atoms.extend([(ss, se)] if se - ss <= size else _pack_words(text, ss, se, size))
    return atoms


def _pack_words(text: str, lo: int, hi: int, size: int) -> list[tuple[int, int]]:
    """Fallback for over-long sentences: pack whitespace-separated words, hard-cut giants."""
    pieces: list[tuple[int, int]] = []
    start: int | None = None
    end = lo
    for ws, we in _spans(text, _WHITESPACE, lo, hi):
        while we - ws > size:  # single token larger than a chunk
            if start is not None:
                pieces.append((start, end))
                start = None
            pieces.append((ws, ws + size))
            ws += size
        if start is None:
            start, end = ws, we
        elif we - start <= size:
            end = we
        else:
            pieces.append((start, end))
            start, end = ws, we
    if start is not None:
        pieces.append((start, end))
    return pieces


def _chunk_segment(text: str, size: int, overlap: int) -> list[tuple[int, int]]:
    """Greedy-pack atomic spans into (start, end) windows; overlap reuses trailing spans."""
    atoms = _atomic_spans(text, size)
    windows: list[tuple[int, int]] = []
    i = 0
    while i < len(atoms):
        j = i + 1
        while j < len(atoms) and atoms[j][1] - atoms[i][0] <= size:
            j += 1
        windows.append((atoms[i][0], atoms[j - 1][1]))
        if j >= len(atoms):
            break
        k = j  # next window starts at the earliest span within `overlap` of this window's end
        while k > i + 1 and atoms[j - 1][1] - atoms[k - 1][0] <= overlap:
            k -= 1
        i = k
    return windows


def chunk_segments(
    segments: list[Segment], checksum: str, size: int, overlap: int
) -> list[ChunkData]:
    """Chunk all segments in order, dropping empty chunks and exact duplicates."""
    chunks: list[ChunkData] = []
    seen: set[str] = set()
    for seg in segments:
        for start, end in _chunk_segment(seg.text, size, overlap):
            content = seg.text[start:end]
            fingerprint = _WHITESPACE.sub(" ", content).casefold()
            if not content.strip() or fingerprint in seen:
                continue
            seen.add(fingerprint)
            index = len(chunks)
            chunks.append(
                ChunkData(
                    id=make_chunk_id(checksum, index),
                    chunk_index=index,
                    content=content,
                    page_number=seg.page_number,
                    section=seg.section,
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks
