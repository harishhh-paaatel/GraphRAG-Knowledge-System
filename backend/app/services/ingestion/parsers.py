"""File parsers producing normalized text segments with page/section metadata."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.services.ingestion.errors import ParseError
from app.services.ingestion.normalize import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A run of text that shares one page number and section (chunks never span segments)."""

    text: str
    page_number: int | None = None
    section: str | None = None


@dataclass
class ParsedDocument:
    """Parser output: ordered segments plus document-level metadata."""

    segments: list[Segment]
    metadata: dict = field(default_factory=dict)


def parse_file(path: Path, file_type: str) -> ParsedDocument:
    """Parse ``path`` according to ``file_type``; raises ParseError, never returns empty."""
    parsers = {"pdf": _parse_pdf, "docx": _parse_docx, "txt": _parse_text, "md": _parse_markdown}
    try:
        parsed = parsers[file_type](path)
    except ParseError:
        raise
    except Exception as exc:  # third-party parsers raise many types on corrupt input
        logger.warning("Parser failed for %s file: %s", file_type, type(exc).__name__)
        raise ParseError(f"The {file_type.upper()} file is corrupted or could not be read") from exc

    parsed.segments = [s for s in parsed.segments if s.text]
    if not parsed.segments:
        raise ParseError("No extractable text found in the document")
    return parsed


def _parse_pdf(path: Path) -> ParsedDocument:
    import pymupdf  # imported lazily to keep app startup light

    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ParseError("The PDF is password-protected")
        segments = [
            Segment(text=normalize_text(page.get_text("text")), page_number=i)
            for i, page in enumerate(pdf, start=1)
        ]
        meta = pdf.metadata or {}
        blank_pages = sum(1 for s in segments if not s.text)
        metadata = {
            "page_count": pdf.page_count,
            "blank_pages": blank_pages,
            "title": meta.get("title") or None,
            "author": meta.get("author") or None,
        }
    if blank_pages == len(segments):
        raise ParseError("No extractable text found (scanned PDFs need OCR, which is unsupported)")
    return ParsedDocument(segments, metadata)


def _parse_docx(path: Path) -> ParsedDocument:
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    doc = Document(str(path))
    segments: list[Segment] = []
    section: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        segments.append(Segment(normalize_text("\n\n".join(buffer)), None, section))
        buffer.clear()

    paragraphs = 0
    for block in doc.iter_inner_content():
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                continue
            paragraphs += 1
            style = (block.style.name if block.style is not None else "") or ""
            if style == "Title" or style.startswith("Heading"):
                flush()
                section = text[:255]
            buffer.append(text)
        elif isinstance(block, Table):
            rows = [" | ".join(c.text.strip() for c in row.cells) for row in block.rows]
            buffer.append("\n".join(r for r in rows if r.strip(" |")))
    flush()

    props = doc.core_properties
    metadata = {
        "paragraph_count": paragraphs,
        "table_count": len(doc.tables),
        "title": props.title or None,
        "author": props.author or None,
    }
    return ParsedDocument(segments, metadata)


def _read_text(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParseError("Text file is not valid UTF-8") from exc


def _parse_text(path: Path) -> ParsedDocument:
    text = normalize_text(_read_text(path))
    return ParsedDocument([Segment(text)], {"char_count": len(text)})


_HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^ {0,3}(```|~~~)")


def _parse_markdown(path: Path) -> ParsedDocument:
    """Split on ATX headings (ignoring fenced code); each section becomes a segment."""
    raw = _read_text(path).replace("\r\n", "\n").replace("\r", "\n")
    segments: list[Segment] = []
    section: str | None = None
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        segments.append(Segment(normalize_text("\n".join(buffer)), None, section))
        buffer.clear()

    for line in raw.split("\n"):
        if _FENCE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line)
        if heading:
            flush()
            section = heading.group(2)[:255]
        buffer.append(line)
    flush()
    return ParsedDocument(segments, {"heading_count": sum(1 for s in segments if s.section)})
