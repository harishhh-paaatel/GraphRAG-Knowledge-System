"""Upload validation: filename sanitization, type allow-list, MIME and content sniffing."""

import re
import unicodedata
import zipfile
from pathlib import Path

from app.core.errors import AppError

# extension -> acceptable declared Content-Types (generic octet-stream is always tolerated,
# because browsers and curl often send it; content sniffing is the real check).
ALLOWED_TYPES: dict[str, frozenset[str]] = {
    "pdf": frozenset({"application/pdf"}),
    "docx": frozenset({"application/vnd.openxmlformats-officedocument.wordprocessingml.document"}),
    "txt": frozenset({"text/plain"}),
    "md": frozenset({"text/markdown", "text/x-markdown", "text/plain"}),
}
_GENERIC_MIME = "application/octet-stream"
_MAX_FILENAME = 255
_MAX_UNCOMPRESSED_DOCX = 200 * 1024 * 1024  # zip-bomb guard


def sanitize_filename(raw: str | None) -> str:
    """Return a safe display filename: no path parts, control chars or odd symbols."""
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1]
    name = unicodedata.normalize("NFKC", name)
    name = "".join(ch for ch in name if unicodedata.category(ch)[0] != "C")
    name = re.sub(r"[^\w .-]", "_", name).strip(" .")
    name = re.sub(r"\s+", " ", name)
    if not name:
        return "document"
    if len(name) > _MAX_FILENAME:
        stem, dot, ext = name.rpartition(".")
        keep = _MAX_FILENAME - len(ext) - 1
        name = f"{stem[:keep]}.{ext}" if dot else name[:_MAX_FILENAME]
    return name


def detect_file_type(filename: str) -> str:
    """Return the allow-listed extension (without dot) or raise a 415."""
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_TYPES:
        raise AppError(
            415,
            "unsupported_file_type",
            f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_TYPES))}",
        )
    return ext


def validate_declared_mime(file_type: str, content_type: str | None) -> None:
    """Reject a declared Content-Type that contradicts the extension."""
    if not content_type:
        return
    mime = content_type.split(";", 1)[0].strip().lower()
    if mime and mime != _GENERIC_MIME and mime not in ALLOWED_TYPES[file_type]:
        raise AppError(
            415,
            "mime_mismatch",
            f"Content-Type '{mime}' does not match a .{file_type} file",
        )


def sniff_content(file_type: str, path: Path) -> None:
    """Check the file's leading bytes/structure match its claimed type (400 on mismatch)."""
    size = path.stat().st_size
    if size == 0:
        raise AppError(400, "empty_file", "The uploaded file is empty")
    with path.open("rb") as fh:
        head = fh.read(8192)

    if file_type == "pdf":
        if b"%PDF-" not in head[:1024]:
            raise AppError(400, "invalid_file_content", "File is not a valid PDF")
    elif file_type == "docx":
        _sniff_docx(path)
    else:  # txt / md
        if b"\x00" in head:
            raise AppError(400, "invalid_file_content", "File appears to be binary, not text")


def _sniff_docx(path: Path) -> None:
    bad = AppError(400, "invalid_file_content", "File is not a valid DOCX document")
    if not zipfile.is_zipfile(path):
        raise bad
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            total = sum(i.file_size for i in zf.infolist())
    except (zipfile.BadZipFile, OSError):
        raise bad from None
    if "word/document.xml" not in names:
        raise bad
    if total > _MAX_UNCOMPRESSED_DOCX:
        raise AppError(400, "invalid_file_content", "DOCX expands to an unsafe size")
