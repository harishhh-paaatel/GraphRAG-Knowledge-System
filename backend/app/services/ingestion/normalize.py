"""Text normalization applied to all extracted text before chunking."""

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HSPACE = re.compile(r"[^\S\n]+")
_BLANKS = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Normalize newlines, drop control chars, collapse spaces, keep paragraph breaks."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = _CONTROL.sub("", text)
    text = _HSPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _BLANKS.sub("\n\n", text)
    return text.strip()
