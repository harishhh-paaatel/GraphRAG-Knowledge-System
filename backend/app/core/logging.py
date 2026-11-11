"""Logging setup with secret redaction."""

import logging
import re

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|token|password|secret)(['\"]?\s*[:=]\s*['\"]?)([^\s'\",}]+)"
)


class RedactingFilter(logging.Filter):
    """Mask values that follow secret-looking keys in log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET_PATTERN.sub(r"\1\2***", record.getMessage())
        record.args = None
        return True


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once; safe to call repeatedly."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(getattr(h, "_graphrag", False) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler._graphrag = True  # type: ignore[attr-defined]
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.addFilter(RedactingFilter())
    root.addHandler(handler)
