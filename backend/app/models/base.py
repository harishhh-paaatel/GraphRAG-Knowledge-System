"""Declarative base and shared helpers."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator


def new_id() -> str:
    """Return a new random UUID string."""
    return uuid.uuid4().hex


class UTCDateTime(TypeDecorator):
    """Store datetimes as UTC; SQLite drops tzinfo, so restore it on read."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):  # noqa: ANN001
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(UTC).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):  # noqa: ANN001
        return value.replace(tzinfo=UTC) if value is not None else None


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class for all ORM models."""
