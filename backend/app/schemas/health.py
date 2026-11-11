"""Health response schemas."""

from typing import Literal

from pydantic import BaseModel


class ComponentStatus(BaseModel):
    """Status of one dependency."""

    status: Literal["ok", "error", "not_configured"]
    detail: str | None = None


class HealthResponse(BaseModel):
    """Overall service health."""

    status: Literal["ok", "degraded"]
    version: str
    environment: str
    components: dict[str, ComponentStatus]
