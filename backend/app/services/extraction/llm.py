"""LLM-backed extraction behind a replaceable text-completion client.

The client is injected; this module never talks to a network or holds credentials. Phase 7
supplies real clients (Ollama / OpenAI-compatible) that satisfy ``TextCompletionClient``.

Model-reported confidence is intentionally ignored: self-reported LLM scores are not
calibrated probabilities, and the requirements forbid presenting uncalibrated confidence.
"""

import json
from typing import Protocol

from pydantic import BaseModel, ValidationError

from app.services.extraction.types import (
    RELATIONSHIP_TYPES,
    ChunkExtraction,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionError,
)


class TextCompletionClient(Protocol):
    """Minimal interface an LLM backend must provide."""

    def complete(self, prompt: str) -> str:
        """Return the model's raw text response; raise on transport errors."""
        ...


class _LLMEntity(BaseModel):
    name: str
    type: str
    surface_form: str | None = None
    description: str | None = None


class _LLMRelationship(BaseModel):
    source: str
    target: str
    type: str
    supporting_text: str


class _LLMOutput(BaseModel):
    entities: list[_LLMEntity] = []
    relationships: list[_LLMRelationship] = []


PROMPT_TEMPLATE = """You extract a knowledge graph from the text below.
Return ONLY a JSON object: {{"entities": [...], "relationships": [...]}}.

Entity: {{"name": canonical name, "type": one of {entity_types}, "surface_form": exact text as
written, "description": short description or null}}.
Relationship: {{"source": entity name, "target": entity name, "type": one of {relationship_types},
"supporting_text": an exact quote from the text that mentions BOTH entities}}.

Rules: only include what the text explicitly states; never infer or invent relationships;
supporting_text must be copied verbatim; omit anything you are unsure about.

TEXT:
\"\"\"
{text}
\"\"\"
"""


class LLMExtractionProvider:
    """Prompts a completion client and parses its JSON; all output is validated downstream."""

    name = "llm"

    def __init__(self, client: TextCompletionClient, entity_types: list[str]) -> None:
        self._client = client
        self._entity_types = list(entity_types)

    def build_prompt(self, text: str) -> str:
        return PROMPT_TEMPLATE.format(
            entity_types=", ".join(self._entity_types),
            relationship_types=", ".join(RELATIONSHIP_TYPES),
            text=text,
        )

    def extract(self, text: str) -> ChunkExtraction:
        try:
            raw = self._client.complete(self.build_prompt(text))
        except Exception as exc:  # transport/provider errors differ per backend
            raise ExtractionError(f"LLM request failed: {type(exc).__name__}") from exc
        payload = _first_json_object(raw)
        try:
            parsed = _LLMOutput.model_validate(payload)
        except ValidationError as exc:
            raise ExtractionError("LLM output did not match the expected schema") from exc
        return ChunkExtraction(
            entities=[
                ExtractedEntity(e.name, e.type, e.surface_form, e.description)
                for e in parsed.entities
            ],
            relationships=[
                ExtractedRelationship(r.source, r.target, r.type, r.supporting_text)
                for r in parsed.relationships
            ],
        )


def _first_json_object(raw: str) -> object:
    """Parse the first JSON object in ``raw`` (tolerates prose or code fences around it)."""
    start = raw.find("{")
    if start < 0:
        raise ExtractionError("LLM output contained no JSON object")
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ExtractionError("LLM output was not valid JSON") from exc
    return obj
