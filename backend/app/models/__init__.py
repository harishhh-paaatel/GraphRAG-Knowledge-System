"""ORM models. Importing this package registers every table on ``Base.metadata``."""

from app.models.base import Base
from app.models.document import Document, DocumentChunk
from app.models.knowledge import Entity, EntityCandidate, EntityMention, Relationship
from app.models.operations import EvaluationRun, GraphState, ProcessingJob, QueryHistory
from app.models.vectors import ChunkEmbedding, VectorState

__all__ = [
    "Base",
    "Document",
    "DocumentChunk",
    "ChunkEmbedding",
    "Entity",
    "EntityCandidate",
    "EntityMention",
    "Relationship",
    "EvaluationRun",
    "GraphState",
    "ProcessingJob",
    "QueryHistory",
    "VectorState",
]
