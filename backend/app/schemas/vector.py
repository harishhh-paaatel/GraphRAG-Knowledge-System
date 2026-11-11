"""Vector search / index API schemas."""

from pydantic import BaseModel, Field, field_validator


class VectorSearchRequest(BaseModel):
    query: str = Field(
        min_length=1, max_length=2000, examples=["Which technologies build Python APIs?"]
    )
    top_k: int = Field(default=5, ge=1, le=500)
    similarity_threshold: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Minimum cosine similarity (inclusive)."
    )
    document_ids: list[str] | None = Field(default=None, max_length=100)

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class ResultSource(BaseModel):
    filename: str
    page_number: int | None
    section: str | None
    chunk_index: int


class VectorSearchResult(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    content: str
    similarity: float = Field(description="Retrieval similarity (cosine). NOT answer confidence.")
    source: ResultSource
    embedding_model: str


class VectorSearchResponse(BaseModel):
    query: str
    results: list[VectorSearchResult]
    embedding_model: str
    embedding_model_revision: str | None
    similarity_metric: str
    similarity_decimals: int
    top_k: int
    similarity_threshold: float | None
    indexed_vectors: int
    index_revision: int | None
    database_revision: int | None
    index_stale: bool = Field(
        description="True when a running sync meant the previous consistent index was used."
    )
    max_tokens: int | None = Field(description="The embedding model's real input limit.")
    query_token_count: int | None = Field(description="Query length in model tokens.")
    query_windows: int | None = Field(
        description=">1: the query exceeded the limit and was pooled."
    )
    unknown_document_ids: list[str]
    timings_ms: dict[str, float]
