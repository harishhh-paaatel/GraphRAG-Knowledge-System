"""Environment-based application configuration."""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.privacy import UnsafeProviderUrl, check_provider_url, valid_env_name

REPO_ROOT = Path(__file__).resolve().parents[3]

# The embedding model and the exact Hugging Face commit this project was verified with
# (384 dimensions, 256-token limit). Pinning the commit makes embeddings reproducible: a
# silently updated upstream model can never change the vectors of an existing index.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Tokens reserved for the system prompt, the question and formatting when checking that the
# configured context budget fits the model window. A conservative placeholder until the real
# prompt size is measured (Phase 7, M2).
PROMPT_OVERHEAD_TOKENS = 400


class Settings(BaseSettings):
    """Runtime settings, read from environment variables prefixed ``GRAPHRAG_``."""

    model_config = SettingsConfigDict(env_prefix="GRAPHRAG_", env_file=".env", extra="ignore")

    app_name: str = "GraphRAG Knowledge Service"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    data_dir: Path = REPO_ROOT / "data"
    database_url: str = ""  # empty -> sqlite file under data_dir

    max_upload_mb: int = Field(default=25, ge=1, le=500)
    cors_origins: list[str] = ["http://localhost:5173"]
    api_key: str | None = None  # optional; never returned by any endpoint

    chunk_size: int = Field(default=800, ge=100, le=8000)  # characters
    chunk_overlap: int = Field(default=100, ge=0)  # characters
    worker_threads: int = Field(default=2, ge=1, le=16)

    # Knowledge extraction. "llm" needs a client wired in Phase 7; selecting it earlier fails fast.
    extraction_provider: Literal["rule_based", "mock", "llm"] = "rule_based"
    entity_types: list[str] = [
        "PERSON",
        "ORGANIZATION",
        "LOCATION",
        "TECHNOLOGY",
        "PRODUCT",
        "CONCEPT",
        "EVENT",
    ]
    fuzzy_match_threshold: float = Field(default=0.88, ge=0.5, le=1.0)

    # Embeddings / vector index. Changing any of these changes the index fingerprint, which
    # invalidates and rebuilds the index (vectors from different models are never mixed).
    embedding_provider: Literal["sentence_transformers", "hashing"] = "sentence_transformers"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    # 40-character HF commit hash, or empty/None to leave the model unpinned (not advised).
    embedding_model_revision: str | None = DEFAULT_EMBEDDING_REVISION
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_device: str = "cpu"
    embedding_max_seq_length: int | None = Field(default=None, ge=8, le=8192)
    embedding_cache_dir: Path | None = None  # default: <data_dir>/models
    embedding_offline: bool = False  # only use the local model cache, never the network
    vector_sync_on_startup: bool = True
    vector_max_top_k: int = Field(default=50, ge=1, le=500)

    # Hard caps for graph traversal (API parameters are validated against these).
    graph_max_hops: int = Field(default=4, ge=1, le=10)
    graph_max_nodes: int = Field(default=500, ge=1, le=10_000)
    graph_max_edges: int = Field(default=1000, ge=1, le=50_000)

    # Hybrid retrieval (Phase 6). Per-request values can only be tighter than these caps.
    hybrid_rrf_k: int = Field(default=60, ge=1, le=1000)
    hybrid_max_limit: int = Field(default=50, ge=1, le=500)
    hybrid_graph_max_chunks: int = Field(default=50, ge=1, le=500)
    hybrid_graph_max_hops: int = Field(default=2, ge=1, le=10)
    hybrid_max_name_tokens: int = Field(default=8, ge=1, le=16)
    # Time budget for each retrieval path in a hybrid search (0 = unlimited). A path that
    # overruns it is reported unavailable and the other path's result is still returned.
    hybrid_path_timeout_seconds: float = Field(default=30.0, ge=0, le=600)

    # Answer generation (Phase 7). The default provider needs no model and no network.
    llm_provider: Literal["extractive", "mock", "ollama", "openai_compatible"] = "extractive"
    llm_base_url: str | None = (
        None  # Ollama default http://127.0.0.1:11434 is applied by the factory
    )
    llm_model: str | None = None
    llm_api_key_env: str | None = None  # the NAME of an environment variable; never the key itself
    llm_allow_remote: bool = False  # a non-loopback provider URL needs this explicit opt-in
    llm_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    llm_retries: int = Field(default=1, ge=0, le=5)
    llm_context_window: int = Field(default=4096, ge=512, le=131_072)
    llm_max_answer_tokens: int = Field(default=512, ge=16, le=8192)
    llm_max_concurrency: int = Field(default=1, ge=1, le=16)
    llm_fallback_to_extractive: bool = False  # never silent: the response says it happened
    llm_debug_prompt: bool = False  # lets a request ask for the prompt back (development only)
    answer_context_tokens: int = Field(default=2048, ge=128, le=65_536)
    answer_chars_per_token: float = Field(default=3.0, ge=0.5, le=10.0)
    answer_grounding_policy: Literal["withhold", "annotate"] = "withhold"
    answer_min_top_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    answer_functional_relationships: list[str] = []
    query_history: bool = True

    @field_validator("embedding_model_revision", mode="before")
    @classmethod
    def _check_revision(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{40}", value.strip()):
            raise ValueError("embedding_model_revision must be a 40-character commit hash")
        return value.strip()

    @model_validator(mode="after")
    def _check_llm(self) -> "Settings":
        if self.llm_base_url is not None:
            try:
                self.llm_base_url = check_provider_url(
                    self.llm_base_url, allow_remote=self.llm_allow_remote
                )
            except UnsafeProviderUrl as exc:
                raise ValueError(f"llm_base_url: {exc}") from exc
        if self.llm_api_key_env is not None and not valid_env_name(self.llm_api_key_env):
            raise ValueError("llm_api_key_env must be an environment variable NAME, not a key")
        if self.llm_provider in ("ollama", "openai_compatible") and not self.llm_model:
            raise ValueError(f"llm_model must be set for llm_provider={self.llm_provider}")
        if self.llm_provider == "openai_compatible" and not self.llm_base_url:
            raise ValueError("llm_base_url must be set for llm_provider=openai_compatible")
        needed = self.answer_context_tokens + self.llm_max_answer_tokens + PROMPT_OVERHEAD_TOKENS
        if needed > self.llm_context_window:
            raise ValueError(
                f"answer_context_tokens ({self.answer_context_tokens}) + llm_max_answer_tokens "
                f"({self.llm_max_answer_tokens}) + {PROMPT_OVERHEAD_TOKENS} tokens of prompt "
                f"overhead exceed llm_context_window ({self.llm_context_window})"
            )
        from app.services.extraction.types import RELATIONSHIP_TYPES

        unknown = sorted(set(self.answer_functional_relationships) - set(RELATIONSHIP_TYPES))
        if unknown:
            raise ValueError(f"answer_functional_relationships: unknown types {unknown}")
        return self

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not self.entity_types or any(
            not re.fullmatch(r"[A-Z][A-Z_]*", t) for t in self.entity_types
        ):
            raise ValueError("entity_types must be non-empty UPPER_SNAKE_CASE names")
        if (
            self.embedding_model != DEFAULT_EMBEDDING_MODEL
            and "embedding_model_revision" not in self.model_fields_set
        ):
            # The default pin belongs to the default model only; never apply it to another.
            self.embedding_model_revision = None
        return self

    @property
    def documents_dir(self) -> Path:
        """Isolated per-document storage root for uploaded files."""
        return self.data_dir / "documents"

    @property
    def effective_hybrid_graph_max_hops(self) -> int:
        """Hybrid retrieval never walks further than the graph API itself allows."""
        return min(self.hybrid_graph_max_hops, self.graph_max_hops)

    @property
    def graph_file(self) -> Path:
        """Persisted knowledge graph (a derived, rebuildable artifact)."""
        return self.data_dir / "graphs" / "knowledge_graph.json"

    @property
    def model_cache_dir(self) -> Path:
        """Embedding model downloads (kept inside data/, i.e. on the project drive)."""
        return self.embedding_cache_dir or (self.data_dir / "models")

    @property
    def vector_index_file(self) -> Path:
        """Persisted FAISS index (a derived, rebuildable artifact)."""
        return self.data_dir / "embeddings" / "vector_index.bin"

    @property
    def tmp_dir(self) -> Path:
        """Staging area for in-flight uploads (same volume, so moves are atomic)."""
        return self.data_dir / "tmp"

    @property
    def resolved_database_url(self) -> str:
        """Database URL, defaulting to a SQLite file in ``data_dir``."""
        if self.database_url:
            return self.database_url
        return f"sqlite:///{(self.data_dir / 'graphrag.db').as_posix()}"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
