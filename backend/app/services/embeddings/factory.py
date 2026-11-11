"""Embedder selection from settings."""

from app.core.config import Settings
from app.services.embeddings.embedder import Embedder, HashingEmbedder, SentenceTransformerEmbedder


def build_embedder(settings: Settings) -> Embedder:
    """Create the configured embedder (no model is loaded until the first embed call)."""
    if settings.embedding_provider == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(
        settings.embedding_model,
        revision=settings.embedding_model_revision,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
        max_seq_length=settings.embedding_max_seq_length,
        cache_dir=settings.model_cache_dir,
        offline=settings.embedding_offline,
    )
