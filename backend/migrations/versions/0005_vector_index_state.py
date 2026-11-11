"""chunk_embeddings + vector_state: identity/sync bookkeeping for the vector index

Revision ID: 0005
Revises: 0004
"""

import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chunk_embeddings",
        sa.Column(
            "chunk_id",
            sa.String(64),
            sa.ForeignKey("document_chunks.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("model_revision", sa.String(64), nullable=True),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("config_fingerprint", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_index("ix_chunk_embeddings_document_id", "chunk_embeddings", ["document_id"])
    op.create_index("ix_chunk_embeddings_status", "chunk_embeddings", ["status"])
    op.create_table(
        "vector_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    # Start at 1 so an index file that predates this table can never look current.
    op.execute("INSERT INTO vector_state (id, revision, updated_at) VALUES (1, 1, CURRENT_TIMESTAMP)")


def downgrade() -> None:
    op.drop_table("vector_state")
    op.drop_table("chunk_embeddings")
