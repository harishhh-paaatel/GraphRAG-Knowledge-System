"""chunk_embeddings: record token count and window count per chunk

The embedding model reads a fixed number of tokens and used to drop the rest silently. Chunks
over the limit are now embedded in windows and pooled (see services/embeddings/windowing.py),
and these two columns make that visible instead of silent. Existing rows keep NULL (unknown)
until their chunk is next embedded or the index is rebuilt.

Revision ID: 0006
Revises: 0005
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("chunk_embeddings") as batch:
        batch.add_column(sa.Column("token_count", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("window_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("chunk_embeddings") as batch:
        batch.drop_column("window_count")
        batch.drop_column("token_count")
