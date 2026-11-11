"""knowledge extraction: provenance columns, mention uniqueness, entity candidates

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("entity_mentions") as batch:
        batch.add_column(
            sa.Column("provenance", sa.String(64), nullable=False, server_default="unknown")
        )
        batch.create_unique_constraint(
            "uq_mention_evidence", ["entity_id", "chunk_id", "surface_form"]
        )
    with op.batch_alter_table("relationships") as batch:
        batch.add_column(
            sa.Column("provenance", sa.String(64), nullable=False, server_default="unknown")
        )
    op.create_table(
        "entity_candidates",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "entity_a_id", sa.String(32), sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "entity_b_id", sa.String(32), sa.ForeignKey("entities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.UniqueConstraint("entity_a_id", "entity_b_id", "reason", name="uq_candidate_pair_reason"),
    )
    op.create_index("ix_entity_candidates_entity_a_id", "entity_candidates", ["entity_a_id"])
    op.create_index("ix_entity_candidates_entity_b_id", "entity_candidates", ["entity_b_id"])


def downgrade() -> None:
    op.drop_table("entity_candidates")
    with op.batch_alter_table("relationships") as batch:
        batch.drop_column("provenance")
    with op.batch_alter_table("entity_mentions") as batch:
        batch.drop_constraint("uq_mention_evidence", type_="unique")
        batch.drop_column("provenance")
