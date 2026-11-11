"""graph_state: change counter used to detect a stale persisted graph

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "graph_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    # Start at 1 so a graph file from before this table existed can never match.
    op.execute("INSERT INTO graph_state (id, revision, updated_at) VALUES (1, 1, CURRENT_TIMESTAMP)")


def downgrade() -> None:
    op.drop_table("graph_state")
