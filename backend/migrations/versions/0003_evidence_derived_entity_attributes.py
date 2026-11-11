"""per-mention extracted name and description

Entity canonical name, description and aliases are now derived from the evidence that
remains (see ``refresh_entities``). That needs the name and description each mention
reported. Existing rows are backfilled conservatively:

* ``extracted_name``  <- the entity's current canonical name (the best information we have);
* ``description``     <- the entity's current description, attached to its earliest mention.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("entity_mentions") as batch:
        batch.add_column(sa.Column("extracted_name", sa.String(255), nullable=True))
        batch.add_column(sa.Column("description", sa.Text(), nullable=True))

    conn = op.get_bind()
    conn.execute(
        sa.text(
            "UPDATE entity_mentions SET extracted_name = "
            "(SELECT e.canonical_name FROM entities e WHERE e.id = entity_mentions.entity_id) "
            "WHERE extracted_name IS NULL"
        )
    )
    described = conn.execute(
        sa.text("SELECT id, description FROM entities WHERE description IS NOT NULL")
    ).all()
    for entity_id, description in described:
        earliest = conn.execute(
            sa.text(
                "SELECT m.id FROM entity_mentions m "
                "JOIN documents d ON d.id = m.document_id "
                "JOIN document_chunks c ON c.id = m.chunk_id "
                "WHERE m.entity_id = :eid "
                "ORDER BY d.upload_timestamp, m.document_id, c.chunk_index, m.surface_form "
                "LIMIT 1"
            ),
            {"eid": entity_id},
        ).scalar()
        if earliest is not None:
            conn.execute(
                sa.text("UPDATE entity_mentions SET description = :d WHERE id = :mid"),
                {"d": description, "mid": earliest},
            )


def downgrade() -> None:
    with op.batch_alter_table("entity_mentions") as batch:
        batch.drop_column("description")
        batch.drop_column("extracted_name")
