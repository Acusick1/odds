"""candidate decisions record their input snapshot ids

Revision ID: d3f8a6b1c5e9
Revises: f3a9c6d2b8e1
Create Date: 2026-10-25 12:00:00.000000

Adds ``candidate_decisions.input_snapshot_ids``, the ids of the snapshots the rule's market
state was built from, so the parity check replays a decision from its recorded inputs.
Existing rows get an empty array and cannot be replayed from inputs.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "d3f8a6b1c5e9"
down_revision = "f3a9c6d2b8e1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column(
        "candidate_decisions",
        sa.Column(
            "input_snapshot_ids",
            postgresql.ARRAY(sa.Integer()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("candidate_decisions", "input_snapshot_ids")
