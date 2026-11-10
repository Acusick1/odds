"""candidate decisions record the events and values their fixture inputs consumed

Revision ID: a6c2e9d4b7f1
Revises: e5b7c9d1f3a2
Create Date: 2026-11-10 23:00:00.000000

Adds ``candidate_decisions.input_event_ids``, the ids of the events a rule's fixture inputs
(each team's previous result, the concurrent kick-offs) were read from, and
``input_fixture``, the values it consumed, which the parity check replays. Existing rows get an
empty array and null: no rule read fixture inputs before.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a6c2e9d4b7f1"
down_revision = "e5b7c9d1f3a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column(
        "candidate_decisions",
        sa.Column(
            "input_event_ids",
            postgresql.ARRAY(sa.BigInteger()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )

    op.add_column(
        "candidate_decisions",
        sa.Column("input_fixture", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("candidate_decisions", "input_fixture")
    op.drop_column("candidate_decisions", "input_event_ids")
