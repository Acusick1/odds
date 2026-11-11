"""candidate decisions record the leagues their candidate version listed

Revision ID: b7d3f1a8c5e2
Revises: a6c2e9d4b7f1
Create Date: 2026-11-11 10:00:00.000000

Adds ``candidate_decisions.sports``, the league keys the candidate version listed when it
decided, which the report counts the version's family by.

Backfill: every existing row gets ``{soccer_epl}``. Every candidate registered so far
(``exchange_gap``, ``polymarket_gap`` and the totals variant) lists EPL alone, and none has ever
changed its leagues, so that is the set each version actually decided with and the backfill is
exact.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b7d3f1a8c5e2"
down_revision = "a6c2e9d4b7f1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column(
        "candidate_decisions",
        sa.Column(
            "sports",
            postgresql.ARRAY(sa.String()),
            server_default=sa.text("'{soccer_epl}'"),
            nullable=False,
        ),
    )
    op.alter_column("candidate_decisions", "sports", server_default=None)


def downgrade() -> None:
    op.drop_column("candidate_decisions", "sports")
