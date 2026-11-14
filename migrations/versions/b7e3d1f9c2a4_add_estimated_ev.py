"""add estimated_ev to paper_trades and candidate_backtest_bets

Revision ID: b7e3d1f9c2a4
Revises: a4c8e2f61b93
Create Date: 2026-11-14 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "b7e3d1f9c2a4"
down_revision = "a4c8e2f61b93"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("paper_trades", sa.Column("estimated_ev", sa.Float(), nullable=True))
    op.add_column("candidate_backtest_bets", sa.Column("estimated_ev", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("candidate_backtest_bets", "estimated_ev")
    op.drop_column("paper_trades", "estimated_ev")
