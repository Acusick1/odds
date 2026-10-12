"""add sharp context and clv to paper_trades

Revision ID: 559c8462bb71
Revises: e6f24ef60549
Create Date: 2026-10-11 21:19:55.719746

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "559c8462bb71"
down_revision = "e6f24ef60549"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("paper_trades", sa.Column("sharp_prob_at_bet", sa.Float(), nullable=True))
    op.add_column(
        "paper_trades",
        sa.Column("sharp_bookmaker_at_bet", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.add_column(
        "paper_trades",
        sa.Column("sharp_snapshot_time_at_bet", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("paper_trades", sa.Column("sharp_spread_at_bet", sa.Float(), nullable=True))
    op.add_column("paper_trades", sa.Column("sharp_prob_at_close", sa.Float(), nullable=True))
    op.add_column(
        "paper_trades",
        sa.Column("sharp_bookmaker_at_close", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.add_column(
        "paper_trades",
        sa.Column("sharp_snapshot_time_at_close", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("paper_trades", sa.Column("clv", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("paper_trades", "clv")
    op.drop_column("paper_trades", "sharp_snapshot_time_at_close")
    op.drop_column("paper_trades", "sharp_bookmaker_at_close")
    op.drop_column("paper_trades", "sharp_prob_at_close")
    op.drop_column("paper_trades", "sharp_spread_at_bet")
    op.drop_column("paper_trades", "sharp_snapshot_time_at_bet")
    op.drop_column("paper_trades", "sharp_bookmaker_at_bet")
    op.drop_column("paper_trades", "sharp_prob_at_bet")
