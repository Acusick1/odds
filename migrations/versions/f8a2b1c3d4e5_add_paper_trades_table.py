"""add paper trades table

Revision ID: f8a2b1c3d4e5
Revises: 3387940c2f83
Create Date: 2026-04-09 12:00:00.000000

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "f8a2b1c3d4e5"
down_revision = "3387940c2f83"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "paper_trades",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("market", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("selection", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("bookmaker", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("odds", sa.Integer(), nullable=False),
        sa.Column("stake", sa.Float(), nullable=False),
        sa.Column("reasoning", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("bankroll_before", sa.Float(), nullable=False),
        sa.Column("bankroll_after", sa.Float(), nullable=True),
        sa.Column("placed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("settled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("result", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_paper_trades_event_id"), "paper_trades", ["event_id"], unique=False)
    op.create_index(op.f("ix_paper_trades_bookmaker"), "paper_trades", ["bookmaker"], unique=False)
    op.create_index(
        "ix_paper_trades_unsettled",
        "paper_trades",
        ["settled_at"],
        unique=False,
        postgresql_where=sa.text("settled_at IS NULL"),
    )
    op.create_index(
        "ix_paper_trades_event_selection",
        "paper_trades",
        ["event_id", "selection"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_paper_trades_event_selection", table_name="paper_trades")
    op.drop_index(
        "ix_paper_trades_unsettled",
        table_name="paper_trades",
        postgresql_where=sa.text("settled_at IS NULL"),
    )
    op.drop_index(op.f("ix_paper_trades_bookmaker"), table_name="paper_trades")
    op.drop_index(op.f("ix_paper_trades_event_id"), table_name="paper_trades")
    op.drop_table("paper_trades")
