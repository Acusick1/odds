"""candidate decisions and flat-stake paper trades

Revision ID: c9d4e1a7b3f2
Revises: e7b2c4d9a1f8
Create Date: 2026-10-24 18:00:00.000000

Adds ``candidate_decisions`` (one row per candidate version and event, written by the
scheduled runner even when no bet results), ``paper_trades.decision_id`` referencing it,
a unique index on a candidate's bet, and makes ``bankroll_before`` nullable for flat-stake
candidate trades.

The downgrade deletes every trade without a bankroll (the runner's), because the old
schema cannot hold them, then drops the decisions table.
"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision = "c9d4e1a7b3f2"
down_revision = "e7b2c4d9a1f8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.create_table(
        "candidate_decisions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("candidate_version", sa.Integer(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("decision_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("kickoff_at_decision", sa.DateTime(timezone=True), nullable=False),
        sa.Column("known_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bets_placed", sa.Integer(), nullable=False),
        sa.Column("missed", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "candidate_name",
            "candidate_version",
            "event_id",
            name="uq_candidate_decisions_candidate_event",
        ),
        sa.CheckConstraint("NOT missed OR bets_placed = 0", name="ck_candidate_decisions_missed"),
    )
    op.create_index(
        op.f("ix_candidate_decisions_candidate_name"),
        "candidate_decisions",
        ["candidate_name"],
        unique=False,
    )
    op.create_index(
        op.f("ix_candidate_decisions_event_id"), "candidate_decisions", ["event_id"], unique=False
    )

    op.add_column("paper_trades", sa.Column("decision_id", sa.Integer(), nullable=True))
    op.create_foreign_key(
        "fk_paper_trades_decision_id",
        "paper_trades",
        "candidate_decisions",
        ["decision_id"],
        ["id"],
    )
    op.create_index(
        op.f("ix_paper_trades_decision_id"), "paper_trades", ["decision_id"], unique=False
    )
    op.alter_column("paper_trades", "bankroll_before", nullable=True)
    op.create_check_constraint(
        "ck_paper_trades_bankroll_or_candidate",
        "paper_trades",
        "bankroll_before IS NOT NULL OR candidate_name IS NOT NULL",
    )
    op.create_index(
        "uq_paper_trades_candidate_bet",
        "paper_trades",
        ["candidate_name", "candidate_version", "event_id", "selection", "bookmaker"],
        unique=True,
        postgresql_where=sa.text("decision_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_paper_trades_candidate_bet",
        table_name="paper_trades",
        postgresql_where=sa.text("decision_id IS NOT NULL"),
    )
    op.drop_constraint("ck_paper_trades_bankroll_or_candidate", "paper_trades", type_="check")
    op.execute(sa.text("DELETE FROM paper_trades WHERE bankroll_before IS NULL"))
    op.alter_column("paper_trades", "bankroll_before", nullable=False)
    op.drop_index(op.f("ix_paper_trades_decision_id"), table_name="paper_trades")
    op.drop_constraint("fk_paper_trades_decision_id", "paper_trades", type_="foreignkey")
    op.drop_column("paper_trades", "decision_id")
    op.drop_index(op.f("ix_candidate_decisions_event_id"), table_name="candidate_decisions")
    op.drop_index(op.f("ix_candidate_decisions_candidate_name"), table_name="candidate_decisions")
    op.drop_table("candidate_decisions")
