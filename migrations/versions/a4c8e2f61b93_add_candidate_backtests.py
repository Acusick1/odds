"""store candidate backtest runs and their bets

Revision ID: a4c8e2f61b93
Revises: d2f8a4c6e1b9
Create Date: 2026-11-13 18:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a4c8e2f61b93"
down_revision = "d2f8a4c6e1b9"
branch_labels = None
depends_on = None

SOURCES = (
    "betfair",
    "espn",
    "football_data_uk",
    "mlb_stats_api",
    "odds_api",
    "oddsportal",
    "polymarket",
)


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.create_table(
        "candidate_backtests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("candidate_name", sa.Text(), nullable=False),
        sa.Column("candidate_version", sa.Integer(), nullable=False),
        sa.Column("issue", sa.Integer(), nullable=False),
        sa.Column("market", sa.Text(), nullable=False),
        sa.Column("rule", sa.Text(), nullable=False),
        sa.Column("rule_sha256", sa.String(length=64), nullable=True),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("window_kind", sa.Text(), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("leagues", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("git_sha", sa.Text(), nullable=False),
        sa.Column("cell_index", sa.Integer(), nullable=True),
        sa.Column("eligible", sa.Boolean(), nullable=True),
        sa.Column("selected", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("blocks", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("events", sa.Integer(), nullable=False),
        sa.Column("bets", sa.Integer(), nullable=False),
        sa.Column("bets_with_close", sa.Integer(), nullable=False),
        sa.Column("stake", sa.Float(), nullable=False),
        sa.Column("pnl", sa.Float(), nullable=False),
        sa.Column("mean_ev", sa.Float(), nullable=True),
        sa.Column("ci_lower", sa.Float(), nullable=True),
        sa.Column("ci_upper", sa.Float(), nullable=True),
        sa.Column("ci_level", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "window_kind IN ('development', 'holdout')",
            name="ck_candidate_backtests_window_kind",
        ),
        sa.CheckConstraint("start_date <= end_date", name="ck_candidate_backtests_range"),
        sa.CheckConstraint(
            "(cell_index IS NULL) = (eligible IS NULL)", name="ck_candidate_backtests_cell"
        ),
        sa.CheckConstraint(
            "(mean_ev IS NULL) = (ci_lower IS NULL) AND (mean_ev IS NULL) = (ci_upper IS NULL) "
            "AND (mean_ev IS NULL) = (ci_level IS NULL)",
            name="ck_candidate_backtests_estimate",
        ),
        sa.UniqueConstraint(
            "candidate_name",
            "candidate_version",
            "window_kind",
            "start_date",
            "end_date",
            "git_sha",
            "cell_index",
            name="uq_candidate_backtests_run",
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index(
        "ix_candidate_backtests_candidate_name", "candidate_backtests", ["candidate_name"]
    )
    op.create_table(
        "candidate_backtest_bets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("backtest_id", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("event_id", sa.BigInteger(), nullable=False),
        sa.Column("league", sa.Text(), nullable=False),
        sa.Column("kickoff", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decision_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("line", sa.Float(), nullable=True),
        sa.Column("bookmaker", sa.Text(), nullable=False),
        sa.Column("price", sa.Float(), nullable=False),
        sa.Column("stake", sa.Float(), nullable=False),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("p_close", sa.Float(), nullable=True),
        sa.Column("ev", sa.Float(), nullable=True),
        sa.Column("close_source", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["backtest_id"], ["candidate_backtests.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("backtest_id", "position", name="uq_candidate_backtest_bets_position"),
        sa.CheckConstraint(
            "close_source IN (" + ", ".join(f"'{s}'" for s in SOURCES) + ")",
            name="ck_candidate_backtest_bets_close_source",
        ),
    )
    op.create_index(
        "ix_candidate_backtest_bets_backtest_id", "candidate_backtest_bets", ["backtest_id"]
    )


def downgrade() -> None:
    op.drop_table("candidate_backtest_bets")
    op.drop_table("candidate_backtests")
