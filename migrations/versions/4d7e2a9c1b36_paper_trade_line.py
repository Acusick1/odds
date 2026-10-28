"""paper trades record the selection's line

Revision ID: 4d7e2a9c1b36
Revises: 152a63c0fae5
Create Date: 2026-10-28 12:00:00.000000

A paper trade on a market with lines, such as totals over 2.5 goals, records the line in
``paper_trades.line``, numeric like ``prices.line``; it is null in a market without lines, which
every existing row is. A candidate version bets each selection, an outcome on a line, of a
match at most once, so the runner's unique bet index adds the line, with nulls not distinct so
a market without lines keeps one bet per outcome.

The downgrade deletes the trades on a line, which the old schema cannot tell apart, and
lowers their decisions' bet counts.
"""

import sqlalchemy as sa
from alembic import op

revision = "4d7e2a9c1b36"
down_revision = "152a63c0fae5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("paper_trades", sa.Column("line", sa.Numeric(), nullable=True))
    op.drop_index(
        "uq_paper_trades_candidate_bet",
        table_name="paper_trades",
        postgresql_where=sa.text("decision_id IS NOT NULL"),
    )
    op.create_index(
        "uq_paper_trades_candidate_bet",
        "paper_trades",
        ["candidate_name", "candidate_version", "event_id", "selection", "line"],
        unique=True,
        postgresql_where=sa.text("decision_id IS NOT NULL"),
        postgresql_nulls_not_distinct=True,
    )


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.execute(
        sa.text(
            "UPDATE candidate_decisions d SET bets_placed = d.bets_placed - t.n"
            " FROM (SELECT decision_id, count(*) AS n FROM paper_trades"
            " WHERE line IS NOT NULL AND decision_id IS NOT NULL GROUP BY decision_id) t"
            " WHERE d.id = t.decision_id"
        )
    )
    op.execute(sa.text("DELETE FROM paper_trades WHERE line IS NOT NULL"))
    op.drop_index(
        "uq_paper_trades_candidate_bet",
        table_name="paper_trades",
        postgresql_where=sa.text("decision_id IS NOT NULL"),
    )
    op.create_index(
        "uq_paper_trades_candidate_bet",
        "paper_trades",
        ["candidate_name", "candidate_version", "event_id", "selection"],
        unique=True,
        postgresql_where=sa.text("decision_id IS NOT NULL"),
    )
    op.drop_column("paper_trades", "line")
