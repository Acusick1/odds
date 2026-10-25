"""candidate decisions are unique per decision time, and runner bets per outcome

Revision ID: 152a63c0fae5
Revises: 9c822a7a3faf
Create Date: 2026-10-25 21:00:00.000000

A mechanical candidate may decide a match at several times, given as minutes before
kick-off. ``candidate_decisions.minutes_before_kickoff`` records which of them a decision is
for, backfilled from the kick-off and decision time every existing row holds, and the table
becomes unique on candidate, version, event and those minutes. A candidate version bets each
outcome of a match at most once, so the runner's unique bet index drops the bookmaker and
becomes unique on candidate, version, event and selection. Existing rows satisfy both: each
candidate version has at most one decision per event, and a rule bets each outcome once.

The downgrade deletes every decision after a candidate version's earliest one on an event,
with its trades, because the old schema cannot hold them.
"""

import sqlalchemy as sa
from alembic import op

revision = "152a63c0fae5"
down_revision = "9c822a7a3faf"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column(
        "candidate_decisions", sa.Column("minutes_before_kickoff", sa.Integer(), nullable=True)
    )
    op.execute(
        sa.text(
            "UPDATE candidate_decisions SET minutes_before_kickoff ="
            " round(extract(epoch FROM kickoff_at_decision - decision_time) / 60)"
        )
    )
    op.alter_column("candidate_decisions", "minutes_before_kickoff", nullable=False)
    op.drop_constraint(
        "uq_candidate_decisions_candidate_event", "candidate_decisions", type_="unique"
    )
    op.create_unique_constraint(
        "uq_candidate_decisions_candidate_event_lead",
        "candidate_decisions",
        ["candidate_name", "candidate_version", "event_id", "minutes_before_kickoff"],
    )
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


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    later = """
        SELECT id FROM (
            SELECT id, row_number() OVER (
                PARTITION BY candidate_name, candidate_version, event_id
                ORDER BY decision_time, id
            ) AS n
            FROM candidate_decisions
        ) ranked
        WHERE n > 1
    """
    op.execute(sa.text(f"DELETE FROM paper_trades WHERE decision_id IN ({later})"))
    op.execute(sa.text(f"DELETE FROM candidate_decisions WHERE id IN ({later})"))
    op.drop_index(
        "uq_paper_trades_candidate_bet",
        table_name="paper_trades",
        postgresql_where=sa.text("decision_id IS NOT NULL"),
    )
    op.create_index(
        "uq_paper_trades_candidate_bet",
        "paper_trades",
        ["candidate_name", "candidate_version", "event_id", "selection", "bookmaker"],
        unique=True,
        postgresql_where=sa.text("decision_id IS NOT NULL"),
    )
    op.drop_constraint(
        "uq_candidate_decisions_candidate_event_lead", "candidate_decisions", type_="unique"
    )
    op.drop_column("candidate_decisions", "minutes_before_kickoff")
    op.create_unique_constraint(
        "uq_candidate_decisions_candidate_event",
        "candidate_decisions",
        ["candidate_name", "candidate_version", "event_id"],
    )
