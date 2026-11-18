"""rename paper_trades to bets

Revision ID: d5a3f7c19e24
Revises: b7e3d1f9c2a4
Create Date: 2026-11-18 12:00:00.000000

Renames the table, its primary key, id sequence, indexes, constraints and the ``traderesult``
enum type to name the bet rather than the paper trade. Privileges follow the table, so no grant
changes. Nothing is dropped or copied; the downgrade renames every object back.
"""

import sqlalchemy as sa
from alembic import op

revision = "d5a3f7c19e24"
down_revision = "b7e3d1f9c2a4"
branch_labels = None
depends_on = None

# (old name, new name) of each index the table holds, including its primary key's.
_INDEXES = (
    ("paper_trades_pkey", "bets_pkey"),
    ("ix_paper_trades_bookmaker", "ix_bets_bookmaker"),
    ("ix_paper_trades_event_id", "ix_bets_event_id"),
    ("ix_paper_trades_candidate_name", "ix_bets_candidate_name"),
    ("ix_paper_trades_decision_id", "ix_bets_decision_id"),
    ("ix_paper_trades_event_selection", "ix_bets_event_selection"),
    ("ix_paper_trades_unsettled", "ix_bets_unsettled"),
    ("uq_paper_trades_candidate_bet", "uq_bets_candidate_bet"),
)
# (old name, new name) of each constraint other than the primary key.
_CONSTRAINTS = (
    ("paper_trades_event_id_fkey", "bets_event_id_fkey"),
    ("fk_paper_trades_decision_id", "fk_bets_decision_id"),
    ("ck_paper_trades_price_decimal_gt_one", "ck_bets_price_decimal_gt_one"),
    ("ck_paper_trades_candidate_pair", "ck_bets_candidate_pair"),
    ("ck_paper_trades_bankroll_or_candidate", "ck_bets_bankroll_or_candidate"),
)


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    _rename(
        "paper_trades", "bets", "paper_trades_id_seq", "bets_id_seq", "traderesult", "betresult"
    )
    for old, new in _INDEXES:
        op.execute(sa.text(f"ALTER INDEX {old} RENAME TO {new}"))
    for old, new in _CONSTRAINTS:
        op.execute(sa.text(f"ALTER TABLE bets RENAME CONSTRAINT {old} TO {new}"))


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    for old, new in _CONSTRAINTS:
        op.execute(sa.text(f"ALTER TABLE bets RENAME CONSTRAINT {new} TO {old}"))
    for old, new in _INDEXES:
        op.execute(sa.text(f"ALTER INDEX {new} RENAME TO {old}"))
    _rename(
        "bets", "paper_trades", "bets_id_seq", "paper_trades_id_seq", "betresult", "traderesult"
    )


def _rename(table: str, new_table: str, seq: str, new_seq: str, enum: str, new_enum: str) -> None:
    op.rename_table(table, new_table)
    op.execute(sa.text(f"ALTER SEQUENCE {seq} RENAME TO {new_seq}"))
    op.execute(sa.text(f"ALTER TYPE {enum} RENAME TO {new_enum}"))
