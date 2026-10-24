"""paper trades: decimal price and candidate fields

Revision ID: e8b1c4d7a2f6
Revises: e1b7c9a4f2d3
Create Date: 2026-10-24 14:00:00.000000

Replaces ``paper_trades.odds`` (integer American odds, which cannot hold a UK
decimal price exactly) with ``price_decimal``, and adds ``candidate_name``,
``candidate_version`` (both set or both null) and ``git_sha``. Existing trades keep
null candidate fields.

Existing prices convert exactly (American odds are exact decimal prices). The
downgrade rounds to the nearest integer American price, so a trade placed at a
decimal price comes back approximate.
"""

import sqlalchemy as sa
from alembic import op

revision = "e8b1c4d7a2f6"
down_revision = "e1b7c9a4f2d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("paper_trades", sa.Column("price_decimal", sa.Float(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE paper_trades SET price_decimal = CASE"
            " WHEN odds > 0 THEN odds / 100.0 + 1"
            " ELSE 100.0 / abs(odds) + 1 END"
        )
    )
    op.alter_column("paper_trades", "price_decimal", nullable=False)
    op.drop_column("paper_trades", "odds")
    op.create_check_constraint(
        "ck_paper_trades_price_decimal_gt_one", "paper_trades", "price_decimal > 1.0"
    )

    op.add_column("paper_trades", sa.Column("candidate_name", sa.String(), nullable=True))
    op.add_column("paper_trades", sa.Column("candidate_version", sa.Integer(), nullable=True))
    op.add_column("paper_trades", sa.Column("git_sha", sa.String(), nullable=True))
    op.create_index(
        op.f("ix_paper_trades_candidate_name"), "paper_trades", ["candidate_name"], unique=False
    )
    op.create_check_constraint(
        "ck_paper_trades_candidate_pair",
        "paper_trades",
        "(candidate_name IS NULL) = (candidate_version IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_paper_trades_candidate_pair", "paper_trades", type_="check")
    op.drop_index(op.f("ix_paper_trades_candidate_name"), table_name="paper_trades")
    op.drop_column("paper_trades", "git_sha")
    op.drop_column("paper_trades", "candidate_version")
    op.drop_column("paper_trades", "candidate_name")

    op.drop_constraint("ck_paper_trades_price_decimal_gt_one", "paper_trades", type_="check")
    op.add_column("paper_trades", sa.Column("odds", sa.Integer(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE paper_trades SET odds = CASE"
            " WHEN price_decimal >= 2.0 THEN round((price_decimal - 1) * 100)"
            " ELSE round(-100 / (price_decimal - 1)) END"
        )
    )
    op.alter_column("paper_trades", "odds", nullable=False)
    op.drop_column("paper_trades", "price_decimal")
