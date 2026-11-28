"""quarantine long retail outliers

Revision ID: d4b7e2a9c1f3
Revises: f8fdc2b8e985
Create Date: 2026-11-28 14:00:00.000000

Creates ``quarantined_prices`` and moves into it every stored price that
``odds_lambda.storage.retail_outliers.long_retail_outliers`` finds, as the writers now do when
they store a snapshot (#888). The downgrade moves them back and drops the table.
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal

import sqlalchemy as sa
from alembic import op
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.models import Price
from odds_core.vocabulary import PriceOutcome, Side
from odds_lambda.storage.retail_outliers import (
    GUARDED_MARKETS,
    MIN_RATIO_TO_NEXT_LONGEST,
    long_retail_outliers,
)

revision = "d4b7e2a9c1f3"
down_revision = "f8fdc2b8e985"
branch_labels = None
depends_on = None

PRICE_COLUMNS = "snapshot_id, bookmaker, outcome, line, side, price, size, matched, changed_at"
REASON = "long_retail_outlier"
BATCH = 1000

# The snapshots with an exchange back price in which a UK retail back price is at least
# MIN_RATIO_TO_NEXT_LONGEST times the next-longest on its outcome and line: a superset of the
# snapshots long_retail_outliers can find an outlier in, which it then reads in full.
CANDIDATE_SNAPSHOTS = sa.text(
    """
    WITH retail AS (
        SELECT p.snapshot_id, p.price,
               lead(p.price) OVER (
                   PARTITION BY p.snapshot_id, p.outcome, p.line ORDER BY p.price DESC, p.id
               ) AS next_longest,
               row_number() OVER (
                   PARTITION BY p.snapshot_id, p.outcome, p.line ORDER BY p.price DESC, p.id
               ) AS rank
        FROM prices p JOIN odds_snapshots s ON s.id = p.snapshot_id
        WHERE p.side = 'back' AND p.bookmaker = ANY(:books) AND s.market = ANY(:markets)
    )
    SELECT DISTINCT r.snapshot_id FROM retail r
    WHERE r.rank = 1 AND r.price >= :ratio * r.next_longest
      AND EXISTS (
          SELECT 1 FROM prices x
          WHERE x.snapshot_id = r.snapshot_id AND x.bookmaker = 'betfair_exchange'
            AND x.side = 'back'
      )
    ORDER BY r.snapshot_id
    """
)


def _outlier_ids(bind: sa.Connection, snapshot_ids: Sequence[int]) -> list[int]:
    rows = (
        bind.execute(
            sa.text(
                "SELECT p.id, p.snapshot_id, s.market, p.bookmaker, p.outcome, p.line, p.side,"
                " p.price FROM prices p JOIN odds_snapshots s ON s.id = p.snapshot_id"
                " WHERE p.snapshot_id = ANY(:ids) ORDER BY p.snapshot_id, p.id"
            ),
            {"ids": list(snapshot_ids)},
        )
        .mappings()
        .all()
    )
    by_snapshot: dict[int, list[sa.RowMapping]] = {}
    for row in rows:
        by_snapshot.setdefault(row["snapshot_id"], []).append(row)
    ids: list[int] = []
    for snapshot_rows in by_snapshot.values():
        prices = [
            Price(
                bookmaker=r["bookmaker"],
                outcome=PriceOutcome(r["outcome"]),
                line=None if r["line"] is None else Decimal(r["line"]),
                side=Side(r["side"]),
                price=Decimal(r["price"]),
            )
            for r in snapshot_rows
        ]
        market = snapshot_rows[0]["market"]
        ids.extend(snapshot_rows[i]["id"] for i in long_retail_outliers(market, prices))
    return ids


def upgrade() -> None:
    op.create_table(
        "quarantined_prices",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("bookmaker", sa.Text(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.Column("line", sa.Numeric(), nullable=True),
        sa.Column("side", sa.Text(), nullable=False),
        sa.Column("price", sa.Numeric(), nullable=False),
        sa.Column("size", sa.Numeric(), nullable=True),
        sa.Column("matched", sa.Numeric(), nullable=True),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("quarantined_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("price > 1.0", name="ck_quarantined_prices_price_gt_one"),
        sa.CheckConstraint(
            "outcome IN ('away', 'draw', 'home', 'over', 'under')",
            name="ck_quarantined_prices_outcome",
        ),
        sa.CheckConstraint(
            "side IN ('back', 'lay', 'mid', 'traded')", name="ck_quarantined_prices_side"
        ),
        sa.CheckConstraint(f"reason IN ('{REASON}')", name="ck_quarantined_prices_reason"),
        sa.ForeignKeyConstraint(["snapshot_id"], ["odds_snapshots.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "snapshot_id",
            "bookmaker",
            "outcome",
            "line",
            "side",
            name="uq_quarantined_prices_snapshot_book_outcome_line_side",
            postgresql_nulls_not_distinct=True,
        ),
    )
    op.create_index(
        op.f("ix_quarantined_prices_snapshot_id"), "quarantined_prices", ["snapshot_id"]
    )
    quarantine_stored_outliers(op.get_bind())


def quarantine_stored_outliers(bind: sa.Connection) -> int:
    """Move every stored outlier into ``quarantined_prices``; returns how many moved."""
    snapshot_ids = list(
        bind.execute(
            CANDIDATE_SNAPSHOTS,
            {
                "books": sorted(UK_RETAIL_BOOKMAKERS),
                "markets": sorted(m.value for m in GUARDED_MARKETS),
                "ratio": MIN_RATIO_TO_NEXT_LONGEST,
            },
        ).scalars()
    )
    moved = 0
    for start in range(0, len(snapshot_ids), BATCH):
        ids = _outlier_ids(bind, snapshot_ids[start : start + BATCH])
        if not ids:
            continue
        bind.execute(
            sa.text(
                f"INSERT INTO quarantined_prices ({PRICE_COLUMNS}, reason, quarantined_at)"
                f" SELECT {PRICE_COLUMNS}, :reason, now() FROM prices WHERE id = ANY(:ids)"
            ),
            {"reason": REASON, "ids": ids},
        )
        bind.execute(sa.text("DELETE FROM prices WHERE id = ANY(:ids)"), {"ids": ids})
        moved += len(ids)
    return moved


def downgrade() -> None:
    op.execute(
        f"INSERT INTO prices ({PRICE_COLUMNS}) SELECT {PRICE_COLUMNS} FROM quarantined_prices"
    )
    op.drop_index(op.f("ix_quarantined_prices_snapshot_id"), table_name="quarantined_prices")
    op.drop_table("quarantined_prices")
