"""Prices a source returned that a writer keeps out of ``prices``, with the reason.

Kept out of ``odds_core.models``, which is in every rule's import closure, so a change here
does not change any rule's hash.
"""

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
)
from sqlmodel import Field, SQLModel

from odds_core.vocabulary import PriceOutcome, Side, StrEnumText, closed_set_check


class QuarantineReason(StrEnum):
    # A UK retail back price at least twice the next-longest and above +50% EV against the
    # snapshot's exchange price.
    LONG_RETAIL_OUTLIER = "long_retail_outlier"


class QuarantinedPrice(SQLModel, table=True):
    """One price of a snapshot that no reader of ``prices`` sees, with the same columns."""

    __tablename__ = "quarantined_prices"

    id: int | None = Field(default=None, sa_column=Column(BigInteger, Identity(), primary_key=True))
    snapshot_id: int | None = Field(
        default=None,
        sa_column=Column(
            Integer, ForeignKey("odds_snapshots.id", ondelete="CASCADE"), nullable=False, index=True
        ),
    )
    bookmaker: str = Field(sa_column=Column(Text, nullable=False))
    outcome: PriceOutcome = Field(sa_column=Column(StrEnumText(PriceOutcome), nullable=False))
    line: Decimal | None = Field(default=None, sa_column=Column(Numeric))
    side: Side = Field(sa_column=Column(StrEnumText(Side), nullable=False))
    price: Decimal = Field(sa_column=Column(Numeric, nullable=False))
    size: Decimal | None = Field(default=None, sa_column=Column(Numeric))
    matched: Decimal | None = Field(default=None, sa_column=Column(Numeric))
    changed_at: datetime | None = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    reason: QuarantineReason = Field(
        sa_column=Column(StrEnumText(QuarantineReason), nullable=False)
    )
    quarantined_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column=Column(DateTime(timezone=True), nullable=False),
    )

    __table_args__ = (
        UniqueConstraint(
            "snapshot_id",
            "bookmaker",
            "outcome",
            "line",
            "side",
            name="uq_quarantined_prices_snapshot_book_outcome_line_side",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint("price > 1.0", name="ck_quarantined_prices_price_gt_one"),
        closed_set_check("outcome", PriceOutcome, "ck_quarantined_prices_outcome"),
        closed_set_check("side", Side, "ck_quarantined_prices_side"),
        closed_set_check("reason", QuarantineReason, "ck_quarantined_prices_reason"),
    )
