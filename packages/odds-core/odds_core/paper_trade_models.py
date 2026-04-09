"""Paper trade model for forward-looking bet tracking and settlement."""

from datetime import datetime
from enum import Enum

from sqlalchemy import Column, DateTime, Index, text
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now


class TradeResult(str, Enum):
    """Outcome of a settled paper trade."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"
    VOID = "void"


class PaperTrade(SQLModel, table=True):
    """Persistent record of a paper bet, placed before an event and settled against results."""

    __tablename__ = "paper_trades"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)

    # Bet details (aligned with BetRecord)
    market: str = Field(description="Market type: h2h")
    selection: str = Field(description="Selection: home, draw, or away")
    bookmaker: str = Field(index=True, description="Bookmaker key")
    odds: int = Field(description="American odds at time of placement")
    stake: float = Field(description="Stake amount")

    # Agent reasoning
    reasoning: str | None = Field(default=None, description="Why this bet was placed")
    confidence: float | None = Field(default=None, description="Agent confidence score (0-1)")

    # Bankroll tracking
    bankroll_before: float = Field(description="Bankroll before this bet")
    bankroll_after: float | None = Field(default=None, description="Bankroll after settlement")

    # Timestamps
    placed_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True)),
        default_factory=utc_now,
        description="When the bet was placed",
    )
    settled_at: datetime | None = Field(
        sa_column=Column(DateTime(timezone=True)),
        default=None,
        description="When the bet was settled",
    )

    # Settlement
    result: TradeResult | None = Field(default=None, description="Settlement outcome")
    pnl: float | None = Field(default=None, description="Profit/loss from this bet")

    __table_args__ = (
        Index(
            "ix_paper_trades_unsettled", "settled_at", postgresql_where=text("settled_at IS NULL")
        ),
        Index("ix_paper_trades_event_selection", "event_id", "selection"),
    )
