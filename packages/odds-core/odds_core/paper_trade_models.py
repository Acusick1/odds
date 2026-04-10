"""Paper trade model for forward-looking bet tracking and settlement."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, Index, text
from sqlmodel import Field, SQLModel

from odds_core.models import utc_now

if TYPE_CHECKING:
    from odds_analytics.backtesting.models import BetRecord

    from odds_core.models import Event


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
    # Bankroll snapshots are per-trade only (bankroll_before at placement, bankroll_after at
    # settlement as bankroll_before + pnl). They do not reflect other trades placed in between.
    bankroll_before: float = Field(description="Bankroll before this bet")
    bankroll_after: float | None = Field(default=None, description="Bankroll after settlement")

    # Timestamps
    placed_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False),
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

    def to_bet_record(self, event: Event) -> BetRecord:
        """Convert a settled PaperTrade to a BetRecord for backtest analysis tools."""
        from odds_analytics.backtesting.models import BetRecord

        assert self.id is not None
        return BetRecord(
            bet_id=self.id,
            event_id=self.event_id,
            event_date=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            market=self.market,
            outcome=self.selection,
            bookmaker=self.bookmaker,
            odds=self.odds,
            line=None,
            decision_time=self.placed_at,
            stake=self.stake,
            bankroll_before=self.bankroll_before,
            strategy_confidence=self.confidence,
            result=self.result.value if self.result else None,
            profit=self.pnl,
            bankroll_after=self.bankroll_after,
            home_score=event.home_score,
            away_score=event.away_score,
            bet_rationale=self.reasoning,
        )
