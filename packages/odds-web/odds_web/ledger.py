"""The candidate page's bet ledger: every trade of one candidate version, void ones included,
each beside its close."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from odds_analytics.candidate_report import close_probabilities, expected_value
from odds_core.models import Event
from odds_core.paper_trade_models import PaperTrade, TradeResult
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

PAGE_SIZE = 25


@dataclass(frozen=True)
class LedgerRow:
    trade_id: int
    event: str
    commence_time: datetime
    selection: str
    line: float | None
    bookmaker: str
    price: float
    stake: float
    result: TradeResult | None
    pnl: float | None
    placed_at: datetime
    p_close: float | None

    @property
    def ev(self) -> float | None:
        """EV of the price taken against the close."""
        return None if self.p_close is None else expected_value(self.p_close, self.price)

    @property
    def is_void(self) -> bool:
        return self.result is TradeResult.VOID


@dataclass(frozen=True)
class LedgerPage:
    rows: Sequence[LedgerRow]
    page: int
    pages: int
    total: int


async def load_ledger_page(
    session: AsyncSession, name: str, version: int, page: int = 1, page_size: int = PAGE_SIZE
) -> LedgerPage:
    """Page ``page`` (from 1, clamped to the last) of the version's trades, newest first.

    Only the page's events are looked up for a close.
    """
    version_trades = (
        PaperTrade.candidate_name == name,
        PaperTrade.candidate_version == version,
    )
    total = await session.scalar(
        select(func.count()).select_from(PaperTrade).where(*version_trades)
    )
    total = total or 0
    pages = max(1, -(-total // page_size))
    page = min(max(page, 1), pages)
    stmt = (
        select(PaperTrade, Event)
        .join(Event, PaperTrade.event_id == Event.id)
        .where(*version_trades)
        .order_by(PaperTrade.placed_at.desc(), PaperTrade.id.desc())  # type: ignore[union-attr]
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    found = [(trade, event) for trade, event in (await session.execute(stmt)).all()]
    p_closes = await close_probabilities(session, found)
    rows = [
        LedgerRow(
            trade_id=trade.id,  # type: ignore[arg-type]
            event=f"{event.home_team} v {event.away_team}",
            commence_time=event.commence_time,
            selection=trade.selection,
            line=trade.line,
            bookmaker=trade.bookmaker,
            price=trade.price_decimal,
            stake=trade.stake,
            result=trade.result,
            pnl=trade.pnl,
            placed_at=trade.placed_at,
            p_close=p_close,
        )
        for (trade, event), p_close in zip(found, p_closes, strict=True)
    ]
    return LedgerPage(rows, page, pages, total)
