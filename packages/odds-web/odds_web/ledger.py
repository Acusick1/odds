"""The candidate page's bet ledger: every bet of one candidate version, void ones included,
each beside its close."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from odds_analytics.candidate_report import close_probabilities, expected_value
from odds_core.bet_models import Bet, BetResult
from odds_core.models import Event
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

PAGE_SIZE = 25


@dataclass(frozen=True)
class LedgerRow:
    bet_id: int
    event: str
    commence_time: datetime
    selection: str
    line: float | None
    bookmaker: str
    price: float
    stake: float
    result: BetResult | None
    pnl: float | None
    placed_at: datetime
    p_close: float | None

    @property
    def ev(self) -> float | None:
        """EV of the price taken against the close."""
        return None if self.p_close is None else expected_value(self.p_close, self.price)

    @property
    def is_void(self) -> bool:
        return self.result is BetResult.VOID


@dataclass(frozen=True)
class LedgerPage:
    rows: Sequence[LedgerRow]
    page: int
    pages: int
    total: int


async def load_ledger_page(
    session: AsyncSession, name: str, version: int, page: int = 1, page_size: int = PAGE_SIZE
) -> LedgerPage:
    """Page ``page`` (from 1, clamped to the last) of the version's bets, newest first.

    Only the page's events are looked up for a close.
    """
    version_bets = (
        Bet.candidate_name == name,
        Bet.candidate_version == version,
    )
    total = await session.scalar(select(func.count()).select_from(Bet).where(*version_bets))
    total = total or 0
    pages = max(1, -(-total // page_size))
    page = min(max(page, 1), pages)
    stmt = (
        select(Bet, Event)
        .join(Event, Bet.event_id == Event.id)
        .where(*version_bets)
        .order_by(Bet.placed_at.desc(), Bet.id.desc())  # type: ignore[union-attr]
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    found = [(bet, event) for bet, event in (await session.execute(stmt)).all()]
    p_closes = await close_probabilities(session, found)
    rows = [
        LedgerRow(
            bet_id=bet.id,  # type: ignore[arg-type]
            event=f"{event.home_team} v {event.away_team}",
            commence_time=event.commence_time,
            selection=bet.selection,
            line=bet.line,
            bookmaker=bet.bookmaker,
            price=bet.price_decimal,
            stake=bet.stake,
            result=bet.result,
            pnl=bet.pnl,
            placed_at=bet.placed_at,
            p_close=None if p_close is None else p_close.proportional,
        )
        for (bet, event), p_close in zip(found, p_closes, strict=True)
    ]
    return LedgerPage(rows, page, pages, total)
