"""Paper trade settlement and portfolio query service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from odds_analytics.utils import calculate_profit_from_odds, determine_h2h_winner
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import PaperTrade, TradeResult
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class PortfolioSummary:
    """Snapshot of paper trading portfolio state."""

    current_bankroll: float
    total_trades: int
    settled_trades: int
    open_trades: int
    total_pnl: float
    total_staked: float
    roi: float
    win_count: int
    loss_count: int
    push_count: int


async def place_trade(
    session: AsyncSession,
    *,
    event_id: str,
    market: str,
    selection: str,
    bookmaker: str,
    odds: int,
    stake: float,
    bankroll: float,
    reasoning: str | None = None,
    confidence: float | None = None,
) -> PaperTrade:
    """Place a new paper trade. Validates stake against bankroll."""
    if stake <= 0:
        raise ValueError("Stake must be positive")
    if stake > bankroll:
        raise ValueError(f"Stake {stake} exceeds available bankroll {bankroll}")
    if selection not in ("home", "draw", "away"):
        raise ValueError(f"Invalid selection '{selection}', must be home/draw/away")

    trade = PaperTrade(
        event_id=event_id,
        market=market,
        selection=selection,
        bookmaker=bookmaker,
        odds=odds,
        stake=stake,
        bankroll_before=bankroll,
        reasoning=reasoning,
        confidence=confidence,
    )
    session.add(trade)
    await session.flush()
    return trade


async def settle_trades(session: AsyncSession) -> list[PaperTrade]:
    """Settle all unsettled trades whose events have final scores. Idempotent."""
    # Fetch unsettled trades joined with their events
    stmt = (
        select(PaperTrade, Event)
        .join(Event, PaperTrade.event_id == Event.id)
        .where(PaperTrade.settled_at.is_(None))  # type: ignore[union-attr]
        .where(Event.status == EventStatus.FINAL)
        .where(Event.home_score.is_not(None))  # type: ignore[union-attr]
        .where(Event.away_score.is_not(None))  # type: ignore[union-attr]
    )
    rows = (await session.execute(stmt)).all()

    settled: list[PaperTrade] = []
    now = datetime.now(UTC)

    for trade, event in rows:
        assert event.home_score is not None
        assert event.away_score is not None

        winner = determine_h2h_winner(event.home_score, event.away_score)
        result = TradeResult.WIN if trade.selection == winner else TradeResult.LOSS
        pnl = (
            calculate_profit_from_odds(trade.stake, trade.odds, won=(result == TradeResult.WIN))
            if result in (TradeResult.WIN, TradeResult.LOSS)
            else 0.0
        )

        trade.result = result
        trade.pnl = pnl
        trade.bankroll_after = trade.bankroll_before + pnl
        trade.settled_at = now

        session.add(trade)
        settled.append(trade)

    await session.flush()
    return settled


async def get_open_trades(session: AsyncSession) -> list[PaperTrade]:
    """Return all unsettled paper trades."""
    stmt = (
        select(PaperTrade)
        .where(PaperTrade.settled_at.is_(None))  # type: ignore[union-attr]
        .order_by(PaperTrade.placed_at.desc())  # type: ignore[union-attr]
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_settled_trades(
    session: AsyncSession,
    *,
    limit: int = 50,
) -> list[PaperTrade]:
    """Return settled paper trades, most recent first."""
    stmt = (
        select(PaperTrade)
        .where(PaperTrade.settled_at.is_not(None))  # type: ignore[union-attr]
        .order_by(PaperTrade.settled_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_portfolio_summary(
    session: AsyncSession,
    *,
    initial_bankroll: float = 1000.0,
) -> PortfolioSummary:
    """Compute aggregate portfolio metrics."""
    # Total counts
    total_stmt = select(func.count()).select_from(PaperTrade)
    total_trades = (await session.execute(total_stmt)).scalar_one()

    # Settled aggregates
    settled_stmt = select(
        func.count(),
        func.coalesce(func.sum(PaperTrade.pnl), 0.0),
        func.coalesce(func.sum(PaperTrade.stake), 0.0),
    ).where(PaperTrade.settled_at.is_not(None))  # type: ignore[union-attr]
    settled_row = (await session.execute(settled_stmt)).one()
    settled_count: int = settled_row[0]
    total_pnl: float = float(settled_row[1])
    total_staked: float = float(settled_row[2])

    # Open trade exposure
    open_stmt = (
        select(func.count(), func.coalesce(func.sum(PaperTrade.stake), 0.0)).where(
            PaperTrade.settled_at.is_(None)
        )  # type: ignore[union-attr]
    )
    open_row = (await session.execute(open_stmt)).one()
    open_count: int = open_row[0]
    open_exposure: float = float(open_row[1])

    # Win/loss/push counts
    win_stmt = (
        select(func.count()).select_from(PaperTrade).where(PaperTrade.result == TradeResult.WIN)
    )
    loss_stmt = (
        select(func.count()).select_from(PaperTrade).where(PaperTrade.result == TradeResult.LOSS)
    )
    push_stmt = (
        select(func.count()).select_from(PaperTrade).where(PaperTrade.result == TradeResult.PUSH)
    )

    win_count = (await session.execute(win_stmt)).scalar_one()
    loss_count = (await session.execute(loss_stmt)).scalar_one()
    push_count = (await session.execute(push_stmt)).scalar_one()

    current_bankroll = initial_bankroll + total_pnl - open_exposure
    roi = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

    return PortfolioSummary(
        current_bankroll=current_bankroll,
        total_trades=total_trades,
        settled_trades=settled_count,
        open_trades=open_count,
        total_pnl=total_pnl,
        total_staked=total_staked,
        roi=roi,
        win_count=win_count,
        loss_count=loss_count,
        push_count=push_count,
    )


async def get_exposure_by_event(session: AsyncSession) -> list[tuple[str, float]]:
    """Return total open stake per event."""
    stmt = (
        select(PaperTrade.event_id, func.sum(PaperTrade.stake))
        .where(PaperTrade.settled_at.is_(None))  # type: ignore[union-attr]
        .group_by(PaperTrade.event_id)
        .order_by(func.sum(PaperTrade.stake).desc())
    )
    rows = (await session.execute(stmt)).all()
    return [(row[0], float(row[1])) for row in rows]
