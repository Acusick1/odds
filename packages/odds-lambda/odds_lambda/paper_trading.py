"""Paper trade settlement and portfolio query service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from odds_core.candidates import CandidateStatus, candidates_dir, load_candidates
from odds_core.git_info import current_git_sha
from odds_core.market_prices import THREE_WAY_MARKET, TWO_WAY_MARKET
from odds_core.models import Event, EventStatus
from odds_core.odds_math import calculate_profit_from_decimal, determine_h2h_winner
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_core.utils import is_three_way_sport
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

SETTLEABLE_MARKETS = frozenset({TWO_WAY_MARKET, THREE_WAY_MARKET})
VOIDING_STATUSES = (EventStatus.CANCELLED,)


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


# Flat-stake candidate trades record no bankroll and stay out of the agent's bankroll maths.
BANKROLL_SIZED = PaperTrade.bankroll_before.is_not(None)  # type: ignore[union-attr]


def _check_candidate(name: str, version: int, event: Event, market: str) -> None:
    """Raise ValueError unless the registry's candidate may place this bet today at this version."""
    candidate = next((c for c in load_candidates(candidates_dir()) if c.name == name), None)
    if candidate is None:
        raise ValueError(f"Candidate '{name}' is not in the registry")
    if candidate.status != CandidateStatus.ACTIVE:
        raise ValueError(f"Candidate '{name}' is {candidate.status}, not active")
    today = datetime.now(UTC).date()
    if candidate.start_date > today:
        raise ValueError(f"Candidate '{name}' starts on {candidate.start_date}, after {today}")
    if candidate.version != version:
        raise ValueError(f"Candidate '{name}' is at version {candidate.version}, not {version}")
    if event.sport_key != candidate.sport:
        raise ValueError(
            f"Candidate '{name}' is for {candidate.sport}, not {event.sport_key} ({event.id})"
        )
    if market == TWO_WAY_MARKET and is_three_way_sport(event.sport_key):
        raise ValueError(f"{event.sport_key} trades use '{THREE_WAY_MARKET}', not '{market}'")


async def place_trade(
    session: AsyncSession,
    *,
    event_id: str,
    market: str,
    selection: str,
    bookmaker: str,
    price_decimal: float,
    stake: float,
    bankroll: float | None,
    reasoning: str | None = None,
    confidence: float | None = None,
    candidate_name: str | None = None,
    candidate_version: int | None = None,
    decision_id: int | None = None,
) -> PaperTrade:
    """Place a new paper trade. Validates stake against bankroll and the candidate.

    ``bankroll=None`` places a flat-stake candidate trade, which records no bankroll.
    """
    if price_decimal <= 1.0:
        raise ValueError(f"Decimal price must exceed 1.0, got {price_decimal}")
    if stake <= 0:
        raise ValueError("Stake must be positive")
    if bankroll is None:
        if candidate_name is None:
            raise ValueError("A trade without a candidate needs a bankroll")
    elif stake > bankroll:
        raise ValueError(f"Stake {stake} exceeds available bankroll {bankroll}")
    if market not in SETTLEABLE_MARKETS:
        raise ValueError(
            f"Unsupported market '{market}', must be one of {sorted(SETTLEABLE_MARKETS)}"
        )
    if selection not in ("home", "draw", "away"):
        raise ValueError(f"Invalid selection '{selection}', must be home/draw/away")
    if selection == "draw" and market == TWO_WAY_MARKET:
        event = await session.get(Event, event_id)
        if event is None:
            raise ValueError(f"Event '{event_id}' not found")
        if not is_three_way_sport(event.sport_key):
            raise ValueError(f"Invalid selection 'draw': {event.sport_key} has no draw")

    if (candidate_name is None) != (candidate_version is None):
        raise ValueError("candidate_name and candidate_version are set together or not at all")
    if candidate_name is not None and candidate_version is not None:
        candidate_event = await session.get(Event, event_id)
        if candidate_event is None:
            raise ValueError(f"Event '{event_id}' not found")
        _check_candidate(candidate_name, candidate_version, candidate_event, market)

    trade = PaperTrade(
        event_id=event_id,
        market=market,
        selection=selection,
        bookmaker=bookmaker,
        price_decimal=price_decimal,
        stake=stake,
        bankroll_before=bankroll,
        reasoning=reasoning,
        confidence=confidence,
        candidate_name=candidate_name,
        candidate_version=candidate_version,
        decision_id=decision_id,
        git_sha=current_git_sha(),
    )
    session.add(trade)
    await session.flush()
    return trade


async def settle_trades(session: AsyncSession) -> list[PaperTrade]:
    """Settle unsettled trades on FINAL events and void those on cancelled ones.

    A FINAL event settles only with both scores. Idempotent.
    """
    played = (Event.status == EventStatus.FINAL) & (
        Event.home_score.is_not(None) & Event.away_score.is_not(None)  # type: ignore[union-attr]
    )
    stmt = (
        select(PaperTrade, Event)
        .join(Event, PaperTrade.event_id == Event.id)
        .where(PaperTrade.settled_at.is_(None))  # type: ignore[union-attr]
        .where(played | Event.status.in_(VOIDING_STATUSES))  # type: ignore[attr-defined]
    )
    rows = (await session.execute(stmt)).all()

    settled: list[PaperTrade] = []
    now = datetime.now(UTC)

    for trade, event in rows:
        if event.status in VOIDING_STATUSES:
            result = TradeResult.VOID
        elif trade.market in SETTLEABLE_MARKETS:
            assert event.home_score is not None
            assert event.away_score is not None
            winner = determine_h2h_winner(event.home_score, event.away_score)
            result = TradeResult.WIN if trade.selection == winner else TradeResult.LOSS
        else:
            logger.error("paper_trade_unsupported_market", trade_id=trade.id, market=trade.market)
            continue

        pnl = (
            calculate_profit_from_decimal(
                trade.stake, trade.price_decimal, won=(result == TradeResult.WIN)
            )
            if result in (TradeResult.WIN, TradeResult.LOSS)
            else 0.0
        )

        trade.result = result
        trade.pnl = pnl
        if trade.bankroll_before is not None:
            trade.bankroll_after = trade.bankroll_before + pnl
        trade.settled_at = now

        session.add(trade)
        settled.append(trade)

    await session.flush()
    return settled


async def get_open_trades(
    session: AsyncSession, *, include_flat_stake: bool = True
) -> list[PaperTrade]:
    """Return all unsettled paper trades, optionally only those sized from a bankroll."""
    stmt = select(PaperTrade).where(PaperTrade.settled_at.is_(None))  # type: ignore[union-attr]
    if not include_flat_stake:
        stmt = stmt.where(BANKROLL_SIZED)
    result = await session.execute(stmt.order_by(PaperTrade.placed_at.desc()))  # type: ignore[union-attr]
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
    """Compute aggregate portfolio metrics over the trades sized from a bankroll."""
    # Total counts
    total_stmt = select(func.count()).select_from(PaperTrade).where(BANKROLL_SIZED)
    total_trades = (await session.execute(total_stmt)).scalar_one()

    # Settled aggregates
    settled_stmt = select(
        func.count(),
        func.coalesce(func.sum(PaperTrade.pnl), 0.0),
        func.coalesce(func.sum(PaperTrade.stake), 0.0),
    ).where(PaperTrade.settled_at.is_not(None), BANKROLL_SIZED)  # type: ignore[union-attr]
    settled_row = (await session.execute(settled_stmt)).one()
    settled_count: int = settled_row[0]
    total_pnl: float = float(settled_row[1])
    total_staked: float = float(settled_row[2])

    # Open trade exposure
    open_stmt = (
        select(func.count(), func.coalesce(func.sum(PaperTrade.stake), 0.0)).where(
            PaperTrade.settled_at.is_(None), BANKROLL_SIZED
        )  # type: ignore[union-attr]
    )
    open_row = (await session.execute(open_stmt)).one()
    open_count: int = open_row[0]
    open_exposure: float = float(open_row[1])

    # Win/loss/push counts
    win_stmt = (
        select(func.count())
        .select_from(PaperTrade)
        .where(PaperTrade.result == TradeResult.WIN, BANKROLL_SIZED)
    )
    loss_stmt = (
        select(func.count())
        .select_from(PaperTrade)
        .where(PaperTrade.result == TradeResult.LOSS, BANKROLL_SIZED)
    )
    push_stmt = (
        select(func.count())
        .select_from(PaperTrade)
        .where(PaperTrade.result == TradeResult.PUSH, BANKROLL_SIZED)
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
    """Return total open stake per event over the trades sized from a bankroll."""
    stmt = (
        select(PaperTrade.event_id, func.sum(PaperTrade.stake))
        .where(PaperTrade.settled_at.is_(None), BANKROLL_SIZED)  # type: ignore[union-attr]
        .group_by(PaperTrade.event_id)
        .order_by(func.sum(PaperTrade.stake).desc())
    )
    rows = (await session.execute(stmt)).all()
    return [(row[0], float(row[1])) for row in rows]
