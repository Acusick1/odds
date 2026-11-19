"""Bet settlement and portfolio query service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from odds_core.bet_models import Bet, BetResult, SharpReading
from odds_core.candidates import CandidateStatus, candidates_dir, load_candidates
from odds_core.git_info import current_git_sha
from odds_core.market_prices import (
    THREE_WAY_MARKET,
    TOTALS_MARKET,
    TWO_WAY_MARKET,
    market_outcomes,
)
from odds_core.models import Event, EventStatus
from odds_core.odds_math import (
    calculate_profit_from_decimal,
    decimal_to_implied_probability,
    selection_won,
)
from odds_core.utils import is_three_way_sport
from odds_core.vocabulary import Market
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.candidate_rules import is_bettable_line

logger = structlog.get_logger()

SETTLEABLE_MARKETS = frozenset({TWO_WAY_MARKET, THREE_WAY_MARKET, TOTALS_MARKET})
VOIDING_STATUSES = (EventStatus.CANCELLED,)


@dataclass(frozen=True)
class PortfolioSummary:
    """Snapshot of betting portfolio state."""

    current_bankroll: float
    total_bets: int
    settled_bets: int
    open_bets: int
    total_pnl: float
    total_staked: float
    roi: float
    win_count: int
    loss_count: int
    push_count: int


# Flat-stake candidate bets record no bankroll and stay out of the agent's bankroll maths.
BANKROLL_SIZED = Bet.bankroll_before.is_not(None)  # type: ignore[union-attr]


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
    if event.sport_key not in candidate.leagues:
        raise ValueError(
            f"Candidate '{name}' is for {', '.join(candidate.leagues)}, not {event.sport_key} "
            f"({event.id})"
        )
    if market == TWO_WAY_MARKET and is_three_way_sport(event.sport_key):
        raise ValueError(f"{event.sport_key} bets use '{THREE_WAY_MARKET}', not '{market}'")


async def place_bet(
    session: AsyncSession,
    *,
    event_id: int,
    market: str,
    selection: str,
    bookmaker: str,
    line: float | None = None,
    price_decimal: float,
    stake: float,
    bankroll: float | None,
    reasoning: str | None = None,
    confidence: float | None = None,
    candidate_name: str | None = None,
    candidate_version: int | None = None,
    decision_id: int | None = None,
    sharp: SharpReading | None = None,
    estimated_ev: float | None = None,
) -> Bet:
    """Place a new bet. Validates stake against bankroll and the candidate.

    ``bankroll=None`` places a flat-stake candidate bet, which records no bankroll. ``line``
    is the selection's line, a half line in a market with lines and None otherwise. ``sharp``
    is the sharp market's reading of the selection at placement, if one was taken and found.
    ``estimated_ev`` is the EV a candidate's rule estimated for the bet at decision time.
    """
    if price_decimal <= 1.0:
        raise ValueError(f"Decimal price must exceed 1.0, got {price_decimal}")
    if stake <= 0:
        raise ValueError("Stake must be positive")
    if bankroll is None:
        if candidate_name is None:
            raise ValueError("A bet without a candidate needs a bankroll")
    elif stake > bankroll:
        raise ValueError(f"Stake {stake} exceeds available bankroll {bankroll}")
    if market not in SETTLEABLE_MARKETS:
        raise ValueError(
            f"Unsupported market '{market}', must be one of {sorted(SETTLEABLE_MARKETS)}"
        )
    # An h2h market of a three-way sport also takes a draw, checked below.
    outcomes = {o.value for o in market_outcomes(market)} | (
        {"draw"} if market == TWO_WAY_MARKET else set()
    )
    if selection not in outcomes:
        raise ValueError(
            f"Invalid selection '{selection}' for market '{market}', must be one of "
            f"{sorted(outcomes)}"
        )
    if not is_bettable_line(Market(market), line):
        raise ValueError(
            f"Invalid line {line} for market '{market}': a half line in a market with lines, "
            "else none"
        )
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

    bet = Bet(
        event_id=event_id,
        market=market,
        selection=selection,
        line=line,
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
        estimated_ev=estimated_ev,
    )
    if sharp is not None:
        bet.sharp_probability = sharp.probability
        bet.sharp_bookmaker = sharp.bookmaker
        bet.sharp_snapshot_time = sharp.snapshot_time
        bet.sharp_spread = decimal_to_implied_probability(price_decimal) - sharp.probability
    session.add(bet)
    await session.flush()
    return bet


async def settle_bets(session: AsyncSession) -> list[Bet]:
    """Settle unsettled bets on FINAL events and void those on cancelled ones.

    A FINAL event settles only with both scores, and a totals bet only on a half line.
    Idempotent.
    """
    played = (Event.status == EventStatus.FINAL) & (
        Event.home_score.is_not(None) & Event.away_score.is_not(None)  # type: ignore[union-attr]
    )
    stmt = (
        select(Bet, Event)
        .join(Event, Bet.event_id == Event.id)
        .where(Bet.settled_at.is_(None))  # type: ignore[union-attr]
        .where(played | Event.status.in_(VOIDING_STATUSES))  # type: ignore[attr-defined]
    )
    rows = (await session.execute(stmt)).all()

    settled: list[Bet] = []
    now = datetime.now(UTC)

    for bet, event in rows:
        if event.status in VOIDING_STATUSES:
            result = BetResult.VOID
        elif bet.market in SETTLEABLE_MARKETS:
            assert event.home_score is not None
            assert event.away_score is not None
            try:
                won = selection_won(bet.selection, bet.line, event.home_score, event.away_score)
            except ValueError:
                logger.error(
                    "bet_unsettleable_line",
                    bet_id=bet.id,
                    market=bet.market,
                    selection=bet.selection,
                    line=bet.line,
                )
                continue
            result = BetResult.WIN if won else BetResult.LOSS
        else:
            logger.error("bet_unsupported_market", bet_id=bet.id, market=bet.market)
            continue

        pnl = (
            calculate_profit_from_decimal(
                bet.stake, bet.price_decimal, won=(result == BetResult.WIN)
            )
            if result in (BetResult.WIN, BetResult.LOSS)
            else 0.0
        )

        bet.result = result
        bet.pnl = pnl
        if bet.bankroll_before is not None:
            bet.bankroll_after = bet.bankroll_before + pnl
        bet.settled_at = now

        session.add(bet)
        settled.append(bet)

    await session.flush()
    return settled


async def get_open_bets(session: AsyncSession, *, include_flat_stake: bool = True) -> list[Bet]:
    """Return all unsettled bets, optionally only those sized from a bankroll."""
    stmt = select(Bet).where(Bet.settled_at.is_(None))  # type: ignore[union-attr]
    if not include_flat_stake:
        stmt = stmt.where(BANKROLL_SIZED)
    result = await session.execute(stmt.order_by(Bet.placed_at.desc()))  # type: ignore[union-attr]
    return list(result.scalars().all())


async def get_settled_bets(
    session: AsyncSession,
    *,
    limit: int = 50,
) -> list[Bet]:
    """Return settled bets, most recent first."""
    stmt = (
        select(Bet)
        .where(Bet.settled_at.is_not(None))  # type: ignore[union-attr]
        .order_by(Bet.settled_at.desc())  # type: ignore[union-attr]
        .limit(limit)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_portfolio_summary(
    session: AsyncSession,
    *,
    initial_bankroll: float = 1000.0,
) -> PortfolioSummary:
    """Compute aggregate portfolio metrics over the bets sized from a bankroll."""
    # Total counts
    total_stmt = select(func.count()).select_from(Bet).where(BANKROLL_SIZED)
    total_bets = (await session.execute(total_stmt)).scalar_one()

    # Settled aggregates
    settled_stmt = select(
        func.count(),
        func.coalesce(func.sum(Bet.pnl), 0.0),
        func.coalesce(func.sum(Bet.stake), 0.0),
    ).where(Bet.settled_at.is_not(None), BANKROLL_SIZED)  # type: ignore[union-attr]
    settled_row = (await session.execute(settled_stmt)).one()
    settled_count: int = settled_row[0]
    total_pnl: float = float(settled_row[1])
    total_staked: float = float(settled_row[2])

    # Open bet exposure
    open_stmt = (
        select(func.count(), func.coalesce(func.sum(Bet.stake), 0.0)).where(
            Bet.settled_at.is_(None), BANKROLL_SIZED
        )  # type: ignore[union-attr]
    )
    open_row = (await session.execute(open_stmt)).one()
    open_count: int = open_row[0]
    open_exposure: float = float(open_row[1])

    # Win/loss/push counts
    win_stmt = (
        select(func.count()).select_from(Bet).where(Bet.result == BetResult.WIN, BANKROLL_SIZED)
    )
    loss_stmt = (
        select(func.count()).select_from(Bet).where(Bet.result == BetResult.LOSS, BANKROLL_SIZED)
    )
    push_stmt = (
        select(func.count()).select_from(Bet).where(Bet.result == BetResult.PUSH, BANKROLL_SIZED)
    )

    win_count = (await session.execute(win_stmt)).scalar_one()
    loss_count = (await session.execute(loss_stmt)).scalar_one()
    push_count = (await session.execute(push_stmt)).scalar_one()

    current_bankroll = initial_bankroll + total_pnl - open_exposure
    roi = (total_pnl / total_staked * 100.0) if total_staked > 0 else 0.0

    return PortfolioSummary(
        current_bankroll=current_bankroll,
        total_bets=total_bets,
        settled_bets=settled_count,
        open_bets=open_count,
        total_pnl=total_pnl,
        total_staked=total_staked,
        roi=roi,
        win_count=win_count,
        loss_count=loss_count,
        push_count=push_count,
    )


async def get_exposure_by_event(session: AsyncSession) -> list[tuple[str, float]]:
    """Return total open stake per event over the bets sized from a bankroll."""
    stmt = (
        select(Bet.event_id, func.sum(Bet.stake))
        .where(Bet.settled_at.is_(None), BANKROLL_SIZED)  # type: ignore[union-attr]
        .group_by(Bet.event_id)
        .order_by(func.sum(Bet.stake).desc())
    )
    rows = (await session.execute(stmt)).all()
    return [(row[0], float(row[1])) for row in rows]
