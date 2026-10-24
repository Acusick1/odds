"""Per-candidate paper-trading results and the checkpoint verdict of the goal page.

A candidate is scored by EV at the price taken against the Betfair Exchange close
(``p_close * price_decimal - 1``), on its first 100, 200, ... settled bets that have
a close, with realised P&L reported beside it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from odds_core.betfair_close import event_close
from odds_core.candidates import Candidate, CandidateStatus
from odds_core.market_prices import THREE_WAY_MARKET, Outcome
from odds_core.models import Event, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade, TradeResult
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.retail_ev_metrics import event_clustered_ratio

CHECKPOINT_SIZE = 100
KILL_AT_BETS = 500
PROMOTE_LEVEL = 0.99
KILL_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 0


class Verdict(StrEnum):
    NOT_YET = "not_yet"
    CONTINUE = "continue"
    PROMOTE = "promote"
    KILL = "kill"


@dataclass(frozen=True)
class ScoredTrade:
    """A settled candidate trade and the devigged close probability of its selection."""

    candidate_name: str
    candidate_version: int
    event_id: str
    stake: float
    pnl: float
    price_decimal: float
    p_close: float | None

    @property
    def ev(self) -> float | None:
        return None if self.p_close is None else self.p_close * self.price_decimal - 1


@dataclass(frozen=True)
class Checkpoint:
    """Mean EV per unit staked over the first ``n`` bets with a close."""

    n: int
    mean_ev: float
    ci95: tuple[float, float]
    ci99: tuple[float, float]


@dataclass(frozen=True)
class CandidateSummary:
    name: str
    version: int
    status: CandidateStatus | None
    settled_bets: int
    bets_with_close: int
    bets_without_close: int
    total_stake: float
    pnl: float
    checkpoint: Checkpoint | None
    verdict: Verdict

    @property
    def roi(self) -> float:
        return self.pnl / self.total_stake if self.total_stake > 0 else 0.0


def selection_close_probability(
    trade: PaperTrade, event: Event, snapshots: Sequence[OddsSnapshot]
) -> float | None:
    """Close probability of the trade's selection; None where no close is defined or exists.

    Only the 1x2 market has a close (MLB's two-way market has none).
    """
    if trade.market != THREE_WAY_MARKET:
        return None
    close = event_close(event, snapshots, trade.market)
    if close is None:
        return None
    return close.fair[Outcome(trade.selection)]


def checkpoint_of(trades: Sequence[ScoredTrade], n_boot: int, seed: int) -> Checkpoint | None:
    """Score the first ``CHECKPOINT_SIZE * k`` trades with a close, the largest such k.

    ``trades`` must be in placement order. None when fewer than one checkpoint exists.
    """
    with_close = [t for t in trades if t.ev is not None]
    n = len(with_close) // CHECKPOINT_SIZE * CHECKPOINT_SIZE
    if n == 0:
        return None
    scored = with_close[:n]
    estimate = event_clustered_ratio(
        np.array([t.event_id for t in scored]),
        np.array([t.stake * t.ev for t in scored]),  # type: ignore[operator]
        np.array([t.stake for t in scored]),
        (KILL_LEVEL, PROMOTE_LEVEL),
        n_boot,
        seed,
    )
    return Checkpoint(
        n=n,
        mean_ev=estimate.point,
        ci95=estimate.intervals[KILL_LEVEL],
        ci99=estimate.intervals[PROMOTE_LEVEL],
    )


def verdict_of(checkpoint: Checkpoint | None) -> Verdict:
    if checkpoint is None:
        return Verdict.NOT_YET
    if checkpoint.ci99[0] > 0:
        return Verdict.PROMOTE
    if checkpoint.ci95[1] <= 0 or checkpoint.n >= KILL_AT_BETS:
        return Verdict.KILL
    return Verdict.CONTINUE


def summarize_candidate(
    name: str,
    version: int,
    status: CandidateStatus | None,
    trades: Sequence[ScoredTrade],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> CandidateSummary:
    """Summarise one candidate version's settled trades, which must be in placement order."""
    checkpoint = checkpoint_of(trades, n_boot, seed)
    with_close = sum(t.p_close is not None for t in trades)
    return CandidateSummary(
        name=name,
        version=version,
        status=status,
        settled_bets=len(trades),
        bets_with_close=with_close,
        bets_without_close=len(trades) - with_close,
        total_stake=sum(t.stake for t in trades),
        pnl=sum(t.pnl for t in trades),
        checkpoint=checkpoint,
        verdict=verdict_of(checkpoint),
    )


def summarize_candidates(
    trades: Sequence[ScoredTrade],
    registry: Sequence[Candidate],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[CandidateSummary]:
    """One summary per (candidate, version) with trades, plus each registered candidate's version.

    ``trades`` must be in placement order. A version other than the registry's has no status.
    """
    by_key: dict[tuple[str, int], list[ScoredTrade]] = defaultdict(list)
    for trade in trades:
        by_key[(trade.candidate_name, trade.candidate_version)].append(trade)
    status = {(c.name, c.version): c.status for c in registry}
    for key in status:
        by_key.setdefault(key, [])
    return [
        summarize_candidate(
            name, version, status.get((name, version)), by_key[(name, version)], n_boot, seed
        )
        for name, version in sorted(by_key)
    ]


async def load_scored_trades(session: AsyncSession) -> list[ScoredTrade]:
    """Settled, non-void candidate trades in placement order, each with its close probability."""
    stmt = (
        select(PaperTrade, Event)
        .join(Event, PaperTrade.event_id == Event.id)
        .where(PaperTrade.candidate_name.is_not(None))  # type: ignore[union-attr]
        .where(PaperTrade.settled_at.is_not(None))  # type: ignore[union-attr]
        .where(PaperTrade.result != TradeResult.VOID)
        .order_by(PaperTrade.placed_at, PaperTrade.id)  # type: ignore[arg-type]
    )
    rows = (await session.execute(stmt)).all()
    event_ids = {event.id for trade, event in rows if trade.market == THREE_WAY_MARKET}
    snapshots: dict[str, list[OddsSnapshot]] = defaultdict(list)
    if event_ids:
        snapshot_rows = await session.execute(
            select(OddsSnapshot).where(OddsSnapshot.event_id.in_(event_ids))  # type: ignore[attr-defined]
        )
        for snapshot in snapshot_rows.scalars():
            snapshots[snapshot.event_id].append(snapshot)

    scored: list[ScoredTrade] = []
    for trade, event in rows:
        assert trade.candidate_name is not None and trade.candidate_version is not None
        assert trade.pnl is not None
        scored.append(
            ScoredTrade(
                candidate_name=trade.candidate_name,
                candidate_version=trade.candidate_version,
                event_id=trade.event_id,
                stake=trade.stake,
                pnl=trade.pnl,
                price_decimal=trade.price_decimal,
                p_close=selection_close_probability(trade, event, snapshots[event.id]),
            )
        )
    return scored


async def candidate_report(
    session: AsyncSession,
    registry: Sequence[Candidate],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[CandidateSummary]:
    return summarize_candidates(await load_scored_trades(session), registry, n_boot, seed)
