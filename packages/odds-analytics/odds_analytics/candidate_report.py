"""Per-candidate paper-trading results and the checkpoint verdict of the goal page.

A candidate is scored by EV at the price taken against the Betfair Exchange close
(``p_close * price_decimal - 1``), on the bets with a close in the prefix of its trades
that is settled and final, at checkpoints of 100, 200, ... bets. Realised P&L is
reported beside it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

import numpy as np
from odds_core.betfair_close import LIVE_CLOSE_WINDOW, event_close
from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.market_prices import THREE_WAY_MARKET, Outcome
from odds_core.models import Event, OddsSnapshot
from odds_core.paper_trade_models import CandidateDecision, PaperTrade, TradeResult
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotSource
from sqlalchemy import Select, and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.retail_ev_metrics import event_clustered_ratio

CHECKPOINT_SIZE = 100
KILL_AT_BETS = 500
# The family-wise false-promote rate; each version's promote level is 1 - this / family size.
PROMOTE_ERROR = 0.01
KILL_LEVEL = 0.95
# A family's promote level reaches 99.9%, whose tail is a few resamples at 2000.
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
# The FDUK closing price can arrive days after the match, so a missing close is final only later.
CLOSE_GRACE = timedelta(days=7)


class Verdict(StrEnum):
    NOT_YET = "not_yet"
    CONTINUE = "continue"
    PROMOTE = "promote"
    KILL = "kill"


@dataclass(frozen=True)
class ScoredBet:
    """A non-void candidate bet, a paper trade or a replayed one, with the devigged close
    probability of its selection."""

    candidate_name: str
    candidate_version: int
    event_id: str
    commence_time: datetime
    stake: float
    pnl: float | None
    price_decimal: float
    p_close: float | None

    @property
    def settled(self) -> bool:
        return self.pnl is not None

    @property
    def ev(self) -> float | None:
        return None if self.p_close is None else self.p_close * self.price_decimal - 1


@dataclass(frozen=True)
class Checkpoint:
    """Mean EV per unit staked over the first ``n`` bets with a close."""

    n: int
    mean_ev: float
    ci95: tuple[float, float]
    promote_level: float
    promote_ci: tuple[float, float]


@dataclass(frozen=True)
class CandidateSummary:
    """``checkpoint`` is the one the verdict fired at, else the last one reached."""

    name: str
    version: int
    status: CandidateStatus | None
    settled_bets: int
    bets_with_close: int
    bets_without_close: int
    pending_bets: int
    total_stake: float
    pnl: float
    checkpoint: Checkpoint | None
    verdict: Verdict
    family_versions: int

    @property
    def roi(self) -> float:
        return self.pnl / self.total_stake if self.total_stake > 0 else 0.0


def selection_close_probability(
    market: str, selection: str, event: Event, snapshots: Sequence[OddsSnapshot]
) -> float | None:
    """Close probability of the selection; None where no close is defined or exists.

    Only the 1x2 market has a close (MLB's two-way market has none).
    """
    if market != THREE_WAY_MARKET:
        return None
    close = event_close(event, snapshots, market)
    if close is None:
        return None
    return close.fair[Outcome(selection)]


def promote_level(family_versions: int) -> float:
    """Confidence of the promote interval: the Bonferroni level for a family of that many versions."""
    return 1 - PROMOTE_ERROR / family_versions


def family_versions(
    rule: str,
    sport: str,
    registry: Sequence[Candidate],
    decided: Collection[tuple[str, int]],
    today: date,
    draft: tuple[str, int] | None = None,
) -> int:
    """Versions in the (rule, sport) family that have started, killed ones included.

    The registry holds each candidate's current version, counted once its start date has
    passed. Earlier versions are counted from ``decided``, the (name, version) pairs that
    have a recorded decision, for the names whose registry entry is in the family. ``draft``
    is an unregistered version being backtested, which counts as one more. A family with
    no started version has size 1, the base bar.
    """
    members = {
        c.name
        for c in registry
        if isinstance(c, MechanicalCandidate) and c.rule == rule and c.sport == sport
    }
    versions = {
        (c.name, c.version) for c in registry if c.name in members and c.start_date <= today
    }
    versions |= {key for key in decided if key[0] in members}
    if draft is not None:
        versions.add(draft)
    return max(1, len(versions))


def checkpoint_at(
    scored: Sequence[ScoredBet], n: int, level: float, n_boot: int, seed: int
) -> Checkpoint:
    """Score the first ``n`` of ``scored``, which are trades with a close in placement order,
    with a promote interval at ``level``."""
    head = scored[:n]
    estimate = event_clustered_ratio(
        np.array([t.event_id for t in head]),
        np.array([t.stake * t.ev for t in head]),  # type: ignore[operator]
        np.array([t.stake for t in head]),
        (KILL_LEVEL, level),
        n_boot,
        seed,
    )
    return Checkpoint(
        n=n,
        mean_ev=estimate.point,
        ci95=estimate.intervals[KILL_LEVEL],
        promote_level=level,
        promote_ci=estimate.intervals[level],
    )


def judge(
    scored: Sequence[ScoredBet], level: float, n_boot: int, seed: int
) -> tuple[Verdict, Checkpoint | None]:
    """The first promote or kill at checkpoints 100, 200, ... 500, and the checkpoint it fired at.

    A verdict stands once reached, so later bets cannot reverse it. Without one, the verdict
    is continue with the last checkpoint reached, or not_yet with none.
    """
    checkpoint: Checkpoint | None = None
    for n in range(CHECKPOINT_SIZE, min(len(scored), KILL_AT_BETS) + 1, CHECKPOINT_SIZE):
        checkpoint = checkpoint_at(scored, n, level, n_boot, seed)
        if checkpoint.promote_ci[0] > 0:
            return Verdict.PROMOTE, checkpoint
        if checkpoint.ci95[1] <= 0 or n >= KILL_AT_BETS:
            return Verdict.KILL, checkpoint
    return (Verdict.NOT_YET if checkpoint is None else Verdict.CONTINUE), checkpoint


def scoreable_prefix(trades: Sequence[ScoredBet], now: datetime) -> tuple[list[ScoredBet], int]:
    """The bets with a close in the settled, final prefix of ``trades``, and its length.

    ``trades`` must be in placement order. The prefix stops at the first trade that is
    unsettled, or has no close and kicked off less than ``CLOSE_GRACE`` ago. Final
    no-close trades inside it are skipped.
    """
    with_close: list[ScoredBet] = []
    length = 0
    for trade in trades:
        if not trade.settled:
            break
        if trade.p_close is None and now - trade.commence_time < CLOSE_GRACE:
            break
        if trade.p_close is not None:
            with_close.append(trade)
        length += 1
    return with_close, length


def summarize_candidate(
    name: str,
    version: int,
    status: CandidateStatus | None,
    trades: Sequence[ScoredBet],
    now: datetime,
    family_versions: int,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> CandidateSummary:
    """Summarise one candidate version's non-void trades, which must be in placement order.

    The promote bar is at ``promote_level(family_versions)``.
    """
    with_close, prefix_length = scoreable_prefix(trades, now)
    verdict, checkpoint = judge(with_close, promote_level(family_versions), n_boot, seed)
    settled = [t for t in trades if t.pnl is not None]
    return CandidateSummary(
        name=name,
        version=version,
        status=status,
        settled_bets=len(settled),
        bets_with_close=len(with_close),
        bets_without_close=sum(t.p_close is None for t in trades[:prefix_length]),
        pending_bets=len(trades) - prefix_length,
        total_stake=sum(t.stake for t in settled),
        pnl=sum(t.pnl for t in settled if t.pnl is not None),
        checkpoint=checkpoint,
        verdict=verdict,
        family_versions=family_versions,
    )


def summarize_candidates(
    trades: Sequence[ScoredBet],
    registry: Sequence[Candidate],
    now: datetime,
    decided: Collection[tuple[str, int]] = (),
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[CandidateSummary]:
    """One summary per (candidate, version) with trades, plus each registered candidate's version.

    ``trades`` must be in placement order. A version other than the registry's has no status.
    Every version of a name takes the family size of the registry's entry for that name; a name
    absent from the registry, and an agent, has a family of one. ``decided`` is as in
    ``family_versions``.
    """
    by_key: dict[tuple[str, int], list[ScoredBet]] = defaultdict(list)
    for trade in trades:
        by_key[(trade.candidate_name, trade.candidate_version)].append(trade)
    status = {(c.name, c.version): c.status for c in registry}
    for key in status:
        by_key.setdefault(key, [])
    current = {c.name: c for c in registry}

    def family_size(name: str) -> int:
        entry = current.get(name)
        if not isinstance(entry, MechanicalCandidate):
            return 1
        return family_versions(entry.rule, entry.sport, registry, decided, now.date())

    return [
        summarize_candidate(
            name,
            version,
            status.get((name, version)),
            by_key[(name, version)],
            now,
            family_size(name),
            n_boot,
            seed,
        )
        for name, version in sorted(by_key)
    ]


def close_candidate_snapshots(event_ids: set[str]) -> Select[tuple[OddsSnapshot]]:
    """The snapshots of ``event_ids`` that ``event_close`` can pick.

    Those are live Betfair prices in the window before kick-off and FDUK closing prices.
    """
    in_window = and_(
        OddsSnapshot.snapshot_time >= Event.commence_time - LIVE_CLOSE_WINDOW,  # type: ignore[operator]
        OddsSnapshot.snapshot_time <= Event.commence_time,  # type: ignore[operator]
    )
    fduk_closing = and_(
        OddsSnapshot.raw_data["source"].as_string() == SnapshotSource.FOOTBALL_DATA_UK.value,  # type: ignore[index]
        OddsSnapshot.raw_data["timing"]["basis"].as_string() == FdukTimeBasis.KICKOFF.value,  # type: ignore[index]
    )
    return (
        select(OddsSnapshot)
        .join(Event, OddsSnapshot.event_id == Event.id)  # type: ignore[arg-type]
        .where(OddsSnapshot.event_id.in_(event_ids))  # type: ignore[attr-defined]
        .where(or_(in_window, fduk_closing))
    )


async def load_scored_trades(session: AsyncSession) -> list[ScoredBet]:
    """Non-void candidate trades in placement order, each with its close probability."""
    stmt = (
        select(PaperTrade, Event)
        .join(Event, PaperTrade.event_id == Event.id)
        .where(PaperTrade.candidate_name.is_not(None))  # type: ignore[union-attr]
        .where(or_(PaperTrade.result.is_(None), PaperTrade.result != TradeResult.VOID))  # type: ignore[union-attr]
        .order_by(PaperTrade.placed_at, PaperTrade.id)  # type: ignore[arg-type]
    )
    rows = (await session.execute(stmt)).all()
    event_ids = {event.id for trade, event in rows if trade.market == THREE_WAY_MARKET}
    snapshots: dict[str, list[OddsSnapshot]] = defaultdict(list)
    if event_ids:
        snapshot_rows = await session.execute(close_candidate_snapshots(event_ids))
        for snapshot in snapshot_rows.scalars():
            snapshots[snapshot.event_id].append(snapshot)

    scored: list[ScoredBet] = []
    for trade, event in rows:
        assert trade.candidate_name is not None and trade.candidate_version is not None
        scored.append(
            ScoredBet(
                candidate_name=trade.candidate_name,
                candidate_version=trade.candidate_version,
                event_id=trade.event_id,
                commence_time=event.commence_time,
                stake=trade.stake,
                pnl=trade.pnl,
                price_decimal=trade.price_decimal,
                p_close=selection_close_probability(
                    trade.market, trade.selection, event, snapshots[event.id]
                ),
            )
        )
    return scored


async def load_decided_versions(session: AsyncSession) -> set[tuple[str, int]]:
    """The (name, version) of every mechanical candidate version with a recorded decision."""
    rows = await session.execute(
        select(CandidateDecision.candidate_name, CandidateDecision.candidate_version).distinct()
    )
    return {(name, version) for name, version in rows.all()}


async def candidate_report(
    session: AsyncSession,
    registry: Sequence[Candidate],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    now: datetime | None = None,
) -> list[CandidateSummary]:
    return summarize_candidates(
        await load_scored_trades(session),
        registry,
        now or datetime.now(UTC),
        await load_decided_versions(session),
        n_boot,
        seed,
    )
