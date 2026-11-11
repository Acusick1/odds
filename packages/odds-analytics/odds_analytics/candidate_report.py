"""Per-candidate paper-trading results and the checkpoint verdict of the goal page.

A candidate is scored by EV at the price taken against the Betfair Exchange close
(``p_close * price_decimal - 1``), on the bets with a close in the prefix of its trades
that is settled and final, at checkpoints of 100, 200, ... bets. Realised P&L is
reported beside it.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from typing import NamedTuple

import numpy as np
from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.close import close_grace, close_quotes, event_close
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.models import Event
from odds_core.paper_trade_models import CandidateDecision, PaperTrade, TradeResult
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Source
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_analytics.retail_ev_metrics import event_clustered_ratio

CHECKPOINT_SIZE = 100
KILL_AT_BETS = 500
# The family-wise false-promote rate; each version's promote level is 1 - this / family size.
PROMOTE_ERROR = 0.01
KILL_LEVEL = 0.95
# Fewer events than this cannot give the current estimate an interval worth drawing.
MIN_CURRENT_EVENTS = 2
# The family's promote levels put the interval tails at a few resamples when this is 2000.
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0


class Verdict(StrEnum):
    NOT_YET = "not_yet"
    CONTINUE = "continue"
    PROMOTE = "promote"
    KILL = "kill"


def expected_value(p_close: float, price_decimal: float) -> float:
    """EV per unit staked of taking ``price_decimal`` against the close probability."""
    return p_close * price_decimal - 1


@dataclass(frozen=True)
class ScoredBet:
    """A non-void candidate bet, a paper trade or a replayed one, with the devigged close
    probability of its selection: ``p_close`` devigged proportionally, the score, and
    ``p_close_power`` by the power method, for information (``CloseProbability``)."""

    candidate_name: str
    candidate_version: int
    event_id: int
    commence_time: datetime
    stake: float
    pnl: float | None
    price_decimal: float
    p_close: float | None
    placed_at: datetime
    # How long after kick-off a missing close may still arrive (``odds_core.close.close_grace``).
    close_grace: timedelta
    sport_key: str
    selection: str
    line: float | None
    bookmaker: str
    p_close_power: float | None

    @property
    def settled(self) -> bool:
        return self.pnl is not None

    @property
    def ev(self) -> float | None:
        return None if self.p_close is None else expected_value(self.p_close, self.price_decimal)

    @property
    def ev_power(self) -> float | None:
        if self.p_close_power is None:
            return None
        return expected_value(self.p_close_power, self.price_decimal)


@dataclass(frozen=True)
class Checkpoint:
    """Mean EV per unit staked over the first ``n`` bets with a close, with the promote
    interval at the level of the ``family_versions`` started when the nth bet was placed."""

    n: int
    mean_ev: float
    ci95: tuple[float, float]
    family_versions: int
    promote_level: float
    promote_ci: tuple[float, float]


@dataclass(frozen=True)
class LeagueBreakdown:
    """One league's bets in a version's series, for information only: the verdict is the pooled
    series' and this never changes it. ``mean_ev`` is over the league's bets with a close in the
    scoreable prefix, ``settled_bets`` and ``pnl`` over all its settled bets."""

    sport_key: str
    settled_bets: int
    bets_with_close: int
    mean_ev: float | None
    pnl: float


@dataclass(frozen=True)
class CandidateSummary:
    """``registered`` is the registry's entry at exactly this version, None for a version the
    registry no longer holds: a version bump overwrites the file, so an old version's kind,
    sport and market are unknown and the current entry's are not its own.

    ``checkpoint`` is the one the verdict fired at, else the last one reached, and
    ``checkpoints`` every one evaluated up to it; each is frozen once reached.

    ``current`` is a separate estimate over all ``bets_with_close`` at the family size now. It
    moves with every bet and carries no verdict. ``leagues`` splits the bets by league, for
    information.
    """

    name: str
    version: int
    registered: Candidate | None
    settled_bets: int
    bets_with_close: int
    bets_without_close: int
    pending_bets: int
    total_stake: float
    pnl: float
    checkpoint: Checkpoint | None
    checkpoints: tuple[Checkpoint, ...]
    current: Checkpoint | None
    verdict: Verdict
    family_versions: int
    leagues: tuple[LeagueBreakdown, ...]

    @property
    def status(self) -> CandidateStatus | None:
        """The registry status of this version, apart from the statistical ``verdict``."""
        return None if self.registered is None else self.registered.status

    @property
    def roi(self) -> float:
        return self.pnl / self.total_stake if self.total_stake > 0 else 0.0


@dataclass(frozen=True)
class RunningScore:
    """Running totals after each bet with a close in the scoreable prefix, in placement order:
    the stake, Σ stake·EV and realised P&L so far."""

    staked: tuple[float, ...]
    expected: tuple[float, ...]
    pnl: tuple[float, ...]

    @property
    def mean_ev(self) -> tuple[float, ...]:
        """The stake-weighted mean EV after each bet, the checkpoint score at every n."""
        return tuple(e / s for e, s in zip(self.expected, self.staked, strict=True))


@dataclass(frozen=True)
class CandidateReport:
    """Every candidate version's summary and running score, from one read of the trades."""

    summaries: list[CandidateSummary]
    running: dict[tuple[str, int], RunningScore]


def power_devig(implied: Sequence[float]) -> tuple[float, ...]:
    """The power devig of implied probabilities: each raised to the one power ``k`` that makes
    them sum to 1, found by bisection. ``ValueError`` unless each is in (0, 1)."""
    if not implied or any(not 0 < q < 1 for q in implied):
        raise ValueError(f"implied probabilities must be in (0, 1), got {list(implied)}")
    low, high = 0.0, 1.0
    while sum(q**high for q in implied) > 1:
        low, high = high, high * 2
    for _ in range(100):
        k = (low + high) / 2
        if sum(q**k for q in implied) > 1:
            low = k
        else:
            high = k
    k = (low + high) / 2
    return tuple(q**k for q in implied)


class CloseProbability(NamedTuple):
    """A selection's close probability devigged proportionally (the score) and by the power
    method (#770, for information); ``power`` is None for a Polymarket close."""

    proportional: float
    power: float | None


def _power_fair(close: ExchangePrice) -> dict[Outcome, float] | None:
    # Polymarket's overround is its asks', not that of the mids it devigs, so its implied
    # probabilities cannot be recovered from ``fair`` and ``overround``.
    if close.source is Source.POLYMARKET:
        return None
    outcomes = list(close.fair)
    implied = power_devig([close.fair[o] * (1 + close.overround) for o in outcomes])
    return dict(zip(outcomes, implied, strict=True))


def selection_close_probability(
    market: Market, selection: str, line: float | None, event: Event, quoted: Sequence[Quote]
) -> CloseProbability | None:
    """Close probability of the selection on ``line`` from the event's close quotes
    (``odds_core.close.close_quotes``), the close on that line; None where no close is
    defined (``odds_core.close.CLOSES``) or exists.
    """
    close = event_close(quoted, event.commence_time, event.sport_key, market, line=line)
    if close is None:
        return None
    outcome = Outcome(selection)
    power = _power_fair(close)
    return CloseProbability(close.fair[outcome], None if power is None else power[outcome])


@dataclass(frozen=True)
class CheckpointProgress:
    """``bets_with_close`` against the next checkpoint, and how far it is from the one before:
    0 just after a checkpoint, 1 at the next."""

    bets_with_close: int
    next_checkpoint: int
    fraction: float


def next_checkpoint(bets_with_close: int) -> int:
    """The first checkpoint with more bets with a close than ``bets_with_close``, capped at
    ``KILL_AT_BETS``."""
    return min(KILL_AT_BETS, (bets_with_close // CHECKPOINT_SIZE + 1) * CHECKPOINT_SIZE)


def checkpoint_progress(bets_with_close: int) -> CheckpointProgress:
    target = next_checkpoint(bets_with_close)
    since = (bets_with_close - (target - CHECKPOINT_SIZE)) / CHECKPOINT_SIZE
    return CheckpointProgress(bets_with_close, target, max(0.0, min(1.0, since)))


def checkpoints_through_next(bets_with_close: int) -> list[int]:
    """Every checkpoint up to the next one."""
    return list(range(CHECKPOINT_SIZE, next_checkpoint(bets_with_close) + 1, CHECKPOINT_SIZE))


def promote_level(family_versions: int) -> float:
    """Confidence of the promote interval: the Bonferroni level for a family of that many versions."""
    return 1 - PROMOTE_ERROR / family_versions


class DecisionStart(NamedTuple):
    """A candidate version's first recorded decision and the leagues it listed when deciding."""

    first: datetime
    sports: frozenset[str]


def family_starts(
    candidate: MechanicalCandidate,
    registry: Sequence[Candidate],
    decisions: Mapping[tuple[str, int], DecisionStart],
    draft: tuple[str, int] | None = None,
    draft_start: datetime | None = None,
) -> list[datetime]:
    """When each version in ``candidate``'s family took its first look, sorted.

    A version counts from its first recorded decision in ``decisions`` when its name's registry
    entry has ``candidate``'s rule and market and the leagues the version decided with
    intersect ``candidate.sports``: not transitive, so a pooled {EPL, Serie A} shares a family
    with EPL alone and with Serie A alone, which do not share one with each other. A version
    with no decision has taken no look, and a ``start_date`` can be back-dated where a decision
    cannot. An unregistered ``draft`` counts from ``draft_start``, unless it already has a
    decision.
    """
    # A name's rule and market never change across versions (the version check refuses it), so
    # its current entry gives every historical version's rule and market.
    members = {
        c.name
        for c in registry
        if isinstance(c, MechanicalCandidate)
        and (c.rule, c.market) == (candidate.rule, candidate.market)
    }
    starts = {
        key: d.first
        for key, d in decisions.items()
        if key[0] in members and not d.sports.isdisjoint(candidate.sports)
    }
    if draft is not None and draft_start is not None:
        starts.setdefault(draft, draft_start)
    return sorted(starts.values())


def family_size_at(starts: Sequence[datetime], when: datetime) -> int:
    """Versions started by ``when`` in a family with the sorted ``starts``; at least 1."""
    return max(1, bisect_right(starts, when))


def checkpoint_at(
    scored: Sequence[ScoredBet], n: int, family_versions: int, n_boot: int, seed: int
) -> Checkpoint:
    """Score the first ``n`` of ``scored``, which are trades with a close in placement order,
    with the promote interval at ``promote_level(family_versions)``."""
    head = scored[:n]
    level = promote_level(family_versions)
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
        family_versions=family_versions,
        promote_level=level,
        promote_ci=estimate.intervals[level],
    )


def judge(
    scored: Sequence[ScoredBet],
    family_size: Callable[[datetime], int],
    n_boot: int,
    seed: int,
) -> tuple[Verdict, list[Checkpoint]]:
    """The first promote or kill at checkpoints 100, 200, ... 500, and the checkpoints evaluated
    up to it, the last being the one it fired at.

    A checkpoint's family size is ``family_size`` at the placement of its nth bet, so a version
    that starts later never changes a verdict already reached. A verdict stands once reached,
    so later bets cannot reverse it. Without one, the verdict is continue with the last
    checkpoint reached, or not_yet with none.
    """
    checkpoints: list[Checkpoint] = []
    for n in range(CHECKPOINT_SIZE, min(len(scored), KILL_AT_BETS) + 1, CHECKPOINT_SIZE):
        checkpoint = checkpoint_at(scored, n, family_size(scored[n - 1].placed_at), n_boot, seed)
        checkpoints.append(checkpoint)
        if checkpoint.promote_ci[0] > 0:
            return Verdict.PROMOTE, checkpoints
        if checkpoint.ci95[1] <= 0 or n >= KILL_AT_BETS:
            return Verdict.KILL, checkpoints
    return (Verdict.CONTINUE if checkpoints else Verdict.NOT_YET), checkpoints


def current_estimate(
    with_close: Sequence[ScoredBet],
    family_versions: int,
    reached: Sequence[Checkpoint],
    n_boot: int,
    seed: int,
) -> Checkpoint | None:
    """The score of every bet with a close at ``family_versions``, None below
    ``MIN_CURRENT_EVENTS`` events, where an interval says nothing.

    The last of ``reached`` is returned when it is the same score, so a version exactly at a
    checkpoint resamples once.
    """
    if len({t.event_id for t in with_close}) < MIN_CURRENT_EVENTS:
        return None
    last = reached[-1] if reached else None
    if last is not None and (last.n, last.family_versions) == (len(with_close), family_versions):
        return last
    return checkpoint_at(with_close, len(with_close), family_versions, n_boot, seed)


def scoreable_prefix(trades: Sequence[ScoredBet], now: datetime) -> tuple[list[ScoredBet], int]:
    """The bets with a close in the settled, final prefix of ``trades``, and its length.

    ``trades`` must be in placement order. The prefix stops at the first trade that is
    unsettled, or has no close and kicked off less than its ``close_grace`` ago. Final
    no-close trades inside it are skipped.
    """
    with_close: list[ScoredBet] = []
    length = 0
    for trade in trades:
        if not trade.settled:
            break
        if trade.p_close is None and now - trade.commence_time < trade.close_grace:
            break
        if trade.p_close is not None:
            with_close.append(trade)
        length += 1
    return with_close, length


def league_breakdown(trades: Sequence[ScoredBet], now: datetime) -> tuple[LeagueBreakdown, ...]:
    """``trades`` split by league, in league-key order; ``trades`` in placement order."""
    with_close, _ = scoreable_prefix(trades, now)
    breakdown: list[LeagueBreakdown] = []
    for league in sorted({t.sport_key for t in trades}):
        scored = [t for t in with_close if t.sport_key == league]
        stake = sum(t.stake for t in scored)
        settled = [t for t in trades if t.sport_key == league and t.pnl is not None]
        breakdown.append(
            LeagueBreakdown(
                sport_key=league,
                settled_bets=len(settled),
                bets_with_close=len(scored),
                mean_ev=sum(t.stake * t.ev for t in scored if t.ev is not None) / stake
                if scored
                else None,
                pnl=sum(t.pnl for t in settled if t.pnl is not None),
            )
        )
    return tuple(breakdown)


def running_score(with_close: Sequence[ScoredBet]) -> RunningScore:
    """The running totals over ``with_close``, bets with a close in placement order."""
    staked = expected = pnl = 0.0
    staked_after: list[float] = []
    expected_after: list[float] = []
    pnl_after: list[float] = []
    for bet in with_close:
        assert bet.ev is not None and bet.pnl is not None
        staked += bet.stake
        expected += bet.stake * bet.ev
        pnl += bet.pnl
        staked_after.append(staked)
        expected_after.append(expected)
        pnl_after.append(pnl)
    return RunningScore(tuple(staked_after), tuple(expected_after), tuple(pnl_after))


def summarize_candidate(
    name: str,
    version: int,
    registered: Candidate | None,
    trades: Sequence[ScoredBet],
    now: datetime,
    family_size: Callable[[datetime], int],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> CandidateSummary:
    """Summarise one candidate version's non-void trades, which must be in placement order.

    ``family_size`` gives the family's started versions at a time. The summary's family size is
    the verdict checkpoint's, or the size at ``now`` before any checkpoint.
    """
    with_close, prefix_length = scoreable_prefix(trades, now)
    verdict, checkpoints = judge(with_close, family_size, n_boot, seed)
    checkpoint = checkpoints[-1] if checkpoints else None
    settled = [t for t in trades if t.pnl is not None]
    return CandidateSummary(
        name=name,
        version=version,
        registered=registered,
        settled_bets=len(settled),
        bets_with_close=len(with_close),
        bets_without_close=sum(t.p_close is None for t in trades[:prefix_length]),
        pending_bets=len(trades) - prefix_length,
        total_stake=sum(t.stake for t in settled),
        pnl=sum(t.pnl for t in settled if t.pnl is not None),
        checkpoint=checkpoint,
        checkpoints=tuple(checkpoints),
        current=current_estimate(with_close, family_size(now), checkpoints, n_boot, seed),
        verdict=verdict,
        family_versions=checkpoint.family_versions if checkpoint else family_size(now),
        leagues=league_breakdown(trades, now),
    )


def unscored_version(
    name: str, version: int, now: datetime
) -> tuple[CandidateSummary, RunningScore]:
    """The summary and running score of a version with no trades and no registry entry, one the
    report omits: its figures are all empty and its family one."""
    return summarize_candidate(name, version, None, [], now, lambda _: 1), running_score([])


def _by_version(
    trades: Sequence[ScoredBet], registry: Sequence[Candidate]
) -> dict[tuple[str, int], list[ScoredBet]]:
    """``trades`` per (name, version), with every registered version present."""
    by_key: dict[tuple[str, int], list[ScoredBet]] = defaultdict(list)
    for trade in trades:
        by_key[(trade.candidate_name, trade.candidate_version)].append(trade)
    for c in registry:
        by_key.setdefault((c.name, c.version), [])
    return dict(sorted(by_key.items()))


def summarize_candidates(
    trades: Sequence[ScoredBet],
    registry: Sequence[Candidate],
    now: datetime,
    decision_starts: Mapping[tuple[str, int], DecisionStart] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[CandidateSummary]:
    """One summary per (candidate, version) with trades, plus each registered candidate's version.

    ``trades`` must be in placement order. A version other than the registry's has no registry entry.
    A version takes the family of the leagues it decided with (``decisions``), or of the
    registry's entry for its name when it has no decision; a name absent from the registry, and
    an agent, has a family of one. ``decisions`` is as in ``family_starts``.
    """
    decisions = decision_starts or {}
    registered = {(c.name, c.version): c for c in registry}
    current = {c.name: c for c in registry}

    def family_size(name: str, version: int) -> Callable[[datetime], int]:
        entry = current.get(name)
        if not isinstance(entry, MechanicalCandidate):
            return lambda _: 1
        recorded = decisions.get((name, version))
        if recorded is not None:
            entry = entry.model_copy(update={"sports": tuple(recorded.sports)})
        starts = family_starts(entry, registry, decisions)
        return lambda when: family_size_at(starts, when)

    return [
        summarize_candidate(
            name,
            version,
            registered.get((name, version)),
            versioned,
            now,
            family_size(name, version),
            n_boot,
            seed,
        )
        for (name, version), versioned in _by_version(trades, registry).items()
    ]


def report_candidates(
    trades: Sequence[ScoredBet],
    registry: Sequence[Candidate],
    now: datetime,
    decision_starts: Mapping[tuple[str, int], DecisionStart] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> CandidateReport:
    """``summarize_candidates`` of ``trades``, with each version's running score over the same
    scoreable prefix at ``now``."""
    return CandidateReport(
        summaries=summarize_candidates(trades, registry, now, decision_starts, n_boot, seed),
        running={
            key: running_score(scoreable_prefix(versioned, now)[0])
            for key, versioned in _by_version(trades, registry).items()
        },
    )


async def close_probabilities(
    session: AsyncSession, rows: Sequence[tuple[PaperTrade, Event]]
) -> list[CloseProbability | None]:
    """The close probability of each trade's selection on its line, in ``rows`` order; None
    where no close is defined or exists (``selection_close_probability``)."""
    events: dict[str, dict[int, Event]] = defaultdict(dict)
    for trade, event in rows:
        events[trade.market][event.id] = event
    closes = {
        market: await close_quotes(session, list(by_id.values()), market)
        for market, by_id in events.items()
    }
    return [
        selection_close_probability(
            Market(trade.market),
            trade.selection,
            trade.line,
            event,
            closes.get(trade.market, {}).get(event.id, []),
        )
        for trade, event in rows
    ]


async def load_scored_trades(session: AsyncSession) -> list[ScoredBet]:
    """Non-void candidate trades in placement order, each with its close probability."""
    stmt = (
        select(PaperTrade, Event)
        .join(Event, PaperTrade.event_id == Event.id)
        .where(PaperTrade.candidate_name.is_not(None))  # type: ignore[union-attr]
        .where(or_(PaperTrade.result.is_(None), PaperTrade.result != TradeResult.VOID))  # type: ignore[union-attr]
        .order_by(PaperTrade.placed_at, PaperTrade.id)  # type: ignore[arg-type]
    )
    rows = [(trade, event) for trade, event in (await session.execute(stmt)).all()]
    p_closes = await close_probabilities(session, rows)

    scored: list[ScoredBet] = []
    for (trade, event), p_close in zip(rows, p_closes, strict=True):
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
                p_close=None if p_close is None else p_close.proportional,
                placed_at=trade.placed_at,
                close_grace=close_grace(event.sport_key, trade.market),
                sport_key=event.sport_key,
                selection=trade.selection,
                line=trade.line,
                bookmaker=trade.bookmaker,
                p_close_power=None if p_close is None else p_close.power,
            )
        )
    return scored


async def load_decision_starts(session: AsyncSession) -> dict[tuple[str, int], DecisionStart]:
    """The first recorded decision time of every candidate version that has one, with the
    leagues it listed. A version's league set is the same on every decision it took."""
    name = col(CandidateDecision.candidate_name)
    version = col(CandidateDecision.candidate_version)
    unnested = select(
        name.label("name"),
        version.label("version"),
        col(CandidateDecision.decision_time).label("decision_time"),
        func.unnest(CandidateDecision.sports).label("sport"),
    ).subquery()
    rows = await session.execute(
        select(
            unnested.c.name,
            unnested.c.version,
            func.min(unnested.c.decision_time),
            func.array_agg(func.distinct(unnested.c.sport)),
        ).group_by(unnested.c.name, unnested.c.version)
    )
    return {
        (name, version): DecisionStart(first, frozenset(sports))
        for name, version, first, sports in rows.all()
    }


async def candidate_report(
    session: AsyncSession,
    registry: Sequence[Candidate],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    now: datetime | None = None,
    executor: Executor | None = None,
) -> CandidateReport:
    """Every candidate version's summary and running score, from one read of the trades and
    decision starts.

    The bootstrap runs on ``executor`` (the loop's default one when None), so a server can hold
    it to a single thread.
    """
    trades = await load_scored_trades(session)
    decision_starts = await load_decision_starts(session)
    return await asyncio.get_running_loop().run_in_executor(
        executor,
        partial(
            report_candidates,
            trades,
            registry,
            now or datetime.now(UTC),
            decision_starts,
            n_boot,
            seed,
        ),
    )
