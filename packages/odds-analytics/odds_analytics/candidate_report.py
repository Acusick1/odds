"""Per-candidate bet results and the checkpoint verdict of the goal page.

A candidate is scored by EV at the price taken against the Betfair Exchange close
(``p_close * price_decimal - 1``), on the bets with a close in the prefix of its bets
that is settled and final, at checkpoints of 100, 200, ... bets. Realised P&L is
reported beside it, and so is the EV of the same bets filled at their book's price some
minutes after placement, which no verdict reads.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from odds_core.bet_models import Bet, BetResult, CandidateDecision
from odds_core.candidates import Candidate, CandidateStatus, RuleCandidate
from odds_core.close import close_grace, close_quotes, event_close
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.odds_math import calculate_profit_from_decimal, selection_won
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source
from odds_lambda.candidate_rules import BetIntent
from odds_lambda.candidate_rules.inputs import RETAIL_SOURCES
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_analytics.retail_ev_metrics import event_clustered_ratio

if TYPE_CHECKING:
    from odds_analytics.book_replays import BookReplay

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
# How long after a decision a bet placed by hand is filled, in the delayed-fill EV (#896).
FILL_DELAYS: tuple[timedelta, ...] = (timedelta(minutes=5), timedelta(minutes=10))


class Verdict(StrEnum):
    NOT_YET = "not_yet"
    CONTINUE = "continue"
    PROMOTE = "promote"
    KILL = "kill"


def expected_value(p_close: float, price_decimal: float) -> float:
    """EV per unit staked of taking ``price_decimal`` against the close probability."""
    return p_close * price_decimal - 1


class FillStatus(StrEnum):
    """What a bet's book showed some minutes after placement (``delayed_fill``)."""

    # The book quoted the selection on the bet's line.
    FILLED = "filled"
    # The book was quoted without the selection on that line, so the delayed bet is not placed.
    VOIDED = "voided"
    # No retail snapshot by kick-off quoted the book, so nothing is known.
    UNOBSERVED = "unobserved"


@dataclass(frozen=True)
class DelayedFill:
    """A bet at its book ``delay`` after placement: ``price_decimal`` is set when ``FILLED``."""

    delay: timedelta
    status: FillStatus
    price_decimal: float | None = None


@dataclass(frozen=True)
class ScoredBet:
    """A non-void candidate bet, a bet or a replayed one, with the devigged close
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
    # Which source's price the close is (``CloseProbability.source``); None without a close.
    close_source: Source | None
    # The EV the rule estimated at decision time (``BetIntent.estimated_ev``); None without one.
    estimated_ev: float | None
    # One per ``FILL_DELAYS``, for a live bet only; empty for a replayed or backtest bet.
    delayed: tuple[DelayedFill, ...] = ()

    @property
    def settled(self) -> bool:
        return self.pnl is not None

    @property
    def ev(self) -> float | None:
        return None if self.p_close is None else expected_value(self.p_close, self.price_decimal)

    def fill(self, delay: timedelta) -> DelayedFill | None:
        """The fill at ``delay``; None on a bet without one (a replayed or backtest bet)."""
        return next((f for f in self.delayed if f.delay == delay), None)

    def delayed_ev(self, delay: timedelta) -> float | None:
        """EV at the price ``delay`` after placement; None without a close or a filled price."""
        fill = self.fill(delay)
        price = None if fill is None else fill.price_decimal
        return (
            None if self.p_close is None or price is None else expected_value(self.p_close, price)
        )

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
class DelayedFillScore:
    """The bets with a close in the scoreable prefix filled ``delay`` after placement, for
    information. Of the ``bets``, ``voided`` were not quoted then and ``unobserved`` had no
    snapshot to tell. Over the filled rest, ``instant_ev`` is the stake-weighted mean EV at the
    price taken and ``mean_ev`` at the delayed price, so their gap is the price change alone;
    both None with none filled."""

    delay: timedelta
    bets: int
    voided: int
    unobserved: int
    instant_ev: float | None
    mean_ev: float | None

    @property
    def minutes(self) -> int:
        return int(self.delay.total_seconds() // 60)

    @property
    def filled(self) -> int:
        return self.bets - self.voided - self.unobserved

    @property
    def void_rate(self) -> float | None:
        """Voided bets per observed bet; None with none observed."""
        observed = self.bets - self.unobserved
        return self.voided / observed if observed else None


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
    # Over every bet with a close, as ``current`` (``estimates_of``).
    estimates: Estimates | None
    # The bets with a close filled at each of ``FILL_DELAYS`` (``delayed_fill_scores``).
    delayed_fills: tuple[DelayedFillScore, ...]

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
class EstimateGap:
    """Σ stake·(estimated EV − EV) / Σ stake over ``n`` bets with a close, and its
    event-clustered interval at ``level``."""

    n: int
    mean_gap: float
    interval: tuple[float, float]
    level: float


@dataclass(frozen=True)
class Estimates:
    """The rule's decision-time estimates over bets with a close in placement order: the
    stake-weighted mean estimated EV after each bet, beside ``RunningScore.mean_ev``, and the
    gap between estimate and EV over them all."""

    mean_estimated_ev: tuple[float, ...]
    gap: EstimateGap


@dataclass(frozen=True)
class CandidateReport:
    """Every candidate version's summary and running score, from one read of the bets.
    ``book_replays`` holds, for information only, the counterfactual book replays of each
    registered rule-candidate version that has them (``candidate_report`` with ``replays``); no
    summary, checkpoint or verdict reads them."""

    summaries: list[CandidateSummary]
    running: dict[tuple[str, int], RunningScore]
    book_replays: dict[tuple[str, int], BookReplay] = field(default_factory=dict)


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
    method (#770, for information), and the source whose price the close is; ``power`` is None
    for a Polymarket close."""

    proportional: float
    power: float | None
    source: Source


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
    return close_probability(
        event_close(quoted, event.commence_time, event.sport_key, market, line=line), selection
    )


def close_probability(close: ExchangePrice | None, selection: str) -> CloseProbability | None:
    """The selection's probability at ``close``; None without one."""
    if close is None:
        return None
    outcome = Outcome(selection)
    power = _power_fair(close)
    return CloseProbability(
        close.fair[outcome], None if power is None else power[outcome], close.source
    )


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
    leagues: frozenset[str]


def family_starts(
    candidate: RuleCandidate,
    registry: Sequence[Candidate],
    decisions: Mapping[tuple[str, int], DecisionStart],
    draft: tuple[str, int] | None = None,
    draft_start: datetime | None = None,
) -> list[datetime]:
    """When each version in ``candidate``'s family took its first look, sorted.

    A version counts from its first recorded decision in ``decisions`` when its name's registry
    entry has ``candidate``'s rule and market and the leagues the version decided with
    intersect ``candidate.leagues``: not transitive, so a pooled {EPL, Serie A} shares a family
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
        if isinstance(c, RuleCandidate) and (c.rule, c.market) == (candidate.rule, candidate.market)
    }
    starts = {
        key: d.first
        for key, d in decisions.items()
        if key[0] in members and not d.leagues.isdisjoint(candidate.leagues)
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
    """Score the first ``n`` of ``scored``, which are bets with a close in placement order,
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


def scoreable_prefix(bets: Sequence[ScoredBet], now: datetime) -> tuple[list[ScoredBet], int]:
    """The bets with a close in the settled, final prefix of ``bets``, and its length.

    ``bets`` must be in placement order. The prefix stops at the first bet that is
    unsettled, or has no close and kicked off less than its ``close_grace`` ago. Final
    no-close bets inside it are skipped.
    """
    with_close: list[ScoredBet] = []
    length = 0
    for bet in bets:
        if not bet.settled:
            break
        if bet.p_close is None and now - bet.commence_time < bet.close_grace:
            break
        if bet.p_close is not None:
            with_close.append(bet)
        length += 1
    return with_close, length


def stake_weighted_mean_ev(with_close: Sequence[ScoredBet]) -> float | None:
    """Σ stake·EV / Σ stake over ``with_close``, the checkpoint score; None without stake."""
    stake = sum(t.stake for t in with_close)
    if not stake:
        return None
    return sum(t.stake * t.ev for t in with_close if t.ev is not None) / stake


def delayed_fill_scores(with_close: Sequence[ScoredBet]) -> tuple[DelayedFillScore, ...]:
    """``with_close``, bets with a close, scored at each delay of ``FILL_DELAYS`` with
    ``expected_value`` and their close probability; empty without a bet or when one of them has
    no fills (a replayed or backtest bet)."""
    if not with_close or any(not t.delayed for t in with_close):
        return ()
    scores: list[DelayedFillScore] = []
    for delay in FILL_DELAYS:
        fills = [t.fill(delay) for t in with_close]
        statuses = [None if f is None else f.status for f in fills]
        filled = [(t, ev) for t in with_close if (ev := t.delayed_ev(delay)) is not None]
        stake = sum(t.stake for t, _ in filled)
        scores.append(
            DelayedFillScore(
                delay=delay,
                bets=len(with_close),
                voided=statuses.count(FillStatus.VOIDED),
                unobserved=statuses.count(FillStatus.UNOBSERVED),
                instant_ev=stake_weighted_mean_ev([t for t, _ in filled]),
                mean_ev=sum(t.stake * ev for t, ev in filled) / stake if stake else None,
            )
        )
    return tuple(scores)


def estimates_of(with_close: Sequence[ScoredBet], n_boot: int, seed: int) -> Estimates | None:
    """The estimates over ``with_close``, bets with a close in placement order, with the gap's
    interval at ``KILL_LEVEL`` like the checkpoint's; None without a bet or when one of them has
    no estimate."""
    estimated = [t.estimated_ev for t in with_close]
    if not with_close or any(e is None for e in estimated):
        return None
    stakes = np.array([t.stake for t in with_close])
    estimated_ev = np.array(estimated, dtype=float)
    ev = np.array([t.ev for t in with_close], dtype=float)
    gap = event_clustered_ratio(
        np.array([t.event_id for t in with_close]),
        stakes * (estimated_ev - ev),
        stakes,
        (KILL_LEVEL,),
        n_boot,
        seed,
    )
    running = np.cumsum(stakes * estimated_ev) / np.cumsum(stakes)
    return Estimates(
        tuple(float(v) for v in running),
        EstimateGap(len(with_close), gap.point, gap.intervals[KILL_LEVEL], KILL_LEVEL),
    )


def scored_intent(
    candidate_name: str,
    candidate_version: int,
    market: Market,
    event: Event,
    intent: BetIntent,
    placed_at: datetime,
    p_close: CloseProbability | None,
) -> ScoredBet:
    """The bet a rule's ``intent`` placed at ``placed_at`` on ``event``, with the close
    ``p_close``, settled on the event's final score as ``settle_bets`` settles a bet and
    unsettled until the event is final with both scores."""
    pnl = None
    if (
        event.status == EventStatus.FINAL
        and event.home_score is not None
        and event.away_score is not None
    ):
        pnl = calculate_profit_from_decimal(
            intent.stake,
            intent.price_decimal,
            won=selection_won(
                intent.outcome.value, intent.line, event.home_score, event.away_score
            ),
        )
    return ScoredBet(
        candidate_name=candidate_name,
        candidate_version=candidate_version,
        event_id=event.id,
        commence_time=event.commence_time,
        stake=intent.stake,
        pnl=pnl,
        price_decimal=intent.price_decimal,
        p_close=None if p_close is None else p_close.proportional,
        placed_at=placed_at,
        close_grace=close_grace(event.sport_key, market),
        sport_key=event.sport_key,
        selection=intent.outcome.value,
        line=intent.line,
        bookmaker=intent.bookmaker,
        p_close_power=None if p_close is None else p_close.power,
        close_source=None if p_close is None else p_close.source,
        estimated_ev=intent.estimated_ev,
    )


def league_breakdown(bets: Sequence[ScoredBet], now: datetime) -> tuple[LeagueBreakdown, ...]:
    """``bets`` split by league, in league-key order; ``bets`` in placement order."""
    with_close, _ = scoreable_prefix(bets, now)
    breakdown: list[LeagueBreakdown] = []
    for league in sorted({t.sport_key for t in bets}):
        scored = [t for t in with_close if t.sport_key == league]
        settled = [t for t in bets if t.sport_key == league and t.pnl is not None]
        breakdown.append(
            LeagueBreakdown(
                sport_key=league,
                settled_bets=len(settled),
                bets_with_close=len(scored),
                mean_ev=stake_weighted_mean_ev(scored),
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
    bets: Sequence[ScoredBet],
    now: datetime,
    family_size: Callable[[datetime], int],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> CandidateSummary:
    """Summarise one candidate version's non-void bets, which must be in placement order.

    ``family_size`` gives the family's started versions at a time. The summary's family size is
    the verdict checkpoint's, or the size at ``now`` before any checkpoint.
    """
    with_close, prefix_length = scoreable_prefix(bets, now)
    verdict, checkpoints = judge(with_close, family_size, n_boot, seed)
    checkpoint = checkpoints[-1] if checkpoints else None
    settled = [t for t in bets if t.pnl is not None]
    return CandidateSummary(
        name=name,
        version=version,
        registered=registered,
        settled_bets=len(settled),
        bets_with_close=len(with_close),
        bets_without_close=sum(t.p_close is None for t in bets[:prefix_length]),
        pending_bets=len(bets) - prefix_length,
        total_stake=sum(t.stake for t in settled),
        pnl=sum(t.pnl for t in settled if t.pnl is not None),
        checkpoint=checkpoint,
        checkpoints=tuple(checkpoints),
        current=current_estimate(with_close, family_size(now), checkpoints, n_boot, seed),
        verdict=verdict,
        family_versions=checkpoint.family_versions if checkpoint else family_size(now),
        leagues=league_breakdown(bets, now),
        estimates=estimates_of(with_close, n_boot, seed),
        delayed_fills=delayed_fill_scores(with_close),
    )


def unscored_version(
    name: str, version: int, now: datetime
) -> tuple[CandidateSummary, RunningScore]:
    """The summary and running score of a version with no bets and no registry entry, one the
    report omits: its figures are all empty and its family one."""
    return summarize_candidate(name, version, None, [], now, lambda _: 1), running_score([])


def _by_version(
    bets: Sequence[ScoredBet], registry: Sequence[Candidate]
) -> dict[tuple[str, int], list[ScoredBet]]:
    """``bets`` per (name, version), with every registered version present."""
    by_key: dict[tuple[str, int], list[ScoredBet]] = defaultdict(list)
    for bet in bets:
        by_key[(bet.candidate_name, bet.candidate_version)].append(bet)
    for c in registry:
        by_key.setdefault((c.name, c.version), [])
    return dict(sorted(by_key.items()))


def summarize_candidates(
    bets: Sequence[ScoredBet],
    registry: Sequence[Candidate],
    now: datetime,
    decision_starts: Mapping[tuple[str, int], DecisionStart] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[CandidateSummary]:
    """One summary per (candidate, version) with bets, plus each registered candidate's version.

    ``bets`` must be in placement order. A version other than the registry's has no registry entry.
    A version takes the family of the leagues it decided with (``decisions``), or of the
    registry's entry for its name when it has no decision; a name absent from the registry, and
    an agent, has a family of one. ``decisions`` is as in ``family_starts``.
    """
    decisions = decision_starts or {}
    registered = {(c.name, c.version): c for c in registry}
    current = {c.name: c for c in registry}

    def family_size(name: str, version: int) -> Callable[[datetime], int]:
        entry = current.get(name)
        if not isinstance(entry, RuleCandidate):
            return lambda _: 1
        recorded = decisions.get((name, version))
        if recorded is not None:
            entry = entry.model_copy(update={"leagues": tuple(recorded.leagues)})
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
        for (name, version), versioned in _by_version(bets, registry).items()
    ]


def report_candidates(
    bets: Sequence[ScoredBet],
    registry: Sequence[Candidate],
    now: datetime,
    decision_starts: Mapping[tuple[str, int], DecisionStart] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> CandidateReport:
    """``summarize_candidates`` of ``bets``, with each version's running score over the same
    scoreable prefix at ``now``."""
    return CandidateReport(
        summaries=summarize_candidates(bets, registry, now, decision_starts, n_boot, seed),
        running={
            key: running_score(scoreable_prefix(versioned, now)[0])
            for key, versioned in _by_version(bets, registry).items()
        },
    )


async def close_probabilities(
    session: AsyncSession, rows: Sequence[tuple[Bet, Event]]
) -> list[CloseProbability | None]:
    """The close probability of each bet's selection on its line, in ``rows`` order; None
    where no close is defined or exists (``selection_close_probability``)."""
    events: dict[str, dict[int, Event]] = defaultdict(dict)
    for bet, event in rows:
        events[bet.market][event.id] = event
    closes = {
        market: await close_quotes(session, list(by_id.values()), market)
        for market, by_id in events.items()
    }
    return [
        selection_close_probability(
            Market(bet.market),
            bet.selection,
            bet.line,
            event,
            closes.get(bet.market, {}).get(event.id, []),
        )
        for bet, event in rows
    ]


class RetailStamp(NamedTuple):
    """A retail snapshot's id, source and time."""

    snapshot_id: int
    source: Source
    snapshot_time: datetime


def delayed_snapshots(
    stamps: Sequence[RetailStamp], after: datetime, kickoff: datetime
) -> list[RetailStamp]:
    """Each source's first of ``stamps`` (an event's retail snapshots) stamped at or
    after ``after`` and by ``kickoff``, earliest first."""
    first: dict[Source, RetailStamp] = {}
    for stamp in sorted(stamps, key=lambda s: (s.snapshot_time, s.snapshot_id)):
        if after <= stamp.snapshot_time <= kickoff:
            first.setdefault(stamp.source, stamp)
    return list(first.values())


def delayed_fill(
    delay: timedelta,
    snapshots: Sequence[RetailStamp],
    prices: Sequence[Quote],
    bookmaker: str,
    selection: str,
    line: float | None,
) -> DelayedFill:
    """The fill at ``delay`` from ``snapshots`` (``delayed_snapshots``, earliest first) and their
    back ``prices``: filled at ``bookmaker``'s price on ``selection`` and ``line`` in the earliest
    snapshot that quotes it, voided when one quotes the book but none that selection, and
    unobserved when none quotes the book.

    A source's first snapshot rather than the book's next quote, so a withdrawn selection voids
    the bet; a source without the book, such as OddsPortal after its last pre-match scrape, says
    nothing about it.
    """
    quoted = False
    for stamp in snapshots:
        for q in prices:
            if q.snapshot_id != stamp.snapshot_id or q.bookmaker != bookmaker:
                continue
            quoted = True
            if q.outcome == selection and q.line == line:
                return DelayedFill(delay, FillStatus.FILLED, q.price)
    return DelayedFill(delay, FillStatus.VOIDED if quoted else FillStatus.UNOBSERVED)


async def retail_stamps(
    session: AsyncSession, windows: Mapping[tuple[int, str], tuple[datetime, datetime]]
) -> dict[tuple[int, str], list[RetailStamp]]:
    """The retail snapshots of each (event id, market) stamped within its window, which ends at
    kick-off,
    inclusive."""
    found: dict[tuple[int, str], list[RetailStamp]] = defaultdict(list)
    if not windows:
        return found
    rows = await session.execute(
        select(
            col(OddsSnapshot.id),
            col(OddsSnapshot.event_id),
            col(OddsSnapshot.market),
            col(OddsSnapshot.source),
            col(OddsSnapshot.snapshot_time),
        ).where(
            col(OddsSnapshot.source).in_(RETAIL_SOURCES),
            or_(
                *(
                    and_(
                        col(OddsSnapshot.event_id) == event_id,
                        col(OddsSnapshot.market) == market,
                        col(OddsSnapshot.snapshot_time).between(since, until),
                    )
                    for (event_id, market), (since, until) in windows.items()
                )
            ),
        )
    )
    for snapshot_id, event_id, market, source, snapshot_time in rows.all():
        found[(event_id, market)].append(RetailStamp(snapshot_id, Source(source), snapshot_time))
    return found


async def delayed_fills(
    session: AsyncSession, rows: Sequence[tuple[Bet, Event]]
) -> list[tuple[DelayedFill, ...]]:
    """Each bet's fill at each of ``FILL_DELAYS``, in ``rows`` order: its book's price in the
    retail snapshots ``delayed_snapshots`` picks from that long after placement
    (``delayed_fill``)."""
    first_placed: dict[tuple[int, str], datetime] = {}
    kickoffs: dict[int, datetime] = {}
    for bet, event in rows:
        key = (event.id, bet.market)
        first_placed[key] = min(bet.placed_at, first_placed.get(key, bet.placed_at))
        kickoffs[event.id] = event.commence_time
    windows = {
        key: (placed + min(FILL_DELAYS), kickoffs[key[0]]) for key, placed in first_placed.items()
    }
    stamps = await retail_stamps(session, windows)
    picked = [
        [
            delayed_snapshots(
                stamps.get((event.id, bet.market), []), bet.placed_at + d, event.commence_time
            )
            for d in FILL_DELAYS
        ]
        for bet, event in rows
    ]
    by_market: dict[str, set[int]] = defaultdict(set)
    for (bet, _), per_delay in zip(rows, picked, strict=True):
        by_market[bet.market].update(s.snapshot_id for snapshots in per_delay for s in snapshots)
    prices: dict[int, list[Quote]] = defaultdict(list)
    for market, ids in by_market.items():
        if not ids:
            continue
        for q in await quotes(
            session,
            {e.id for b, e in rows if b.market == market},
            Market(market),
            bookmakers={b.bookmaker for b, _ in rows if b.market == market},
            sides=[Side.BACK],
            snapshot_ids=ids,
        ):
            prices[q.snapshot_id].append(q)
    return [
        tuple(
            delayed_fill(
                delay,
                snapshots,
                [q for s in snapshots for q in prices[s.snapshot_id]],
                bet.bookmaker,
                bet.selection,
                bet.line,
            )
            for delay, snapshots in zip(FILL_DELAYS, per_delay, strict=True)
        )
        for (bet, _), per_delay in zip(rows, picked, strict=True)
    ]


async def load_scored_bets(session: AsyncSession) -> list[ScoredBet]:
    """Non-void candidate bets in placement order, each with its close probability and its
    delayed fills."""
    stmt = (
        select(Bet, Event)
        .join(Event, Bet.event_id == Event.id)
        .where(Bet.candidate_name.is_not(None))  # type: ignore[union-attr]
        .where(or_(Bet.result.is_(None), Bet.result != BetResult.VOID))  # type: ignore[union-attr]
        .order_by(Bet.placed_at, Bet.id)  # type: ignore[arg-type]
    )
    rows = [(bet, event) for bet, event in (await session.execute(stmt)).all()]
    p_closes = await close_probabilities(session, rows)
    fills = await delayed_fills(session, rows)

    scored: list[ScoredBet] = []
    for (bet, event), p_close, delayed in zip(rows, p_closes, fills, strict=True):
        assert bet.candidate_name is not None and bet.candidate_version is not None
        scored.append(
            ScoredBet(
                candidate_name=bet.candidate_name,
                candidate_version=bet.candidate_version,
                event_id=bet.event_id,
                commence_time=event.commence_time,
                stake=bet.stake,
                pnl=bet.pnl,
                price_decimal=bet.price_decimal,
                p_close=None if p_close is None else p_close.proportional,
                placed_at=bet.placed_at,
                close_grace=close_grace(event.sport_key, bet.market),
                sport_key=event.sport_key,
                selection=bet.selection,
                line=bet.line,
                bookmaker=bet.bookmaker,
                p_close_power=None if p_close is None else p_close.power,
                close_source=None if p_close is None else p_close.source,
                estimated_ev=bet.estimated_ev,
                delayed=delayed,
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
        func.unnest(CandidateDecision.leagues).label("sport"),
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
        (name, version): DecisionStart(first, frozenset(leagues))
        for name, version, first, leagues in rows.all()
    }


async def candidate_report(
    session: AsyncSession,
    registry: Sequence[Candidate],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    now: datetime | None = None,
    executor: Executor | None = None,
    replays: bool = False,
) -> CandidateReport:
    """Every candidate version's summary and running score, from one read of the bets and
    decision starts.

    The bootstrap runs on ``executor`` (the loop's default one when None), so a server can hold
    it to a single thread. With ``replays``, each registered rule-candidate version's book replays
    are added (``odds_analytics.book_replays``), which read every recorded decision's inputs.
    """
    # book_replays scores with this module's functions, so it cannot be imported above.
    from odds_analytics.book_replays import book_replays

    bets = await load_scored_bets(session)
    decision_starts = await load_decision_starts(session)
    now = now or datetime.now(UTC)
    report = await asyncio.get_running_loop().run_in_executor(
        executor,
        partial(report_candidates, bets, registry, now, decision_starts, n_boot, seed),
    )
    if not replays:
        return report
    return replace(
        report,
        book_replays=await book_replays(
            session, [s.registered for s in report.summaries], bets, now
        ),
    )
