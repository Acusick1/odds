"""Per-candidate paper-trading results and the checkpoint verdict of the goal page.

A candidate is scored by EV at the price taken against the Betfair Exchange close
(``p_close * price_decimal - 1``), on the bets with a close in the prefix of its trades
that is settled and final, at checkpoints of 100, 200, ... bets. Realised P&L is
reported beside it.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import numpy as np
from odds_core.betfair_close import close_quotes, event_close
from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.market_prices import THREE_WAY_MARKET, Outcome
from odds_core.models import Event
from odds_core.paper_trade_models import CandidateDecision, PaperTrade, TradeResult
from odds_core.quotes import Quote
from odds_core.vocabulary import Market
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_analytics.retail_ev_metrics import event_clustered_ratio

CHECKPOINT_SIZE = 100
KILL_AT_BETS = 500
# The family-wise false-promote rate; each version's promote level is 1 - this / family size.
PROMOTE_ERROR = 0.01
KILL_LEVEL = 0.95
# The family's promote levels put the interval tails at a few resamples when this is 2000.
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 0
# The FDUK closing price can arrive days after the match, so a missing close is final only later.
CLOSE_GRACE = timedelta(days=7)
# Markets with a Betfair Exchange close; MLB's two-way market has none.
CLOSE_MARKETS: frozenset[Market] = frozenset({Market.ONE_X_TWO, Market.TOTALS})


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
    placed_at: datetime

    @property
    def settled(self) -> bool:
        return self.pnl is not None

    @property
    def ev(self) -> float | None:
        return None if self.p_close is None else self.p_close * self.price_decimal - 1


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
    market: Market, selection: str, line: float | None, event: Event, quoted: Sequence[Quote]
) -> float | None:
    """Close probability of the selection on ``line`` from the event's close quotes
    (``odds_core.betfair_close.close_quotes``), the close on that line; None where no close is
    defined (a market outside ``CLOSE_MARKETS``) or exists.
    """
    if market not in CLOSE_MARKETS:
        return None
    close = event_close(quoted, event.commence_time, market, line=line)
    if close is None:
        return None
    return close.fair[Outcome(selection)]


def promote_level(family_versions: int) -> float:
    """Confidence of the promote interval: the Bonferroni level for a family of that many versions."""
    return 1 - PROMOTE_ERROR / family_versions


# A family of candidate versions: their rule, sport and market.
Family = tuple[str, str, Market]


def family(candidate: MechanicalCandidate) -> Family:
    return (candidate.rule, candidate.sport, candidate.market)


def family_starts(
    key: Family,
    registry: Sequence[Candidate],
    decision_starts: Mapping[tuple[str, int], datetime],
    draft: tuple[str, int] | None = None,
    draft_start: datetime | None = None,
) -> list[datetime]:
    """When each version in the family ``key`` took its first look, sorted.

    A version counts from its first recorded decision in ``decision_starts``, for names whose
    registry entry is in the family: a version with no decision has taken no look, and a
    ``start_date`` can be back-dated where a decision cannot. An unregistered ``draft`` counts
    from ``draft_start``, unless it already has a decision.
    """
    members = {c.name for c in registry if isinstance(c, MechanicalCandidate) and family(c) == key}
    starts = {key: first for key, first in decision_starts.items() if key[0] in members}
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
) -> tuple[Verdict, Checkpoint | None]:
    """The first promote or kill at checkpoints 100, 200, ... 500, and the checkpoint it fired at.

    A checkpoint's family size is ``family_size`` at the placement of its nth bet, so a version
    that starts later never changes a verdict already reached. A verdict stands once reached,
    so later bets cannot reverse it. Without one, the verdict is continue with the last
    checkpoint reached, or not_yet with none.
    """
    checkpoint: Checkpoint | None = None
    for n in range(CHECKPOINT_SIZE, min(len(scored), KILL_AT_BETS) + 1, CHECKPOINT_SIZE):
        checkpoint = checkpoint_at(scored, n, family_size(scored[n - 1].placed_at), n_boot, seed)
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
    family_size: Callable[[datetime], int],
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> CandidateSummary:
    """Summarise one candidate version's non-void trades, which must be in placement order.

    ``family_size`` gives the family's started versions at a time. The summary's family size is
    the verdict checkpoint's, or the size at ``now`` before any checkpoint.
    """
    with_close, prefix_length = scoreable_prefix(trades, now)
    verdict, checkpoint = judge(with_close, family_size, n_boot, seed)
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
        family_versions=checkpoint.family_versions if checkpoint else family_size(now),
    )


def summarize_candidates(
    trades: Sequence[ScoredBet],
    registry: Sequence[Candidate],
    now: datetime,
    decision_starts: Mapping[tuple[str, int], datetime] | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[CandidateSummary]:
    """One summary per (candidate, version) with trades, plus each registered candidate's version.

    ``trades`` must be in placement order. A version other than the registry's has no status.
    Every version of a name takes the family of the registry's entry for that name; a name
    absent from the registry, and an agent, has a family of one. ``decision_starts`` is as in
    ``family_starts``.
    """
    by_key: dict[tuple[str, int], list[ScoredBet]] = defaultdict(list)
    for trade in trades:
        by_key[(trade.candidate_name, trade.candidate_version)].append(trade)
    status = {(c.name, c.version): c.status for c in registry}
    for key in status:
        by_key.setdefault(key, [])
    current = {c.name: c for c in registry}

    def family_size(name: str) -> Callable[[datetime], int]:
        entry = current.get(name)
        if not isinstance(entry, MechanicalCandidate):
            return lambda _: 1
        starts = family_starts(family(entry), registry, decision_starts or {})
        return lambda when: family_size_at(starts, when)

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
    events = {event.id: event for trade, event in rows if trade.market == THREE_WAY_MARKET}
    closes = await close_quotes(session, list(events.values()), THREE_WAY_MARKET)

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
                # A paper trade records no line: the live runner bets only markets without lines.
                p_close=selection_close_probability(
                    Market(trade.market), trade.selection, None, event, closes.get(event.id, [])
                ),
                placed_at=trade.placed_at,
            )
        )
    return scored


async def load_decision_starts(session: AsyncSession) -> dict[tuple[str, int], datetime]:
    """The first recorded decision time of every candidate version that has one."""
    name = col(CandidateDecision.candidate_name)
    version = col(CandidateDecision.candidate_version)
    rows = await session.execute(
        select(name, version, func.min(CandidateDecision.decision_time)).group_by(name, version)
    )
    return {(name, version): first for name, version, first in rows.all()}


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
        await load_decision_starts(session),
        n_boot,
        seed,
    )
