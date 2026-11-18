"""The candidate page's content for one candidate version.

Every figure comes from the report data: the tiles, verdict and intervals from
``odds_analytics.candidate_report.CandidateSummary`` and the series from
``odds_analytics.candidate_report.RunningScore``. This module only lays them out.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from odds_analytics.book_replays import BookReplay
from odds_analytics.candidate_report import (
    CandidateSummary,
    Checkpoint,
    CheckpointProgress,
    EstimateGap,
    Estimates,
    LeagueBreakdown,
    RunningScore,
    checkpoints_through_next,
)
from odds_core.candidates import AgentCandidate, Candidate, RuleCandidate
from odds_lambda.candidate_rules import EstimateBasis
from odds_lambda.candidate_rules.registry import estimate_basis

from odds_web.decisions import DecisionHistory
from odds_web.ledger import LedgerPage
from odds_web.portfolio import (
    is_agent,
    no_ev_reason,
    progress_of,
    scope_of,
    status_of,
    verdict_of,
)

# Running means over fewer bets than this swing too wildly to set the EV axis.
AXIS_FROM_BETS = 10
MIN_AXIS_PERCENT = 5.0

# What the estimate line's legend says each basis is.
BASIS_LABELS: dict[EstimateBasis, str] = {
    EstimateBasis.EXCHANGE: "exchange at decision",
    EstimateBasis.POLYMARKET: "Polymarket at decision",
    EstimateBasis.RETAIL_CONSENSUS: "retail consensus at decision",
    EstimateBasis.MODEL: "model prediction at decision",
}
ESTIMATE_OMITTED = "No estimate: not every bet in the series has one recorded."


def name_basis(summaries: Sequence[CandidateSummary], name: str) -> EstimateBasis | None:
    """The estimate basis of the rule of ``name``'s registered rule-candidate entry, which every
    version of a name shares; None for an agent, a rule without one or a name the registry no
    longer holds."""
    rule = next(
        (
            s.registered.rule
            for s in summaries
            if s.name == name and isinstance(s.registered, RuleCandidate)
        ),
        None,
    )
    return None if rule is None else estimate_basis(rule)


def estimate_label(basis: EstimateBasis | None) -> str:
    """The estimate line's legend: its basis when known."""
    return "Estimate" if basis is None else f"Estimate ({BASIS_LABELS[basis]})"


@dataclass(frozen=True)
class Interval:
    """The intervals of an estimate over the first ``n`` bets with a close, in percent."""

    n: int
    mean_percent: float
    ci_percent: tuple[float, float]
    promote_percent: tuple[float, float] | None
    at_checkpoint: bool
    # The confidence level of ``ci_percent``; a stored backtest's is not always the report's 95%.
    level: float = 0.95
    # What the estimate is called in the chart's tooltip, when not "checkpoint" or "current".
    label: str | None = None


@dataclass(frozen=True)
class Series:
    """Per bet with a close in the scoreable prefix, in placement order: the running mean EV in
    percent, and the cumulative Σ stake·EV and realised P&L in units of a 1.0 stake."""

    running_ev_percent: list[float]
    expected_units: list[float]
    pnl_units: list[float]
    intervals: list[Interval]
    checkpoints: list[int]
    axis_percent: float
    # The running mean estimated EV over the same bets, in percent, and its legend; None when a
    # bet has no estimate (``odds_analytics.candidate_report.estimates_of``).
    estimate_percent: list[float] | None
    estimate_label: str

    @property
    def estimate_omitted(self) -> str | None:
        """Why the series has no estimate line, None when it has one."""
        return ESTIMATE_OMITTED if self.estimate_percent is None else None

    def ev_chart(self) -> dict[str, Any]:
        """The running mean EV and estimated EV, the intervals and the checkpoint markers, for
        ``charts.js``."""
        return {
            "running": [[n, ev] for n, ev in enumerate(self.running_ev_percent, 1)],
            "estimate": None
            if self.estimate_percent is None
            else [[n, ev] for n, ev in enumerate(self.estimate_percent, 1)],
            "intervals": [
                {
                    "n": i.n,
                    "mean": i.mean_percent,
                    "ci": list(i.ci_percent),
                    "level": i.level,
                    "promote": None if i.promote_percent is None else list(i.promote_percent),
                    "checkpoint": i.at_checkpoint,
                    "label": i.label,
                }
                for i in self.intervals
            ],
            "checkpoints": self.checkpoints,
            "axis": self.axis_percent,
        }

    def cumulative_chart(self) -> dict[str, Any]:
        """The cumulative Σ stake·EV and P&L over the same bets, for ``charts.js``."""
        return {
            "expected": [[n, v] for n, v in enumerate(self.expected_units, 1)],
            "pnl": [[n, v] for n, v in enumerate(self.pnl_units, 1)],
        }


@dataclass(frozen=True)
class BookReplayView:
    """A version's counterfactual book replays, for information only, with the axis of the
    running mean EV of every book set drawn together."""

    replay: BookReplay
    axis_percent: float

    def chart(self) -> dict[str, Any]:
        """The running mean EV of each book set, for ``charts.js``."""
        return {
            "sets": [
                {
                    "name": s.books.value,
                    "registered": s.registered,
                    "running": [[n, ev * 100] for n, ev in enumerate(s.running_mean_ev, 1)],
                }
                for s in self.replay.sets
            ],
            "axis": self.axis_percent,
        }


@dataclass(frozen=True)
class CandidatePage:
    name: str
    version: int
    scope: str
    is_agent: bool
    status: str
    verdict: str
    verdict_note: str
    progress: CheckpointProgress | None
    bets_with_close: int
    current: Checkpoint | None
    no_ev_reason: str
    pnl: float
    roi: float
    settled_bets: int
    pending_bets: int
    series: Series | None
    estimate_gap: EstimateGap | None
    # Shown only for a version that bet in more than one league; it never changes the verdict.
    leagues: tuple[LeagueBreakdown, ...]
    # Shown only when the report holds replays of the version's decisions; it never changes the
    # verdict.
    book_replay: BookReplayView | None
    decisions: DecisionHistory | None
    ledger: LedgerPage


def _interval(point: Checkpoint, at_checkpoint: bool) -> Interval:
    return Interval(
        n=point.n,
        mean_percent=point.mean_ev * 100,
        ci_percent=(point.ci95[0] * 100, point.ci95[1] * 100),
        promote_percent=(point.promote_ci[0] * 100, point.promote_ci[1] * 100),
        at_checkpoint=at_checkpoint,
    )


def axis_percent(running: Sequence[float], intervals: Sequence[Interval]) -> float:
    """The EV axis' half-width: the intervals and the settled part of the running mean,
    rounded up to a multiple of ``MIN_AXIS_PERCENT``."""
    reach = [abs(bound) for i in intervals for bound in (*i.ci_percent, *(i.promote_percent or ()))]
    reach += [abs(v) for v in running[AXIS_FROM_BETS - 1 :]] or [abs(v) for v in running]
    widest = max(reach, default=0.0) * 1.1
    return max(MIN_AXIS_PERCENT, math.ceil(widest / MIN_AXIS_PERCENT) * MIN_AXIS_PERCENT)


def build_book_replay(replay: BookReplay | None) -> BookReplayView | None:
    """The view of ``replay``; None without one or without a decision to replay."""
    if replay is None or not replay.decisions:
        return None
    settled = [
        ev * 100
        for s in replay.sets
        for ev in s.running_mean_ev[AXIS_FROM_BETS - 1 :] or s.running_mean_ev
    ]
    return BookReplayView(replay, axis_percent(settled, ()))


def running_series(
    running: RunningScore,
    intervals: list[Interval],
    checkpoints: list[int],
    estimates: Estimates | None,
    basis: EstimateBasis | None,
) -> Series:
    """The series of ``running``, with the estimate line of ``estimates`` over the same bets;
    the axis covers both lines."""
    running_percent = [ev * 100 for ev in running.mean_ev]
    estimate_percent = (
        None if estimates is None else [ev * 100 for ev in estimates.mean_estimated_ev]
    )
    return Series(
        running_ev_percent=running_percent,
        expected_units=list(running.expected),
        pnl_units=list(running.pnl),
        intervals=intervals,
        checkpoints=checkpoints,
        axis_percent=max(
            axis_percent(line, intervals) for line in (running_percent, estimate_percent or [])
        ),
        estimate_percent=estimate_percent,
        estimate_label=estimate_label(basis),
    )


def build_series(
    summary: CandidateSummary, running: RunningScore, basis: EstimateBasis | None
) -> Series:
    """The version's running score, with an interval at each checkpoint reached and at the
    current estimate (``CandidateSummary.checkpoints`` and ``current``), never per bet. The
    current estimate is drawn beside a checkpoint at the same n unless it is that checkpoint."""
    intervals = [_interval(c, True) for c in summary.checkpoints]
    if summary.current is not None and summary.current not in summary.checkpoints:
        intervals.append(_interval(summary.current, False))
    return running_series(
        running,
        intervals,
        checkpoints_through_next(len(running.staked)),
        summary.estimates,
        basis,
    )


def build_candidate_page(
    summary: CandidateSummary,
    running: RunningScore,
    decisions: DecisionHistory | None,
    ledger: LedgerPage,
    book_replay: BookReplay | None = None,
    basis: EstimateBasis | None = None,
) -> CandidatePage:
    """``basis`` labels the version's estimate (``name_basis``)."""
    entry = summary.registered
    verdict, note = verdict_of(summary)
    series = build_series(summary, running, basis) if summary.bets_with_close else None
    return CandidatePage(
        name=summary.name,
        version=summary.version,
        scope=scope_of(entry),
        is_agent=is_agent(entry),
        status=status_of(summary),
        verdict=verdict,
        verdict_note=note,
        progress=progress_of(summary),
        bets_with_close=summary.bets_with_close,
        current=summary.current,
        no_ev_reason=no_ev_reason(summary, entry),
        pnl=summary.pnl,
        roi=summary.roi,
        settled_bets=summary.settled_bets,
        pending_bets=summary.pending_bets,
        series=series,
        estimate_gap=None if series is None or summary.estimates is None else summary.estimates.gap,
        leagues=summary.leagues if len(summary.leagues) > 1 else (),
        book_replay=build_book_replay(book_replay),
        decisions=shown_decisions(entry, decisions),
        ledger=ledger,
    )


def reads_decisions(entry: Candidate | None) -> bool:
    """Whether to read the version's decision rows: every version but an agent's, which records
    none. A version the registry no longer holds has no known kind, so its rows decide."""
    return not isinstance(entry, AgentCandidate)


def shown_decisions(
    entry: Candidate | None, history: DecisionHistory | None
) -> DecisionHistory | None:
    """The timeline of a rule-candidate version, or of an earlier one that recorded decisions."""
    if history is None or (entry is None and history.total == 0):
        return None
    return history
