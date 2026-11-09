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

from odds_analytics.candidate_report import (
    CHECKPOINT_SIZE,
    KILL_AT_BETS,
    CandidateSummary,
    Checkpoint,
    RunningScore,
)
from odds_core.candidates import Candidate, MechanicalCandidate

from odds_web.decisions import DecisionHistory
from odds_web.ledger import LedgerPage
from odds_web.portfolio import (
    Progress,
    is_agent,
    no_ev_reason,
    progress_of,
    scope_of,
    verdict_of,
)

# Running means over fewer bets than this swing too wildly to set the EV axis.
AXIS_FROM_BETS = 10
MIN_AXIS_PERCENT = 5.0


@dataclass(frozen=True)
class Interval:
    """The intervals of an estimate over the first ``n`` bets with a close, in percent."""

    n: int
    mean_percent: float
    ci95_percent: tuple[float, float]
    promote_percent: tuple[float, float]
    at_checkpoint: bool


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

    def ev_chart(self) -> dict[str, Any]:
        """The running mean EV, its intervals and the checkpoint markers, for ``charts.js``."""
        return {
            "running": [[n, ev] for n, ev in enumerate(self.running_ev_percent, 1)],
            "intervals": [
                {
                    "n": i.n,
                    "mean": i.mean_percent,
                    "ci95": list(i.ci95_percent),
                    "promote": list(i.promote_percent),
                    "checkpoint": i.at_checkpoint,
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
class CandidatePage:
    name: str
    version: int
    scope: str
    is_agent: bool
    not_current: bool
    verdict: str
    verdict_note: str
    progress: Progress | None
    bets_with_close: int
    current: Checkpoint | None
    no_ev_reason: str
    pnl: float
    roi: float
    settled_bets: int
    pending_bets: int
    series: Series | None
    decisions: DecisionHistory | None
    ledger: LedgerPage


def _interval(point: Checkpoint, at_checkpoint: bool) -> Interval:
    return Interval(
        n=point.n,
        mean_percent=point.mean_ev * 100,
        ci95_percent=(point.ci95[0] * 100, point.ci95[1] * 100),
        promote_percent=(point.promote_ci[0] * 100, point.promote_ci[1] * 100),
        at_checkpoint=at_checkpoint,
    )


def _axis_percent(running: Sequence[float], intervals: Sequence[Interval]) -> float:
    """The EV axis' half-width: the intervals and the settled part of the running mean,
    rounded up to a multiple of ``MIN_AXIS_PERCENT``."""
    reach = [abs(bound) for i in intervals for bound in (*i.ci95_percent, *i.promote_percent)]
    reach += [abs(v) for v in running[AXIS_FROM_BETS - 1 :]] or [abs(v) for v in running]
    widest = max(reach, default=0.0) * 1.1
    return max(MIN_AXIS_PERCENT, math.ceil(widest / MIN_AXIS_PERCENT) * MIN_AXIS_PERCENT)


def build_series(summary: CandidateSummary, running: RunningScore) -> Series:
    """The version's running score, with an interval at each checkpoint reached and at the
    current estimate (``CandidateSummary.checkpoints`` and ``current``), never per bet. The
    current estimate is drawn beside a checkpoint at the same n unless it is that checkpoint."""
    running_percent = [ev * 100 for ev in running.mean_ev]
    intervals = [_interval(c, True) for c in summary.checkpoints]
    if summary.current is not None and summary.current not in summary.checkpoints:
        intervals.append(_interval(summary.current, False))
    reach = min(KILL_AT_BETS, (len(running_percent) // CHECKPOINT_SIZE + 1) * CHECKPOINT_SIZE)
    return Series(
        running_ev_percent=running_percent,
        expected_units=list(running.expected),
        pnl_units=list(running.pnl),
        intervals=intervals,
        checkpoints=list(range(CHECKPOINT_SIZE, reach + 1, CHECKPOINT_SIZE)),
        axis_percent=_axis_percent(running_percent, intervals),
    )


def build_candidate_page(
    summary: CandidateSummary,
    entry: Candidate | None,
    running: RunningScore,
    decisions: DecisionHistory | None,
    ledger: LedgerPage,
) -> CandidatePage:
    agent = is_agent(entry)
    verdict, note = verdict_of(summary)
    return CandidatePage(
        name=summary.name,
        version=summary.version,
        scope=scope_of(entry),
        is_agent=agent,
        not_current=summary.status is None,
        verdict=verdict,
        verdict_note=note,
        progress=progress_of(summary, agent),
        bets_with_close=summary.bets_with_close,
        current=summary.current,
        no_ev_reason=no_ev_reason(summary, entry),
        pnl=summary.pnl,
        roi=summary.roi,
        settled_bets=summary.settled_bets,
        pending_bets=summary.pending_bets,
        series=build_series(summary, running) if summary.bets_with_close else None,
        decisions=decisions,
        ledger=ledger,
    )


def has_decisions(entry: Candidate | None) -> bool:
    """Only a mechanical candidate records decisions; an agent's trades have none."""
    return isinstance(entry, MechanicalCandidate)
