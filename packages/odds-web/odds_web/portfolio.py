"""The portfolio page's rows, one per candidate version, from the report data.

Every number comes from ``odds_analytics.candidate_report``; this module only picks what to show
and lays out the forest plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from odds_analytics.candidate_report import (
    CHECKPOINT_SIZE,
    KILL_AT_BETS,
    CandidateSummary,
    Checkpoint,
    Verdict,
)
from odds_core.candidates import Candidate, MechanicalCandidate

# The forest plot's EV axis, shared by every row so intervals compare; an interval running past it
# is clipped and marked.
AXIS_PERCENT = 10.0
FOREST_WIDTH = 140


@dataclass(frozen=True)
class Forest:
    """A row's current estimate: positions in pixels along the shared axis, and the figures
    in percent for the label and tooltip."""

    mean_percent: float
    ci95_percent: tuple[float, float]
    promote_percent: tuple[float, float]
    n: int
    zero_x: float
    mean_x: float
    ci95_x: tuple[float, float]
    promote_x: tuple[float, float]
    clipped_low: bool
    clipped_high: bool


@dataclass(frozen=True)
class Progress:
    bets_with_close: int
    next_checkpoint: int


@dataclass(frozen=True)
class PortfolioRow:
    name: str
    version: int
    scope: str
    is_agent: bool
    not_current: bool
    progress: Progress | None
    forest: Forest | None
    no_ev_reason: str
    pnl: float
    roi: float
    verdict: str
    verdict_note: str


def forest_for(current: Checkpoint) -> Forest:
    def x(percent: float) -> float:
        clamped = max(-AXIS_PERCENT, min(AXIS_PERCENT, percent))
        return round((clamped + AXIS_PERCENT) / (2 * AXIS_PERCENT) * FOREST_WIDTH, 1)

    ci95 = (current.ci95[0] * 100, current.ci95[1] * 100)
    promote = (current.promote_ci[0] * 100, current.promote_ci[1] * 100)
    reach = (min(ci95[0], promote[0]), max(ci95[1], promote[1]))
    return Forest(
        mean_percent=current.mean_ev * 100,
        ci95_percent=ci95,
        promote_percent=promote,
        n=current.n,
        zero_x=x(0.0),
        mean_x=x(current.mean_ev * 100),
        ci95_x=(x(ci95[0]), x(ci95[1])),
        promote_x=(x(promote[0]), x(promote[1])),
        clipped_low=reach[0] < -AXIS_PERCENT,
        clipped_high=reach[1] > AXIS_PERCENT,
    )


def _scope(entry: Candidate | None) -> str:
    if entry is None:
        return "—"
    if isinstance(entry, MechanicalCandidate):
        return f"{entry.sport} · {entry.market}"
    return f"{entry.sport} · agent"


def _progress(summary: CandidateSummary, is_agent: bool) -> Progress | None:
    if is_agent and summary.bets_with_close == 0:
        return None
    reached = summary.verdict in (Verdict.PROMOTE, Verdict.KILL)
    if reached:
        return None
    n = summary.bets_with_close
    return Progress(n, min(KILL_AT_BETS, (n // CHECKPOINT_SIZE + 1) * CHECKPOINT_SIZE))


def _verdict(summary: CandidateSummary) -> tuple[str, str]:
    """The verdict's label and the checkpoint it rests on; ``no close`` when no bet has one."""
    if summary.bets_with_close == 0 and summary.bets_without_close > 0:
        return "no close", ""
    checkpoint = summary.checkpoint
    note = f"at {checkpoint.n} bets" if checkpoint is not None else ""
    return summary.verdict.value.replace("_", " "), note


def _no_ev_reason(summary: CandidateSummary) -> str:
    if summary.bets_with_close == 0:
        return "EV not scoreable" if summary.bets_without_close else "no bets with a close"
    return "too few events"


def portfolio_rows(
    summaries: Sequence[CandidateSummary], registry: Sequence[Candidate]
) -> list[PortfolioRow]:
    by_name = {c.name: c for c in registry}
    rows: list[PortfolioRow] = []
    for s in summaries:
        entry = by_name.get(s.name)
        is_agent = entry is not None and entry.kind == "agent"
        verdict, note = _verdict(s)
        rows.append(
            PortfolioRow(
                name=s.name,
                version=s.version,
                scope=_scope(entry),
                is_agent=is_agent,
                not_current=s.status is None,
                progress=_progress(s, is_agent),
                forest=None if s.current is None else forest_for(s.current),
                no_ev_reason=_no_ev_reason(s),
                pnl=s.pnl,
                roi=s.roi,
                verdict=verdict,
                verdict_note=note,
            )
        )
    return rows
