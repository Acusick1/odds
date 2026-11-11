"""The portfolio page's rows, one per candidate version, from the report data.

Every number comes from ``odds_analytics.candidate_report``; this module only picks what to show
and lays out the forest plot.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from odds_analytics.candidate_report import (
    CandidateSummary,
    Checkpoint,
    CheckpointProgress,
    Verdict,
    checkpoint_progress,
)
from odds_core.candidates import Candidate, CandidateStatus, MechanicalCandidate
from odds_core.close import close_spec

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
class PortfolioRow:
    name: str
    version: int
    scope: str
    is_agent: bool
    status: str
    progress: CheckpointProgress | None
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


def scope_of(entry: Candidate | None) -> str:
    """``entry`` is the registry's entry at the summary's own version; a version the registry
    no longer holds has no leagues or market of its own to show."""
    if entry is None:
        return "—"
    leagues = ", ".join(entry.sports)
    if isinstance(entry, MechanicalCandidate):
        return f"{leagues} · {entry.market}"
    return f"{leagues} · agent"


def status_of(summary: CandidateSummary) -> str:
    """The registry status of the version, apart from its verdict; ``not current`` for a version
    the registry does not hold."""
    return "not current" if summary.status is None else summary.status.value


def progress_of(summary: CandidateSummary) -> CheckpointProgress | None:
    """Progress to the next checkpoint, none for a version that will not reach one: one the
    registry no longer holds, an agent with no bet that has a close, a candidate that is no
    longer active (the runner places no more bets for it), or a version whose verdict has fired."""
    if summary.registered is None:
        return None
    if is_agent(summary.registered) and summary.bets_with_close == 0:
        return None
    if summary.status is not CandidateStatus.ACTIVE:
        return None
    if summary.verdict in (Verdict.PROMOTE, Verdict.KILL):
        return None
    return checkpoint_progress(summary.bets_with_close)


def verdict_of(summary: CandidateSummary) -> tuple[str, str]:
    """The verdict's label and the checkpoint it rests on; ``no close`` when no bet has one."""
    if summary.bets_with_close == 0 and summary.bets_without_close > 0:
        return "no close", ""
    checkpoint = summary.checkpoint
    note = f"at {checkpoint.n} bets" if checkpoint is not None else ""
    return summary.verdict.value.replace("_", " "), note


def is_agent(entry: Candidate | None) -> bool:
    return entry is not None and entry.kind == "agent"


def no_ev_reason(summary: CandidateSummary, entry: Candidate | None = None) -> str:
    """Why there is no current estimate: the market has no close (known from a mechanical
    candidate's registry entry, else from its bets lacking one), or the events are too few."""
    if summary.bets_with_close == 0:
        no_close = summary.bets_without_close > 0 or (
            isinstance(entry, MechanicalCandidate)
            and all(close_spec(sport, entry.market) is None for sport in entry.sports)
        )
        return "EV not scoreable" if no_close else "no bets with a close"
    return "too few events"


def portfolio_rows(summaries: Sequence[CandidateSummary]) -> list[PortfolioRow]:
    rows: list[PortfolioRow] = []
    for s in summaries:
        entry = s.registered
        verdict, note = verdict_of(s)
        rows.append(
            PortfolioRow(
                name=s.name,
                version=s.version,
                scope=scope_of(entry),
                is_agent=is_agent(entry),
                status=status_of(s),
                progress=progress_of(s),
                forest=None if s.current is None else forest_for(s.current),
                no_ev_reason=no_ev_reason(s, entry),
                pnl=s.pnl,
                roi=s.roi,
                verdict=verdict,
                verdict_note=note,
            )
        )
    return rows
