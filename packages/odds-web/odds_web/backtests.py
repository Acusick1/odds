"""Stored backtest runs for the pages: a candidate version's development and holdout runs, the
portfolio's backtest estimate and the killed ideas' grids.

The runs are written by ``odds_analytics.backtest_store``; this module reads them and lays them
out. A killed idea's best cell is picked after the fact, so it is reported as a figure and never
drawn as a series.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime

from odds_analytics.backtest_store import kill_reason, stored_with_close
from odds_analytics.candidate_report import EstimateGap, estimate_gap, running_score
from odds_core.candidate_backtest_models import (
    CandidateBacktest,
    CandidateBacktestBet,
    WindowKind,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_web.candidate_page import Interval, Series, rule_basis, series_of
from odds_web.portfolio import BacktestForest, backtest_forest


@dataclass(frozen=True)
class Estimate:
    """A run's mean EV over its bets with a close and the interval around it, in fractions."""

    n: int
    mean_ev: float
    ci: tuple[float, float]
    level: float


@dataclass(frozen=True)
class BlockFigure:
    name: str
    start: date
    end: date
    leagues: tuple[str, ...]
    bets: int
    mean_ev: float | None


@dataclass(frozen=True)
class RunFigures:
    """What a stored run shows: its window, tiles and, for a run that kept its bets, the series."""

    kind: WindowKind
    start: date
    end: date
    leagues: tuple[str, ...]
    git_sha: str
    stored_at: datetime
    params: dict[str, object]
    bets: int
    bets_with_close: int
    events: int
    stake: float
    pnl: float
    estimate: Estimate | None
    blocks: tuple[BlockFigure, ...]
    series: Series | None
    # Over the series' bets; None without a series or when one of its bets has no estimate.
    estimate_gap: EstimateGap | None

    @property
    def roi(self) -> float:
        return self.pnl / self.stake if self.stake > 0 else 0.0


def estimate_of(run: CandidateBacktest) -> Estimate | None:
    if run.mean_ev is None or run.ci_lower is None or run.ci_upper is None or run.ci_level is None:
        return None
    return Estimate(run.bets_with_close, run.mean_ev, (run.ci_lower, run.ci_upper), run.ci_level)


def run_series(
    run: CandidateBacktest, bets: Sequence[CandidateBacktestBet], estimate: Estimate | None
) -> tuple[Series | None, EstimateGap | None]:
    """The running mean EV and estimated EV and cumulative Σ stake·EV and P&L over the run's
    settled bets with a close, with the run's estimate as the one interval drawn, and the gap
    between the estimate and the EV over those bets."""
    with_close = stored_with_close(run, bets)
    if not with_close or estimate is None:
        return None, None
    intervals = [
        Interval(
            n=len(with_close),
            mean_percent=estimate.mean_ev * 100,
            ci_percent=(estimate.ci[0] * 100, estimate.ci[1] * 100),
            promote_percent=None,
            at_checkpoint=False,
            level=estimate.level,
            label=f"{run.window_kind.value} estimate",
        )
    ]
    series = series_of(running_score(with_close), intervals, [], rule_basis(run.rule))
    return series, estimate_gap(with_close) if series.estimate_percent is not None else None


def figures_of(run: CandidateBacktest, bets: Sequence[CandidateBacktestBet]) -> RunFigures:
    estimate = estimate_of(run)
    series, gap = run_series(run, bets, estimate)
    return RunFigures(
        kind=run.window_kind,
        start=run.start_date,
        end=run.end_date,
        leagues=tuple(run.leagues),
        git_sha=run.git_sha,
        stored_at=run.created_at,
        params=run.params,
        bets=run.bets,
        bets_with_close=run.bets_with_close,
        events=run.events,
        stake=run.stake,
        pnl=run.pnl,
        estimate=estimate,
        blocks=tuple(
            BlockFigure(
                name=b["name"],
                start=date.fromisoformat(b["start"]),
                end=date.fromisoformat(b["end"]),
                leagues=tuple(b["leagues"]),
                bets=b["bets"],
                mean_ev=b["mean_ev"],
            )
            for b in run.blocks or ()
        ),
        series=series,
        estimate_gap=gap,
    )


def shown(run: CandidateBacktest) -> bool:
    """Whether a page shows the run: any replay, and of a grid only the selected cell."""
    return run.cell_index is None or run.selected


async def load_runs(session: AsyncSession, name: str, version: int) -> list[RunFigures]:
    """The version's latest development run and latest holdout run, development first."""
    latest: dict[WindowKind, CandidateBacktest] = {}
    found = await session.execute(
        select(CandidateBacktest)
        .where(
            col(CandidateBacktest.candidate_name) == name,
            col(CandidateBacktest.candidate_version) == version,
        )
        .order_by(col(CandidateBacktest.created_at), col(CandidateBacktest.id))
    )
    for run in found.scalars():
        if shown(run):
            latest[run.window_kind] = run
    figures: list[RunFigures] = []
    for kind in WindowKind:
        run = latest.get(kind)
        if run is None:
            continue
        bets = await session.execute(
            select(CandidateBacktestBet)
            .where(col(CandidateBacktestBet.backtest_id) == run.id)
            .order_by(col(CandidateBacktestBet.position))
        )
        figures.append(figures_of(run, list(bets.scalars())))
    return figures


@dataclass(frozen=True)
class KilledIdea:
    """A grid with no selected cell: the cells it scored, the best of them by the selection
    rule's own order (the highest interval lower bound) and why none was eligible."""

    name: str
    version: int
    issue: int
    market: str
    leagues: tuple[str, ...]
    cells: int
    best: Estimate | None
    best_cell: dict[str, object] | None
    reason: str
    stored_at: datetime


@dataclass(frozen=True)
class BacktestOverview:
    """The portfolio's stored-backtest figures: the estimate shown for each version and the
    killed ideas."""

    estimates: dict[tuple[str, int], BacktestForest]
    killed: list[KilledIdea]


def killed_ideas(cells: Sequence[CandidateBacktest]) -> list[KilledIdea]:
    """The ideas among the stored grid cells whose latest stored grid selected no cell."""
    grids: dict[tuple[str, int, str], list[CandidateBacktest]] = {}
    for cell in cells:
        grids.setdefault((cell.candidate_name, cell.candidate_version, cell.git_sha), []).append(
            cell
        )
    latest: dict[tuple[str, int], list[CandidateBacktest]] = {}
    for grid in grids.values():
        key = (grid[0].candidate_name, grid[0].candidate_version)
        if key not in latest or grid[0].created_at > latest[key][0].created_at:
            latest[key] = grid
    killed: list[KilledIdea] = []
    for (name, version), grid in sorted(latest.items()):
        if any(c.selected for c in grid):
            continue
        scored = [c for c in grid if estimate_of(c) is not None]
        top = max(
            scored, key=lambda c: (c.ci_lower or 0.0, c.mean_ev or 0.0, c.events), default=None
        )
        first = grid[0]
        killed.append(
            KilledIdea(
                name=name,
                version=version,
                issue=first.issue,
                market=first.market,
                leagues=tuple(first.leagues),
                cells=len(grid),
                best=None if top is None else estimate_of(top),
                best_cell=None if top is None else top.params,
                reason=kill_reason(grid),
                stored_at=first.created_at,
            )
        )
    return killed


async def load_overview(session: AsyncSession) -> BacktestOverview:
    """The estimate of each version's latest holdout run, else of its latest development run,
    and the killed ideas."""
    found = await session.execute(
        select(CandidateBacktest).order_by(
            col(CandidateBacktest.created_at), col(CandidateBacktest.id)
        )
    )
    runs = list(found.scalars())
    latest: dict[tuple[str, int, WindowKind], CandidateBacktest] = {}
    for run in runs:
        if shown(run):
            latest[(run.candidate_name, run.candidate_version, run.window_kind)] = run
    estimates: dict[tuple[str, int], BacktestForest] = {}
    for kind in (WindowKind.DEVELOPMENT, WindowKind.HOLDOUT):
        for (name, version, run_kind), run in latest.items():
            estimate = estimate_of(run)
            if run_kind is kind and estimate is not None:
                estimates[(name, version)] = backtest_forest(
                    kind, estimate.n, estimate.mean_ev, estimate.ci, estimate.level
                )
    return BacktestOverview(estimates, killed_ideas([r for r in runs if r.cell_index is not None]))
