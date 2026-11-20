"""The /perf page's readers: the statements that cost the database the most time, and how long
each scheduled job's recent runs took against the time between them.

The ``odds_web`` role reads the statements only through the functions ``odds_web.grants``
defines, which run as their owner and return every role's statements in the current database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import groupby, pairwise
from statistics import median, quantiles

from odds_core.job_run_models import JobOutcome, JobRun
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from odds_web.status import ChipState

TOP = 20
# The runs of each job the durations and the sparkline are taken over.
RECENT_RUNS = 50
# A job whose p95 duration reaches these fractions of its interval is late (amber) or stale (red).
LATE_FRACTION = 0.5
STALE_FRACTION = 0.8
SPARK_WIDTH = 160
SPARK_HEIGHT = 24


@dataclass(frozen=True)
class QueryStat:
    """One normalised statement as one role ran it, with its times in milliseconds."""

    query: str
    role: str
    calls: int
    total_ms: float
    mean_ms: float
    rows: int


@dataclass(frozen=True)
class QueryStats:
    by_total: list[QueryStat]
    by_mean: list[QueryStat]
    reset_at: datetime | None


def _top(order: str) -> str:
    return (
        "SELECT query, role, calls, total_exec_time, mean_exec_time, rows "
        f"FROM odds_web_query_stats() ORDER BY {order} DESC, queryid LIMIT :limit"
    )


async def load_query_stats(session: AsyncSession, limit: int = TOP) -> QueryStats:
    """The ``limit`` statements with the most total and the highest mean execution time, and when
    the statistics were last reset."""

    async def top(order: str) -> list[QueryStat]:
        rows = await session.execute(text(_top(order)), {"limit": limit})
        return [QueryStat(*row) for row in rows.tuples()]

    reset_at = await session.scalar(text("SELECT odds_web_query_stats_reset()"))
    return QueryStats(await top("total_exec_time"), await top("mean_exec_time"), reset_at)


@dataclass(frozen=True)
class RunPoint:
    started_at: datetime
    duration_ms: float
    outcome: JobOutcome

    @property
    def succeeded(self) -> bool:
        return self.outcome is JobOutcome.SUCCEEDED


@dataclass(frozen=True)
class JobDurations:
    """A job's recent runs, oldest first, and their durations against its interval, the median
    time between the starts of consecutive runs."""

    job_name: str
    runs: list[RunPoint]

    @property
    def latest_ms(self) -> float:
        return self.runs[-1].duration_ms

    @property
    def p50_ms(self) -> float:
        return median(r.duration_ms for r in self.runs)

    @property
    def p95_ms(self) -> float:
        durations = [r.duration_ms for r in self.runs]
        if len(durations) < 2:
            return durations[0]
        return quantiles(durations, n=20, method="inclusive")[-1]

    @property
    def interval_ms(self) -> float | None:
        gaps = [
            (b.started_at - a.started_at).total_seconds() * 1000 for a, b in pairwise(self.runs)
        ]
        return median(gaps) if gaps else None

    @property
    def p95_fraction(self) -> float | None:
        """The p95 duration as a fraction of the interval; None without an interval."""
        interval = self.interval_ms
        return self.p95_ms / interval if interval else None

    @property
    def state(self) -> ChipState:
        fraction = self.p95_fraction
        if fraction is None:
            return ChipState.UNKNOWN
        if fraction >= STALE_FRACTION:
            return ChipState.STALE
        if fraction >= LATE_FRACTION:
            return ChipState.LATE
        return ChipState.OK

    def spark_points(self) -> list[tuple[float, float, RunPoint]]:
        """Each run's (x, y) in a ``SPARK_WIDTH`` by ``SPARK_HEIGHT`` box, the longest at the top."""
        longest = max(r.duration_ms for r in self.runs) or 1.0
        step = SPARK_WIDTH / max(len(self.runs) - 1, 1)
        return [
            (round(i * step, 1), round(SPARK_HEIGHT * (1 - r.duration_ms / longest), 1), r)
            for i, r in enumerate(self.runs)
        ]


async def load_job_durations(session: AsyncSession, limit: int = RECENT_RUNS) -> list[JobDurations]:
    """The newest ``limit`` runs of every job with a recorded run, by job name."""
    recency = func.row_number().over(
        partition_by=JobRun.job_name, order_by=JobRun.started_at.desc()
    )
    recent = select(
        JobRun.job_name,
        JobRun.started_at,
        JobRun.finished_at,
        JobRun.outcome,
        recency.label("recency"),
    ).subquery()
    rows = await session.execute(
        select(recent.c.job_name, recent.c.started_at, recent.c.finished_at, recent.c.outcome)
        .where(recent.c.recency <= limit)
        .order_by(recent.c.job_name, recent.c.started_at)
    )
    return [
        JobDurations(
            name,
            [
                RunPoint(started, (finished - started).total_seconds() * 1000, JobOutcome(outcome))
                for _, started, finished, outcome in runs
            ],
        )
        for name, runs in groupby(rows.tuples(), key=lambda row: row[0])
    ]
