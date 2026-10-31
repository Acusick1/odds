"""Odds API historical calls planned from an acquisition manifest.

A manifest's coverage bands each promise that a decision at any whole minute in a range of
leads before kick-off finds a snapshot at most the band's max age old, the inputs a candidate
deciding then reads (``odds_lambda.candidate_rules.market_state.load_market_state``).
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml
from odds_core.models import Event, EventStatus
from odds_core.sports import SportKey
from odds_core.vocabulary import Market
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveInt, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# The Odds API bills a historical odds call 10 units per region per market, and each group of
# up to 10 named bookmakers as one region.
UNITS_PER_REGION_MARKET = 10
BOOKMAKERS_PER_REGION = 10
# Decisions fall on whole minutes before kick-off, like a candidate's minutes_before_kickoff.
MINUTE = timedelta(minutes=1)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# The Odds API's market keys and regions, in the request dialect.
ApiMarket = Literal[Market.H2H, Market.TOTALS]
Region = Literal["us", "us2", "uk", "au", "eu"]


class Season(_Frozen):
    """Fixtures kicking off in ``[start, end)``."""

    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def _starts_before_end(self) -> Season:
        if self.start >= self.end:
            raise ValueError(f"season starts at or after its end: {self.start} >= {self.end}")
        return self


class MarketRequest(_Frozen):
    key: ApiMarket
    line: float | None = None


class Band(_Frozen):
    """Decisions ``min_lead_minutes`` to ``max_lead_minutes`` before kick-off, inclusive, each
    with an input at most ``max_input_age_minutes`` old."""

    min_lead_minutes: PositiveInt
    max_lead_minutes: PositiveInt
    max_input_age_minutes: PositiveInt

    @model_validator(mode="after")
    def _ordered(self) -> Band:
        if self.min_lead_minutes > self.max_lead_minutes:
            raise ValueError(
                f"min lead {self.min_lead_minutes} is after max lead {self.max_lead_minutes}"
            )
        return self

    @property
    def max_age(self) -> timedelta:
        return timedelta(minutes=self.max_input_age_minutes)

    def decision_span(self, kickoff: datetime) -> tuple[datetime, datetime]:
        """The first and last decision time, inclusive."""
        return (
            kickoff - timedelta(minutes=self.max_lead_minutes),
            kickoff - timedelta(minutes=self.min_lead_minutes),
        )


class GridStep(_Frozen):
    """Snapshots every ``step_minutes`` from ``start``, until the next step starts."""

    start: AwareDatetime
    step_minutes: PositiveInt


class Probe(_Frozen):
    """The snapshot times to request before the full run, and what each must return: fixtures
    at least ``min_fixture_lead_hours`` ahead, ``required_bookmakers`` on each market, and a
    bill of ``units_per_call``."""

    snapshots: tuple[AwareDatetime, ...] = Field(min_length=1)
    min_fixture_lead_hours: PositiveInt
    required_bookmakers: dict[ApiMarket, tuple[str, ...]]
    units_per_call: PositiveInt


class Manifest(_Frozen):
    sport: SportKey
    seasons: dict[str, Season] = Field(min_length=1)
    markets: tuple[MarketRequest, ...] = Field(min_length=1)
    bookmakers: tuple[str, ...] = ()
    regions: tuple[Region, ...] = ()
    coverage: tuple[Band, ...] = Field(min_length=1)
    snapshot_grid: tuple[GridStep, ...] = Field(min_length=1)
    probe: Probe

    @model_validator(mode="after")
    def _books_or_regions(self) -> Manifest:
        if bool(self.bookmakers) == bool(self.regions):
            raise ValueError("a manifest names either bookmakers or regions")
        return self

    @model_validator(mode="after")
    def _distinct_markets(self) -> Manifest:
        keys = [market.key for market in self.markets]
        if len(set(keys)) != len(keys):
            raise ValueError(f"each market is requested once, got {keys}")
        return self

    @model_validator(mode="after")
    def _seasons_disjoint(self) -> Manifest:
        ordered = sorted(self.seasons.items(), key=lambda item: item[1].start)
        for (earlier, a), (later, b) in pairwise(ordered):
            if b.start < a.end:
                raise ValueError(f"seasons {earlier} and {later} overlap")
        return self

    @model_validator(mode="after")
    def _probe_matches(self) -> Manifest:
        longest = max(
            timedelta(minutes=band.max_lead_minutes) + band.max_age for band in self.coverage
        )
        if timedelta(hours=self.probe.min_fixture_lead_hours) < longest:
            raise ValueError(
                f"the probe's fixture lead is shorter than the earliest coverage, {longest}"
            )
        if self.probe.units_per_call != self.units_per_call:
            raise ValueError(
                f"the probe expects {self.probe.units_per_call} units per call, "
                f"the manifest requests {self.units_per_call}"
            )
        requested = {market.key for market in self.markets}
        for market, books in self.probe.required_bookmakers.items():
            if market not in requested:
                raise ValueError(f"the probe requires books on unrequested market {market}")
            if self.bookmakers and not set(books) <= set(self.bookmakers):
                raise ValueError(f"the probe requires unrequested books on {market}")
        return self

    @model_validator(mode="after")
    def _bands_disjoint(self) -> Manifest:
        ordered = sorted(self.coverage, key=lambda band: band.min_lead_minutes)
        for nearer, farther in pairwise(ordered):
            if farther.min_lead_minutes <= nearer.max_lead_minutes:
                raise ValueError(
                    f"coverage bands overlap at {farther.min_lead_minutes} minutes before kick-off"
                )
        return self

    @model_validator(mode="after")
    def _grid_in_order(self) -> Manifest:
        starts = [step.start for step in self.snapshot_grid]
        if starts != sorted(starts) or len(set(starts)) != len(starts):
            raise ValueError("snapshot grid steps must start in increasing order")
        return self

    @classmethod
    def load(cls, path: Path) -> Manifest:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    @property
    def region_count(self) -> int:
        """The regions a call is billed for."""
        if self.bookmakers:
            return math.ceil(len(self.bookmakers) / BOOKMAKERS_PER_REGION)
        return len(self.regions)

    @property
    def units_per_call(self) -> int:
        return len(self.markets) * self.region_count * UNITS_PER_REGION_MARKET

    def band_cuts(self) -> list[tuple[Band, Manifest | None]]:
        """Each band, nearest kick-off first, with this manifest without it (None when it is
        the only one)."""
        return [
            (
                band,
                self.model_copy(update={"coverage": tuple(b for b in self.coverage if b != band)})
                if len(self.coverage) > 1
                else None,
            )
            for band in sorted(self.coverage, key=lambda band: band.min_lead_minutes)
        ]

    def floor_to_grid(self, t: datetime) -> datetime:
        """The latest snapshot time at or before ``t``."""
        steps = [step for step in self.snapshot_grid if step.start <= t]
        if not steps:
            raise ValueError(f"{t.isoformat()} is before the first snapshot")
        step = steps[-1]
        size = timedelta(minutes=step.step_minutes)
        return step.start + (t - step.start) // size * size


@dataclass(frozen=True)
class Fixture:
    event_id: int
    kickoff: datetime

    def __post_init__(self) -> None:
        # The planner steps decision time in whole minutes from a minute boundary, so a
        # kick-off between minutes would put its decisions off that lattice.
        if self.kickoff.second or self.kickoff.microsecond:
            raise ValueError(f"kick-off {self.kickoff.isoformat()} is not on a whole minute")


@dataclass(frozen=True)
class Target:
    """One fixture's band: calls from ``start`` to ``end`` can be its decisions' inputs."""

    fixture: Fixture
    band: Band
    start: datetime
    end: datetime

    def contains(self, t: datetime) -> bool:
        return self.start <= t <= self.end


@dataclass(frozen=True)
class Call:
    """One historical odds request at ``time`` and the fixture bands it gives inputs to."""

    time: datetime
    serves: tuple[Target, ...]


def targets(manifest: Manifest, fixtures: Sequence[Fixture]) -> list[Target]:
    out: list[Target] = []
    for fixture in fixtures:
        for band in manifest.coverage:
            first, last = band.decision_span(fixture.kickoff)
            out.append(Target(fixture, band, first - band.max_age, last))
    return out


def _required_ages(
    manifest: Manifest, fixtures: Sequence[Fixture]
) -> list[tuple[datetime, datetime, timedelta]]:
    """Disjoint ``[start, end)`` stretches of decision time, each with the strictest max age of
    the bands covering it, in time order."""
    edges: dict[datetime, list[tuple[timedelta, int]]] = {}
    for fixture in {f.kickoff for f in fixtures}:
        for band in manifest.coverage:
            first, last = band.decision_span(fixture)
            edges.setdefault(first, []).append((band.max_age, 1))
            edges.setdefault(last + MINUTE, []).append((band.max_age, -1))
    active: Counter[timedelta] = Counter()
    stretches: list[tuple[datetime, datetime, timedelta]] = []
    times = sorted(edges)
    for t, next_t in pairwise(times):
        for age, change in edges[t]:
            active[age] += change
        ages = [age for age, count in active.items() if count > 0]
        if ages:
            stretches.append((t, next_t, min(ages)))
    return stretches


def plan(manifest: Manifest, fixtures: Sequence[Fixture]) -> list[Call]:
    """The fewest grid-aligned calls that give every decision minute of every fixture's bands
    an input at most the band's max age old.

    Greedy by decision time: at the first decision minute the last call is too old for, a new
    call goes at the latest snapshot at or before it.
    """
    times: list[datetime] = []
    for start, end, age in _required_ages(manifest, fixtures):
        t = start
        while t < end:
            if times and times[-1] >= t - age:
                t = times[-1] + age + MINUTE
                continue
            call = manifest.floor_to_grid(t)
            if call < t - age:
                raise ValueError(f"no snapshot in [{(t - age).isoformat()}, {t.isoformat()}]")
            times.append(call)
    by_time: dict[datetime, list[Target]] = {t: [] for t in times}
    for target in targets(manifest, fixtures):
        lo = bisect.bisect_left(times, target.start)
        hi = bisect.bisect_right(times, target.end)
        for t in times[lo:hi]:
            by_time[t].append(target)
    return [Call(t, tuple(by_time[t])) for t in times]


async def load_fixtures(session: AsyncSession, sport: str, season: Season) -> list[Fixture]:
    """The season's FINAL stored events."""
    rows = await session.execute(
        select(Event.id, Event.commence_time)
        .where(
            Event.sport_key == sport,
            Event.status == EventStatus.FINAL,
            Event.commence_time >= season.start,
            Event.commence_time < season.end,
        )
        .order_by(Event.commence_time, Event.id)
    )
    return [Fixture(event_id, kickoff) for event_id, kickoff in rows.all()]
