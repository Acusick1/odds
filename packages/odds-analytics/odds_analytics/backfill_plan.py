"""Odds API historical calls planned from an acquisition manifest.

A manifest window is a candidate's decision time with its max input age: a call at ``t``
serves a fixture kicking off at ``K`` on a window ``(d, a)`` when ``t`` lies in
``[K - d - a, K - d]``, the inputs a candidate deciding ``d`` before kick-off reads
(``odds_lambda.candidate_rules.market_state.load_market_state``).
"""

from __future__ import annotations

import math
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
from odds_lambda.candidate_rules.rules import RuleParams, decision_times, max_input_age
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveInt, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# The Odds API bills a historical odds call 10 units per region per market, and each group of
# up to 10 named bookmakers as one region.
UNITS_PER_REGION_MARKET = 10
BOOKMAKERS_PER_REGION = 10


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
    windows: tuple[RuleParams, ...] = Field(min_length=1)
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
            timedelta(minutes=minutes) + max_input_age(params)
            for params in self.windows
            for minutes in params.minutes_before_kickoff
        )
        if timedelta(hours=self.probe.min_fixture_lead_hours) < longest:
            raise ValueError(
                f"the probe's fixture lead is shorter than the longest window, {longest}"
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
    def _one_max_age_per_lead(self) -> Manifest:
        leads = [m for params in self.windows for m in params.minutes_before_kickoff]
        if len(set(leads)) != len(leads):
            raise ValueError(f"each offset belongs to one window group, got {sorted(leads)}")
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

    def window_cuts(self) -> list[tuple[Window, Manifest | None]]:
        """Each window, latest first, with this manifest without it (None when it is the only
        one)."""
        cuts: list[tuple[Window, Manifest | None]] = []
        for i, params in enumerate(self.windows):
            for minutes in params.minutes_before_kickoff:
                rest = tuple(m for m in params.minutes_before_kickoff if m != minutes)
                kept = params.model_copy(update={"minutes_before_kickoff": rest})
                groups = (*self.windows[:i], *((kept,) if rest else ()), *self.windows[i + 1 :])
                window = Window(timedelta(minutes=minutes), max_input_age(params))
                cuts.append(
                    (window, self.model_copy(update={"windows": groups}) if groups else None)
                )
        return sorted(cuts, key=lambda cut: cut[0].lead)

    def floor_to_grid(self, t: datetime) -> datetime:
        """The latest snapshot time at or before ``t``."""
        steps = [step for step in self.snapshot_grid if step.start <= t]
        if not steps:
            raise ValueError(f"{t.isoformat()} is before the first snapshot")
        step = steps[-1]
        size = timedelta(minutes=step.step_minutes)
        return step.start + (t - step.start) // size * size


@dataclass(frozen=True)
class Window:
    """Inputs taken ``lead`` before kick-off, at most ``max_age`` old."""

    lead: timedelta
    max_age: timedelta


@dataclass(frozen=True)
class Fixture:
    event_id: int
    kickoff: datetime


@dataclass(frozen=True)
class Target:
    """One fixture's window, ``[start, end]``."""

    fixture: Fixture
    window: Window
    start: datetime
    end: datetime

    def contains(self, t: datetime) -> bool:
        return self.start <= t <= self.end


def targets(manifest: Manifest, fixtures: Sequence[Fixture]) -> list[Target]:
    """Each fixture's window at each of the manifest's decision times."""
    out: list[Target] = []
    for fixture in fixtures:
        for params in manifest.windows:
            age = max_input_age(params)
            for minutes, decision in decision_times(fixture.kickoff, params):
                window = Window(timedelta(minutes=minutes), age)
                out.append(Target(fixture, window, decision - age, decision))
    return out


@dataclass(frozen=True)
class Call:
    """One historical odds request at ``time`` and the targets it serves."""

    time: datetime
    serves: tuple[Target, ...]


def plan(manifest: Manifest, fixtures: Sequence[Fixture]) -> list[Call]:
    """The fewest grid-aligned calls whose times fall inside every fixture's every window.

    Greedy by window end: a window the last call misses gets a new call at the latest snapshot
    at or before its end, which serves every later-ending window that starts by then.
    """
    times: list[datetime] = []
    served: list[list[Target]] = []
    for target in sorted(targets(manifest, fixtures), key=lambda t: t.end):
        if times and times[-1] >= target.start:
            served[-1].append(target)
            continue
        t = manifest.floor_to_grid(target.end)
        if t < target.start:
            raise ValueError(
                f"no snapshot in [{target.start.isoformat()}, {target.end.isoformat()}] "
                f"for event {target.fixture.event_id}"
            )
        times.append(t)
        served.append([target])
    return [Call(t, tuple(s)) for t, s in zip(times, served, strict=True)]


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
