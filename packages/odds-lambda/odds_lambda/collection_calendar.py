from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from odds_core.database import async_session_maker
from odds_core.mlb_data_models import MlbProbablePitchers
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.snapshot_sources import SnapshotSource
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.scheduling.decision import OVERNIGHT_WINDOWS, CadenceConfig, OvernightWindow

MLB_SPORT_KEY = "baseball_mlb"
MLB_PRICED_GAME_TYPES = ("R", "F", "D", "L", "W")
MLB_SCHEDULE_MAX_AGE = timedelta(hours=36)
ERRORS_SHOWN = 3


class EmptyCollectionError(RuntimeError):
    """A collection job wrote no rows while fixtures it should price are scheduled."""


@dataclass(frozen=True, slots=True)
class CollectionSource:
    """A snapshot source and the schedule its job promises to keep.

    ``cadence`` is the job's own polling cadence, so the expected gap between
    rows follows the tier of the next fixture. ``fixture_horizons`` is how far
    ahead a scheduled fixture of each sport must be before the source is
    expected to price it. ``snapshot_source`` is the ``raw_data["source"]`` tag
    of its snapshots; ``None`` is The Odds API, which writes none.
    """

    key: str
    job_name: str
    snapshot_source: SnapshotSource | None
    cadence: CadenceConfig
    fixture_horizons: Mapping[str, timedelta]
    skips_overnight: bool

    def fixture_horizon(self, sport_key: str) -> timedelta:
        return self.fixture_horizons[sport_key]

    def overnight_window(self, sport_key: str) -> OvernightWindow | None:
        return OVERNIGHT_WINDOWS.get(sport_key) if self.skips_overnight else None


@dataclass(frozen=True, slots=True)
class FixtureCalendar:
    """Scheduled fixtures of one sport after a reference time."""

    next_kickoff: datetime | None
    kickoffs_in_horizon: int


async def fixture_calendar(
    session: AsyncSession, sport_key: str, *, now: datetime, horizon: timedelta
) -> FixtureCalendar:
    """Upcoming fixtures of a sport, MLB from the MLB Stats API schedule.

    The MLB calendar comes from ``mlb_probable_pitchers`` so it does not depend
    on the collectors being monitored, which create the ``events`` rows.
    """
    if sport_key == MLB_SPORT_KEY:
        return await _mlb_schedule_calendar(session, now=now, horizon=horizon)
    scheduled = (
        Event.sport_key == sport_key,
        Event.status == EventStatus.SCHEDULED,
        Event.commence_time > now,
    )
    next_kickoff = (
        await session.execute(select(func.min(Event.commence_time)).where(*scheduled))
    ).scalar_one()
    in_horizon = (
        await session.execute(
            select(func.count(Event.id)).where(*scheduled, Event.commence_time <= now + horizon)
        )
    ).scalar_one()
    return FixtureCalendar(next_kickoff=next_kickoff, kickoffs_in_horizon=in_horizon)


async def _mlb_schedule_calendar(
    session: AsyncSession, *, now: datetime, horizon: timedelta
) -> FixtureCalendar:
    scheduled = (
        MlbProbablePitchers.fetched_at >= now - MLB_SCHEDULE_MAX_AGE,
        MlbProbablePitchers.commence_time > now,
        MlbProbablePitchers.game_type.in_(MLB_PRICED_GAME_TYPES),  # type: ignore[attr-defined]
    )
    next_kickoff = (
        await session.execute(select(func.min(MlbProbablePitchers.commence_time)).where(*scheduled))
    ).scalar_one()
    in_horizon = (
        await session.execute(
            select(func.count(func.distinct(MlbProbablePitchers.game_pk))).where(
                *scheduled, MlbProbablePitchers.commence_time <= now + horizon
            )
        )
    ).scalar_one()
    return FixtureCalendar(next_kickoff=next_kickoff, kickoffs_in_horizon=in_horizon)


async def latest_snapshot_time(
    session: AsyncSession, sport_key: str, source: SnapshotSource | None
) -> datetime | None:
    tag = OddsSnapshot.raw_data["source"].as_string()  # type: ignore[index]
    return (
        await session.execute(
            select(func.max(OddsSnapshot.snapshot_time))
            .join(Event, Event.id == OddsSnapshot.event_id)
            .where(
                Event.sport_key == sport_key,
                tag.is_(None) if source is None else tag == source.value,
            )
        )
    ).scalar_one()


def active_hours_between(
    start: datetime, end: datetime, overnight: OvernightWindow | None
) -> float:
    """Hours from ``start`` to ``end`` outside the daily overnight window."""
    if end <= start:
        return 0.0
    total = end - start
    if overnight is not None:
        window_start, window_resume = overnight
        wraps = window_start > window_resume
        day = start.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        day -= timedelta(days=1)
        while day < end:
            opens = day + timedelta(hours=window_start)
            closes = day + timedelta(hours=window_resume, days=1 if wraps else 0)
            overlap = min(end, closes) - max(start, opens)
            if overlap > timedelta(0):
                total -= overlap
            day += timedelta(days=1)
    return total.total_seconds() / 3600


async def require_rows_for_fixtures(
    source: CollectionSource,
    rows_by_sport: Mapping[str, int],
    *,
    errors_by_sport: Mapping[str, Sequence[str]] | None = None,
    now: datetime | None = None,
) -> None:
    """Raise when a sport got no rows although fixtures inside the source's horizon are scheduled.

    Called inside the job's alert context so the failure skips the heartbeat and
    raises a critical alert. An empty fixture calendar (international break,
    off-season) never raises. The run's own errors for the sport join the message.
    """
    moment = now or datetime.now(UTC)
    async with async_session_maker() as session:
        for sport_key, rows_written in rows_by_sport.items():
            if rows_written > 0 or sport_key not in source.fixture_horizons:
                continue
            horizon = source.fixture_horizon(sport_key)
            calendar = await fixture_calendar(session, sport_key, now=moment, horizon=horizon)
            if calendar.kickoffs_in_horizon:
                errors = list((errors_by_sport or {}).get(sport_key, ()))[:ERRORS_SHOWN]
                detail = f"; errors: {'; '.join(errors)}" if errors else ""
                raise EmptyCollectionError(
                    f"{source.key} wrote 0 rows for {sport_key} with "
                    f"{calendar.kickoffs_in_horizon} fixture(s) scheduled within "
                    f"{horizon} (next kickoff {calendar.next_kickoff}){detail}"
                )
