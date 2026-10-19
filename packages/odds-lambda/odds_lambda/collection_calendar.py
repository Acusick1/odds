from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.scheduling.decision import OVERNIGHT_WINDOWS, CadenceConfig, OvernightWindow


class EmptyCollectionError(RuntimeError):
    """A collection job wrote no rows while fixtures it should price are scheduled."""


@dataclass(frozen=True, slots=True)
class CollectionSource:
    """A snapshot source and the schedule its job promises to keep.

    ``cadence`` is the job's own polling cadence, so the expected gap between
    rows follows the tier of the next fixture. ``fixture_horizon`` is how far
    ahead a scheduled fixture must be before the source is expected to price it.
    """

    key: str
    job_name: str
    request_id_prefix: str
    cadence: CadenceConfig
    fixture_horizon: timedelta
    skips_overnight: bool

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


async def latest_snapshot_time(
    session: AsyncSession, sport_key: str, request_id_prefix: str
) -> datetime | None:
    return (
        await session.execute(
            select(func.max(OddsSnapshot.snapshot_time))
            .join(Event, Event.id == OddsSnapshot.event_id)
            .where(
                Event.sport_key == sport_key,
                OddsSnapshot.api_request_id.startswith(request_id_prefix),  # type: ignore[union-attr]
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
    now: datetime | None = None,
) -> None:
    """Raise when a sport got no rows although fixtures inside the source's horizon are scheduled.

    Called inside the job's alert context so the failure skips the heartbeat and
    raises a critical alert. An empty fixture calendar (international break,
    off-season) never raises.
    """
    moment = now or datetime.now(UTC)
    async with async_session_maker() as session:
        for sport_key, rows_written in rows_by_sport.items():
            if rows_written > 0:
                continue
            calendar = await fixture_calendar(
                session, sport_key, now=moment, horizon=source.fixture_horizon
            )
            if calendar.kickoffs_in_horizon:
                raise EmptyCollectionError(
                    f"{source.key} wrote 0 rows for {sport_key} with "
                    f"{calendar.kickoffs_in_horizon} fixture(s) scheduled within "
                    f"{source.fixture_horizon} (next kickoff {calendar.next_kickoff})"
                )
