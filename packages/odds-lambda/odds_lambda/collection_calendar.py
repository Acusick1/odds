from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from odds_core.database import async_session_maker
from odds_core.epl_data_models import ESPN_STATUS_POSTPONED, EspnFixture
from odds_core.mlb_data_models import UNPLAYED_GAME_STATUSES, MlbProbablePitchers
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.vocabulary import Source, TimeBasis
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.espn_fixture_fetcher import LEAGUE_SLUGS, PRIMARY_LEAGUE_SLUG
from odds_lambda.scheduling.decision import (
    DORMANCY_HORIZON_DAYS,
    OVERNIGHT_WINDOWS,
    CadenceConfig,
    KickoffLookup,
    Lead,
    OvernightWindow,
    ScheduleDecision,
    decide_forward_resilient,
)
from odds_lambda.scheduling.helpers import get_next_kickoff

MLB_SPORT_KEY = "baseball_mlb"
EPL_SPORT_KEY = "soccer_epl"
MLB_PRICED_GAME_TYPES = ("R", "F", "D", "L", "W")
MLB_SCHEDULE_MAX_AGE = timedelta(hours=36)
ERRORS_SHOWN = 3


@dataclass(frozen=True, slots=True)
class EventKey:
    """A fixture of an event-backed sport, by its ``events`` row."""

    event_id: int


@dataclass(frozen=True, slots=True)
class MlbGameKey:
    """An MLB fixture, by the ``gamePk`` of the schedule that is its calendar."""

    game_pk: int


FixtureKey = EventKey | MlbGameKey


class EmptyCollectionError(RuntimeError):
    """A collection job wrote no rows while fixtures it should price are scheduled."""


@dataclass(frozen=True, slots=True)
class CollectionSource:
    """A snapshot source and the schedule its job promises to keep.

    ``cadence`` is the job's own polling cadence, so the expected gap between
    rows follows the tier of the next fixture. ``fixture_horizons`` is how far
    ahead a scheduled fixture of each sport must be before the source is
    expected to price it. ``source`` is the ``source`` of the snapshots its job
    writes, all of them ``observed``.
    """

    key: str
    job_name: str
    source: Source
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
    session: AsyncSession,
    sport_key: str,
    *,
    now: datetime,
    horizon: timedelta,
    excluding: Collection[FixtureKey] = (),
) -> FixtureCalendar:
    """Upcoming fixtures of a sport, MLB from the MLB Stats API schedule.

    The MLB calendar comes from ``mlb_probable_pitchers`` so it does not depend
    on the collectors being monitored, which create the ``events`` rows.
    ``excluding`` leaves out fixtures by key (see ``FixtureKey``).
    """
    if sport_key == MLB_SPORT_KEY:
        return await _mlb_schedule_calendar(session, now=now, horizon=horizon, excluding=excluding)
    return await event_calendar(session, sport_key, now=now, horizon=horizon, excluding=excluding)


async def event_calendar(
    session: AsyncSession,
    sport_key: str,
    *,
    now: datetime,
    horizon: timedelta,
    excluding: Collection[FixtureKey] = (),
) -> FixtureCalendar:
    """Upcoming SCHEDULED ``events`` of a sport, whichever collector created them."""
    scheduled = (
        Event.sport_key == sport_key,
        Event.status == EventStatus.SCHEDULED,
        Event.commence_time > now,
        Event.id.notin_([k.event_id for k in excluding if isinstance(k, EventKey)]),  # type: ignore[attr-defined]
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
    session: AsyncSession,
    *,
    now: datetime,
    horizon: timedelta,
    excluding: Collection[FixtureKey] = (),
) -> FixtureCalendar:
    """Games of the latest schedule fetch, which is the job's only write path.

    Every row of a fetch shares one ``fetched_at``, so the latest fetch is the
    current schedule; a fetch older than ``MLB_SCHEDULE_MAX_AGE`` is no calendar.
    """
    latest = (await session.execute(select(func.max(MlbProbablePitchers.fetched_at)))).scalar_one()
    if latest is None or latest < now - MLB_SCHEDULE_MAX_AGE:
        return FixtureCalendar(next_kickoff=None, kickoffs_in_horizon=0)
    scheduled = (
        MlbProbablePitchers.fetched_at == latest,
        MlbProbablePitchers.commence_time > now,
        MlbProbablePitchers.game_type.in_(MLB_PRICED_GAME_TYPES),  # type: ignore[attr-defined]
        MlbProbablePitchers.start_time_tbd.is_(False),  # type: ignore[attr-defined]
        MlbProbablePitchers.game_pk.notin_(
            [k.game_pk for k in excluding if isinstance(k, MlbGameKey)]
        ),  # type: ignore[attr-defined]
        or_(
            MlbProbablePitchers.game_status.is_(None),  # type: ignore[union-attr]
            MlbProbablePitchers.game_status.notin_(UNPLAYED_GAME_STATUSES),  # type: ignore[union-attr]
        ),
    )
    next_kickoff = (
        await session.execute(select(func.min(MlbProbablePitchers.commence_time)).where(*scheduled))
    ).scalar_one()
    in_horizon = (
        await session.execute(
            select(func.count(MlbProbablePitchers.game_pk)).where(
                *scheduled, MlbProbablePitchers.commence_time <= now + horizon
            )
        )
    ).scalar_one()
    return FixtureCalendar(next_kickoff=next_kickoff, kickoffs_in_horizon=in_horizon)


async def _espn_next_fixture(session: AsyncSession, *, now: datetime) -> datetime | None:
    return (
        await session.execute(
            select(func.min(EspnFixture.date)).where(
                EspnFixture.competition == LEAGUE_SLUGS[PRIMARY_LEAGUE_SLUG],
                EspnFixture.status != ESPN_STATUS_POSTPONED,
                EspnFixture.date > now,
            )
        )
    ).scalar_one()


async def published_schedule_kickoff(
    session: AsyncSession, sport_key: str, *, now: datetime
) -> datetime | None:
    """Next kickoff in the sport's published schedule, which no collection job writes.

    EPL reads the ESPN fixtures of the daily ``fetch-espn-fixtures`` cron and
    MLB the Stats API schedule of the daily ``fetch-mlb-probables`` cron. NBA has no
    table of its own: its ``fetch-espn-fixtures`` run creates the events, which the
    stored lookup reads. A sport without a published schedule here has none.
    """
    if sport_key == MLB_SPORT_KEY:
        calendar = await _mlb_schedule_calendar(session, now=now, horizon=timedelta(0))
        return calendar.next_kickoff
    if sport_key == EPL_SPORT_KEY:
        return await _espn_next_fixture(session, now=now)
    return None


async def next_fixture_kickoff(
    sport_key: str, *, now: datetime, stored_kickoff: KickoffLookup = get_next_kickoff
) -> datetime | None:
    """Earliest upcoming kickoff in the stored events (read by ``stored_kickoff``) or the
    published schedule.

    A collector that goes dormant stops creating events, so the stored events
    alone could never end the dormancy. The published schedule wakes it, and the
    stored events keep a sport awake when the schedule source is down.
    """
    stored = await stored_kickoff(sport_key, now=now)
    async with async_session_maker() as session:
        published = await published_schedule_kickoff(session, sport_key, now=now)
    return min((k for k in (stored, published) if k is not None), default=None)


async def decide_collection_run(
    sport_keys: list[str],
    cadence: CadenceConfig,
    *,
    overnight: OvernightWindow | None,
    now: datetime | None = None,
    stored_only: bool = False,
    stored_kickoff: KickoffLookup | None = None,
    leads: Sequence[Lead] = (),
    run_started: datetime | None = None,
) -> ScheduleDecision:
    """Next run of a collector, dormant while no fixture is within ``DORMANCY_HORIZON_DAYS``.

    ``stored_only`` looks at stored events alone, for a job that only pairs
    with events other jobs create. ``stored_kickoff`` replaces the stored events'
    ``commence_time`` as their kickoff, for the cadence and the lead runs.
    ``leads`` adds a run before each kickoff that the run started at
    ``run_started`` did not serve.
    """
    lookup: KickoffLookup | None = None if stored_only else next_fixture_kickoff
    if stored_kickoff is not None:
        stored = stored_kickoff

        async def with_stored(sport_key: str, *, now: datetime) -> datetime | None:
            return await next_fixture_kickoff(sport_key, now=now, stored_kickoff=stored)

        lookup = stored if stored_only else with_stored
    return await decide_forward_resilient(
        sport_keys,
        cadence,
        overnight=overnight,
        lookahead_days=DORMANCY_HORIZON_DAYS,
        now=now,
        next_kickoff_lookup=lookup,
        leads=leads,
        lead_kickoff_lookup=stored_kickoff,
        run_started=run_started,
    )


async def latest_snapshot_time(
    session: AsyncSession, sport_key: str, source: Source
) -> datetime | None:
    """The time of the latest ``observed`` snapshot ``source`` wrote for the sport."""
    return (
        await session.execute(
            select(OddsSnapshot.snapshot_time)
            .join(Event, Event.id == OddsSnapshot.event_id)
            .where(
                Event.sport_key == sport_key,
                OddsSnapshot.source == source,
                OddsSnapshot.time_basis == TimeBasis.OBSERVED,
            )
            .order_by(OddsSnapshot.snapshot_time.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


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
    accounted_by_sport: Mapping[str, Collection[FixtureKey]] | None = None,
    now: datetime | None = None,
) -> None:
    """Raise when a sport got no rows although fixtures inside the source's horizon are scheduled.

    Called inside the job's alert context so the failure skips the heartbeat and
    raises a critical alert. An empty fixture calendar (international break,
    off-season) never raises. The run's own errors for the sport join the message.
    ``accounted_by_sport`` holds the keys of fixtures the run identified but rightly
    did not price (a market already in play); the calendar leaves them out, so the
    check raises only for fixtures beyond them.
    """
    moment = now or datetime.now(UTC)
    async with async_session_maker() as session:
        for sport_key, rows_written in rows_by_sport.items():
            if rows_written > 0 or sport_key not in source.fixture_horizons:
                continue
            horizon = source.fixture_horizon(sport_key)
            calendar = await fixture_calendar(
                session,
                sport_key,
                now=moment,
                horizon=horizon,
                excluding=(accounted_by_sport or {}).get(sport_key, ()),
            )
            if calendar.kickoffs_in_horizon:
                errors = list((errors_by_sport or {}).get(sport_key, ()))[:ERRORS_SHOWN]
                detail = f"; errors: {'; '.join(errors)}" if errors else ""
                raise EmptyCollectionError(
                    f"{source.key} wrote 0 rows for {sport_key} with "
                    f"{calendar.kickoffs_in_horizon} fixture(s) scheduled within "
                    f"{horizon} (next kickoff {calendar.next_kickoff}){detail}"
                )
