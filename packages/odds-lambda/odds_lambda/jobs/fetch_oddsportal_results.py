"""Settle events from OddsPortal's results pages.

Reads finished matches over HTTP from the results page's archive feed (no
browser; see docs/adr/0009-oddsportal-feed-client.md) and sets matching
unsettled events FINAL with their scores, or CANCELLED for cancelled matches.
Sport-aware: the results page path comes from the shared ``LeagueSpec``
registry in ``fetch_oddsportal``.

Idempotency: only events past kick-off that are SCHEDULED, LIVE, POSTPONED, or FINAL
without scores are pending, so re-runs after settling find nothing to process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.database import async_session_maker
from odds_core.models import AlertHistory, Event, EventStatus
from odds_core.team import normalize_team_name
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT
from odds_lambda.oddsportal_client import (
    CancelledMatch,
    FetchedResults,
    FinishedMatch,
    OddsPortalClient,
    OddsPortalError,
)
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.event_merge import EVENT_MATCH_WINDOW, find_match_id
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

DEFAULT_SPORT_KEY = "soccer_epl"
SCHEDULED_LOOKBACK = timedelta(days=14)
STALE_AFTER = timedelta(hours=24)


@dataclass
class ResultsStats:
    """Tracks results of a single results-collection run."""

    matches_fetched: int = 0
    events_updated: int = 0
    events_cancelled: int = 0
    events_unresolved: int = 0
    unresolved_kickoffs: list[datetime] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PendingEvent:
    """Plain-data snapshot of a pending event, decoupled from any DB session.

    The results job reads pending events into these dataclasses so the DB
    session can be closed before the network fetch begins.
    """

    id: str
    commence_time: datetime
    status: EventStatus


async def get_pending_events(
    session: AsyncSession, sport_key: str = DEFAULT_SPORT_KEY, since: datetime | None = None
) -> list[PendingEvent]:
    """Events of the sport past kick-off (and not before ``since``) that are not FINAL with scores."""
    conditions = [
        Event.sport_key == sport_key,
        Event.commence_time < datetime.now(UTC),
        or_(
            Event.status.in_([EventStatus.SCHEDULED, EventStatus.LIVE, EventStatus.POSTPONED]),
            and_(
                Event.status == EventStatus.FINAL,
                or_(Event.home_score.is_(None), Event.away_score.is_(None)),
            ),
        ),
    ]
    if since is not None:
        conditions.append(Event.commence_time >= since)
    result = await session.execute(
        select(Event.id, Event.commence_time, Event.status).where(and_(*conditions))
    )
    return [
        PendingEvent(id=row.id, commence_time=row.commence_time, status=row.status)
        for row in result.all()
    ]


def _raise_if_feed_parsed_nothing(fetched: FetchedResults, pending: list[PendingEvent]) -> None:
    """A feed with rows but no finished or cancelled match, while old events wait, means the parser broke."""
    stale_before = datetime.now(UTC) - STALE_AFTER
    stale = sum(1 for event in pending if event.commence_time < stale_before)
    if stale and fetched.rows_seen and not fetched.matches and not fetched.cancelled:
        raise OddsPortalError(
            f"results feed returned {fetched.rows_seen} rows but no finished or cancelled match, "
            f"with {stale} pending events older than {STALE_AFTER}; page format may have changed"
        )


async def process_results(
    sport: str | None = None,
    lookback: timedelta | None = SCHEDULED_LOOKBACK,
    fetched: FetchedResults | None = None,
) -> ResultsStats:
    """Set pending events FINAL with the scores of their finished matches.

    The job runs in three phases with distinct session scopes so no DB session
    is held across the network fetch (Neon closes idle connections, which would
    crash the write phase):

    1. Read phase: open a session, load pending events (detached), close session.
    2. Fetch phase: read the results pages back to the oldest pending kick-off.
    3. Write phase: open a fresh session, update each matched event, commit per event.

    Args:
        sport: Sport key (e.g. "soccer_epl"). Defaults to DEFAULT_SPORT_KEY.
        lookback: How far back pending events are considered; None for all of them.
        fetched: Results replacing the fetch. For testing.
    """
    stats = ResultsStats()
    sport_key = sport or DEFAULT_SPORT_KEY

    spec = _LEAGUE_SPEC_BY_SPORT.get(sport_key)
    if spec is None:
        logger.error(
            "unknown_sport_key",
            sport_key=sport_key,
            available=list(_LEAGUE_SPEC_BY_SPORT),
        )
        return stats

    since = datetime.now(UTC) - lookback if lookback is not None else None
    async with async_session_maker() as session:
        pending_events = await get_pending_events(session, sport_key=sport_key, since=since)

    if not pending_events:
        logger.info("no_pending_events", sport_key=sport_key)
        return stats

    logger.info("pending_events_found", count=len(pending_events))

    if fetched is None:
        oldest = min(event.commence_time for event in pending_events)
        async with OddsPortalClient() as client:
            fetched = await client.fetch_results(
                f"{spec.listing_path}results/", since=oldest - EVENT_MATCH_WINDOW
            )
    _raise_if_feed_parsed_nothing(fetched, pending_events)
    if not fetched.complete:
        logger.warning("results_fetch_incomplete", matches=len(fetched.matches))
    stats.matches_fetched = len(fetched.matches)

    matches = [
        FinishedMatch(
            home_team=normalize_team_name(m.home_team),
            away_team=normalize_team_name(m.away_team),
            start_time=m.start_time,
            home_score=m.home_score,
            away_score=m.away_score,
        )
        for m in fetched.matches
    ]
    cancelled = [
        CancelledMatch(
            home_team=normalize_team_name(m.home_team),
            away_team=normalize_team_name(m.away_team),
            start_time=m.start_time,
        )
        for m in fetched.cancelled
    ]
    starts_by_teams: dict[tuple[str, str], list[datetime]] = {}
    for match in (*matches, *cancelled):
        starts_by_teams.setdefault((match.home_team, match.away_team), []).append(match.start_time)
    pending_by_id = {event.id: event for event in pending_events}

    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for match in sorted((*matches, *cancelled), key=lambda m: m.start_time):
            event_id = await find_match_id(
                session, match.home_team, match.away_team, match.start_time, sport_key
            )
            event = pending_by_id.get(event_id) if event_id is not None else None
            if event is None:
                continue
            nearest = min(
                starts_by_teams[(match.home_team, match.away_team)],
                key=lambda start: abs(start - event.commence_time),
            )
            if nearest != match.start_time:
                continue
            if isinstance(match, CancelledMatch) and not event.status.can_become(
                EventStatus.CANCELLED
            ):
                continue
            try:
                if isinstance(match, FinishedMatch):
                    await writer.update_event_status(
                        event_id=event.id,
                        status=EventStatus.FINAL,
                        home_score=match.home_score,
                        away_score=match.away_score,
                    )
                else:
                    await writer.update_event_status(
                        event_id=event.id, status=EventStatus.CANCELLED
                    )
                await session.commit()
            except Exception as e:
                msg = f"{match.home_team} vs {match.away_team}: {e}"
                stats.errors.append(msg)
                logger.error("result_processing_failed", error=msg, exc_info=True)
                try:
                    await session.rollback()
                except Exception:
                    logger.warning("session_rollback_failed", exc_info=True)
                continue

            if isinstance(match, FinishedMatch):
                stats.events_updated += 1
            else:
                stats.events_cancelled += 1
            del pending_by_id[event.id]

    stats.events_unresolved = len(pending_by_id)
    stats.unresolved_kickoffs = [event.commence_time for event in pending_by_id.values()]
    logger.info(
        "results_collection_complete",
        matches_fetched=stats.matches_fetched,
        events_updated=stats.events_updated,
        events_cancelled=stats.events_cancelled,
        events_unresolved=stats.events_unresolved,
        errors=len(stats.errors),
    )

    return stats


def _settled_alert_type(sport_key: str) -> str:
    return f"results_settled:{sport_key}"


async def _last_settled_at(sport_key: str) -> datetime | None:
    async with async_session_maker() as session:
        return (
            await session.execute(
                select(func.max(AlertHistory.sent_at)).where(
                    AlertHistory.alert_type == _settled_alert_type(sport_key)
                )
            )
        ).scalar_one()


async def record_run_outcome(sport_key: str, stats: ResultsStats, now: datetime) -> None:
    """Record that the run settled events, or warn when it settled none and events are overdue.

    A run is stalled when it settled nothing while events that kicked off after
    the job's last settling run are still pending more than ``STALE_AFTER`` on.
    Unmatched events from before that run do not count, so one unresolvable
    match does not alert while the job keeps settling others. Without a
    settling run on record the job has no baseline and stays quiet.
    """
    from odds_core.alerts import record_to_alert_history, send_job_warning

    if stats.events_updated:
        await record_to_alert_history(
            _settled_alert_type(sport_key),
            "info",
            f"Settled {stats.events_updated} {sport_key} events",
        )
        return
    last_settled = await _last_settled_at(sport_key)
    if last_settled is None:
        return
    overdue = [
        kickoff
        for kickoff in stats.unresolved_kickoffs
        if kickoff < now - STALE_AFTER and kickoff > last_settled
    ]
    if overdue:
        await send_job_warning(
            f"results_stalled:{sport_key}",
            f"⚠️ OddsPortal results settled nothing: {len(overdue)} {sport_key} events "
            f"kicked off since {min(overdue):%Y-%m-%d %H:%M} UTC are still pending "
            f"(last settling run: {last_settled:%Y-%m-%d %H:%M} UTC)",
        )


async def main(ctx: JobContext) -> None:
    """Main job entry point — runs results collection, then self-schedules for tomorrow.

    Self-scheduling runs in a ``finally`` block so a failure inside
    ``process_results`` cannot leave the EventBridge rule disabled.
    """
    from odds_core.alerts import job_alert_context, send_scheduling_failure
    from odds_core.config import get_settings

    settings = get_settings()
    sport = require_sport(ctx, "fetch-oddsportal-results")
    schedule_job_name = make_compound_job_name("fetch-oddsportal-results", sport)

    run_raised = True
    try:
        async with job_alert_context(schedule_job_name):
            logger.info("fetch_oddsportal_results_started", sport=sport)
            stats = await process_results(sport=sport)
            await record_run_outcome(sport, stats, datetime.now(UTC))
            if stats.errors:
                raise RuntimeError(
                    f"{schedule_job_name} failed to settle {len(stats.errors)} event(s); "
                    f"first: {stats.errors[0]}"
                )
        run_raised = False
    finally:
        # Self-schedule: run again tomorrow at 08:00 UTC, regardless of outcome.
        # This is a once-a-day results sweep, not a proximity-driven fetch, so
        # it intentionally uses a fixed wake time and does NOT apply the
        # overnight skip (08:00 UTC is outside every configured overnight
        # window anyway).
        now = datetime.now(UTC)
        tomorrow_8am = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)

        try:
            await self_schedule(
                job_name=schedule_job_name,
                next_time=tomorrow_8am,
                dry_run=settings.scheduler.dry_run,
                sport=sport,
                reason="daily results sweep (fixed 08:00 UTC)",
            )
        except Exception as e:
            logger.error("fetch_oddsportal_results_scheduling_failed", error=str(e), exc_info=True)
            await send_scheduling_failure(schedule_job_name, e, final=not run_raised)
