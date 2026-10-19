"""Settle events from OddsPortal's results pages.

Reads finished matches over HTTP from the results page's archive feed (no
browser; see docs/adr/0009-oddsportal-feed-client.md) and sets matching
unsettled events FINAL with their scores. Sport-aware: the results page path
comes from the shared ``LeagueSpec`` registry in ``fetch_oddsportal``.

Idempotency: only events past kick-off that are SCHEDULED, LIVE, or FINAL
without scores are pending, so re-runs after settling find nothing to process.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.jobs.fetch_oddsportal import _LEAGUE_SPEC_BY_SPORT
from odds_lambda.oddsportal_client import FinishedMatch, OddsPortalClient
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

DEFAULT_SPORT_KEY = "soccer_epl"
MATCH_WINDOW = timedelta(hours=24)


@dataclass
class ResultsStats:
    """Tracks results of a single results-collection run."""

    matches_fetched: int = 0
    events_updated: int = 0
    events_unresolved: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class PendingEvent:
    """Plain-data snapshot of a pending event, decoupled from any DB session.

    The results job reads pending events into these dataclasses so the DB
    session can be closed before the network fetch begins.
    """

    id: str
    home_team: str
    away_team: str
    commence_time: datetime


async def get_pending_events(
    session: AsyncSession, sport_key: str = DEFAULT_SPORT_KEY
) -> list[PendingEvent]:
    """Events of the sport past kick-off that are not FINAL with scores."""
    query = select(Event.id, Event.home_team, Event.away_team, Event.commence_time).where(
        and_(
            Event.sport_key == sport_key,
            Event.commence_time < datetime.now(UTC),
            or_(
                Event.status.in_([EventStatus.SCHEDULED, EventStatus.LIVE]),
                and_(
                    Event.status == EventStatus.FINAL,
                    or_(Event.home_score.is_(None), Event.away_score.is_(None)),
                ),
            ),
        )
    )
    result = await session.execute(query)
    return [
        PendingEvent(
            id=row.id,
            home_team=row.home_team,
            away_team=row.away_team,
            commence_time=row.commence_time,
        )
        for row in result.all()
    ]


def _match_to_event(
    match: FinishedMatch, pending_events: list[PendingEvent]
) -> PendingEvent | None:
    """The pending event with the same teams whose kick-off is nearest the match's, within the window."""
    candidates = [
        event
        for event in pending_events
        if event.home_team == match.home_team
        and event.away_team == match.away_team
        and abs(event.commence_time - match.start_time) <= MATCH_WINDOW
    ]
    return min(candidates, key=lambda e: abs(e.commence_time - match.start_time), default=None)


async def process_results(
    sport: str | None = None, matches: list[FinishedMatch] | None = None
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
        matches: Finished matches replacing the fetch. For testing.
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

    async with async_session_maker() as session:
        pending_events = await get_pending_events(session, sport_key=sport_key)

    if not pending_events:
        logger.info("no_pending_events", sport_key=sport_key)
        return stats

    logger.info("pending_events_found", count=len(pending_events))

    if matches is None:
        oldest = min(event.commence_time for event in pending_events)
        async with OddsPortalClient() as client:
            matches = await client.fetch_results(
                f"{spec.listing_path}results/", since=oldest - MATCH_WINDOW
            )
    stats.matches_fetched = len(matches)

    async with async_session_maker() as session:
        writer = OddsWriter(session)

        for match in sorted(matches, key=lambda m: m.start_time):
            event = _match_to_event(match, pending_events)
            if event is None:
                continue
            try:
                await writer.update_event_status(
                    event_id=event.id,
                    status=EventStatus.FINAL,
                    home_score=match.home_score,
                    away_score=match.away_score,
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

            stats.events_updated += 1
            pending_events.remove(event)

    stats.events_unresolved = len(pending_events)
    logger.info(
        "results_collection_complete",
        matches_fetched=stats.matches_fetched,
        events_updated=stats.events_updated,
        events_unresolved=stats.events_unresolved,
        errors=len(stats.errors),
    )

    return stats


async def main(ctx: JobContext) -> None:
    """Main job entry point — runs results collection, then self-schedules for tomorrow.

    Self-scheduling runs in a ``finally`` block so a failure inside
    ``process_results`` cannot leave the EventBridge rule disabled.
    """
    from odds_core.alerts import job_alert_context
    from odds_core.config import get_settings

    settings = get_settings()
    sport = ctx.sport
    schedule_job_name = make_compound_job_name("fetch-oddsportal-results", sport)

    try:
        async with job_alert_context(schedule_job_name):
            logger.info("fetch_oddsportal_results_started", sport=sport)
            await process_results(sport=sport)
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
            # Escalate to Discord so the failure is visible. Alert dispatch is
            # best-effort — any error here must not mask an upstream exception
            # propagating through this ``finally`` block.
            try:
                from odds_core.alerts import send_critical

                await send_critical(
                    f"🚨 fetch-oddsportal-results self-schedule failed "
                    f"(job={schedule_job_name}): {type(e).__name__}: {e}"
                )
            except Exception:
                logger.warning("fetch_oddsportal_results_scheduling_alert_failed", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
