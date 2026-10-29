"""What the daily results jobs share: the pending-event query, run stats, the stalled-run
alert and the run-then-book-tomorrow wrapper."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.database import async_session_maker
from odds_core.models import AlertHistory, Event, EventExternalId, EventStatus
from odds_core.sports import SportKey
from odds_core.vocabulary import Source
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.scheduling.helpers import self_schedule

logger = structlog.get_logger()

SCHEDULED_LOOKBACK = timedelta(days=14)
STALE_AFTER = timedelta(hours=24)


@dataclass
class ResultsStats:
    """Tracks results of a single results-collection run.

    ``unresolved_kickoffs`` are the kick-offs of the pending events the run left unresolved. The
    OddsPortal job counts every event it did not settle FINAL or CANCELLED, POSTPONED ones
    included. The MLB job also accounts for events it marked or kept POSTPONED or moved to a
    replay (``events_postponed``, ``events_moved``), which it leaves out.
    """

    matches_fetched: int = 0
    events_updated: int = 0
    events_cancelled: int = 0
    events_postponed: int = 0
    events_moved: int = 0
    events_unresolved: int = 0
    unresolved_kickoffs: list[datetime] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PendingEvent:
    """Plain-data snapshot of a pending event, decoupled from any DB session.

    The results jobs read pending events into these dataclasses so the DB
    session can be closed before the network fetch begins.
    """

    id: int
    commence_time: datetime
    status: EventStatus
    game_pk: int | None = None


async def get_pending_events(
    session: AsyncSession, sport_key: SportKey, since: datetime | None = None
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
        select(Event.id, Event.commence_time, Event.status, EventExternalId.external_id)
        .outerjoin(
            EventExternalId,
            and_(
                EventExternalId.event_id == Event.id,  # type: ignore[arg-type]
                EventExternalId.source == Source.MLB_STATS_API,
            ),
        )
        .where(and_(*conditions))
    )
    return [
        PendingEvent(
            id=row.id,
            commence_time=row.commence_time,
            status=row.status,
            game_pk=None if row.external_id is None else int(row.external_id),
        )
        for row in result.all()
    ]


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


async def record_run_outcome(
    job_name: str, sport_key: str, stats: ResultsStats, now: datetime
) -> None:
    """Record that the run settled events, or warn when it settled none and events are overdue.

    A run is stalled when it settled nothing while events that kicked off after
    the sport's last settling run are still pending more than ``STALE_AFTER`` on.
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
            f"⚠️ {job_name} settled nothing: {len(overdue)} {sport_key} events "
            f"kicked off since {min(overdue):%Y-%m-%d %H:%M} UTC are still pending "
            f"(last settling run: {last_settled:%Y-%m-%d %H:%M} UTC)",
        )


async def run_daily_results_job(
    job_name: str,
    sport_key: SportKey,
    settle: Callable[[], Awaitable[ResultsStats]],
    schedule_sport: SportKey | None,
) -> None:
    """Run ``settle`` under the job's alert context, then book tomorrow 08:00 UTC.

    The booking runs in ``finally`` so a failed run cannot end the chain. It is a once-a-day
    sweep, so it takes a fixed wake time and no overnight skip (08:00 UTC is outside every
    configured overnight window). ``schedule_sport`` is the booked run's payload sport, ``None``
    for a job that is not per-sport.
    """
    from odds_core.alerts import job_alert_context, send_scheduling_failure
    from odds_core.config import get_settings

    settings = get_settings()
    run_raised = True
    try:
        async with job_alert_context(job_name):
            logger.info("results_job_started", job_name=job_name, sport=sport_key)
            stats = await settle()
            await record_run_outcome(job_name, sport_key, stats, datetime.now(UTC))
            if stats.errors:
                raise RuntimeError(
                    f"{job_name} failed to settle {len(stats.errors)} event(s); "
                    f"first: {stats.errors[0]}"
                )
        run_raised = False
    finally:
        now = datetime.now(UTC)
        tomorrow_8am = (now + timedelta(days=1)).replace(hour=8, minute=0, second=0, microsecond=0)
        try:
            await self_schedule(
                job_name=job_name,
                next_time=tomorrow_8am,
                dry_run=settings.scheduler.dry_run,
                sport=schedule_sport,
                reason="daily results sweep (fixed 08:00 UTC)",
            )
        except Exception as e:
            logger.error(
                "results_job_scheduling_failed", job_name=job_name, error=str(e), exc_info=True
            )
            await send_scheduling_failure(job_name, e, final=not run_raised)
