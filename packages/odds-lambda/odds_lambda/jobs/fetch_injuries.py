"""Fetch the official NBA injury report before tip-offs.

Reads the newest published report (``odds_lambda.injury_fetcher.fetch_latest_injury_report``) and
stores it through ``InjuryWriter``, which links each ``(team, game_date)`` to its event.
Self-schedules from the stored NBA events (``odds_lambda.jobs.fetch_espn_fixtures`` creates them):
a run every ``CADENCE`` hours while games are ahead, plus runs ``LEADS`` before each kickoff, so
every tip-off cluster gets a report from about two hours and half an hour out. With no game in
the stored events within ``DORMANCY_HORIZON_DAYS`` the run only books its next check-in.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.alerts import job_alert_context, send_job_warning
from odds_core.config import get_settings
from odds_core.database import async_session_maker

from odds_lambda.collection_calendar import decide_collection_run, event_calendar
from odds_lambda.injury_fetcher import fetch_latest_injury_report_isolated
from odds_lambda.scheduling.decision import OVERNIGHT_WINDOWS, CadenceConfig, Lead
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name, require_sport
from odds_lambda.storage.injury_writer import InjuryWriter

logger = structlog.get_logger()

SPORT_KEY = "basketball_nba"

# One interval at every tier: the cadence run is the morning-ET anchor (the overnight window
# resumes at 15:00 UTC, 11:00 ET) and a catch-up; the lead runs do the pre-tip work.
CADENCE = CadenceConfig(
    closing=6.0,
    pregame=6.0,
    sharp=6.0,
    early=6.0,
    opening=6.0,
    no_game=6.0,
    db_fallback=1.0,
)

# About two hours out the starters' status is mostly filed, and half an hour out is the last
# report slot before tip-off that is already published. A run that falls behind the earlier
# lead still serves it until an hour out, the later one until ten minutes out.
LEADS = (
    Lead(timedelta(hours=2), until=timedelta(hours=1)),
    Lead(timedelta(minutes=30), until=timedelta(minutes=10)),
)

# Slots to search back. A slot appears a few minutes after its time, and the NBA has stopped
# publishing for hours (every slot from 06:15 to 15:45 ET on 2026-10-30 was unavailable), so the
# search reaches far enough to say how old the newest report is.
MAX_SLOTS_BACK = 24

# A game this close should have a report no older than ``STALE_REPORT_AGE``.
REPORT_DUE_HORIZON = timedelta(hours=3)
STALE_REPORT_AGE = timedelta(hours=1)


async def _warn_if_report_stale(now: datetime, newest: datetime | None) -> None:
    """Warn, without failing the run, when a game is near and the newest report is old or absent.

    A publishing gap is the NBA's, and ``nbainjuries`` cannot tell it from a network error.
    """
    async with async_session_maker() as session:
        calendar = await event_calendar(session, SPORT_KEY, now=now, horizon=REPORT_DUE_HORIZON)
    if not calendar.kickoffs_in_horizon:
        return
    if newest is None:
        found = f"no report in the last {MAX_SLOTS_BACK + 1} slots"
    elif now - newest > STALE_REPORT_AGE:
        found = f"the newest report is the {newest:%Y-%m-%d %H:%M} UTC slot, {now - newest} old"
    else:
        return
    await send_job_warning(
        "injury_report_stale:fetch-injuries-nba",
        f"fetch-injuries-nba: {found} with {calendar.kickoffs_in_horizon} game(s) within "
        f"{REPORT_DUE_HORIZON}",
    )


async def collect_report(now: datetime) -> int:
    """Store the newest report; returns the rows written (0 when none is published)."""
    found = await fetch_latest_injury_report_isolated(now, max_slots_back=MAX_SLOTS_BACK)
    if found is None:
        logger.info("fetch_injuries_no_report", now=now)
        await _warn_if_report_stale(now, None)
        return 0
    slot, records = found
    await _warn_if_report_stale(now, slot)
    async with async_session_maker() as session:
        result = await InjuryWriter(session).upsert_injury_reports(records)
        await session.commit()
    logger.info(
        "fetch_injuries_stored",
        slot=slot,
        rows=result.rows,
        unmatched=len(result.unmatched),
    )
    if result.unmatched:
        pairs = ", ".join(f"{team} {day}" for team, day in result.unmatched[:5])
        await send_job_warning(
            "injury_unmatched:fetch-injuries-nba",
            f"fetch-injuries-nba: {len(result.unmatched)} (team, game date) pair(s) in the "
            f"{slot:%Y-%m-%d %H:%M} UTC report match no event: {pairs}",
        )
    return result.rows


async def main(ctx: JobContext) -> None:
    """Fetch the newest injury report, then self-schedule.

    Raises:
        ValueError: If ``ctx.sport`` is unset or is not NBA.
    """
    sport = require_sport(ctx, "fetch-injuries")
    if sport != SPORT_KEY:
        raise ValueError(f"fetch-injuries supports {SPORT_KEY}, got sport={sport!r}")

    settings = get_settings()
    compound_job_name = make_compound_job_name("fetch-injuries", sport)
    started = datetime.now(UTC)

    # Book the next run before the fetch so a hung JVM or failed fetch keeps the chain.
    decision = await decide_collection_run(
        [sport],
        CADENCE,
        overnight=OVERNIGHT_WINDOWS[sport],
        stored_only=True,
        leads=LEADS,
        run_started=started,
    )
    assert decision.next_execution is not None  # noqa: S101
    await self_schedule(
        job_name=compound_job_name,
        next_time=decision.next_execution,
        dry_run=settings.scheduler.dry_run,
        sport=sport,
        reason=f"pre-schedule: {decision.reason}",
    )

    async with job_alert_context(compound_job_name):
        if not decision.should_execute:
            logger.info("fetch_injuries_dormant", reason=decision.reason)
            return
        await collect_report(started)


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport=SPORT_KEY)))
