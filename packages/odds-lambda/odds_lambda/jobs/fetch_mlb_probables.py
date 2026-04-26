"""Fetch MLB probable-pitcher snapshots — backstop for days the agent doesn't run.

Flow:
1. Hit MLB Stats API ``/schedule`` with the ``probablePitcher`` hydration for
   today through D+3 (UTC dates).
2. Append snapshot rows to ``mlb_probable_pitchers`` via ``MlbPitcherWriter``
   (idempotent on ``(game_pk, fetched_at)``).
3. Alerts on failure via ``job_alert_context``.

The MCP ``get_probable_pitchers`` tool is the primary write path; this cron
exists to keep the announcement-timing series populated on days the agent
doesn't fire. The 3-day window matches MLBAM's empirical populate rate
(D+0 fully populated, D+1..D+3 partial, D+4 onwards empty).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime

import structlog
from odds_core.config import get_settings
from odds_core.database import async_session_maker

from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher, dates_for_window
from odds_lambda.scheduling.jobs import JobContext
from odds_lambda.storage.mlb_pitcher_writer import MlbPitcherWriter

logger = structlog.get_logger()

# Sport this job serves.
SPORT_KEY = "baseball_mlb"

# Core UTC window: today + next 3 days. ``dates_for_window`` pads ±1
# calendar day to absorb the ET-anchored MLBAM ``date=`` parameter.
_LOOKAHEAD_HOURS = 24 * 3


async def main(ctx: JobContext) -> None:
    """Fetch probable pitchers for today + next 3 UTC days.

    Raises:
        ValueError: If ``ctx.sport`` is provided and does not match the sport
            this job serves.
    """
    from odds_core.alerts import job_alert_context

    if ctx.sport is not None and ctx.sport != SPORT_KEY:
        raise ValueError(f"fetch-mlb-probables only supports {SPORT_KEY}, got sport={ctx.sport!r}")

    app_settings = get_settings()
    now = datetime.now(UTC)
    target_dates = dates_for_window(now, _LOOKAHEAD_HOURS)

    logger.info(
        "fetch_mlb_probables_started",
        backend=app_settings.scheduler.backend,
        sport=SPORT_KEY,
        date_count=len(target_dates),
    )

    async with job_alert_context("fetch-mlb-probables"):
        record_count = await _fetch_and_store(now, target_dates)
        logger.info(
            "fetch_mlb_probables_completed",
            sport=SPORT_KEY,
            records=record_count,
        )


async def _fetch_and_store(now: datetime, target_dates: list[date]) -> int:
    """Fetch the probable-pitcher snapshot for the given dates and persist it."""
    async with MlbStatsFetcher() as fetcher:
        records = await fetcher.fetch_dates(target_dates, fetched_at=now)

    if not records:
        logger.warning("fetch_mlb_probables_empty", date_count=len(target_dates))
        return 0

    async with async_session_maker() as session:
        writer = MlbPitcherWriter(session)
        count = await writer.insert_snapshots(records)
        await session.commit()

    return count


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport=SPORT_KEY)))
