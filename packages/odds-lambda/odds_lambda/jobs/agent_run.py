"""Run the betting agent and self-schedule based on fixture proximity.

Scheduling strategy: query DB for next kickoff, compute wake interval from
proximity tiers, pre-schedule before launching the agent subprocess (survives
crashes). After the agent exits, check the agent_wakeups table for an
agent-requested override and reschedule if it's sooner.

Wake interval tiers:
  > 48h to KO   -> 24h  (far out)
  24-48h        -> 12h  (research window opening)
  6-24h         ->  4h  (active research)
  1.5-6h        ->  1h  (lineups dropping)
  < 1.5h        ->  3h  (too close — check back post-match)
  no fixtures   -> 12h  (off-season check-in)
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.config import get_settings

from odds_lambda.scheduling.helpers import apply_overnight_skip, get_next_kickoff, self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name

logger = structlog.get_logger()

# Wake interval tiers (hours)
TIER_FAR_HOURS = 24.0  # > 48h to KO
TIER_RESEARCH_HOURS = 12.0  # 24-48h to KO
TIER_ACTIVE_HOURS = 4.0  # 6-24h to KO
TIER_LINEUP_HOURS = 1.0  # 1.5-6h to KO
TIER_NO_FIXTURES_HOURS = 12.0  # no fixtures within 7 days
SKIP_THRESHOLD_HOURS = 1.5  # too close to KO — don't wake

# Overnight suppression windows (start_utc, resume_utc) per sport
# EPL: last KO ~20:00 UTC, suppress 22:00-06:00
# MLB: last pitch ~04:00 UTC, suppress 06:00-14:00
OVERNIGHT_WINDOWS: dict[str, tuple[int, int]] = {
    "soccer_epl": (22, 6),
    "baseball_mlb": (6, 14),
}

# Agent subprocess limits
AGENT_TIMEOUT_SECONDS = 15 * 60  # 15 minutes

# Post-match check-in (too close to KO — match about to start / in progress)
TIER_TOO_CLOSE_HOURS = 3.0

# Horizon for "no fixtures" classification
FIXTURE_HORIZON_DAYS = 7


def _compute_wake_interval(hours_until_ko: float | None) -> float:
    """Return wake interval in hours based on fixture proximity tier."""
    if hours_until_ko is None:
        return TIER_NO_FIXTURES_HOURS
    if hours_until_ko > 48:
        return TIER_FAR_HOURS
    if hours_until_ko > 24:
        return TIER_RESEARCH_HOURS
    if hours_until_ko > 6:
        return TIER_ACTIVE_HOURS
    if hours_until_ko > SKIP_THRESHOLD_HOURS:
        return TIER_LINEUP_HOURS
    # Too close — match is about to start; check back after it ends
    return TIER_TOO_CLOSE_HOURS


def _should_skip_run(hours_until_ko: float | None) -> bool:
    """Return True if too close to kickoff to justify a wake-up."""
    return hours_until_ko is not None and hours_until_ko <= SKIP_THRESHOLD_HOURS


async def _run_claude_agent(sport: str) -> int:
    """Spawn ``claude -p`` subprocess and return exit code.

    Stdout is logged line-by-line at INFO level. Timeout after
    ``AGENT_TIMEOUT_SECONDS``. Returns -1 on timeout.
    """
    cmd = ["claude", "-p", f"/agent {sport}", "--dangerously-skip-permissions"]

    settings = get_settings()
    logger.info("agent_subprocess_starting", sport=sport, cmd=cmd)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(settings.project_root),
    )

    try:
        async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
            assert proc.stdout is not None  # noqa: S101
            async for line in proc.stdout:
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.info("agent_output", line=text)

            await proc.wait()
    except TimeoutError:
        logger.warning("agent_subprocess_timeout", sport=sport, timeout=AGENT_TIMEOUT_SECONDS)
        proc.kill()
        await proc.wait()
        return -1

    exit_code = proc.returncode or 0
    logger.info("agent_subprocess_finished", sport=sport, exit_code=exit_code)
    return exit_code


async def _check_agent_requested_wakeup(sport_key: str) -> datetime | None:
    """Read and consume any agent-requested wake-up from the agent_wakeups table.

    Returns the requested time if a row exists for this sport, then marks it
    consumed. Returns None if no pending request.
    """
    from odds_core.agent_wakeup_models import AgentWakeup
    from odds_core.database import async_session_maker
    from sqlalchemy import select
    from sqlmodel import col

    async with async_session_maker() as session:
        result = await session.execute(
            select(AgentWakeup).where(
                col(AgentWakeup.sport_key) == sport_key,
                col(AgentWakeup.consumed_at).is_(None),
            )
        )
        wakeup = result.scalar_one_or_none()

        if wakeup is None:
            return None

        requested_time = wakeup.requested_time
        logger.info(
            "agent_wakeup_found",
            sport_key=sport_key,
            requested_time=requested_time.isoformat(),
            reason=wakeup.reason,
        )

        wakeup.consumed_at = datetime.now(UTC)
        session.add(wakeup)
        await session.commit()

    return requested_time


async def schedule_next(sport: str) -> tuple[datetime, float | None]:
    """Compute and schedule the next agent wake-up for a sport.

    Queries the DB for the next kickoff, computes the wake interval from
    proximity tiers, and schedules via self_schedule. Returns a tuple of
    (scheduled_next_time, hours_until_ko).

    This is the crash-safe pre-scheduling step — called both during bootstrap
    (without launching the agent) and at the start of a full agent run.
    """
    settings = get_settings()
    compound_job_name = make_compound_job_name("agent-run", sport)

    # --- Resolve overnight window for this sport ---
    if sport not in OVERNIGHT_WINDOWS:
        raise ValueError(
            f"No overnight window configured for {sport} — add it to OVERNIGHT_WINDOWS"
        )
    overnight_start_utc, overnight_resume_utc = OVERNIGHT_WINDOWS[sport]

    # --- Determine default wake interval from fixture proximity ---
    hours_until_ko: float | None = None
    try:
        next_kickoff = await get_next_kickoff(sport)
        if next_kickoff is not None:
            hours_until_ko = (next_kickoff - datetime.now(UTC)).total_seconds() / 3600
        logger.info(
            "agent_proximity",
            sport=sport,
            next_kickoff=next_kickoff.isoformat() if next_kickoff else None,
            hours_until_ko=round(hours_until_ko, 2) if hours_until_ko is not None else None,
        )
    except Exception as e:
        logger.warning("agent_kickoff_query_failed", error=str(e), exc_info=True)

    default_interval = _compute_wake_interval(hours_until_ko)

    # --- Schedule next wake-up ---
    default_next_time = apply_overnight_skip(
        datetime.now(UTC) + timedelta(hours=default_interval),
        overnight_start_utc=overnight_start_utc,
        overnight_resume_utc=overnight_resume_utc,
    )
    try:
        await self_schedule(
            job_name=compound_job_name,
            next_time=default_next_time,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
            interval_hours=default_interval,
            reason="pre-schedule (default tier)",
        )
    except Exception as e:
        logger.error("agent_run_preschedule_failed", error=str(e), exc_info=True)
        raise

    return default_next_time, hours_until_ko


async def main(ctx: JobContext) -> None:
    """Orchestrate agent wake-up: schedule, run, check override."""
    settings = get_settings()
    sport = ctx.sport

    if not sport:
        logger.error("agent_run_no_sport", msg="sport is required for agent-run job")
        return

    compound_job_name = make_compound_job_name("agent-run", sport)

    # --- Resolve overnight window for this sport ---
    if sport not in OVERNIGHT_WINDOWS:
        raise ValueError(
            f"No overnight window configured for {sport} — add it to OVERNIGHT_WINDOWS"
        )
    overnight_start_utc, overnight_resume_utc = OVERNIGHT_WINDOWS[sport]

    # --- Pre-schedule before work (crash-safe) ---
    default_next_time, hours_until_ko = await schedule_next(sport)
    skip_run = _should_skip_run(hours_until_ko)

    # --- Run the agent (unless too close to KO) ---
    if skip_run:
        logger.info(
            "agent_run_skipped",
            hours_until_ko=round(hours_until_ko, 2) if hours_until_ko is not None else None,
            reason="too close to kickoff",
        )
        return

    exit_code = await _run_claude_agent(sport)

    if exit_code != 0:
        logger.warning("agent_run_nonzero_exit", exit_code=exit_code, sport=sport)

    # --- Check for agent-requested wake-up override ---
    try:
        requested_time = await _check_agent_requested_wakeup(sport)
    except Exception as e:
        logger.warning("agent_wakeup_check_failed", error=str(e), exc_info=True)
        requested_time = None

    if requested_time is not None and requested_time <= datetime.now(UTC):
        logger.warning(
            "agent_wakeup_in_past",
            requested_time=requested_time.isoformat(),
            msg="ignoring agent-requested wakeup that is in the past",
        )
        requested_time = None

    if requested_time is not None and requested_time < default_next_time:
        override_next_time = apply_overnight_skip(
            requested_time,
            overnight_start_utc=overnight_start_utc,
            overnight_resume_utc=overnight_resume_utc,
        )
        try:
            await self_schedule(
                job_name=compound_job_name,
                next_time=override_next_time,
                dry_run=settings.scheduler.dry_run,
                sport=sport,
                reason="agent-requested override",
            )
            logger.info(
                "agent_run_rescheduled",
                default_time=default_next_time.isoformat(),
                override_time=override_next_time.isoformat(),
            )
        except Exception as e:
            logger.error("agent_run_override_schedule_failed", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport="soccer_epl")))
