"""Run the betting agent and self-schedule with agent-driven wake times.

The agent decides when to wake up next via the ``schedule_next_wakeup`` MCP
tool, which writes to the ``agent_wakeups`` table. This module reads that
table after the agent exits and schedules accordingly.

A fallback pre-schedule fires before the agent runs (crash protection). If
the agent requests a specific wake time, it overrides the fallback.
Proximity-tier guidance lives in the agent prompts, not in code.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import structlog
from odds_core.config import get_settings

from odds_lambda.scheduling.helpers import apply_overnight_skip, self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name

logger = structlog.get_logger()

# Overnight suppression windows (start_utc, resume_utc) per sport
# EPL: last KO ~20:00 UTC, suppress 22:00-06:00
# MLB: last pitch ~04:00 UTC, suppress 06:00-14:00
OVERNIGHT_WINDOWS: dict[str, tuple[int, int]] = {
    "soccer_epl": (22, 6),
    "baseball_mlb": (6, 14),
}

# Agent subprocess limits
AGENT_TIMEOUT_SECONDS = 15 * 60  # 15 minutes

# Fallback interval when agent crashes or fails to request a wake time
FALLBACK_INTERVAL_HOURS = 12.0


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


async def main(ctx: JobContext) -> None:
    """Orchestrate agent run: fallback pre-schedule, run agent, read wake request."""
    settings = get_settings()
    sport = ctx.sport

    if not sport:
        logger.error("agent_run_no_sport", msg="sport is required for agent-run job")
        return

    compound_job_name = make_compound_job_name("agent-run", sport)

    # Validate overnight window
    if sport not in OVERNIGHT_WINDOWS:
        raise ValueError(
            f"No overnight window configured for {sport} — add it to OVERNIGHT_WINDOWS"
        )
    overnight_start_utc, overnight_resume_utc = OVERNIGHT_WINDOWS[sport]

    # --- Pre-schedule fallback (crash protection) ---
    fallback_time = apply_overnight_skip(
        datetime.now(UTC) + timedelta(hours=FALLBACK_INTERVAL_HOURS),
        overnight_start_utc=overnight_start_utc,
        overnight_resume_utc=overnight_resume_utc,
    )
    await self_schedule(
        job_name=compound_job_name,
        next_time=fallback_time,
        dry_run=settings.scheduler.dry_run,
        sport=sport,
        interval_hours=FALLBACK_INTERVAL_HOURS,
        reason="fallback (crash protection)",
    )

    # --- Run the agent ---
    exit_code = await _run_claude_agent(sport)
    if exit_code != 0:
        logger.warning(
            "agent_run_nonzero_exit",
            exit_code=exit_code,
            sport=sport,
            msg="agent exited with non-zero code — fallback schedule remains",
        )
        return  # fallback schedule remains

    # --- Read agent's requested next wake time ---
    try:
        requested_time = await _check_agent_requested_wakeup(sport)
    except Exception as e:
        logger.warning("agent_wakeup_check_failed", error=str(e), exc_info=True)
        return  # fallback remains

    if requested_time is None:
        logger.warning(
            "agent_no_wakeup_requested",
            sport=sport,
            msg="agent did not request a wake time — fallback schedule remains",
        )
        return

    if requested_time <= datetime.now(UTC):
        logger.warning(
            "agent_wakeup_in_past",
            requested_time=requested_time.isoformat(),
            msg="ignoring agent-requested wakeup that is in the past",
        )
        return  # fallback remains

    # --- Override fallback with agent's requested time (overnight clamped) ---
    override_time = apply_overnight_skip(
        requested_time,
        overnight_start_utc=overnight_start_utc,
        overnight_resume_utc=overnight_resume_utc,
    )
    await self_schedule(
        job_name=compound_job_name,
        next_time=override_time,
        dry_run=settings.scheduler.dry_run,
        sport=sport,
        reason="agent-requested",
    )


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport="soccer_epl")))
