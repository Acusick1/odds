"""Run the betting agent and self-schedule based on fixture proximity.

Scheduling strategy: query DB for next kickoff, compute wake interval from
proximity tiers, pre-schedule before launching the agent subprocess (survives
crashes). After the agent exits, check the agent_wakeups table for an
agent-requested override and reschedule if it's sooner.

Wake interval tiers:
  > 48h to KO   -> 24h  (far out)
  24-48h        -> 12h  (research window opening)
  6-24h         ->  4h  (active research)
  < 6h          ->  1h  (lineups dropping, final decisions)
  no fixtures   -> 12h  (off-season check-in)
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import structlog
from odds_core.config import get_settings
from odds_core.sports import SportKey

from odds_lambda.scheduling.helpers import apply_overnight_skip, get_next_kickoff, self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name

logger = structlog.get_logger()

# Wake interval tiers (hours)
TIER_FAR_HOURS = 24.0  # > 48h to KO
TIER_RESEARCH_HOURS = 12.0  # 24-48h to KO
TIER_ACTIVE_HOURS = 4.0  # 6-24h to KO
TIER_LINEUP_HOURS = 1.0  # < 6h to KO
TIER_NO_FIXTURES_HOURS = 12.0  # no fixtures within 7 days

# Overnight suppression windows (start_utc, resume_utc) per sport
# EPL: last KO ~20:00 UTC, suppress 22:00-06:00
# MLB: last pitch ~04:00 UTC, suppress 06:00-14:00
OVERNIGHT_WINDOWS: dict[SportKey, tuple[int, int]] = {
    "soccer_epl": (22, 6),
    "baseball_mlb": (6, 14),
}

# Agent subprocess limits
AGENT_TIMEOUT_SECONDS = 15 * 60  # 15 minutes

# Retain this many per-run JSONL logs per sport; older ones are pruned on write.
AGENT_RUN_LOG_KEEP = 50

# Upper bound on any single stream-json line. The asyncio default (64 KiB) is
# too small — a single ``tool_result`` block (e.g. a large Read or WebFetch
# payload) routinely exceeds it and would raise ``LimitOverrunError``.
AGENT_STREAM_LINE_LIMIT = 8 * 1024 * 1024

# Horizon for "no fixtures" classification
FIXTURE_HORIZON_DAYS = 7


class ScheduleResult(NamedTuple):
    """Values computed by schedule_next(), reused by main() for overrides."""

    next_time: datetime
    hours_until_ko: float | None
    compound_job_name: str
    overnight_start_utc: int
    overnight_resume_utc: int


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
    return TIER_LINEUP_HOURS


def _preview_tool_input(d: dict[str, Any] | None, *, max_value_chars: int = 60) -> str | None:
    """Return a compact ``key=value`` summary of tool input for live logging."""
    if not d:
        return None
    parts: list[str] = []
    for k, v in d.items():
        s = str(v).replace("\n", " ").replace("\r", " ")
        if len(s) > max_value_chars:
            s = s[:max_value_chars] + "..."
        parts.append(f"{k}={s}")
    return " ".join(parts)


def _prune_agent_run_logs(log_dir: Path, sport: SportKey, keep: int) -> None:
    """Delete all but the newest ``keep`` JSONL files for ``sport``."""
    files = sorted(log_dir.glob(f"{sport}_*.jsonl"), reverse=True)
    for stale in files[keep:]:
        try:
            stale.unlink()
        except OSError as e:
            logger.warning("agent_run_log_prune_failed", file=str(stale), error=str(e))


def _log_stream_message(msg: dict[str, Any]) -> None:
    """Emit a structlog event for notable stream-json message types.

    Surfaces tool calls and the final result; other message types (assistant
    text / thinking blocks, tool results, system init) are captured in the
    per-run JSONL file but omitted from the main log to keep it readable.
    """
    msg_type = msg.get("type")
    if msg_type == "assistant":
        message = msg.get("message")
        if not isinstance(message, dict):
            return
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                logger.info(
                    "agent_tool_use",
                    tool=block.get("name"),
                    input_preview=_preview_tool_input(block.get("input")),
                )
    elif msg_type == "result":
        logger.info(
            "agent_run_summary",
            text=msg.get("result"),
            num_turns=msg.get("num_turns"),
            duration_ms=msg.get("duration_ms"),
            cost_usd=msg.get("total_cost_usd"),
        )


async def _run_claude_agent(sport: SportKey) -> int:
    """Spawn ``claude -p`` subprocess and return exit code.

    Full stream-json trace is written to
    ``logs/agent_runs/{sport}_<ts>.jsonl``; tool calls and the final result
    are tee'd into the main structlog log for live visibility. Timeout after
    ``AGENT_TIMEOUT_SECONDS``; returns -1 on timeout.
    """
    cmd = [
        "claude",
        "-p",
        f"/agent {sport}",
        "--output-format",
        "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]

    settings = get_settings()

    log_dir = settings.project_root / "logs" / "agent_runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    _prune_agent_run_logs(log_dir, sport, keep=AGENT_RUN_LOG_KEEP)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # PID suffix avoids collisions when two invocations start in the same second.
    log_path = log_dir / f"{sport}_{timestamp}_{os.getpid()}.jsonl"

    logger.info("agent_subprocess_starting", sport=sport, cmd=cmd, log_file=str(log_path))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(settings.project_root),
        limit=AGENT_STREAM_LINE_LIMIT,
    )

    with log_path.open("ab") as log_file:
        try:
            async with asyncio.timeout(AGENT_TIMEOUT_SECONDS):
                assert proc.stdout is not None  # noqa: S101
                async for line in proc.stdout:
                    log_file.write(line)
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    _log_stream_message(msg)
                await proc.wait()
        except TimeoutError:
            logger.warning(
                "agent_subprocess_timeout",
                sport=sport,
                timeout=AGENT_TIMEOUT_SECONDS,
                log_file=str(log_path),
            )
            proc.kill()
            await proc.wait()
            return -1

    exit_code = proc.returncode or 0
    logger.info(
        "agent_subprocess_finished",
        sport=sport,
        exit_code=exit_code,
        log_file=str(log_path),
    )
    return exit_code


async def _check_agent_requested_wakeup(sport_key: SportKey) -> datetime | None:
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


async def schedule_next(sport: SportKey) -> ScheduleResult:
    """Compute and schedule the next agent wake-up for a sport.

    Queries the DB for the next kickoff, computes the wake interval from
    proximity tiers, and schedules via self_schedule.

    This is the crash-safe pre-scheduling step — called both during bootstrap
    (without launching the agent) and at the start of a full agent run.

    Raises ValueError if the sport has no configured overnight window.
    """
    if sport not in OVERNIGHT_WINDOWS:
        raise ValueError(
            f"No overnight window configured for {sport} — add it to OVERNIGHT_WINDOWS"
        )

    settings = get_settings()
    compound_job_name = make_compound_job_name("agent-run", sport)
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

    return ScheduleResult(
        next_time=default_next_time,
        hours_until_ko=hours_until_ko,
        compound_job_name=compound_job_name,
        overnight_start_utc=overnight_start_utc,
        overnight_resume_utc=overnight_resume_utc,
    )


async def main(ctx: JobContext) -> None:
    """Orchestrate agent wake-up: schedule, run, check override."""
    settings = get_settings()
    sport = ctx.sport

    if not sport:
        logger.error("agent_run_no_sport", msg="sport is required for agent-run job")
        return

    # --- Pre-schedule before work (crash-safe) ---
    result = await schedule_next(sport)

    # --- Run the agent ---
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

    if requested_time is not None:
        try:
            await self_schedule(
                job_name=result.compound_job_name,
                next_time=requested_time,
                dry_run=settings.scheduler.dry_run,
                sport=sport,
                reason="agent-requested override",
            )
            logger.info(
                "agent_run_rescheduled",
                default_time=result.next_time.isoformat(),
                override_time=requested_time.isoformat(),
            )
        except Exception as e:
            logger.error("agent_run_override_schedule_failed", error=str(e), exc_info=True)


if __name__ == "__main__":
    asyncio.run(main(JobContext(sport="soccer_epl")))
