"""Run the betting agent and self-schedule based on fixture proximity.

Scheduling strategy: query DB for next kickoff, compute the wake interval via
the unified decision engine (proximity → cadence on canonical FetchTier
boundaries), pre-schedule before launching the agent subprocess (survives
crashes). After the agent exits, check the agent_wakeups table for an
agent-requested override and reschedule if it's sooner.

Wake cadence (canonical FetchTier boundaries, see ``CADENCE`` below):
  72h+ to KO    -> 24h  (far out)
  24-72h        -> 12h  (research window opening)
  3-24h         ->  4h  (active research)
  < 3h          ->  1h  (lineups dropping, final decisions)
  no fixtures   -> 12h  (off-season check-in)
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

import structlog
from odds_core.config import get_settings
from odds_core.sports import SportKey

from odds_lambda.scheduling.decision import (
    OVERNIGHT_WINDOWS,
    CadenceConfig,
    decide_forward_resilient,
)
from odds_lambda.scheduling.helpers import self_schedule
from odds_lambda.scheduling.jobs import JobContext, make_compound_job_name

logger = structlog.get_logger()

# Wake cadence (hours) keyed by canonical FetchTier proximity bands.
CADENCE = CadenceConfig(
    closing=1.0,  # < 3h to KO — lineups dropping, final decisions
    pregame=4.0,  # 3-12h
    sharp=4.0,  # 12-24h — active research
    early=12.0,  # 24-72h — research window opening
    opening=24.0,  # 72h+ — far out
    no_game=12.0,  # no fixtures within lookahead — off-season check-in
)

# Agent subprocess limits
AGENT_TIMEOUT_SECONDS = 15 * 60  # 15 minutes

# Retain this many per-run JSONL logs per sport; older ones are pruned on write.
AGENT_RUN_LOG_KEEP = 50

# Upper bound on any single stream-json line. The asyncio default (64 KiB) is
# too small — a single ``tool_result`` block (e.g. a large Read or WebFetch
# payload) routinely exceeds it and would raise ``LimitOverrunError``.
AGENT_STREAM_LINE_LIMIT = 8 * 1024 * 1024


class ScheduleResult(NamedTuple):
    """Values computed by schedule_next(), reused by main() for overrides."""

    next_time: datetime
    compound_job_name: str
    should_execute: bool


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


def _build_agent_subprocess_env() -> dict[str, str]:
    """Return an environment dict for the agent subprocess.

    When ``AGENT_DATABASE_URL`` is set, its value replaces ``DATABASE_URL``
    so the agent connects as the read-mostly ``odds_agent`` role rather
    than inheriting the scraper/scheduler's write-capable DSN. When it is
    unset the subprocess inherits the parent's ``DATABASE_URL`` and a
    warning is logged — the security control is then disabled but the
    agent continues to function, preserving pre-control behaviour for
    operators who haven't provisioned the role yet.
    """
    env = os.environ.copy()
    agent_dsn = env.get("AGENT_DATABASE_URL")
    if agent_dsn:
        env["DATABASE_URL"] = agent_dsn
    else:
        logger.warning(
            "agent_database_url_not_set",
            msg=(
                "AGENT_DATABASE_URL not set — agent will run with scraper's "
                "write-capable DSN (security control disabled)"
            ),
        )
    return env


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
        "--model",
        "claude-sonnet-4-6",
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

    subprocess_env = _build_agent_subprocess_env()

    logger.info("agent_subprocess_starting", sport=sport, cmd=cmd, log_file=str(log_path))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(settings.project_root),
        limit=AGENT_STREAM_LINE_LIMIT,
        env=subprocess_env,
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

    Uses the unified decision engine to map fixture proximity to a wake
    interval (canonical FetchTier boundaries, agent ``CADENCE`` values) with
    the sport's overnight suppression applied. The returned ``should_execute``
    is the season gate: ``False`` when the next fixture is beyond the lead
    window (or absent), in which case the precise wake is scheduled and the
    caller must skip launching the agent.

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

    decision = await decide_forward_resilient(
        [sport],
        CADENCE,
        overnight=OVERNIGHT_WINDOWS[sport],
        lookahead_days=settings.scheduler.lead_days_for(sport),
    )
    assert decision.next_execution is not None  # noqa: S101
    logger.info("agent_proximity", sport=sport, reason=decision.reason)

    try:
        await self_schedule(
            job_name=compound_job_name,
            next_time=decision.next_execution,
            dry_run=settings.scheduler.dry_run,
            sport=sport,
            reason=f"pre-schedule: {decision.reason}",
        )
    except Exception as e:
        logger.error("agent_run_preschedule_failed", error=str(e), exc_info=True)
        raise

    return ScheduleResult(
        next_time=decision.next_execution,
        compound_job_name=compound_job_name,
        should_execute=decision.should_execute,
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

    # --- Season gate: skip the agent run when no fixture is within the lead ---
    if not result.should_execute:
        logger.info(
            "agent_run_season_gated",
            sport=sport,
            next_execution=result.next_time.isoformat(),
        )
        return

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
