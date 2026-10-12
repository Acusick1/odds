---
type: component
title: agent-run job and wake-up scheduling
summary: "The local-only job that runs one agent wake-up: fixture-proximity cadence, agent_wakeups overrides, the AGENT_DATABASE_URL role and the odds agent run command."
sources:
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::CADENCE
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::schedule_next
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::main
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_check_agent_requested_wakeup
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::OVERNIGHT_WINDOWS
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_BOOTSTRAP_MAP
  - packages/odds-core/odds_core/agent_wakeup_models.py::AgentWakeup
  - packages/odds-cli/odds_cli/commands/agent.py::run
  - packages/odds-mcp/odds_mcp/server.py::schedule_next_wakeup
  - docs/BETTING_AGENT.md#architecture
  - docs/ARCHITECTURE.md#agent-run-job
  - docs/CLI.md#agent-commands
  - docs/REPORTING.md#8-storage-backing-reports
verified_at: 7887039
---

# agent-run job and wake-up scheduling

`packages/odds-lambda/odds_lambda/jobs/agent_run.py` runs one [betting agent](betting-agent.md) wake-up per invocation and books the next one. It is local-only: no EventBridge rule exists for it, and the default `SCHEDULER_BOOTSTRAP_JOBS` omits it ([Job deployment matrix](job-deployment-matrix.md), [Local scheduler](local-scheduler.md)).

## Flow (`main`)

1. Requires `ctx.sport`; otherwise logs and returns.
2. `schedule_next(sport)` pre-schedules the next wake-up before any work, so a crash does not break the chain ([Self-scheduling](self-scheduling.md)). It calls `decide_forward_resilient` with `CADENCE` and the sport's overnight window, and uses only `next_execution` (the run gate is ignored: the agent always wakes). It raises `ValueError` for a sport missing from `OVERNIGHT_WINDOWS` (EPL 22:00-06:00 UTC, MLB 05:00-14:00).
3. `_run_claude_agent(sport)` runs the subprocess; a non-zero exit is logged, not raised.
4. `_check_agent_requested_wakeup(sport)` reads and consumes the `agent_wakeups` row; a failure is logged and swallowed.
5. A requested time in the past is ignored; otherwise `self_schedule` overrides the pre-scheduled time.

## Cadence

| Hours to kickoff | Interval | Field |
|---|---|---|
| under 3 | 1h | `closing` |
| 3 to 12, 12 to 24 | 4h | `pregame`, `sharp` |
| 24 to 72 | 12h | `early` |
| over 72 | 24h | `opening` |
| no fixture within 7 days | 12h | `no_game` |

Bands come from [Fetch tier](fetch-tier.md) boundaries via the [decision engine](scheduling-decision-engine.md). Bootstrap uses `_JOB_BOOTSTRAP_MAP`: `scheduler start` calls `schedule_next`, not `main`, so starting the scheduler never launches the agent.

## agent_wakeups

`AgentWakeup` has one row per sport (`uq_agent_wakeup_sport_key`). The MCP tool `schedule_next_wakeup` upserts it, clamps the delay to 0.5-168 hours and resets `consumed_at` ([scheduling tools](mcp-scrape-and-schedule-tools.md)). The job reads rows where `consumed_at IS NULL`.

## Database role

If `AGENT_DATABASE_URL` is set, the subprocess gets it as `DATABASE_URL`. `.env.example` documents the restricted `odds_agent` role: `SELECT` everywhere, `INSERT, UPDATE` on `paper_trades`, `match_briefs` and `agent_wakeups`, and write access to the APScheduler tables (`(inferred)`: the tables `refresh_scrape` writes through APScheduler). Unset, the agent runs with the parent's write-capable DSN and a warning is logged.

## `odds agent run`

`odds agent run --sport S [--db NAME]` calls `_run_claude_agent` directly: no pre-schedule, no wake-up consumption, no reschedule. `--db` swaps the database name in `DATABASE_URL` (default `odds`); a timeout exits 1.

## Hazards

- The module docstring and the `schedule_next_wakeup` docstring say the override applies "if it's sooner". The code reschedules to any requested time in the future, so a later request delays the wake-up past the proximity schedule.
- `--db` has no effect when `AGENT_DATABASE_URL` is set: the subprocess environment replaces `DATABASE_URL` with it (`_build_agent_subprocess_env`).
- `packages/odds-cli/odds_cli/commands/agent.py` imports the underscore-private `_run_claude_agent` across packages.

## Related

- [Agent run traces](agent-run-traces.md)
- [Matchday wake-up](matchday-wakeup.md)
