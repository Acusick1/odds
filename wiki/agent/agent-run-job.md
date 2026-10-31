---
type: reference
title: Agent runtime and the agent-run job
description: The claude subprocess, the agent-run job flow and cadence, agent_wakeups, odds agent run and the database role.
sources:
  - odds_lambda.jobs.agent_run._run_claude_agent
  - odds_lambda.jobs.agent_run._build_agent_subprocess_env
  - odds_lambda.jobs.agent_run.CADENCE
  - odds_lambda.jobs.agent_run.AGENT_TIMEOUT_SECONDS
  - odds_lambda.jobs.agent_run.schedule_next
  - odds_lambda.jobs.agent_run.main
  - odds_lambda.jobs.agent_run._check_agent_requested_wakeup
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_lambda.scheduling.jobs._SMOKE_UNSAFE_JOBS
  - odds_lambda.scheduling.jobs._JOB_BOOTSTRAP_MAP
  - odds_core.agent_wakeup_models.AgentWakeup
  - odds_core.match_brief_models.MatchBrief
  - odds_cli.commands.agent.run
  - odds_mcp.server.schedule_next_wakeup
  - odds_mcp.server.refresh_scrape
  - .mcp.json
  - .gitignore
status: stable
---

# Agent runtime and the agent-run job

The runtime of the [betting agent](betting-agent.md) lives in `odds_lambda.jobs.agent_run`.

## Runtime

`odds_lambda.jobs.agent_run._run_claude_agent` spawns `claude -p "/agent {sport}"` with `--model claude-sonnet-4-6` (hard-coded in the function, #414), `--output-format stream-json --verbose` and `--dangerously-skip-permissions`, working directory the repo root, a 15-minute timeout (`odds_lambda.jobs.agent_run.AGENT_TIMEOUT_SECONDS`). A cancelled run (the scheduler's own 20-minute limit, or shutdown) kills the subprocess before the cancellation propagates. The `/agent` slash command is not in the repository; `.gitignore` excludes the whole `.claude/` directory (#479). The function passes the full sport key (`soccer_epl`, `baseball_mlb`), and what the command does with it cannot be reviewed from the repo. The subprocess:

- reaches the pipeline through the `odds` MCP server registered in `.mcp.json` (`uv run fastmcp run odds_mcp.server`), see [odds-mcp server](mcp-server.md);
- researches with web search and Playwright;
- runs under `AGENT_DATABASE_URL` when set, otherwise inherits the parent DSN with a logged warning (`odds_lambda.jobs.agent_run._build_agent_subprocess_env`), see [database role](#database-role);
- writes a JSONL trace per run ([run traces](agent-run-traces.md)).

Two entry paths: the scheduled `agent-run` job, and `odds agent run --sport S` for one wake-up with no scheduling side effects. `agent-run` is in `_SMOKE_UNSAFE_JOBS` because it places paper bets (as does [`run-candidates`](../evaluation/run-candidates-job.md), which has its own `--include-candidates`), so `odds scheduler smoke` skips it unless `--include-agent` ([local scheduler](../scheduling/local-scheduler.md#cli)).

## The agent-run job

The job is opt-in and local-only: no EventBridge rule exists for it, and the default `SCHEDULER_BOOTSTRAP_JOBS` has `agent-run` commented out, so the loop starts only when you add it ([where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs), [configuration](../foundations/configuration.md)). Its bootstrap function in `_JOB_BOOTSTRAP_MAP` is `odds_lambda.jobs.agent_run.schedule_next`, not `odds_lambda.jobs.agent_run.main`, so `odds scheduler start` never launches the agent itself, and it calls even that only when no wake-up is pending ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).

### Flow

1. Requires `ctx.sport`; otherwise raises `ValueError`.
2. `schedule_next(sport)` pre-schedules the next wake-up before any work, so a crash does not break the chain ([self-scheduling](../scheduling/self-scheduling.md)). It calls `odds_lambda.scheduling.decision.decide_forward_resilient` with `odds_lambda.jobs.agent_run.CADENCE`, the sport's overnight window and no leads (so no `run_started`), and uses only `next_execution` (the run gate is ignored: the agent always wakes); kickoffs are the stored events' `commence_time`. Its off-season check-in does not apply the overnight skip ([decision engine](../scheduling/decision-engine.md#forward-decisions)). It raises `ValueError` for a sport missing from `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS` (EPL 22:00-06:00 UTC, MLB 05:00-14:00, NBA 07:00-15:00).
3. `odds_lambda.jobs.agent_run._run_claude_agent(sport)` runs the subprocess; a non-zero exit is logged, not raised.
4. `_check_agent_requested_wakeup(sport)` reads and consumes the `agent_wakeups` row; a failure is logged and swallowed.
5. A requested time in the past is ignored; otherwise `odds_lambda.scheduling.helpers.self_schedule` overrides the pre-scheduled time.

### Cadence

The wake-up interval is the `agent-run` row of the [cadence table](../scheduling/decision-engine.md#cadence): bands come from [fetch tier](../storage/fetch-tier.md) boundaries via the decision engine, with 12 h when no fixture is within 7 days.

## Wake-up requests

`odds_core.agent_wakeup_models.AgentWakeup` has one row per sport (`uq_agent_wakeup_sport_key`). The MCP tool `odds_mcp.server.schedule_next_wakeup` upserts it, clamps the delay to 0.5-168 hours and resets `consumed_at` ([scheduling tools](mcp-server.md#scrape-and-scheduling-tools)). The job reads rows where `consumed_at IS NULL`. Any future requested time overrides the proximity schedule, later as well as sooner (#449).

### `odds agent run`

`odds agent run --sport S [--db NAME]` calls `odds_lambda.jobs.agent_run._run_claude_agent` directly: no pre-schedule, no wake-up consumption, no reschedule. `--db` swaps the database name in `DATABASE_URL` (default `odds`, the database the local scheduler writes scraper output to); a timeout exits 1. `odds_cli.commands.agent` imports the underscore-private `odds_lambda.jobs.agent_run._run_claude_agent` across packages ([Workspace packages](../foundations/workspace-packages.md)).

## Database role

> [!WARNING]
> Without `AGENT_DATABASE_URL` exported in the shell, the agent subprocess runs with `--dangerously-skip-permissions` against the parent's write-capable `DATABASE_URL` (#444).

If `AGENT_DATABASE_URL` is set, the subprocess gets it as `DATABASE_URL`. `.env.example` documents the restricted `odds_agent` role ([Databases and environments](../foundations/databases-and-environments.md#environments)): `SELECT` everywhere, `INSERT, UPDATE` on `paper_trades`, `match_briefs` and `agent_wakeups`, and write access to the APScheduler tables (`(inferred)`: the tables `odds_mcp.server.refresh_scrape` writes to through APScheduler). Unset, the agent runs with the parent's write-capable DSN and a warning is logged. `AGENT_DATABASE_URL` is read with `os.getenv`, not a field of `odds_core.config.Settings`, so it must be exported in the shell, not only set in `.env` ([configuration](../foundations/configuration.md#variables-read-outside-settings)). When it is set, `--db` has no effect, because the subprocess environment replaces `DATABASE_URL` with it (`odds_lambda.jobs.agent_run._build_agent_subprocess_env`, #445).

## Run traces

Each run writes its `stream-json` output to a JSONL file under `logs/agent_runs/`, described in [Agent run traces](agent-run-traces.md).

## Related

- [The betting agent](betting-agent.md)
- [Agent prompts](agent-prompts.md)
- [Agent run traces](agent-run-traces.md)
- [Match briefs](match-briefs.md) and [paper trading](paper-trading.md), what the agent writes, each row keyed to an event by its integer id
