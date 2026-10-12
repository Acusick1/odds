---
type: component
title: Betting agent runtime and the agent-run job
description: How each agent wake-up runs as a claude subprocess over odds-mcp, how the opt-in, local-only agent-run job schedules wake-ups and honours agent_wakeups, the agent's database role, and the JSONL run traces.
sources:
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_run_claude_agent
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_build_agent_subprocess_env
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::CADENCE
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::schedule_next
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::main
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_check_agent_requested_wakeup
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_log_stream_message
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_prune_agent_run_logs
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::AGENT_RUN_LOG_KEEP
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::OVERNIGHT_WINDOWS
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_SMOKE_UNSAFE_JOBS
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_BOOTSTRAP_MAP
  - packages/odds-core/odds_core/agent_wakeup_models.py::AgentWakeup
  - packages/odds-cli/odds_cli/commands/agent.py::run
  - packages/odds-mcp/odds_mcp/server.py::mcp
  - packages/odds-mcp/odds_mcp/server.py::schedule_next_wakeup
  - experiments/scripts/show_agent_summaries.py::load_summaries
  - docs/BETTING_AGENT.md
  - docs/BETTING_AGENT.md#design-principles
  - docs/BETTING_AGENT.md#architecture
  - docs/BETTING_AGENT.md#role-of-the-xgboost-model
  - docs/ARCHITECTURE.md#agent-run-job
  - docs/CLI.md#agent-commands
  - docs/REPORTING.md#7-agent-run-logs--packagesodds-lambdaodds_lambdajobsagent_runpy
  - docs/REPORTING.md#8-storage-backing-reports
  - docs/REPORTING.md#9-known-limitations
  - docs/audits/comprehension-audit-2026-06-09.md#a14-put-the-agents-prompt-composition-contract-in-version-control
verified: "7887039"
status: stable
---

# Betting agent runtime and the agent-run job

An LLM agent per sport (EPL and MLB) synthesises odds, sharp-market prices and unstructured match context and records bets as paper trades. Both agents are in interactive evaluation, not autonomous production (`docs/BETTING_AGENT.md`). The thesis is breadth and speed of information synthesis ([pivot to information edge](decision-information-edge.md)). The runtime lives in `packages/odds-lambda/odds_lambda/jobs/agent_run.py`.

## Runtime

`_run_claude_agent` spawns `claude -p "/agent {sport}"` with `--model claude-sonnet-4-6`, `--output-format stream-json --verbose` and `--dangerously-skip-permissions`, working directory the repo root, a 15-minute timeout (`AGENT_TIMEOUT_SECONDS`). The subprocess:

- reaches the pipeline through the `odds` MCP server registered in `.mcp.json` (`uv run fastmcp run packages/odds-mcp/odds_mcp/server.py`), see [odds-mcp server](mcp-server.md);
- researches with web search and Playwright;
- runs under `AGENT_DATABASE_URL` when set, otherwise inherits the parent DSN with a logged warning (`_build_agent_subprocess_env`), see [database role](#database-role);
- writes a JSONL trace per run ([run traces](#run-traces)).

Two entry paths: the scheduled `agent-run` job, and `odds agent run --sport S` for one wake-up with no scheduling side effects. `agent-run` is in `_SMOKE_UNSAFE_JOBS` because it places paper bets, so `odds scheduler smoke` skips it unless `--include-agent` ([local scheduler](../scheduling/local-scheduler.md)).

## Prompts

The agent's behaviour is defined by prompt files in `packages/odds-mcp/agents/`: a common prompt (strategy and the wake-up workflow) plus one base prompt per sport (`epl` and `mlb` subdirectories). Those files are authoritative; the docs summarise them ([wake-up workflow](agent-wakeup-workflow.md)).

## The agent-run job

The job is opt-in and local-only: no EventBridge rule exists for it, and the default `SCHEDULER_BOOTSTRAP_JOBS` has `agent-run` commented out, so the loop starts only when you add it ([jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md#where-each-job-runs), [configuration](../foundations/configuration.md)). Its bootstrap function in `_JOB_BOOTSTRAP_MAP` is `schedule_next`, not `main`, so `odds scheduler start` never launches the agent itself.

### Flow (`main`)

1. Requires `ctx.sport`; otherwise logs and returns.
2. `schedule_next(sport)` pre-schedules the next wake-up before any work, so a crash does not break the chain ([self-scheduling](../scheduling/self-scheduling.md)). It calls `decide_forward_resilient` with `CADENCE` and the sport's overnight window, and uses only `next_execution` (the run gate is ignored: the agent always wakes). It raises `ValueError` for a sport missing from `OVERNIGHT_WINDOWS` (EPL 22:00-06:00 UTC, MLB 05:00-14:00).
3. `_run_claude_agent(sport)` runs the subprocess; a non-zero exit is logged, not raised.
4. `_check_agent_requested_wakeup(sport)` reads and consumes the `agent_wakeups` row; a failure is logged and swallowed.
5. A requested time in the past is ignored; otherwise `self_schedule` overrides the pre-scheduled time.

### Cadence

| Hours to kickoff | Interval | Field |
|---|---|---|
| under 3 | 1h | `closing` |
| 3 to 12, 12 to 24 | 4h | `pregame`, `sharp` |
| 24 to 72 | 12h | `early` |
| over 72 | 24h | `opening` |
| no fixture within 7 days | 12h | `no_game` |

Bands come from [fetch tier](../storage/fetch-tier.md) boundaries via the decision engine ([self-scheduling](../scheduling/self-scheduling.md)).

### agent_wakeups

`AgentWakeup` has one row per sport (`uq_agent_wakeup_sport_key`). The MCP tool `schedule_next_wakeup` upserts it, clamps the delay to 0.5-168 hours and resets `consumed_at` ([scheduling tools](mcp-server.md#scrape-and-scheduling-tools)). The job reads rows where `consumed_at IS NULL`. Any future requested time overrides the pre-scheduled one, later as well as sooner.

### `odds agent run`

`odds agent run --sport S [--db NAME]` calls `_run_claude_agent` directly: no pre-schedule, no wake-up consumption, no reschedule. `--db` swaps the database name in `DATABASE_URL` (default `odds`); a timeout exits 1.

## Database role

If `AGENT_DATABASE_URL` is set, the subprocess gets it as `DATABASE_URL`. `.env.example` documents the restricted `odds_agent` role: `SELECT` everywhere, `INSERT, UPDATE` on `paper_trades`, `match_briefs` and `agent_wakeups`, and write access to the APScheduler tables (`(inferred)`: the tables `refresh_scrape` writes through APScheduler). Unset, the agent runs with the parent's write-capable DSN and a warning is logged. `AGENT_DATABASE_URL` is read with `os.getenv`, not `Settings`, so it must be exported in the shell, not only set in `.env` ([configuration](../foundations/configuration.md)).

## Run traces

Each run writes its full `stream-json` output to a file: the only durable record of the run's reasoning, with no database component (`docs/REPORTING.md`).

- Every line goes to `<project_root>/logs/agent_runs/{sport}_{YYYYMMDDTHHMMSSZ}_{pid}.jsonl`. The PID suffix avoids collisions when two runs start in the same second. `logs/` is gitignored.
- The asyncio stream line limit is raised to 8 MiB (`AGENT_STREAM_LINE_LIMIT`), because one `tool_result` block can exceed the 64 KiB default.
- Only some messages are teed to structlog by `_log_stream_message`: assistant `tool_use` blocks become `agent_tool_use` (tool name and a 60-character-per-value input preview), and the `result` message becomes `agent_run_summary` (text, turns, duration, cost).
- `_prune_agent_run_logs` runs before writing and deletes all but the newest `AGENT_RUN_LOG_KEEP` (50) files matching `{sport}_*.jsonl`, so up to 51 files exist per sport after a run.
- The subprocess is killed after `AGENT_TIMEOUT_SECONDS`; the partial trace stays.

The viewer `experiments/scripts/show_agent_summaries.py` prints the `result` summaries: `uv run python experiments/scripts/show_agent_summaries.py --sport epl --since 24h --limit 3`, or `--file <path>`. `--sport` takes the short names `epl` or `mlb`. One file can hold several sessions; the script tracks a session index per `result` line. The filename is the only schema linking producer and consumer: `FILENAME_RE` in the viewer expects `{sport_key}_{timestamp}_{pid}.jsonl`, so renaming either side breaks the other silently. Other report channels are in [reporting](../operations/reporting-and-daily-digest.md).

## Role of the CLV model

A supplementary signal: weak, unprofitable alone, and its strongest feature is the sharp-retail spread the agent sees directly via `find_retail_edges`. The prompt treats models as sanity checks ([CLV model findings](../evaluation/clv-model-findings.md), [sharp benchmark](../modelling/sharp-benchmark.md)).

## Design principles

Recorded without a decision record, so the reasons are `(inferred)`: sharp price is a strong reference, not gospel; breadth of information is the edge; MCP is the tool interface so the runtime can be swapped; tools return structured data; the database is shared state; local-first, never on AWS.

## Hazards

- The `/agent` slash command lives under the `.claude/` directory, which is gitignored, so it is not in the repo. `_run_claude_agent` passes the full sport key (`soccer_epl`, `baseball_mlb`), and what the command does with it cannot be reviewed from the repo.
- The model id is hardcoded in `_run_claude_agent`.
- **Override docstrings are wrong.** The module docstring and the `schedule_next_wakeup` docstring say the override applies "if it's sooner"; the code reschedules to any future time, so a later request delays the wake-up past the proximity schedule.
- **`--db` has no effect when `AGENT_DATABASE_URL` is set**: the subprocess environment replaces `DATABASE_URL` with it (`_build_agent_subprocess_env`).
- `packages/odds-cli/odds_cli/commands/agent.py` imports the underscore-private `_run_claude_agent` across packages.
- Deleting `logs/agent_runs/` loses the traces; the agent's own persistent memory is `match_briefs` ([briefs](match-briefs-and-paper-trades.md)).
- The viewer's `--since` filters on the file's start time taken from its name, so every session in a file that started before the cutoff is dropped, although a code comment says later sessions are still walked (`load_summaries`).

## Related

- [Wake-up workflow and staking rules](agent-wakeup-workflow.md), [Agent data sources](agent-data-sources.md)
- [Match briefs and paper trades](match-briefs-and-paper-trades.md) — what the agent writes
- [Project overview](../foundations/project-overview.md)
