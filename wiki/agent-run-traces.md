---
type: component
title: Agent run traces
summary: "JSONL stream traces under logs/agent_runs, the structlog tee, pruning, and the show_agent_summaries viewer."
sources:
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_run_claude_agent
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_log_stream_message
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_prune_agent_run_logs
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::AGENT_RUN_LOG_KEEP
  - experiments/scripts/show_agent_summaries.py::load_summaries
  - docs/REPORTING.md#7-agent-run-logs--packagesodds-lambdaodds_lambdajobsagent_runpy
  - docs/REPORTING.md#9-known-limitations
verified_at: 7887039
---

# Agent run traces

Each [agent-run](job-agent-run.md) subprocess writes its full `stream-json` output to a file. This is the only durable record of the run's reasoning; there is no database component ([REPORTING.md](../docs/REPORTING.md#7-agent-run-logs--packagesodds-lambdaodds_lambdajobsagent_runpy)).

## Producer

`_run_claude_agent` in `packages/odds-lambda/odds_lambda/jobs/agent_run.py`:

- Writes every line to `<project_root>/logs/agent_runs/{sport}_{YYYYMMDDTHHMMSSZ}_{pid}.jsonl`. The PID suffix avoids collisions when two runs start in the same second. `logs/` is gitignored.
- Raises the asyncio stream line limit to 8 MiB (`AGENT_STREAM_LINE_LIMIT`), because one `tool_result` block can exceed the 64 KiB default.
- Tees only some messages to structlog via `_log_stream_message`: assistant `tool_use` blocks become `agent_tool_use` (tool name and a 60-character-per-value input preview), and the `result` message becomes `agent_run_summary` (text, turns, duration, cost). Everything else stays in the file.
- Prunes before writing: `_prune_agent_run_logs` deletes all but the newest `AGENT_RUN_LOG_KEEP` (50) files matching `{sport}_*.jsonl`, so up to 51 files exist per sport after a run.
- Kills the subprocess after `AGENT_TIMEOUT_SECONDS` (15 minutes); the partial trace stays.

## Consumer

`experiments/scripts/show_agent_summaries.py` prints the `result` summaries: `uv run python experiments/scripts/show_agent_summaries.py --sport epl --since 24h --limit 3`, or `--file <path>`. `--sport` takes the short names `epl` or `mlb` and maps them to sport keys. One file can hold several sessions; the script tracks a session index per `result` line.

## Contract

The filename is the only schema linking producer and consumer: `FILENAME_RE` in the viewer expects `{sport_key}_{timestamp}_{pid}.jsonl`. Renaming either side breaks the other silently (the viewer skips non-matching names).

## Hazards

- Deleting `logs/agent_runs/` loses the traces; the agent's own persistent memory is `match_briefs` ([Match briefs](match-briefs.md)).
- `--since` filters on the file's start time taken from its name, so every session in a file that started before the cutoff is dropped, although a code comment says later sessions are still walked (`load_summaries`). See [Reporting surfaces](reporting-surfaces.md) for the other report channels.

## Related

- [Betting agent](betting-agent.md)
- [Code conventions](code-conventions.md#logging) — structlog setup
