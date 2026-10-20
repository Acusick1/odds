---
type: reference
title: Agent run traces
description: Where each agent run's stream-json trace is written, how files are named, limited and pruned, which messages reach structlog, and what the summary viewer reads.
sources:
  - odds_lambda.jobs.agent_run._run_claude_agent
  - odds_lambda.jobs.agent_run._log_stream_message
  - odds_lambda.jobs.agent_run._prune_agent_run_logs
  - odds_lambda.jobs.agent_run.AGENT_RUN_LOG_KEEP
  - odds_lambda.jobs.agent_run.AGENT_STREAM_LINE_LIMIT
  - odds_lambda.jobs.agent_run.AGENT_TIMEOUT_SECONDS
  - odds_core.match_brief_models.MatchBrief
  - experiments/scripts/show_agent_summaries.py
status: stable
---

# Agent run traces

Each run of the [agent runtime](agent-run-job.md#runtime) writes its full `stream-json` output to a file. The file is the only durable record of the run's reasoning and has no database component: `odds_lambda.jobs.agent_run._run_claude_agent` writes the file only, and the agent's persistent memory is `odds_core.match_brief_models.MatchBrief` ([match briefs](match-briefs.md)). Deleting `logs/agent_runs/` loses the traces.

## Files

- Every line goes to `<project_root>/logs/agent_runs/{sport}_{YYYYMMDDTHHMMSSZ}_{pid}.jsonl`. `logs/` is gitignored.
- The asyncio stream line limit is `odds_lambda.jobs.agent_run.AGENT_STREAM_LINE_LIMIT` (8 MiB); the asyncio default is 64 KiB.
- `odds_lambda.jobs.agent_run._prune_agent_run_logs` runs before writing and deletes all but the newest `odds_lambda.jobs.agent_run.AGENT_RUN_LOG_KEEP` (50) files matching `{sport}_*.jsonl`, so up to 51 files exist per sport after a run.
- The subprocess is killed after `odds_lambda.jobs.agent_run.AGENT_TIMEOUT_SECONDS`, or when the run is cancelled; the partial trace stays.

## Structlog events

`odds_lambda.jobs.agent_run._log_stream_message` tees only some messages: assistant `tool_use` blocks become `agent_tool_use` (tool name and a 60-character-per-value input preview), and the `result` message becomes `agent_run_summary` (text, turns, duration, cost).

## Viewer

`experiments/scripts/show_agent_summaries.py` prints the `result` summaries:

- `uv run python experiments/scripts/show_agent_summaries.py --sport epl --since 24h --limit 3`, or `--file <path>`.
- `--sport` takes the short names `epl` or `mlb`.
- One file can hold several sessions; the script tracks a session index per `result` line.
- `FILENAME_RE` in the viewer expects `{sport_key}_{timestamp}_{pid}.jsonl`, the same pattern the producer writes, so the two are coupled only by filename.
- `--since` filters on the file's start time taken from its name, so every session in a file that started before the cutoff is dropped (#479).

Other report channels are in [reporting](../operations/reporting-surfaces.md).
