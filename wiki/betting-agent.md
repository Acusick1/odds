---
type: component
title: Betting agent
summary: "The per-sport LLM agent run as a claude subprocess over odds-mcp, web search and Playwright: runtime, prompt composition, design principles and status."
sources:
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_run_claude_agent
  - packages/odds-lambda/odds_lambda/jobs/agent_run.py::_build_agent_subprocess_env
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_SMOKE_UNSAFE_JOBS
  - packages/odds-cli/odds_cli/commands/agent.py::run
  - packages/odds-mcp/odds_mcp/server.py::mcp
  - docs/BETTING_AGENT.md
  - docs/BETTING_AGENT.md#design-principles
  - docs/BETTING_AGENT.md#architecture
  - docs/BETTING_AGENT.md#role-of-the-xgboost-model
  - docs/audits/comprehension-audit-2026-06-09.md#a14-put-the-agents-prompt-composition-contract-in-version-control
verified_at: 7887039
---

# Betting agent

An LLM agent per sport (EPL and MLB) that synthesises odds, sharp-market prices and unstructured match context and records bets as paper trades. Both agents are in interactive evaluation, not autonomous production ([BETTING_AGENT.md](../docs/BETTING_AGENT.md)). The thesis is breadth and speed of information synthesis, see [Pivot to information edge](decision-pivot-to-information-edge.md).

## Runtime

`packages/odds-lambda/odds_lambda/jobs/agent_run.py::_run_claude_agent` spawns `claude -p "/agent {sport}"` with `--model claude-sonnet-4-6`, `--output-format stream-json --verbose` and `--dangerously-skip-permissions`, working directory the repo root, a 15-minute timeout (`AGENT_TIMEOUT_SECONDS`). The subprocess:

- reaches the pipeline through the `odds` MCP server registered in `.mcp.json` (`uv run fastmcp run packages/odds-mcp/odds_mcp/server.py`), see [odds-mcp server](mcp-server.md);
- researches with web search and Playwright;
- runs under `AGENT_DATABASE_URL` when set, otherwise inherits the parent DSN with a logged warning (`_build_agent_subprocess_env`), see [Run the agent job](job-agent-run.md);
- writes a JSONL trace per run ([Agent run traces](agent-run-traces.md)).

Two entry paths: the scheduled `agent-run` job (self-scheduling by fixture proximity) and `odds agent run --sport S` for one wake-up with no scheduling side effects. `agent-run` is in `_SMOKE_UNSAFE_JOBS` because it places paper bets.

## Prompts

The agent's behaviour is defined by prompt files in `packages/odds-mcp/agents/`: a common prompt (strategy and the wake-up workflow) plus one base prompt per sport (`epl` and `mlb` subdirectories). Those files are authoritative; the doc summarises them ([Matchday wake-up](matchday-wakeup.md), [Conviction tiers](conviction-tiers.md)).

## Design principles

Recorded without a decision record, so the reasons are `(inferred)`: sharp price is a strong reference, not gospel; breadth of information is the edge; MCP is the tool interface so the runtime can be swapped; tools return structured data; the database is shared state; local-first, never on AWS ([Job deployment matrix](job-deployment-matrix.md)).

## Role of the CLV model

A supplementary signal: weak, unprofitable alone, and its strongest feature is the sharp-retail spread the agent sees directly via `find_retail_edges`. The prompt treats models as sanity checks ([CLV model findings](clv-model-findings.md), [Retail-sharp divergence](retail-sharp-divergence.md)).

## Hazards

- The `/agent` slash command lives under the `.claude/` directory, which is gitignored, so it is not in the repo. `_run_claude_agent` passes the full sport key (`soccer_epl`, `baseball_mlb`), and what the command does with it cannot be reviewed from the repo.
- The model id is hardcoded in `_run_claude_agent`.
- The default `SCHEDULER_BOOTSTRAP_JOBS` has `agent-run` commented out, so the loop starts only when you add it ([Configuration](configuration.md)).

## Related

- [Agent data sources](agent-data-sources.md) — what the agent reads
- [Match briefs](match-briefs.md), [Paper trading](paper-trading.md) — what it writes
- [Project pivots](project-pivots.md)
