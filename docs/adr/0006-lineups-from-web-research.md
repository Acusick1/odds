---
status: proposed
---

# Source lineups from live web research, not a lineup API

## Context

Lineups land about 60 minutes before kick-off (`packages/odds-mcp/agents/epl/base.md`). As of 2026-04-13 every structured lineup API investigated is paywalled, blocks current-season data on its free tier, or has unconfirmed pre-kick-off timing. Most of this evidence is external and cannot be verified from the repo.

## Decision

The agent reads lineups from BBC Sport and club sites via web search and Playwright at about KO-60. No scheduled lineup job exists. ESPN stays in use for fixtures, and the manual ESPN lineup script (`scripts/ingest_espn_lineups.py`) feeds training data only.

## Considered options

- **API-Football.** The free tier covers 2022-2024 only and the lineups endpoint needs the gated fixture ids; Pro costs about $20 per month (#295, #296, closed unmerged).
- **football-data.org.** Lineups cost €29 per month and the timing is undocumented.
- **ESPN API.** Lineups appear around KO-60 but non-monotonically (present at KO-60 and gone by KO-45 in a 2026-04-14 Champions League test; nothing at KO-59 in a 2026-04-13 EPL test), and cloud and datacenter IPs get a 403. No results file records the probes.

## Consequences

- No live lineup data is stored in the database.
- `epl_lineup` features exist only for historical training and carry a look-ahead caveat: the XI is public about KO-60 while training samples 3-24 h out.
- Lineup freshness depends on the agent's wake-up cadence placing a run inside the lineup window `(inferred)`.

## Evidence

- PRs: #295, #296.
- Prior text: `wiki/agent/agent-data-sources.md` at f2a0862; `docs/AGENT_DATA_SOURCES.md` at origin/main; `packages/odds-mcp/agents/epl/base.md`.
- Lineup-window cadence: `wiki/agent/agent-wakeup-workflow.md` at f2a0862, which labels it `(inferred)`.
- Wiki: [Agent data sources and rejected sources](../../wiki/agent/agent-data-sources.md#rejected-sources), [The betting agent](../../wiki/agent/betting-agent.md), [EPL context data](../../wiki/collection/epl-context-data.md).
