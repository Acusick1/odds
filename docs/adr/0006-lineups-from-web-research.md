---
status: proposed
---

# Source lineups from live web research, not a lineup API

## Context

Lineups land about 60 minutes before kick-off (`packages/odds-mcp/agents/epl/base.md`). As of 2026-04-13 every structured lineup API investigated is paywalled, blocks current-season data on its free tier, or has unconfirmed timing:

- API-Football: the free tier covers 2022-2024 only (#295, #296, closed unmerged).
- football-data.org: lineups cost €29 per month and the timing is undocumented.
- ESPN: lineups were non-monotonic at KO-60 and KO-45 in manual probes on 2026-04-13 and 2026-04-14, and cloud IPs get a 403. No results file records the probes.

Most of this evidence is external and cannot be verified from the repo.

## Decision

The agent reads lineups from BBC Sport and club sites via web search and Playwright at about KO-60. No scheduled lineup job exists. ESPN stays in use for fixtures, and the manual ESPN lineup script (`scripts/ingest_espn_lineups.py`) feeds training data only.

## Consequences

- No live lineup data is stored in the database.
- `epl_lineup` features exist only for historical training and carry a look-ahead caveat: the XI is public about KO-60 while training samples 3-24 h out.
- Lineup freshness depends on the agent's wake-up cadence placing a run inside the lineup window `(inferred)`.

## References

- Explanation: [Agent data sources and rejected sources](../../wiki/agent/agent-data-sources.md#rejected-sources), [The betting agent](../../wiki/agent/betting-agent.md), [EPL context data](../../wiki/collection/epl-context-data.md)
