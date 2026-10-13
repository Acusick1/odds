---
status: accepted
---

# Source lineups from live web research, not a lineup API

## Context

Confirmed lineups land about 60 minutes before kick-off (`packages/odds-mcp/agents/epl/base.md`). The conclusion recorded on 2026-04-13 is that every structured lineup API investigated "is either paywalled, blocks current-season data on free tier, or has unconfirmed pre-KO timing" (`docs/AGENT_DATA_SOURCES.md`, added in 3f74022). The prices, plan tiers and timings below are external facts as recorded on those dates, and the repo cannot verify them. No probe script or results file records the tests.

## Decision

The agent reads lineups through web search and Playwright. In the recorded words, "BBC Sport publishes confirmed lineups within minutes of the official team sheet (KO-60), and the agent can read those directly. No new infrastructure needed." (3f74022).

The code matches the decision:

- No job or MCP tool fetches lineups (`odds_lambda.scheduling.jobs._JOB_MODULE_MAP`, `odds_mcp.server`).
- The only lineup writer, `odds_lambda.storage.espn_lineup_writer.EspnLineupWriter`, is called from the manual `scripts/ingest_espn_lineups.py`, which supplies the historical lineups behind the `epl_lineup` features.
- ESPN stays in use for fixtures (`odds_lambda.jobs.fetch_espn_fixtures`).

## Considered options

- **API-Football.** #295 (2026-04-12) planned a `get_confirmed_lineups` MCP tool on the free tier, citing 100 requests per day and lineups 55-75 minutes before kick-off. #296 implemented the tool and was closed unmerged. The recorded reasons for rejecting it (3f74022):
  - The free tier blocks current-season (2025-26) fixture lookup and is limited to 2022-2024.
  - The lineups endpoint needs fixture ids from that gated search.
  - The Pro plan, about $20 per month, is required.
- **football-data.org.** Lineups are paywalled at €29 per month (Deep Data plan), and their timing is undocumented even when paid (3f74022).
- **ESPN API.** The first record found the summary endpoint empty about 10 h before kick-off and KO-60 timing unconfirmed (3f74022). The record revised on 2026-04-14 (c0ecdf5) adds two tests:
  - In a Champions League test (2026-04-14), lineups appeared at KO-60 and had gone by KO-45.
  - In an EPL test (2026-04-13), nothing appeared at KO-59.

  The endpoint also returns 403 from cloud and datacenter IPs. These were single manual tests.

## Consequences

- No live lineup data is stored in the database.
- The `epl_lineup` features compare a team's XI for the match itself with its previous XI. That XI is announced about 75 minutes before kick-off, but training rows are sampled 3-24 h out (#244). #244 planned to split the group and gate the match-specific part, but it was closed without that change (#481). The recorded signal check (fa40196) found that:
  - the match-specific features reach CV R² 0.060 but "require post-lineup-announcement execution";
  - the tendency features score R² -0.017.
- How fresh the agent's lineup information is depends on its wake-up cadence. Inside 3 h of kick-off, `odds_lambda.jobs.agent_run.CADENCE` wakes the agent hourly, with the comment "lineups dropping, final decisions" (#322). No wake-up is scheduled at the announcement itself.

## Evidence

- PRs and issues: #244, #295, #296, #322, #481.
- Commits: 3f74022 (`docs/AGENT_DATA_SOURCES.md` added, 2026-04-13), c0ecdf5 (ESPN row revised, 2026-04-14), fa40196 (lineup signal check).
- Prior text: `git show 3f74022:docs/AGENT_DATA_SOURCES.md` ("Evaluated and Rejected", "Lineup Data Source Conclusion"); `git show origin/main:docs/AGENT_DATA_SOURCES.md`.
- Code and prompts: `packages/odds-mcp/agents/epl/base.md`, `odds_lambda.jobs.agent_run.CADENCE`, `odds_lambda.scheduling.jobs._JOB_MODULE_MAP`, `odds_lambda.storage.espn_lineup_writer.EspnLineupWriter`, `scripts/ingest_espn_lineups.py`.
- Wiki: [Agent data sources and rejected sources](../../wiki/agent/agent-data-sources.md#rejected-sources), [The betting agent](../../wiki/agent/betting-agent.md), [EPL context data](../../wiki/collection/epl-context-data.md).
