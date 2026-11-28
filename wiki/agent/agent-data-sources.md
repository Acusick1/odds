---
type: explanation
title: Agent data sources and rejected sources
description: Which sources are collected on a schedule and exposed through MCP versus researched live each wake-up, why lineups come from web search, and why every evaluated source was rejected.
sources:
  - scripts.ingest_espn_lineups
  - packages/odds-mcp/agents/epl/base.md
  - packages/odds-mcp/agents/mlb/base.md
status: stable
---

# Agent data sources and rejected sources

The [betting agent](betting-agent.md) draws on two kinds of source. The split decides where a new source belongs: stable structured data is collected and queried through MCP, while qualitative and fast-changing context is researched live. The reasons below are external facts the repo cannot verify, as of the dates given (April 2026); prices and tier names in the rejected-source table are likewise external. The reasons were recorded in the agent data-sources doc this wiki replaced (added in 3f74022 on 2026-04-13, ESPN row revised in c0ecdf5 on 2026-04-14); only FPL (#255) has a repo experiment behind it. #295 and #296 record the attempted API-Football lineup tool (#296 closed unmerged), not the reasons it was rejected. The lineup decision is [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md) and the Polymarket decision is [ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md).

## Scheduled collection

Every odds feed, and what each supplies, is on [data sources](../collection/data-sources.md); the agent reads them through the [MCP tools](mcp-server.md). Beside the odds feeds, the agent's tools read ESPN games (the all-competition EPL schedule, [EPL context](../collection/epl-context-data.md)) and the MLB Stats API's probable pitchers ([MLB pitchers](../collection/mlb-probable-pitchers.md)). Every scheduled collector runs on the production host ([job deployment matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). ESPN lineups are loaded only by a manual script (`scripts.ingest_espn_lineups`); the lineup rejection below applies to lineup timing for the agent, not to ESPN games.

## On-demand research

Not collected on a schedule; the agent searches and browses these each wake-up with web search and Playwright (the list is in `packages/odds-mcp/agents/epl/base.md`; MLB sources are in `packages/odds-mcp/agents/mlb/base.md`). Nothing in the repo fetches these sites: only the prompt files define what the agent does with them ([prompts](betting-agent.md#prompts)). Odds for an event come through the MCP tools under its one integer event id, whichever source stored them ([market tools](mcp-market-tools.md)).

| Source | Provides | Use |
|---|---|---|
| BBC Sport | Confirmed lineups, previews, injury updates | Team news, lineup confirmation near kick-off |
| Club websites and Twitter | Official lineups, press conference quotes | Primary lineup confirmation |
| RotoWire | Predicted and confirmed lineups | Cross-reference |
| Understat | xG, shot maps, form | Match preview context |
| Reddit r/soccer | Match threads, qualitative news | Supplementary context, sentiment |
| OddsShark | Consensus picks | Crude public-sentiment proxy |

## Rejected sources

| Source | Reason recorded |
|---|---|
| Betfair Exchange volume | Needs a £499 Live App Key and active betting; the free tier gives prices only |
| Twitter/X API | $100-5,000 per month under 2023 pricing, economically unviable |
| Action Network, BetQL | US sportsbook data; the "public" they track does not move EPL lines |
| Accumulator popularity | No source publishes most-backed acca legs with volume; tipster output is not crowd data |
| Tipster consensus (Predictz, Forebet) | Model-driven, lags sharp money, no signal for anticipating line movement |
| FPL API ownership and transfers | Tested as XGBoost features (#255): R² below the baseline without them; transfer spikes lag the news sources the agent searches directly |
| API-Football | Free tier blocks current-season fixture lookup (2022-2024 only) and the lineups endpoint needs the gated fixture ids; Pro plan about $20 per month |
| football-data.org | A different site from football-data.co.uk; lineups cost €29 per month and timing is undocumented |
| ESPN API (for lineups) | Lineups appear around KO-60 but non-monotonically (present at KO-60, gone by KO-45 in a 2026-04-14 Champions League test; nothing at KO-59 in a 2026-04-13 EPL test); 403 from cloud and datacenter IPs |
| Polymarket (as a sentiment source) | The EPL book tracks Betfair Exchange within 0.35-0.5 pp ([#518](https://github.com/Acusick1/odds/issues/518)), so it adds a price reference, not independent sentiment; UK trading is blocked. See [Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md) and [ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md) |

### Lineup sourcing (2026-04-13)

The record is the agent data-sources doc this wiki replaced, at 3f74022 (2026-04-13) and c0ecdf5 (the ESPN timing tests of 2026-04-13/14), plus the base prompt's "Lineup drops land roughly 60 minutes before kick-off" (`packages/odds-mcp/agents/epl/base.md`). #296, an API-Football lineup tool, was closed unmerged. No probe script or results file records the API or ESPN tests.

Every structured lineup API investigated is paywalled, blocks current-season data on its free tier, or has unconfirmed pre-kick-off timing. The agent's web search and Playwright are the lineup source: BBC Sport publishes confirmed lineups within minutes of the official team sheet (KO-60), and the agent reads them directly ([agent prompts](agent-prompts.md)). The record concludes "No new infrastructure needed", and no scheduled lineup job exists; the stored historical lineups are manually ingested and feed no feature group ([EPL context data](../collection/epl-context-data.md)). The decision is recorded in [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md).

## Related

- [Goal and strategy](../foundations/goal-and-strategy.md)
- [The betting agent](betting-agent.md)
- [About the system and its pivots](../foundations/project-overview.md)
