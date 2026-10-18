---
type: explanation
title: Agent data sources and rejected sources
description: Which sources are collected on a schedule and exposed through MCP versus researched live each wake-up, why lineups come from web search, and why each evaluated source was rejected.
sources:
  - odds_lambda.jobs.fetch_oddsportal.CADENCE
  - odds_lambda.jobs.fetch_betfair_exchange.CADENCE
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher
  - scripts/ingest_football_data_uk.py
  - scripts/ingest_espn_lineups.py
  - packages/odds-mcp/agents/epl/base.md
  - packages/odds-mcp/agents/mlb/base.md
status: stable
---

# Agent data sources and rejected sources

The [betting agent](betting-agent.md) draws on two kinds of source. The split decides where a new source belongs: stable structured data is collected and queried through MCP, while qualitative and fast-changing context is researched live. The reasons below are external facts the repo cannot verify, as of the dates given (April 2026); prices and tier names in the rejected-source table are likewise external. The reasons were recorded in the agent data-sources doc this wiki replaced (added in 3f74022 on 2026-04-13, ESPN row revised in c0ecdf5 on 2026-04-14); only FPL (#255) has a repo experiment behind it. #295 and #296 record the attempted API-Football lineup tool (#296 closed unmerged), not the reasons it was rejected. The lineup decision is [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md) and the Polymarket decision is [ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md).

## Scheduled collection

Collected by jobs or ingestion scripts into the database and available through [MCP tools](mcp-server.md). Where each job runs is in the [job deployment matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs).

| Source | Provides | Collection | Page |
|---|---|---|---|
| OddsPortal | UK multi-bookmaker odds (`odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`), upcoming and closing | Self-scheduling by kickoff proximity (`odds_lambda.jobs.fetch_oddsportal.CADENCE`) | [OddsPortal](../collection/oddsportal-scraper.md) |
| Betfair Exchange | Best back/lay Match Odds, the sharp reference | Self-scheduling by kickoff proximity (`odds_lambda.jobs.fetch_betfair_exchange.CADENCE`) | [Betfair](../collection/betfair-exchange.md) |
| The Odds API | US bookmaker odds | Self-scheduling by fetch tier, EPL only (`odds_lambda.scheduling.jobs._JOB_SPORTS`); 500 units per month per free key (the only quota that is a code default, `odds_core.config.APIConfig`; a paid key changes it) | [Odds API quota](../collection/odds-api-quota.md) |
| football-data.co.uk | Historical EPL opening and closing `1x2` odds from Pinnacle, Betfair Exchange and UK retail books | Manual script | [historical loads](../collection/historical-ingest-scripts.md) |
| ESPN fixtures | All-competition EPL schedule | Daily job (cron in `_JOB_CRON_MAP`) | [EPL context](../collection/epl-context-data.md) |
| ESPN lineups | Starting XIs | Manual script | [EPL context](../collection/epl-context-data.md) |
| MLB Stats API | Probable pitchers | Daily job | [MLB pitchers](../collection/mlb-probable-pitchers.md) |
| Polymarket | Volume, order book depth, imbalance, price velocity | Pipeline built, deprioritized | [Polymarket](../collection/polymarket.md) |

The cadence column summarises code; the `odds_lambda.jobs.fetch_oddsportal.CADENCE` and `odds_lambda.scheduling.jobs._JOB_CRON_MAP` values are authoritative. Every scheduled collector runs on the mini-PC; the ESPN and MLB collectors are daily crons. The Betfair `*/30` cron floor is replaced by self-scheduling once the first self-schedule runs ([local scheduler](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)). ESPN is still used in the pipeline for fixtures (`odds_lambda.espn_fixture_fetcher.EspnFixtureFetcher`) and a manual lineup ingestion script exists (`scripts/ingest_espn_lineups.py`); the lineup rejection below applies to lineup timing for the agent, not to those feeds. The [pipeline-side view](../collection/data-sources.md) of the same feeds lists every bookmaker and key.

## On-demand research

Not collected on a schedule; the agent searches and browses these each wake-up with web search and Playwright (the list is in `packages/odds-mcp/agents/epl/base.md`; MLB sources are in `packages/odds-mcp/agents/mlb/base.md`). Nothing in the repo fetches these sites: what the agent does with them is defined only by the prompt files ([prompts](betting-agent.md#prompts)).

| Source | Provides | Use |
|---|---|---|
| BBC Sport | Confirmed lineups, previews, injury updates | Team news, lineup confirmation near kickoff |
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
| FPL API ownership and transfers | As XGBoost features (2026-03-20): R² -0.017 under sliding-380 and 0.022 under expanding:150 against baselines of 0.026 and 0.052; transfer spikes lag the news sources the agent searches directly. See [EPL ablation](../evaluation/exp-epl-feature-ablation-and-search-space.md) |
| API-Football | Free tier blocks current-season fixture lookup (2022-2024 only) and the lineups endpoint needs the gated fixture ids; Pro plan about $20 per month |
| football-data.org | A different site from football-data.co.uk; lineups cost €29 per month and timing is undocumented |
| ESPN API (for lineups) | Lineups appear around KO-60 but non-monotonically (present at KO-60, gone by KO-45 in a 2026-04-14 Champions League test; nothing at KO-59 in a 2026-04-13 EPL test); 403 from cloud and datacenter IPs |
| Polymarket | EPL volume thin ($10-100K per match) and AMM-driven, so the book reflects maker parameters, not public sentiment; UK trading is blocked. See [Polymarket](../collection/polymarket.md) and [ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md) |

### Lineup sourcing (2026-04-13)

The record is the agent data-sources doc this wiki replaced, at 3f74022 (2026-04-13) and c0ecdf5 (the ESPN timing tests of 2026-04-13/14), plus the base prompt's "Lineup drops land roughly 60 minutes before kick-off" (`packages/odds-mcp/agents/epl/base.md`). #296, an API-Football lineup tool, was closed unmerged. No probe script or results file records the API or ESPN tests.

Every structured lineup API investigated is paywalled, blocks current-season data on its free tier, or has unconfirmed pre-kickoff timing. The agent's web search and Playwright are the lineup source: BBC Sport publishes confirmed lineups within minutes of the official team sheet (KO-60), and the agent reads them directly ([agent prompts](agent-prompts.md)). The record concludes "No new infrastructure needed", and no scheduled lineup job exists; the stored historical lineups are manually ingested and feed no feature group ([EPL context data](../collection/epl-context-data.md)). The decision is recorded in [ADR-0006](../../docs/adr/0006-lineups-from-web-research.md).

## Related

- [ADR-0005: information edge](../../docs/adr/0005-information-edge.md)
- [The betting agent](betting-agent.md)
- [About the system and its pivots](../foundations/project-overview.md)
