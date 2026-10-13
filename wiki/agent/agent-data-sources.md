---
type: concept
title: Agent data sources and rejected sources
description: Which sources are collected on a schedule and exposed through MCP versus researched live each wake-up, why lineups come from web search, and why each evaluated source was rejected.
sources:
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::CADENCE
  - packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py::CADENCE
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::BOOKMAKER_KEY_MAP
  - packages/odds-lambda/odds_lambda/espn_fixture_fetcher.py::EspnFixtureFetcher
  - scripts/ingest_football_data_uk.py
  - scripts/ingest_espn_lineups.py
  - packages/odds-mcp/agents/epl/base.md
  - packages/odds-mcp/agents/mlb/base.md
status: stable
---

# Agent data sources and rejected sources

The [betting agent](betting-agent.md) draws on two kinds of source. The split decides where a new source belongs: stable structured data is collected and queried through MCP, while qualitative and fast-changing context is researched live. No ADR records the rejections; the reasons below are external facts the repo cannot verify, as of the dates given (April 2026). Only FPL (#255) and API-Football (#295, #296, closed unmerged) have repo evidence.

## Scheduled collection

Collected by jobs or ingestion scripts into the database and available through [MCP tools](mcp-server.md). Where each job runs is in the [job deployment matrix](../scheduling/jobs-and-deployment-matrix.md#where-each-job-runs).

| Source | Provides | Collection | Page |
|---|---|---|---|
| OddsPortal | UK multi-bookmaker odds (`BOOKMAKER_KEY_MAP`), upcoming and closing | Self-scheduling by kickoff proximity, 30 min to 2 h (`CADENCE`) | [OddsPortal](../collection/oddsportal.md) |
| Betfair Exchange | Best back/lay Match Odds, the sharp reference | Self-scheduling, 15 min to 2 h (`CADENCE`) | [Betfair](../collection/betfair-exchange.md) |
| The Odds API | US bookmaker odds | Self-scheduling by fetch tier; 500 units per month per free key | [Odds API client](../collection/odds-api-client-and-quota.md) |
| football-data.co.uk | Historical EPL, Pinnacle and Betfair closing odds | Manual script | [historical loads](../collection/howto-load-historical-odds.md) |
| ESPN fixtures | All-competition EPL schedule | Daily job (cron in `_JOB_CRON_MAP`) | [EPL context](../collection/epl-context-data.md) |
| ESPN lineups | Starting XIs | Manual script | [EPL context](../collection/epl-context-data.md) |
| MLB Stats API | Probable pitchers | Daily job | [MLB pitchers](../collection/mlb-probable-pitchers.md) |
| Polymarket | Volume, order book depth, imbalance, price velocity | Pipeline built, deprioritized | [Polymarket](../collection/polymarket.md) |

The ESPN and MLB collectors are local cron only. The Betfair `*/30` cron floor is replaced by self-scheduling once the first self-schedule runs ([local scheduler](../scheduling/local-scheduler.md)). The [pipeline-side view](../collection/sources-and-keys.md) of the same feeds lists every bookmaker and key.

## On-demand research

Not collected on a schedule; the agent searches and browses these each wake-up with web search and Playwright (the list is in `packages/odds-mcp/agents/epl/base.md`; MLB sources are in `packages/odds-mcp/agents/mlb/base.md`).

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
| Polymarket | EPL volume thin ($10-100K per match) and AMM-driven, so the book reflects maker parameters, not public sentiment; UK trading is blocked. See [Polymarket](../collection/polymarket.md) |

### Lineup sourcing (2026-04-13)

Evidence is limited to #296 (an API-Football lineup tool, closed unmerged) and the base prompt's "Lineup drops land roughly 60 minutes before kick-off" (`packages/odds-mcp/agents/epl/base.md`).

Every structured lineup API investigated is paywalled, blocks current-season data on its free tier, or has unconfirmed pre-kickoff timing. The agent's web search and Playwright are the lineup source: BBC Sport publishes confirmed lineups within minutes of the official team sheet (KO-60), and the agent reads them directly ([wake-up workflow](agent-wakeup-workflow.md)).

## Hazards

- The cadence column above summarises code; the `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::CADENCE` and `packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP` values win.
- The on-demand table is not backed by code: nothing in the repo fetches those sites, and what the agent does with them is defined in the prompt files under `packages/odds-mcp/agents/` ([betting agent](betting-agent.md#prompts)).
- Only the 500-unit quota is a code default (`packages/odds-core/odds_core/config.py::APIConfig`); a paid key changes it ([quota](../collection/odds-api-client-and-quota.md)).
- ESPN is still used in the pipeline for fixtures (`packages/odds-lambda/odds_lambda/espn_fixture_fetcher.py::EspnFixtureFetcher`) and a manual lineup ingestion script exists (`scripts/ingest_espn_lineups.py`); the rejection applies to lineup timing for the agent, not to those feeds ([EPL context data](../collection/epl-context-data.md)).
- Prices and tier names in the rejected-source table are external facts the repo cannot verify.

## Related

- [Pivot to information edge](decision-information-edge.md)
- [Project overview](../foundations/project-overview.md)
