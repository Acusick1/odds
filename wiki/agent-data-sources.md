---
type: concept
title: Agent data sources
summary: "Sources collected on a schedule and exposed through MCP, versus sources the agent researches on demand at each wake-up."
sources:
  - docs/AGENT_DATA_SOURCES.md#scheduled-collection
  - docs/AGENT_DATA_SOURCES.md#on-demand-agent-research
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::CADENCE
  - packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py::CADENCE
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::BOOKMAKER_KEY_MAP
  - scripts/ingest_football_data_uk.py
  - scripts/ingest_espn_lineups.py
verified_at: 7887039
---

# Agent data sources

The [betting agent](betting-agent.md) draws on two kinds of source ([AGENT_DATA_SOURCES.md](../docs/AGENT_DATA_SOURCES.md)). The split decides where a new source belongs: stable structured data is collected and queried through MCP, while qualitative and fast-changing context is researched live.

## Scheduled collection

Collected by jobs or ingestion scripts into the database and available through [MCP tools](mcp-server.md). Where each job runs is in the [Job deployment matrix](job-deployment-matrix.md).

| Source | Provides | Collection | Page |
|---|---|---|---|
| OddsPortal | UK multi-bookmaker odds (`BOOKMAKER_KEY_MAP`), upcoming and closing | Self-scheduling by kickoff proximity, 30 min to 2 h (`CADENCE`) | [OddsPortal scraper](oddsportal-scraper.md) |
| Betfair Exchange | Best back/lay Match Odds, the sharp reference | Self-scheduling, 15 min to 2 h (`CADENCE`) | [Betfair client](betfair-exchange-client.md) |
| The Odds API | US bookmaker odds | Self-scheduling by fetch tier; 500 units per month per free key | [Odds API client](odds-api-client.md) |
| football-data.co.uk | Historical EPL, Pinnacle and Betfair closing odds | Manual script | [Historical ingestion](historical-odds-ingestion.md) |
| ESPN fixtures | All-competition EPL schedule | Daily job (cron in `_JOB_CRON_MAP`) | [EPL context](epl-context-data.md) |
| ESPN lineups | Starting XIs | Manual script | [EPL context](epl-context-data.md) |
| MLB Stats API | Probable pitchers | Daily job | [MLB pitchers](mlb-probable-pitchers.md) |
| Polymarket | Volume, order book depth, imbalance, price velocity | Pipeline built, deprioritized | [Polymarket](polymarket-pipeline.md) |

## On-demand research

Not collected on a schedule; the agent searches and browses these each wake-up with web search and Playwright.

| Source | Provides | Use |
|---|---|---|
| BBC Sport | Confirmed lineups, previews, injury updates | Team news, lineup confirmation near kickoff |
| Club websites and Twitter | Official lineups, press conference quotes | Primary lineup confirmation |
| RotoWire | Predicted and confirmed lineups | Cross-reference |
| Understat | xG, shot maps, form | Match preview context |
| Reddit r/soccer | Match threads, qualitative news | Supplementary context, sentiment |
| OddsShark | Consensus picks | Crude public-sentiment proxy |

Sources evaluated and turned down are in [Rejected data sources](decision-rejected-data-sources.md).

## Hazards

- The doc's cadence and key-file columns are summaries; the `CADENCE` and `_JOB_CRON_MAP` values in code win. `fetch-betfair-exchange` also has a `*/30` cron floor locally ([Self-scheduling](self-scheduling.md)).
- The on-demand table is not backed by code: nothing in the repo fetches those sites, and what the agent does with them is defined in the prompt files under `packages/odds-mcp/agents/` ([Betting agent](betting-agent.md)).
- Only the 500-unit quota is a code default (`APIConfig`); a paid key changes it ([Odds API quota](odds-api-quota.md)).

## Related

- [Data sources](data-sources.md) — the pipeline-side view of the same feeds
- [Project overview](project-overview.md)
