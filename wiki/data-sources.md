---
type: concept
title: Data sources and bookmaker coverage
summary: Every external odds feed the pipeline ingests, what each supplies, how it is collected, and which bookmaker and market keys come from where.
sources:
  - packages/odds-core/odds_core/config.py::DataCollectionConfig
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::BOOKMAKER_KEY_MAP
  - packages/odds-lambda/odds_lambda/betfair/constants.py::SPORT_CONFIG
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py::convert_upcoming_matches
  - scripts/ingest_football_data_uk.py::BOOKMAKER_PREFIXES
  - docs/ARCHITECTURE.md#data-sources
  - docs/ARCHITECTURE.md#bookmaker-coverage
  - docs/ARCHITECTURE.md#markets-collected
  - docs/DATA_MODELS.md#sportsbook-odds
  - docs/AGENT_DATA_SOURCES.md#scheduled-collection
verified_at: 7887039
---

# Data sources and bookmaker coverage

Four feeds write `raw_data` snapshots in The Odds API shape ([Odds writer](odds-writer.md), [odds_snapshots](table-odds-snapshots.md)). Each is tagged by `Event.id` prefix or `OddsSnapshot.api_request_id`.

| Source | Supplies | Collected by | Event ids |
|---|---|---|---|
| The Odds API | US books incl. Pinnacle, dense snapshots | `fetch-odds` job, see [Odds API client](odds-api-client.md) | API hex ids |
| OddsPortal | UK books, upcoming and closing odds | `fetch-oddsportal` job, see [OddsPortal scraper](oddsportal-scraper.md) | `op_live_*` (`OddsWriter.LIVE_EVENT_ID_PREFIX`) |
| Betfair Exchange | Best back price per outcome | `fetch-betfair-exchange` job, see [Betfair client](betfair-exchange-client.md) | joins existing events |
| football-data.co.uk (FDUK) | Historical EPL, Pinnacle and exchange closing odds | manual script, see [Historical ingestion](historical-odds-ingestion.md) | `fduk_*` / `op_<season>_*` |

Non-odds feeds (ESPN, MLB, Polymarket) live in [EPL context data](epl-context-data.md), [MLB probable pitchers](mlb-probable-pitchers.md) and [Polymarket pipeline](polymarket-pipeline.md). Agent-facing view: [Agent data sources](agent-data-sources.md). Where jobs run: [Job deployment matrix](job-deployment-matrix.md).

## Odds API scope

`DataCollectionConfig` defaults (`packages/odds-core/odds_core/config.py::DataCollectionConfig`): sports `soccer_epl`, `baseball_mlb`; markets `h2h`, `spreads`, `totals`; region `us`; bookmakers pinnacle, circasports, draftkings, fanduel, betmgm, williamhill_us, betrivers, bovada. These filters apply only to Odds API requests. Sport keys: [Sport and market keys](sport-and-market-keys.md).

## OddsPortal and FDUK coverage

- OddsPortal names map through `BOOKMAKER_KEY_MAP` (32 entries; unmapped names fall back to `slugify`). The docs list only bet365, Betway, Betfred, bwin; the map also holds Pinnacle, BetMGM, DraftKings, FanDuel, Betfair Exchange, William Hill, Ladbrokes and others.
- FDUK maps column prefixes in `BOOKMAKER_PREFIXES`: pinnacle, bet365, bwin, betfair_exchange (`BFE`), betfair_sportsbook (`BF`), interwetten, williamhill, betvictor, ladbrokes, onexbet, plus pseudo-books `market_max` / `market_avg` (`AGGREGATE_PREFIXES`).

## Hazards

- **Football three-way market key is not uniform.** Odds API and FDUK store `h2h` with a Draw outcome (`_extract_h2h_odds`); OddsPortal live (`_convert_1x2_match`), historical OddsPortal ingest (`MARKET_CONFIGS["1x2"]`) and Betfair EPL (`SPORT_CONFIG["soccer_epl"]`) store `1x2`. `ARCHITECTURE.md` says h2h only. Readers must check both.
- **`betfair_exchange` has three writers**, not two: FDUK `BFE*` columns, the Betfair job, and OddsPortal's "Betfair Exchange" row (`BOOKMAKER_KEY_MAP`, fractional odds parsed by `parse_betfair_odds`).
- **Betfair covers MLB too** (`SPORT_CONFIG["baseball_mlb"]`, market `h2h`); the docs only mention football.
- Event counts, season coverage and the Pinnacle gap after July 2025 in the docs are data facts the repo cannot confirm; query the DB. Deployed schedule: `odds scheduler list-jobs --backend aws`.

## Related

- [Odds ingestion flow](odds-ingestion-flow.md)
- [Fetch tier](fetch-tier.md) — tag on every snapshot
- [CLV model findings](clv-model-findings.md) — which source each experiment used
