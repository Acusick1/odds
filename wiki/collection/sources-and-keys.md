---
type: concept
title: Data sources, sport keys and market keys
description: "Every odds feed and what it supplies, the bookmaker coverage per source, SportKey and job-name suffixes, the h2h/1x2/OddsHarvester market-key dialects, and what adding a sport, bookmaker or market touches."
sources:
  - packages/odds-core/odds_core/config.py::DataCollectionConfig
  - packages/odds-core/odds_core/sports.py::SportKey
  - packages/odds-core/odds_core/utils.py::raw_data_has_market
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::BOOKMAKER_KEY_MAP
  - packages/odds-lambda/odds_lambda/betfair/constants.py::SPORT_CONFIG
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py::convert_upcoming_matches
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py::_MARKET_CONVERTERS
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_SPORT_SUFFIX_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::make_compound_job_name
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LEAGUE_SPECS
  - packages/odds-mcp/odds_mcp/server.py::MarketKey
  - scripts/ingest_football_data_uk.py::BOOKMAKER_PREFIXES
  - packages/odds-lambda/odds_lambda/jobs/daily_digest.py
status: stable
---

# Data sources, sport keys and market keys

Four feeds write `raw_data` snapshots in The Odds API shape ([odds-storage](../storage/odds-storage.md)). Each is tagged by `Event.id` prefix or `OddsSnapshot.api_request_id` ([events](../storage/events.md#id-prefix-identifies-the-source)).

| Source | Supplies | Collected by | Where it runs | Event ids |
|---|---|---|---|---|
| The Odds API | US books incl. Pinnacle, dense snapshots | `fetch-odds` job ([odds-api-ingestion](odds-api-ingestion.md), [client](odds-api-client-and-quota.md)) | AWS only | API hex ids |
| OddsPortal | UK books, upcoming and closing odds | `fetch-oddsportal` job ([oddsportal](oddsportal.md)) | scraper Lambda if enabled, or local | `op_live_*` (`OddsWriter.LIVE_EVENT_ID_PREFIX`) |
| Betfair Exchange | Best back price per outcome | `fetch-betfair-exchange` job ([betfair-exchange](betfair-exchange.md)) | scheduler Lambda if `betfair_enabled`, or local | joins existing events |
| football-data.co.uk (FDUK) | Historical EPL, Pinnacle and exchange closing odds | manual script ([howto-load-historical-odds](howto-load-historical-odds.md)) | manual | `fduk_*` / `op_<season>_*` |

ESPN, MLB and Polymarket are non-odds feeds: [epl-context-data](epl-context-data.md), [mlb-probable-pitchers](mlb-probable-pitchers.md), [polymarket](polymarket.md). The ESPN and MLB collectors run only as a local cron. The deployed placement of every job is on [jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md); the agent-facing view is [agent-data-sources](../agent/agent-data-sources.md).

## Odds API scope

`packages/odds-core/odds_core/config.py::DataCollectionConfig` defaults (`packages/odds-core/odds_core/config.py::DataCollectionConfig`): sports `soccer_epl`, `baseball_mlb`; markets `h2h`, `spreads`, `totals`; region `us`; bookmakers pinnacle, circasports, draftkings, fanduel, betmgm, williamhill_us, betrivers, bovada. These filters apply only to Odds API requests ([configuration](../foundations/configuration.md)). Requests go to `APIConfig.base_url` (`packages/odds-core/odds_core/config.py::APIConfig`) under `sports/{sport_key}/odds` and `sports/{sport_key}/scores`, with sport keys from `packages/odds-core/odds_core/sports.py::SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`).

Bookmaker roles (inferred): Pinnacle and Circa Sports are sharp, low-margin books (Circa is Vegas-based); DraftKings, FanDuel, BetMGM and Caesars (`williamhill_us`) are major US retail; BetRivers is regional retail; Bovada is offshore. UK books come from OddsPortal: bet365 is the primary target ([decision-bet365-target](../modelling/decision-bet365-target.md)); Betway, Betfred and bwin are retail.

## OddsPortal and FDUK coverage

- OddsPortal names map through `BOOKMAKER_KEY_MAP` (32 entries; unmapped names fall back to `packages/odds-lambda/odds_lambda/oddsportal_common.py::slugify`). The map holds Pinnacle, BetMGM, DraftKings, FanDuel, Betfair Exchange, William Hill, Ladbrokes and others.
- FDUK maps column prefixes in `BOOKMAKER_PREFIXES`: pinnacle, bet365, bwin, betfair_exchange (`BFE`), betfair_sportsbook (`BF`), interwetten, williamhill, betvictor, ladbrokes, onexbet, plus pseudo-books `market_max` / `market_avg` (`AGGREGATE_PREFIXES`).
- **Pinnacle gap.** Pinnacle's Odds API coverage of EPL stopped in January 2026 (71 events lost), and FDUK's Pinnacle odds are unreliable after July 2025; Betfair Exchange is the sharp fallback ([decision-hybrid-sharp-reference](../modelling/decision-hybrid-sharp-reference.md)). Event counts and season coverage are data facts the repo cannot confirm; query the database.

## Sport keys

`packages/odds-core/odds_core/sports.py::SportKey` is `Literal["soccer_epl", "baseball_mlb", "basketball_nba"]`, matching The Odds API's convention. It is the `Event.sport_key` value and the `sport` argument of jobs, MCP tools and CLI commands.

Job names use a short suffix: `_SPORT_SUFFIX_MAP` maps `epl` to `soccer_epl` and `mlb` to `baseball_mlb`, and `make_compound_job_name("fetch-odds", "soccer_epl")` gives `fetch-odds-epl` only for jobs in `_PER_SPORT_JOBS` ([jobs-and-deployment-matrix](../scheduling/jobs-and-deployment-matrix.md)). NBA has no suffix, so its compound names stay bare. The suffix names EventBridge rules, heartbeat keys and `packages/odds-mcp/odds_mcp/server.py::get_scheduled_jobs` filters.

CLI `--sport` defaults differ by group (`basketball_nba` in `fetch current`, `fetch scores`, `packages/odds-cli/odds_cli/commands/discover.py` and `quality`; `soccer_epl` in `copy from-prod`; see [packages-and-cli](../foundations/packages-and-cli.md)). Several NBA-era defaults remain ([project-overview](../foundations/project-overview.md)), so an NBA default and an empty EPL window look alike.

## Market-key dialects

A market key is whatever string sits in `raw_data.bookmakers[].markets[].key`, and readers match it exactly (`packages/odds-core/odds_core/utils.py::raw_data_has_market`). Three writers use different keys for the same bet:

| Writer | Football 3-way | Two-way moneyline | Totals |
|---|---|---|---|
| The Odds API | `h2h` (with a draw outcome) | `h2h` | `totals` |
| OddsPortal adapter | `1x2` | `h2h` (scraped as `home_away`) | `totals`, point 2.5 (scraped as `over_under_2_5`) |
| Betfair (`SPORT_CONFIG`) | `1x2` (EPL) | `h2h` (MLB) | none |

EPL data is `1x2` from OddsPortal and Betfair, but `h2h` from the Odds API and FDUK; MLB is `h2h`. OddsHarvester scrape names (`1x2`, `over_under_2_5`, `home_away`) are the only keys `_MARKET_CONVERTERS` accepts; they are inputs to `packages/odds-mcp/odds_mcp/server.py::refresh_scrape` and `odds scrape --market`, not stored keys. MCP read tools accept `MarketKey = h2h | 1x2 | totals | spreads` and match it exactly, so EPL queries need `1x2`, MLB queries need `h2h`, and a wrong key returns an empty result rather than an error ([mcp-market-tools](../agent/mcp-market-tools.md)).

Bookmaker keys: OddsPortal display names map through `BOOKMAKER_KEY_MAP` with a slug fallback; the default Odds API list is above.

## Adding a sport, bookmaker or market

- Bookmakers and markets for The Odds API are config lists (`bookmakers`, `markets` on `packages/odds-core/odds_core/config.py::DataCollectionConfig`), no code.
- A sport needs a `SportKey` member, then (from reading the code) entries in `_SPORT_SUFFIX_MAP`, `OVERNIGHT_WINDOWS` in `packages/odds-lambda/odds_lambda/scheduling/decision.py` ([self-scheduling](../scheduling/self-scheduling.md)), `LEAGUE_SPECS`, Betfair `SPORT_CONFIG`, team aliases in `packages/odds-core/odds_core/team.py` ([events](../storage/events.md#matchers)), and any `_JOB_CRON_MAP` allowlists.

## Hazards

- **Football three-way market key is not uniform.** Odds API and FDUK store `h2h` with a Draw outcome (`_extract_h2h_odds`); OddsPortal live (`packages/odds-lambda/odds_lambda/oddsportal_adapter.py::_convert_1x2_match`), historical OddsPortal ingest (`MARKET_CONFIGS["1x2"]`) and Betfair EPL (`SPORT_CONFIG["soccer_epl"]`) store `1x2`. Readers must check both.
- **`betfair_exchange` has three writers**, not two: FDUK `BFE*` columns, the Betfair job, and OddsPortal's "Betfair Exchange" row (`BOOKMAKER_KEY_MAP`, fractional odds parsed by `packages/odds-lambda/odds_lambda/oddsportal_adapter.py::parse_betfair_odds`).
- **Betfair covers MLB too** (`SPORT_CONFIG["baseball_mlb"]`, market `h2h`).
- **CLI market name.** The scrape market key is `over_under_2_5` (not `over_under_2.5`), and `packages/odds-lambda/odds_lambda/oddsportal_adapter.py::convert_upcoming_matches` raises `ValueError("Unsupported market")` for anything else ([oddsportal](oddsportal.md)).
- **Four vocabularies in MCP.** The MCP surface mixes `SportKey` (`league` params), OddsHarvester league slugs (`packages/odds-mcp/odds_mcp/server.py::refresh_scrape`), stored market keys and scrape market names ([mcp-server](../agent/mcp-server.md)).
- **Sport naming is ad hoc.** The digest job is `daily-digest-epl`, other places use `soccer_epl`, and heartbeat keys, CloudWatch metrics, embed titles and CLI names each translate separately (`packages/odds-lambda/odds_lambda/jobs/daily_digest.py`, `packages/odds-lambda/odds_lambda/scheduling/jobs.py::_SPORT_SUFFIX_MAP`).
- Deployed schedule is not in the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Project overview](../foundations/project-overview.md), for the NBA assumptions behind several defaults
- [Odds API ingestion](odds-api-ingestion.md)
- [CLV model findings](../evaluation/clv-model-findings.md), for which source each experiment used
