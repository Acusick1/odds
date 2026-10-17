---
type: reference
title: Sport, market and bookmaker keys
description: SportKey, job-name suffixes, the market-key dialects per writer, bookmaker key maps, and DataCollectionConfig defaults.
sources:
  - odds_core.config.DataCollectionConfig
  - odds_core.config.APIConfig
  - odds_core.sports.SportKey
  - odds_core.utils.raw_data_has_market
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.oddsportal_common.slugify
  - odds_lambda.betfair.constants.SPORT_CONFIG
  - odds_lambda.oddsportal_adapter.convert_market
  - odds_lambda.oddsportal_client.FEED_MARKETS
  - odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP
  - odds_lambda.scheduling.jobs._PER_SPORT_JOBS
  - odds_lambda.scheduling.jobs.make_compound_job_name
  - odds_lambda.jobs.daily_digest
  - odds_mcp.server.MarketKey
  - scripts/ingest_football_data_uk.py
status: stable
---

# Sport, market and bookmaker keys

## Odds API scope

`odds_core.config.DataCollectionConfig` defaults: sports `soccer_epl`, `baseball_mlb`; markets `h2h`, `spreads`, `totals`; region `us`; bookmakers pinnacle, circasports, draftkings, fanduel, betmgm, williamhill_us, betrivers, bovada. These filters apply only to Odds API requests ([configuration](../foundations/configuration.md)). Requests go to `odds_core.config.APIConfig.base_url` under `sports/{sport_key}/odds` and `sports/{sport_key}/scores`, with sport keys from `odds_core.sports.SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`).

## OddsPortal and FDUK coverage

- OddsPortal names map through `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP` (32 entries; unmapped names fall back to `odds_lambda.oddsportal_common.slugify`). The map holds Pinnacle, BetMGM, DraftKings, FanDuel, Betfair Exchange, William Hill, Ladbrokes and others.
- FDUK maps column prefixes in `BOOKMAKER_PREFIXES` (`scripts/ingest_football_data_uk.py`): pinnacle, bet365, bwin, betfair_exchange (`BFE`), betfair_sportsbook (`BF`), interwetten, williamhill, betvictor, ladbrokes, onexbet, plus pseudo-books `market_max` / `market_avg` (`AGGREGATE_PREFIXES`).
- Betfair Exchange covers EPL and MLB (`odds_lambda.betfair.constants.SPORT_CONFIG`).

## Sport keys

`odds_core.sports.SportKey` is `Literal["soccer_epl", "baseball_mlb", "basketball_nba"]`, matching The Odds API's convention. It is the `odds_core.models.Event.sport_key` value and the `sport` argument of jobs, MCP tools and CLI commands.

Job names use a short suffix: `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` maps `epl` to `soccer_epl` and `mlb` to `baseball_mlb`, and `odds_lambda.scheduling.jobs.make_compound_job_name("fetch-odds", "soccer_epl")` gives `fetch-odds-epl` only for jobs in `odds_lambda.scheduling.jobs._PER_SPORT_JOBS` ([job-registry](../scheduling/job-registry.md#compound-job-names)). NBA has no suffix, so its compound names stay bare. The suffix names EventBridge rules, heartbeat keys and `odds_mcp.server.get_scheduled_jobs` filters.

Sport naming is translated separately per surface: compound job names (`daily-digest-epl`), sport keys (`soccer_epl`), heartbeat keys, CloudWatch metrics, embed titles and CLI names each have their own table (`odds_lambda.jobs.daily_digest`, `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP`; #384).

CLI `--sport` defaults differ by command ([odds-cli](../foundations/odds-cli.md#sport-defaults)). Several NBA-era defaults remain ([project-overview](../foundations/project-overview.md)), so an NBA default and an empty EPL window look alike.

## Market-key dialects

A market key is whatever string sits in `raw_data.bookmakers[].markets[].key`, and readers match it exactly (`odds_core.utils.raw_data_has_market`). Three writers use different keys for the same bet:

| Writer | Football 3-way | Two-way moneyline | Totals |
|---|---|---|---|
| The Odds API | `h2h` (with a draw outcome) | `h2h` | `totals` |
| OddsPortal adapter | `1x2` | `h2h` (scraped as `home_away`) | `totals`, point 2.5 (scraped as `over_under_2_5`) |
| Betfair (`odds_lambda.betfair.constants.SPORT_CONFIG`) | `1x2` (EPL) | `h2h` (MLB) | none |

- The football three-way key is not uniform. The Odds API and FDUK store `h2h` with a Draw outcome (`_extract_h2h_odds` in `scripts/ingest_football_data_uk.py`). OddsPortal live (`odds_lambda.oddsportal_adapter.convert_market`), historical OddsPortal ingest (`MARKET_CONFIGS["1x2"]` in `scripts/ingest_oddsportal.py`) and Betfair EPL store `1x2`. EPL data is therefore `1x2` from OddsPortal and Betfair but `h2h` from the Odds API and FDUK; MLB is `h2h`. Readers must check both for football.
- `betfair_exchange` has three writers: FDUK `BFE*` columns, the Betfair job, and OddsPortal's "Betfair Exchange" row (`odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`, with matched volume from the feed).
- OddsPortal scrape names (`1x2`, `over_under_2_5`, `home_away`) are the keys of `odds_lambda.oddsportal_client.FEED_MARKETS`, the only ones the live scrape accepts. They are inputs to `odds_mcp.server.refresh_scrape` and `odds scrape --market`, not stored keys. The totals spelling is `over_under_2_5` (not `over_under_2.5`), and `odds_lambda.oddsportal_adapter.convert_market` raises `ValueError("Unsupported market")` for anything else ([oddsportal-scraper](oddsportal-scraper.md)).
- MCP read tools accept `odds_mcp.server.MarketKey` (`h2h`, `1x2`, `totals`, `spreads`) and match it exactly, so EPL queries need `1x2`, MLB queries need `h2h`, and a wrong key returns an empty result rather than an error (#433; [mcp-market-tools](../agent/mcp-market-tools.md)).

Bookmaker keys: OddsPortal display names map through `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP` with a slug fallback; the default Odds API list is under [Odds API scope](#odds-api-scope).

## Related

- [Data sources](data-sources.md)
- [Add a sport](howto-add-sport.md)
- [Project overview](../foundations/project-overview.md), for the NBA assumptions behind several defaults
