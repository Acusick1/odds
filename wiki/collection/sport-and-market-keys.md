---
type: reference
title: Sport, market and bookmaker keys
description: SportKey, job-name suffixes, the market-key dialects per writer, bookmaker key maps, and DataCollectionConfig defaults.
sources:
  - odds_core.config.DataCollectionConfig
  - odds_core.config.APIConfig
  - odds_core.sports.SportKey
  - odds_core.quotes.quotes
  - odds_core.utils.relabel_three_way_markets
  - odds_core.team.normalize_team_name
  - odds_lambda.data_fetcher.TheOddsAPIClient.get_odds
  - odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds
  - odds_core.odds_math.outcome_price_decimal
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.oddsportal_common.slugify
  - odds_lambda.betfair.constants.SPORT_CONFIG
  - odds_lambda.oddsportal_adapter.convert_market
  - odds_lambda.oddsportal_client.FEED_MARKETS
  - odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP
  - odds_lambda.scheduling.jobs._PER_SPORT_JOBS
  - odds_lambda.scheduling.jobs.make_compound_job_name
  - odds_mcp.server.MarketKey
  - scripts/ingest_football_data_uk.py
status: stable
---

# Sport, market and bookmaker keys

## Odds API scope

`odds_core.config.DataCollectionConfig` defaults: sports `soccer_epl`, `baseball_mlb`, `basketball_nba`; markets `h2h`, `spreads`, `totals`; region `us`. `odds_core.config.DataCollectionConfig.bookmakers` maps each sport to the Odds API bookmaker keys it requests, read through `odds_core.config.DataCollectionConfig.bookmakers_for`, which raises `ValueError` for a sport with no entry:

- `soccer_epl`: pinnacle, betfair_ex_uk, williamhill, paddypower, skybet, ladbrokes_uk, coral, betfred_uk, betvictor, sport888.
- `baseball_mlb` and `basketball_nba`: pinnacle, circasports, draftkings, fanduel, betmgm, williamhill_us, betrivers, bovada.

The Odds API gives `bookmakers` priority over `regions`, so the region setting has no effect on these requests, and it bills each group of up to 10 named bookmakers as one region ([Odds API v4 guide](https://the-odds-api.com/liveapi/guides/v4/)). The Odds API carries no bet365 for EPL in the `uk` or `eu` region (checked 2026-10-18); bet365 comes from OddsPortal and FDUK.

`odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` rewrite response bookmaker keys through `odds_core.bookmakers.ODDS_API_BOOKMAKER_KEY_MAP` (also part of the Odds API alias map, `odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE`) (`odds_lambda.data_fetcher.canonicalize_bookmaker_keys`) before anything stores them, so the same book carries the same key across sources: `betfair_ex_uk` and `betfair_ex_eu` become `betfair_exchange`, `ladbrokes_uk` becomes `ladbrokes`, `betfred_uk` becomes `betfred` and `sport888` becomes `888sport`. The other EPL keys already match the OddsPortal and FDUK keys below.

These filters apply only to Odds API requests ([configuration](../foundations/configuration.md)). Requests go to `odds_core.config.APIConfig.base_url` ([credentials](odds-api-client.md#retry-and-rotation)) under `sports/{sport_key}/odds` and `sports/{sport_key}/scores`, with sport keys from `odds_core.sports.SportKey` (`soccer_epl`, `baseball_mlb`, `basketball_nba`).

## OddsPortal and FDUK coverage

- OddsPortal names map through `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP` (32 entries; unmapped names fall back to `odds_lambda.oddsportal_common.slugify`). The map holds Pinnacle, BetMGM, DraftKings, FanDuel, Betfair Exchange, William Hill, Ladbrokes and others.
- FDUK issues no id of its own for an event ([events](../storage/events.md#external-ids)). Its snapshots carry source `football_data_uk` (the envelope's `source` is the same string), and FDUK maps its column prefixes in `BOOKMAKER_PREFIXES` (`scripts/ingest_football_data_uk.py`): pinnacle, bet365, bwin, betfair_exchange (`BFE`), betfair_sportsbook (`BF`), interwetten, williamhill, betvictor (`VC`, `BV`), betfred (`BFD`), paddypower (`PP`), skybet (`SKB`), betmgm (`BMGM`), coral (`CL`), ladbrokes, onexbet, plus pseudo-books `market_max` / `market_avg` (`AGGREGATE_PREFIXES`).
- Betfair Exchange covers EPL, MLB and NBA (`odds_lambda.betfair.constants.SPORT_CONFIG`): match odds for all three, and over/under 2.5 goals for EPL (the line is on `odds_lambda.betfair.constants.BetfairMarketSpec`).

## Sport keys

`odds_core.sports.SportKey` is `Literal["soccer_epl", "baseball_mlb", "basketball_nba"]`, matching The Odds API's convention. It is the `odds_core.models.Event.sport_key` value and the `sport` argument of jobs, MCP tools and CLI commands.

Job names use a short suffix: `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` maps `epl` to `soccer_epl`, `mlb` to `baseball_mlb` and `nba` to `basketball_nba`, and `odds_lambda.scheduling.jobs.make_compound_job_name("fetch-odds", "soccer_epl")` gives `fetch-odds-epl`, and only for jobs in `odds_lambda.scheduling.jobs._PER_SPORT_JOBS` (`run-candidates` among them) ([job-registry](../scheduling/job-registry.md#compound-job-names)). A per-sport job given no sport raises `ValueError`. The suffix names EventBridge rules, heartbeat keys and `odds_mcp.server.get_scheduled_jobs` filters.

Sport naming is translated separately per surface: compound job names (`fetch-odds-epl`), sport keys (`soccer_epl`), heartbeat keys, CloudWatch metrics, embed titles and CLI names each have their own table (`odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` is the job-name one; #384).

CLI `--sport` defaults differ by command ([odds-cli](../foundations/odds-cli.md#sport-defaults)). Several NBA-era defaults remain ([project-overview](../foundations/project-overview.md)), so an NBA default and an empty EPL window look alike. NBA team names are The Odds API's full names; OddsPortal and Betfair spell them the same, and `odds_core.team.normalize_team_name` maps the Polymarket "LA" spellings of the two Los Angeles teams ([event-matching](../storage/event-matching.md)).

## Market-key dialects

A market key is whatever string sits in an envelope's `bookmakers[].markets[].key`; the writers store it as the snapshot's `market`, which readers match exactly (`odds_core.quotes.quotes`, which selects events by their integer ids). Writers use different keys for the same two-way bet, and one key for the football three-way bet:

| Writer | Football 3-way | Two-way moneyline | Totals |
|---|---|---|---|
| The Odds API | `1x2` (relabelled from `h2h` for soccer) | `h2h` | `totals` |
| FDUK loader (`scripts/ingest_football_data_uk.py`, opening and closing `1x2` only; skips a game whose event match is ambiguous) | `1x2` | none | none |
| OddsPortal adapter | `1x2` | `h2h` (scraped as `home_away`) | `totals`, point 2.5 (scraped as `over_under_2_5`) |
| Betfair (`odds_lambda.betfair.constants.SPORT_CONFIG`) | `1x2` (EPL) | `h2h` (MLB, NBA) | `totals`, point 2.5 (EPL, `OVER_UNDER_25`) |
| Polymarket (each entry of `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`) | `1x2` (EPL), `h2h` (NBA) | none | none |

- The football three-way key is `1x2` in every source. The Odds API names that market `h2h` and spells teams in full ("Newcastle United"); `odds_core.utils.relabel_three_way_markets` rewrites an `h2h` market with a Draw outcome to `1x2` and its team outcomes to the canonical names of `odds_core.team.normalize_team_name`, for soccer sport keys, as `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` return it. OddsPortal live (`odds_lambda.oddsportal_adapter.convert_market`, which takes the pipeline key from `odds_lambda.oddsportal_client.FEED_MARKETS`), historical OddsPortal ingest (`MARKET_CONFIGS["1x2"]` in `scripts/ingest_oddsportal.py`, which also loads basketball and skips an NBA game with an unknown side) and Betfair EPL write `1x2` directly, and FDUK writes it too (`MARKET_KEY` in `scripts/ingest_football_data_uk.py`). MLB and NBA stay `h2h`. The OddsPortal NBA moneyline is the `home_away` feed and Betfair's is `MATCH_ODDS`.
- `betfair_exchange` has three writers: FDUK `BFE*` columns, the Betfair job, and OddsPortal's "Betfair Exchange" row (`odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`, with matched volume from the feed).
- OddsPortal scrape names (`1x2`, `over_under_2_5`, `home_away`) are the keys of `odds_lambda.oddsportal_client.FEED_MARKETS`, the only ones the live scrape accepts; `odds_mcp.server.refresh_scrape` rejects any other value. They are inputs to `odds_mcp.server.refresh_scrape` and `odds scrape --market`, not stored keys. The totals spelling is `over_under_2_5` (not `over_under_2.5`), and `odds_lambda.oddsportal_adapter.convert_market` raises `ValueError("Unsupported market")` for anything else ([oddsportal-scraper](oddsportal-scraper.md)).
- MCP read tools accept `odds_mcp.server.MarketKey` (`h2h`, `1x2`, `totals`, `spreads`) and match it exactly, so EPL queries need `1x2`, MLB and NBA queries need `h2h`, and a wrong key returns an empty result rather than an error (#433; [mcp-market-tools](../agent/mcp-market-tools.md)).

A market's outcomes are priced by `price` (integer American, The Odds API) or `price_decimal` (FDUK, OddsPortal, Betfair, and Polymarket from the current ingest code; archived envelopes can still carry `price`, see [Price encodings](../foundations/odds-math.md#price-encodings)); price rows are built through `odds_core.odds_math.outcome_price_decimal`.

Bookmaker keys: OddsPortal display names map through `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP` with a slug fallback; the default Odds API list is under [Odds API scope](#odds-api-scope).

## Related

- [Data sources](data-sources.md)
- [Add a sport](howto-add-sport.md)
- [Project overview](../foundations/project-overview.md), for the NBA assumptions behind several defaults
