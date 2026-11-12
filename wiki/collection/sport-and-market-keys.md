---
type: reference
title: Sport, market and bookmaker keys
description: SportKey, job-name suffixes, the market-key dialects per writer, bookmaker key maps, the UK retail roster and why three stored books are left out of it, and DataCollectionConfig defaults.
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
  - odds_core.job_specs.JOB_SPECS
  - odds_lambda.scheduling.jobs.make_compound_job_name
  - odds_mcp.server.MarketKey
  - scripts.ingest_football_data_uk.BOOKMAKER_PREFIXES
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
  - scripts.ingest_football_data_uk.MARKET_KEY
  - scripts.ingest_football_data_uk._ingest_one_match
  - scripts.ingest_oddsportal.MARKET_CONFIGS
status: stable
---

# Sport, market and bookmaker keys

## Odds API scope

`odds_core.config.DataCollectionConfig` defaults: sports `soccer_epl`, `baseball_mlb`, `basketball_nba`; markets `h2h`, `spreads`, `totals`; region `us`. A call spends the key of its sport, except a backfill call, which spends the configured acquisition key; the key's plan size is read from the API's response headers, not configured (`odds_core.config.APIConfig.key_for`, `odds_core.config.APIConfig.acquisition_key`, [keys](odds-api-client.md#retry-and-keys)). `odds_core.config.DataCollectionConfig.bookmakers` maps each sport to the Odds API bookmaker keys it requests, read through `odds_core.config.DataCollectionConfig.bookmakers_for`, which raises `ValueError` for a sport with no entry:

- `soccer_epl`: pinnacle, betfair_ex_uk, williamhill, paddypower, skybet, ladbrokes_uk, coral, betfred_uk, betvictor, sport888.
- `baseball_mlb`: pinnacle, circasports, draftkings, fanduel, betmgm, williamhill_us, betrivers, bovada.
- `basketball_nba`: pinnacle, williamhill, ladbrokes_uk, coral, sport888, betvictor, paddypower, skybet, unibet_uk, betway. A `uk` region request for NBA on 2026-10-31 (12 games) returned these UK books: betano_uk, betfair_ex_uk, betfair_sb_uk, betvictor, betway, boylesports, casumo, coral, grosvenor, ladbrokes_uk, leovegas, livescorebet, matchbook, paddypower, skybet, smarkets, sport888, unibet_uk, virginbet, williamhill. Only William Hill, Ladbrokes, Coral, 888sport, LeoVegas and Matchbook priced `spreads`; `totals` came from William Hill, Ladbrokes, Coral, 888sport, BetVictor, Betano, LeoVegas, Virgin Bet, LiveScore Bet, Grosvenor, Casumo and Matchbook. The list is ten of the [UK retail](#uk-retail-roster) books that priced NBA, so the request bills as one region, and the exchanges are left to Betfair and OddsPortal. Pinnacle is not in the `uk` region and is named through `bookmakers` like EPL's, which the API honours whatever the region (EPL's stored Pinnacle prices from the same call shape confirm it; no NBA Pinnacle price was requested in the probe, so its NBA coverage is `(inferred)` until the first scheduled run).

The Odds API gives `bookmakers` priority over `regions`, so the region setting has no effect on these requests, and it bills each group of up to 10 named bookmakers as one region ([Odds API v4 guide](https://the-odds-api.com/liveapi/guides/v4/)). The Odds API carries no bet365 for EPL in the `uk` or `eu` region (checked 2026-10-18); bet365 comes from OddsPortal and FDUK.

`odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` rewrite response bookmaker keys through `odds_core.bookmakers.ODDS_API_BOOKMAKER_KEY_MAP` (also part of the Odds API alias map, `odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE`) (`odds_lambda.data_fetcher.canonicalize_bookmaker_keys`) before anything stores them (a historical request sends only the markets and the bookmakers or regions its caller names, not the configured defaults), so the same book carries the same key across sources: `betfair_ex_uk`, `betfair_ex_eu` and the legacy `betfair` of early `uk`-region history (the exchange; [facts](odds-api-facts.md#coverage)) become `betfair_exchange`, `ladbrokes_uk` becomes `ladbrokes`, `betfred_uk` becomes `betfred` and `sport888` becomes `888sport`. The other EPL keys already match the OddsPortal and FDUK keys below. A manifest's probe lists its required books under the request keys, and `odds backfill probe` compares them after this rewrite ([probe](odds-api-backfill.md#probing-a-manifest)).

These filters apply only to Odds API requests ([configuration](../foundations/configuration.md)). Requests go to `odds_core.config.APIConfig.base_url` ([credentials](odds-api-client.md#retry-and-keys)) under `sports/{sport_key}/odds` and `sports/{sport_key}/scores`, with sport keys from `odds_core.sports.SportKey`.

## OddsPortal and FDUK coverage

- OddsPortal names map through `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP` (32 entries; unmapped names fall back to `odds_lambda.oddsportal_common.slugify`). The map holds Pinnacle, BetMGM, DraftKings, FanDuel, Betfair Exchange, William Hill, Ladbrokes and others.
- FDUK issues no id of its own for an event ([events](../storage/events.md#external-ids)). Its snapshots carry source `football_data_uk` (the envelope's `source` is the same string), and FDUK maps its column prefixes in `scripts.ingest_football_data_uk.BOOKMAKER_PREFIXES`: pinnacle, bet365, bwin, betfair_exchange (`BFE`), betfair_sportsbook (`BF`), interwetten, williamhill, betvictor (`VC`, `BV`), betfred (`BFD`), paddypower (`PP`), skybet (`SKB`), betmgm (`BMGM`), coral (`CL`), ladbrokes, onexbet, plus pseudo-books `market_max` / `market_avg` (`AGGREGATE_PREFIXES`).
- Betfair Exchange covers the six football leagues, MLB and NBA (`odds_lambda.betfair.constants.SPORT_CONFIG`): match odds for all of them, and over/under 2.5 goals for each football league (the line is on `odds_lambda.betfair.constants.BetfairMarketSpec`), each sport with its own fixture horizon.

## UK retail roster

`odds_core.bookmakers.UK_RETAIL_BOOKMAKERS` lists, by canonical key, the sportsbooks a UK bettor can take a price from: every rule, model and screen that reads a retail price reads only these books. A book belongs in it when its operator holds a UK Gambling Commission remote betting licence ([public register](https://www.gamblingcommission.gov.uk/public-register/)) and its prices are its own.

Three `uk`-region Odds API keys are stored but left out:

| Key | Why it is left out |
|---|---|
| `virginbet` | Its `1x2` and `h2h` back prices equal `livescorebet`'s in 80 to 100% of co-quoted snapshots in every sampled month (first to fourth day) from 2023-05 to 2026-05 except 2024-06 and 2024-07 (3%). Both are LiveScore Group books on Kambi, and the register flags Virgin Bet's domain as a white label ([account 54310](https://www.gamblingcommission.gov.uk/public-register/business/detail/54310)). Counting both would count one price twice in a retail consensus. |
| `grosvenor` | Its prices equal `casumo`'s in 97 to 99% of co-quoted snapshots from 2025-10 to 2026-05, and in 30 to 98% from 2024-04 to 2025-09 (the first to fourth day of each month sampled), while no book in the roster matches either in more than 16% (in the windows sampled below). Both are Kambi sportsbooks. `casumo` is kept because it covers more EPL fixtures. |
| `mrgreen` | Mr Green Limited ([account 39264](https://www.gamblingcommission.gov.uk/public-register/business/detail/39264)) holds no remote betting licence, and its UK sportsbook closed in September 2022 ([report](https://www.casino.org/news/mr-green-shutting-down-uk-sportsbook-operations-igaming-safe-for-now/)), so its stored prices (to 2024-11-01) are not prices a UK bettor could take. |

The kept `casumo` ([account 61549](https://www.gamblingcommission.gov.uk/public-register/business/detail/61549)), `leovegas` ([account 39198](https://www.gamblingcommission.gov.uk/public-register/business/detail/39198)) and `livescorebet` ([account 56859](https://www.gamblingcommission.gov.uk/public-register/business/detail/56859)) hold remote betting licences that started before August 2024 (register checked 2026-11-12). Their prices match no book in the roster in more than 5% of co-quoted snapshots (sampled 2024-10-01 to 14, 2025-07-01 to 07 and 2026-03-01 to 14).

## Sport keys

`odds_core.sports.SportKey` is a `Literal` of The Odds API's sport keys: `soccer_epl`, `soccer_efl_champ` (the Championship), `soccer_spain_la_liga`, `soccer_italy_serie_a`, `soccer_germany_bundesliga`, `soccer_france_ligue_one`, `baseball_mlb` and `basketball_nba`. It is the `odds_core.models.Event.sport_key` value and the `sport` argument of jobs, MCP tools and CLI commands. The five football leagues after the EPL have no Odds API fetch plan, so no scheduled job spends quota on them; OddsPortal, Betfair, the ESPN schedule and the OddsPortal results job cover them as they do the EPL once they are in `data_collection.sports` ([add a sport](howto-add-sport.md)).

Job names use a short suffix: `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` maps `epl` to `soccer_epl`, `mlb` to `baseball_mlb`, `nba` to `basketball_nba` and `championship`, `laliga`, `seriea`, `bundesliga` and `ligue1` to the other football leagues' keys, and `odds_lambda.scheduling.jobs.make_compound_job_name("fetch-odds", "soccer_epl")` gives `fetch-odds-epl`, and only for per-sport jobs in `odds_core.job_specs.JOB_SPECS` (`run-candidates` and `fetch-injuries` among them) ([job-registry](../scheduling/job-registry.md#compound-job-names)). A per-sport job given no sport raises `ValueError`. The suffix names EventBridge rules, heartbeat keys and the sport filter of `odds_mcp.server.get_scheduled_jobs`.

Sport naming is translated separately per surface: compound job names (`fetch-odds-epl`), sport keys (`soccer_epl`), heartbeat keys, CloudWatch metrics, embed titles and CLI names each have their own table (`odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` is the job-name one; #384).

CLI `--sport` defaults differ by command ([odds-cli](../foundations/odds-cli.md#sport-defaults)). Several NBA-era defaults remain ([project-overview](../foundations/project-overview.md)), so an NBA default and an empty EPL window look alike. NBA team names are The Odds API's full names; OddsPortal and Betfair spell them the same, and `odds_core.team.normalize_team_name` maps the Polymarket "LA" spellings of the two Los Angeles teams ([event-matching](../storage/event-matching.md)).

## Market-key dialects

A market key is whatever string sits in an envelope's `bookmakers[].markets[].key`; the writers store it as the snapshot's `market`, which readers match exactly (`odds_core.quotes.quotes`, which selects events by their integer ids). Writers use different keys for the same two-way bet, and one key for the football three-way bet:

| Writer | Football 3-way | Two-way moneyline | Totals |
|---|---|---|---|
| The Odds API | `1x2` (relabelled from `h2h` for soccer) | `h2h` | `totals` |
| FDUK loader (`scripts.ingest_football_data_uk._ingest_one_match`, the opening and closing `1x2` and the Betfair Exchange closing `totals`, a snapshot each; skips a game whose event match is ambiguous) | `1x2` | none | `totals`, point 2.5, Betfair Exchange close only |
| OddsPortal adapter | `1x2` | `h2h` (scraped as `home_away`) | `totals`, point 2.5 (scraped as `over_under_2_5`), or per-game main-line ladders (NBA, scraped as `totals_ladder`); `spreads` per-game lines (NBA, scraped as `spreads_ladder`) |
| Betfair (`odds_lambda.betfair.constants.SPORT_CONFIG`) | `1x2` (football) | `h2h` (MLB, NBA) | `totals`, point 2.5 (football, `OVER_UNDER_25`) |
| Polymarket (each entry of `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`) | `1x2` (EPL), `h2h` (NBA) | none | none |

- The football three-way key is `1x2` in every source. The Odds API names that market `h2h` and spells teams in full ("Newcastle United"); `odds_core.utils.relabel_three_way_markets` rewrites an `h2h` market with a Draw outcome to `1x2` and its team outcomes to the canonical names of `odds_core.team.normalize_team_name`, for soccer sport keys, as `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` return it. OddsPortal live (`odds_lambda.oddsportal_adapter.convert_market`, which takes the pipeline key from `odds_lambda.oddsportal_client.FEED_MARKETS`), historical OddsPortal ingest (the `1x2` entry of `scripts.ingest_oddsportal.MARKET_CONFIGS`, which also loads basketball and skips an NBA game with an unknown side) and Betfair EPL write `1x2` directly, and FDUK writes it too (`scripts.ingest_football_data_uk.MARKET_KEY`; it reads dates and times in the league's timezone, UK local time by default, `scripts.ingest_football_data_uk.parse_kickoff`, and takes the sport key of each league it loads, `scripts.ingest_football_data_uk.FdukLeague`). MLB and NBA stay `h2h`. The OddsPortal NBA moneyline is the `home_away` feed and Betfair's is `MATCH_ODDS`.
- `betfair_exchange` has three writers: FDUK `BFE*` columns, the Betfair job, and OddsPortal's "Betfair Exchange" row (`odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`, with matched volume from the feed).
- OddsPortal scrape names (`1x2`, `over_under_2_5`, `home_away`, `totals_ladder`, `spreads_ladder`) are the keys of `odds_lambda.oddsportal_client.FEED_MARKETS`, the only ones the live scrape accepts; `odds_mcp.server.refresh_scrape` rejects any other value. They are inputs to `odds_mcp.server.refresh_scrape` and `odds scrape --market`, not stored keys. The totals spelling is `over_under_2_5` (not `over_under_2.5`), and `odds_lambda.oddsportal_adapter.convert_market` raises `ValueError("Unsupported market")` for anything else ([oddsportal-scraper](oddsportal-scraper.md)).
- MCP read tools accept `odds_mcp.server.MarketKey` (`h2h`, `1x2`, `totals`, `spreads`) and match it exactly, so EPL queries need `1x2`, MLB and NBA queries need `h2h`, and a wrong key returns an empty result rather than an error (#433; [mcp-market-tools](../agent/mcp-market-tools.md)).

A market's outcomes are priced by `price` (integer American, The Odds API) or `price_decimal` (FDUK, OddsPortal, Betfair, and Polymarket from the current ingest code; archived envelopes can still carry `price`, see [Price encodings](../foundations/odds-math.md#price-encodings)); price rows are built through `odds_core.odds_math.outcome_price_decimal`.

Bookmaker keys: OddsPortal display names map through `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP` with a slug fallback; the default Odds API list is under [Odds API scope](#odds-api-scope).

## Related

- [Data sources](data-sources.md)
- [Add a sport](howto-add-sport.md)
- [Project overview](../foundations/project-overview.md), for the NBA assumptions behind several defaults
