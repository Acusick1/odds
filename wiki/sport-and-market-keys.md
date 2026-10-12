---
type: concept
title: Sport and market keys
summary: "SportKey values, short sport suffixes in job names, the market-key dialects (h2h vs 1x2, OddsHarvester names) and what touching a new sport, bookmaker or market involves."
sources:
  - packages/odds-core/odds_core/sports.py::SportKey
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_SPORT_SUFFIX_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::make_compound_job_name
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py::_MARKET_CONVERTERS
  - packages/odds-lambda/odds_lambda/betfair/constants.py::SPORT_CONFIG
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py::LEAGUE_SPECS
  - packages/odds-core/odds_core/utils.py::raw_data_has_market
  - packages/odds-mcp/odds_mcp/server.py::MarketKey
  - docs/DEVELOPMENT.md#extensibility
  - docs/ARCHITECTURE.md#markets-collected
  - docs/REPORTING.md#9-known-limitations
verified_at: 7887039
---

# Sport and market keys

## Sport keys

`packages/odds-core/odds_core/sports.py::SportKey` is `Literal["soccer_epl", "baseball_mlb", "basketball_nba"]`, matching The Odds API's convention. It is the `Event.sport_key` value and the `sport` argument of jobs, MCP tools and CLI commands.

Job names use a short suffix: `_SPORT_SUFFIX_MAP` maps `epl` to `soccer_epl` and `mlb` to `baseball_mlb`, and `make_compound_job_name("fetch-odds", "soccer_epl")` gives `fetch-odds-epl` only for jobs in `_PER_SPORT_JOBS` ([Job registry](job-registry.md)). NBA has no suffix, so its compound names stay bare. The suffix names EventBridge rules, heartbeat keys and `get_scheduled_jobs` filters.

## Market-key dialects

A market key is whatever string sits in `raw_data.bookmakers[].markets[].key`, and readers match it exactly (`raw_data_has_market`). Three writers use different keys for the same bet:

| Writer | Football 3-way | Two-way moneyline | Totals |
|---|---|---|---|
| The Odds API | `h2h` (with a draw outcome) | `h2h` | `totals` |
| OddsPortal adapter | `1x2` | `h2h` (scraped as `home_away`) | `totals`, point 2.5 (scraped as `over_under_2_5`) |
| Betfair (`SPORT_CONFIG`) | `1x2` (EPL) | `h2h` (MLB) | none |

OddsHarvester scrape names (`1x2`, `over_under_2_5`, `home_away`) are the only keys `_MARKET_CONVERTERS` accepts; they are inputs to `refresh_scrape` and `odds scrape --market`, not stored keys. MCP read tools accept `MarketKey = h2h | 1x2 | totals | spreads` ([market-data tools](mcp-market-data-tools.md)). So EPL queries need `1x2` and MLB queries need `h2h`.

Bookmaker keys: OddsPortal display names map through `BOOKMAKER_KEY_MAP` with a slug fallback ([Data sources](data-sources.md)); the default Odds API list is in [Configuration](configuration.md).

## Adding a sport, bookmaker or market

- Bookmakers and markets for The Odds API are config lists (`BOOKMAKERS`, `MARKETS`), no code ([DEVELOPMENT.md](../docs/DEVELOPMENT.md#extensibility)).
- A sport needs a `SportKey` member, then (from reading the code) entries in `_SPORT_SUFFIX_MAP`, `OVERNIGHT_WINDOWS` in `packages/odds-lambda/odds_lambda/scheduling/decision.py` ([decision engine](scheduling-decision-engine.md)), `LEAGUE_SPECS`, Betfair `SPORT_CONFIG`, team aliases in `packages/odds-core/odds_core/team.py` ([Event matching](event-matching.md)), and any `_JOB_CRON_MAP` allowlists.

## Hazards

- `docs/CLI.md` shows `--market over_under_2.5`; the converter key is `over_under_2_5`, and `convert_upcoming_matches` raises `ValueError("Unsupported market")` for anything else ([OddsPortal scraper](oddsportal-scraper.md)).
- The MCP surface mixes four vocabularies: `SportKey` (`league` params), OddsHarvester league slugs (`refresh_scrape`), stored market keys and scrape market names ([scrape tools](mcp-scrape-and-schedule-tools.md)).
- Sport naming is ad hoc: the digest job is `daily-digest-epl`, other places use `soccer_epl`, and heartbeat keys, CloudWatch metrics, embed titles and CLI names each translate separately ([REPORTING.md](../docs/REPORTING.md#9-known-limitations), item 13).
- CLI `--sport` defaults differ by group, see [odds CLI](cli-overview.md#hazards).

## Related

- [Project overview](project-overview.md)
- [Project pivots](project-pivots.md) — NBA assumptions behind several defaults
