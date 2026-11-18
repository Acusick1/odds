---
type: howto
title: Add a league
description: "Steps to add a league: SportKey, timezone, kick-off authority, suffix, overnight window, league spec and Betfair config with their event horizons, Odds API fetch plan, candidate markets, close and fallback, team aliases, job league allowlists, Terraform rules, heartbeat keys, the configured leagues, and what odds-web then shows."
sources:
  - odds_core.sports.SportKey
  - odds_core.sports.LEAGUE_TIMEZONES
  - odds_lambda.storage.event_kickoff.KICKOFF_AUTHORITY
  - odds_core.config.DataCollectionConfig
  - odds_core.config.AlertConfig.heartbeat_expectations
  - odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP
  - odds_core.job_specs.JOB_SPECS
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS
  - odds_lambda.jobs.fetch_odds.PLANS
  - odds_lambda.jobs.fetch_odds.FetchPlan
  - odds_lambda.betfair.constants.SPORT_CONFIG
  - odds_lambda.betfair.constants.SportBetfairConfig
  - odds_lambda.jobs.fetch_oddsportal.LeagueSpec
  - odds_lambda.collection_calendar.SportCollection
  - odds_lambda.candidate_rules.registry.SPORT_MARKETS
  - odds_core.close.CLOSES
  - odds_core.close.FallbackSpec
  - odds_core.team
  - deployment/aws/terraform/eventbridge.tf
mentions:
  - odds_core.alerts.job_alert_context
  - odds_lambda.health_readers.collection_sources
status: stable
---

# Add a league

The Odds API books and markets are config (`bookmakers`, a per-league map, and `markets` on `odds_core.config.DataCollectionConfig`), and its fetch needs a plan (step 10); a league with no `bookmakers` entry fails its Odds API fetch, and a league that needs its own Odds API key adds its index to `odds_core.config.APIConfig.key_index_by_sport` ([keys](odds-api-client.md#retry-and-keys)). A new league needs the steps below; skip a step only when the league has no feed or job of that kind. A league loaded only for the Odds API historical backfill needs steps 1, 7, 8 and 9 ([historical games](../storage/event-matching.md#historical-odds-api-games)).

1. Add the league to `odds_core.sports.SportKey`, its timezone to `odds_core.sports.LEAGUE_TIMEZONES` and its kick-off sources, best first, to `odds_lambda.storage.event_kickoff.KICKOFF_AUTHORITY`.
   - It becomes the `sport` argument of jobs, MCP tools and CLI commands ([sport-and-market-keys](sport-and-market-keys.md#sport-keys)); the timezone sets its game day ([timezones](../foundations/timezones.md)) and the order which source may move its kick-offs ([kick-off authority](../storage/kickoff-authority.md#authority-order)).
2. Add a suffix in `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP` (a single word such as `laliga`; no suffix may end another).
   - The suffix names EventBridge rules, heartbeat keys, the league filter of `odds_mcp.server.get_scheduled_jobs` and the league of each odds-web status chip; without one, compound job names stay bare ([job-registry](../scheduling/job-registry.md#compound-job-names)).
3. Add an entry to `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS`.
   - The decision engine skips runs inside the window ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).
4. If OddsPortal covers it, add a `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` to `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`, with the league's OddsPortal `league` slug, `listing_path`, `sport_key`, `sport_title`, `fixture_horizon` and markets that are keys of `odds_lambda.oddsportal_client.FEED_MARKETS` (a market not in that table needs a new `odds_lambda.oddsportal_client.FeedMarketSpec` row first; a league whose totals or spreads line differs per game uses a ladder row, and its feed scopes and line values (`handicapValue`, not the lossy key) are probed on a live game first, [ladder markets](oddsportal-scraper.md#ladder-markets)).
   - The job reads its overnight window from step 3 ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues)).
   - The spec's horizon and markets become the league's `odds_lambda.collection_calendar.SportCollection` in the job's source: a run that stores nothing while a event lies inside the horizon fails, and the stale-data check watches each market ([stale-data check](../operations/health-monitor.md#stale-data-check)).
5. If Betfair lists it, add an entry to `odds_lambda.betfair.constants.SPORT_CONFIG`, with its match odds market, any other market types to collect (a totals market carries its line) and its `fixture_horizon`, as for OddsPortal.
   - The Betfair allowlist in `odds_core.job_specs.JOB_SPECS` is derived from that map, so no allowlist edit is needed ([betfair-exchange](betfair-exchange.md)).
6. If Polymarket lists its match markets, add an entry to `odds_lambda.polymarket_sports.POLYMARKET_SPORTS` (`has_draw` for a three-way league, and a `team_name` function); the `fetch-polymarket-odds` allowlist and the source's leagues follow that dict ([Polymarket odds snapshots](polymarket-odds-snapshots.md#sport-config-and-discovery)).
7. Add the markets its candidates can decide to `odds_lambda.candidate_rules.registry.SPORT_MARKETS`.
   - Without an entry, `odds_lambda.candidate_rules.registry.resolve_rule` raises for every candidate that lists the league, as it also does for a rule whose extra markets the league does not decide (a rule's [fixture inputs](../evaluation/candidate-rules.md#fixture-context) read the league's FINAL events and the season `odds_core.time.season_start` gives, so a league whose season does not start on 1 August or 1 January needs that function changed): `odds candidates list` and `odds candidates report` exit with code 1, the run-candidates job skips the candidate and alerts, and the odds-web data-health page's coverage section fails ([params validation](../evaluation/candidate-rules.md#params-validation)).
   - A market whose outcomes `odds_core.market_prices.market_outcomes` does not define needs them added there. Every rule imports that module, so it changes the `odds_lambda.candidate_rules.pinning.rule_sha256` of each rule. Record the rehash with `odds candidates rehash`, which checks that no bet changed, or give each registered candidate a new version ([rehash ledger](../evaluation/candidate-registry.md#rehash-ledger)).
8. Add each of those (league, market) pairs to `odds_core.close.CLOSES`: `None` (no close) until an ADR gives the league a close, as [ADR-0019](../../docs/adr/0019-nba-h2h-close-betfair-with-polymarket-fallback.md) did for NBA `h2h`, then an `odds_core.close.CloseSpec`, which needs Betfair to collect the market (step 5). MLB's markets stay `None` although Betfair collects its `h2h`.
   - An `odds_core.close.CloseSpec` takes the live Betfair close; its `fallback`, an `odds_core.close.FallbackSpec`, names the source whose price stands in when Betfair has none in the window (`odds_core.close.FDUK_CLOSE` for a football league, also listed in `odds_core.close.FDUK_CLOSED_LEAGUES`; `odds_core.close.POLYMARKET_CLOSE` for NBA), or is `None` when only Betfair can close it. The fallback's source must be ingested for the league: Football-Data.co.uk files are loaded by hand per league ([historical ingest scripts](historical-ingest-scripts.md)), and Polymarket by step 6.
   - The fallback's `arrives_after` is the grace: a bet with no close stays pending that long after kick-off (7 days for FDUK, none for Polymarket), then has none ([close](../evaluation/candidate-registry.md#odds-candidates-report)).
   - A pair whose entry is `None` is reported without EV: odds-web shows its candidate as "EV not scoreable" and its bets' close as unsupported ([candidate page](../operations/odds-web.md#candidate-page)).
9. Add team aliases in `odds_core.team`, the one alias map (`odds_lambda.polymarket_sports.PolymarketSportConfig.team_name` only picks the Polymarket spelling), for every spelling each source uses, ESPN's included if ESPN schedules the league ([ESPN football schedule](espn-football-schedule.md)). The map ignores the league, so an alias must not name another club in any league.
   - Event matching compares normalised names; unknown names pass through unchanged and surface as duplicate events or NULL `event_id` ([event-matching](../storage/event-matching.md)).
10. If the Odds API prices it, add an `odds_lambda.jobs.fetch_odds.FetchPlan` to `odds_lambda.jobs.fetch_odds.PLANS`, and the league to the `fetch-odds` allowlist in step 11.
    - Without a plan, the league's `fetch-odds` run raises `ValueError: fetch-odds has no plan for <sport>` before any work ([fetch-odds job](fetch-odds-job.md)); the plan's `source` is a `odds_lambda.collection_calendar.CollectionSource` whose `odds_lambda.collection_calendar.SportCollection` holds the horizon and markets its empty-run check uses, and its cadence is `odds_core.job_specs.job_cadence` (a league with its own cadence adds it to `sport_cadences` in the job record).
    - The stale-source check reads only `odds_lambda.jobs.fetch_odds.SOURCE`, so a league that polls at its cadence (as EPL does) is added to that source's `sports`; a league with sparse runs gets a source of its own, as NBA does, and is left out of the check ([fetch-odds job](fetch-odds-job.md)).
11. Add the league to the `sports` allowlist of each `odds_core.job_specs.JOB_SPECS` record whose job should run for it.
    - A job with an allowlist skips every league not on it, so a new league gets none of those jobs until added; `fetch-oddsportal-results` lists the football leagues there, so a league settled from OddsPortal results is added to it; per-league jobs without an allowlist run for every league in `data_collection.sports` ([local-scheduler](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
12. Skip the `sport_configs` local entry in `deployment/aws/terraform/eventbridge.tf` while AWS is dormant.
    - Rules are created per `sport_configs` entry, and production runs on a single local host ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#dormant-aws-rules)).
13. Add a heartbeat expectation for the league to the `heartbeats` of each job record in `odds_core.job_specs.JOB_SPECS` that keys one per league, sized like the other leagues' entries for that job ([heartbeat table](../operations/health-monitor.md#heartbeats)).
    - Keys are compound names (`fetch-oddsportal-<suffix>`), and a key with no history is never alerted ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).
14. Run `uv run pytest tests/unit/test_sport_registrations.py`.
    - It fails naming the missing entry when a `SportKey` lacks a timezone, kick-off authority or candidate markets, a `fetch-odds` league lacks a plan, a candidate market lacks its `odds_core.close.CLOSES` entry, or a league a collector covers (`odds_lambda.health_readers.collection_sources`) lacks a suffix or, for a collector with an overnight pause, an overnight window.
15. Add the league to `SPORTS` (a JSON array, `odds_core.config.DataCollectionConfig`) in the production host's `.env`, and deploy ([configuration](../foundations/configuration.md#list-fields-and-the-template)).
    - The scheduler bootstraps each per-league job for it. The odds-web [status strip](../operations/odds-web.md#status-strip) then shows its job chips and a source chip per collected market, unknown until the first snapshot of that market is stored.
16. Pass `--sport` and the league's market key (`1x2` or `h2h`) in every manual command and check the first fetch.
    - Several CLI defaults are NBA ([odds-cli](../foundations/odds-cli.md#sport-defaults)).

## Related

- [Sport, market and book keys](sport-and-market-keys.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
