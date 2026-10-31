---
type: howto
title: Add a sport
description: "Steps to add a sport: SportKey, suffix, overnight window, league spec, Betfair config, team aliases, job sport allowlists, Terraform rules, heartbeat keys."
sources:
  - odds_core.sports.SportKey
  - odds_core.config.DataCollectionConfig
  - odds_core.config.AlertConfig.heartbeat_expectations
  - odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS
  - odds_lambda.betfair.constants.SPORT_CONFIG
  - odds_core.team
  - deployment/aws/terraform/eventbridge.tf
mentions:
  - odds_core.alerts.job_alert_context
status: stable
---

# Add a sport

The Odds API bookmakers and markets are config (`bookmakers`, a per-sport map, and `markets` on `odds_core.config.DataCollectionConfig`) and need no other code; a sport with no `bookmakers` entry fails its Odds API fetch. A new sport needs the steps below; skip a step only when the sport has no feed or job of that kind.

1. Add the sport to `odds_core.sports.SportKey`.
   - It becomes the `sport` argument of jobs, MCP tools and CLI commands ([sport-and-market-keys](sport-and-market-keys.md#sport-keys)).
2. Add a suffix in `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP`.
   - The suffix names EventBridge rules, heartbeat keys and `odds_mcp.server.get_scheduled_jobs` filters; without one, compound job names stay bare ([job-registry](../scheduling/job-registry.md#compound-job-names)).
3. Add an entry to `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS`.
   - The decision engine skips runs inside the window ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).
4. If OddsPortal covers it, add a `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` to `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`, with the sport's OddsPortal `league` slug, `listing_path`, `sport_key`, `sport_title` and markets that are keys of `odds_lambda.oddsportal_client.FEED_MARKETS` (a market not in that table needs a new `odds_lambda.oddsportal_client.FeedMarketSpec` row first).
   - The job reads its overnight window from step 3 ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues)).
5. If Betfair lists it, add an entry to `odds_lambda.betfair.constants.SPORT_CONFIG`, with its match odds market and any other market types to collect (a totals market carries its line).
   - The Betfair allowlist in `odds_lambda.scheduling.jobs._JOB_SPORTS` is derived from that map, so no allowlist edit is needed ([betfair-exchange](betfair-exchange.md)).
6. If Polymarket lists its match markets, add an entry to `odds_lambda.polymarket_sports.POLYMARKET_SPORTS` (`has_draw` for a three-way sport); the `fetch-polymarket-odds` allowlist follows that dict ([Polymarket odds snapshots](polymarket-odds-snapshots.md#sport-config-and-discovery)).
7. Add team aliases in `odds_core.team`.
   - Event matching compares normalised names; unknown names pass through unchanged and surface as duplicate events or NULL `event_id` ([event-matching](../storage/event-matching.md)).
8. Add the sport to each allowlist in `odds_lambda.scheduling.jobs._JOB_SPORTS` whose job should run for it.
   - A job with an allowlist skips every sport not on it, so a new sport gets none of those jobs until added; `fetch-oddsportal-results` is EPL-only there, so a sport settled from OddsPortal results is added to it; per-sport jobs without an allowlist run for every sport in `data_collection.sports` ([local-scheduler](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
9. Skip the `sport_configs` local entry in `deployment/aws/terraform/eventbridge.tf` while AWS is dormant.
   - Rules are created per `sport_configs` entry, and production runs on a single local host ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#dormant-aws-rules)).
10. Add heartbeat expectations (allowed hours between runs, longer for jobs with an overnight pause) to `odds_core.config.AlertConfig.heartbeat_expectations` for each job that records a heartbeat through [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery) (on clean exit), keyed by compound name (a daily job needs 26 to 30 h, as `fetch-espn-fixtures` and `fetch-mlb-probables` have; `fetch-betfair-exchange`, `fetch-polymarket-odds` and `fetch-espn-fixtures` (an NBA key beside the EPL one) have an entry per sport; `fetch-odds` and `fetch-scores` have an EPL entry only, since both are EPL-only). The local scheduler cancels any run that exceeds `odds_lambda.scheduling.runner.JOB_TIMEOUT`, so a sport's job must finish within it.
   - Keys are compound names (`fetch-oddsportal-<suffix>`), and a key with no history is never alerted ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).
11. Pass `--sport` and the sport's market key (`1x2` or `h2h`) in every manual command and check the first fetch.
    - Several CLI defaults are NBA ([odds-cli](../foundations/odds-cli.md#sport-defaults)).

## Related

- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
