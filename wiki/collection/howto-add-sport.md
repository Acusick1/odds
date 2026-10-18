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
status: stable
---

# Add a sport

The Odds API bookmakers and markets are config (`bookmakers`, a per-sport map, and `markets` on `odds_core.config.DataCollectionConfig`) and need no other code; a sport with no `bookmakers` entry fails its Odds API fetch. A new sport needs the steps below; skip a step only when the sport has no feed or job of that kind.

1. Add the sport to `odds_core.sports.SportKey`.
   - It becomes the `sport` argument of jobs, MCP tools and CLI commands ([sport-and-market-keys](sport-and-market-keys.md#sport-keys)).
2. Add a suffix in `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP`.
   - The suffix names EventBridge rules, heartbeat keys and `odds_mcp.server.get_scheduled_jobs` filters; without one, compound job names stay bare as for NBA ([job-registry](../scheduling/job-registry.md#compound-job-names)).
3. Add an entry to `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS`.
   - The decision engine skips runs inside the window ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).
4. If OddsPortal covers it, add a `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` to `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`, with the sport's OddsPortal `listing_path` and markets that are keys of `odds_lambda.oddsportal_client.FEED_MARKETS` (a market not in that table needs a new `odds_lambda.oddsportal_client.FeedMarketSpec` row first).
   - The spec carries its own overnight fields, so keep them equal to step 3 ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues)).
5. If Betfair lists it, add an entry to `odds_lambda.betfair.constants.SPORT_CONFIG`.
   - The Betfair allowlist in `odds_lambda.scheduling.jobs._JOB_SPORTS` is derived from that map, so no allowlist edit is needed ([betfair-exchange](betfair-exchange.md)).
6. If Polymarket lists three-way match markets for it, add an entry to `odds_lambda.polymarket_sports.POLYMARKET_SPORTS` with `three_way_markets=True`; the `fetch-polymarket-odds` allowlist follows `odds_lambda.polymarket_sports.MATCH_ODDS_SPORTS` ([Polymarket odds snapshots](polymarket-odds-snapshots.md#sport-config-and-discovery)).
7. Add team aliases in `odds_core.team`.
   - Event matching compares normalised names; unknown names pass through unchanged and surface as duplicate events or NULL `event_id` ([event-matching](../storage/event-matching.md)).
8. Add the sport to each allowlist in `odds_lambda.scheduling.jobs._JOB_SPORTS` whose job should run for it.
   - A job with an allowlist skips every sport not on it, so a new sport gets none of those jobs by default; per-sport jobs without an allowlist run for every sport in `data_collection.sports` ([local-scheduler](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
9. Skip `sport_configs` in `deployment/aws/terraform/eventbridge.tf` while AWS is dormant.
   - Rules are created per `sport_configs` entry, and production runs on the mini-PC ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#dormant-aws-rules)).
10. Add heartbeat expectations (allowed hours between runs, longer for jobs with an overnight pause) to `odds_core.config.AlertConfig.heartbeat_expectations` for each job that uses `odds_core.alerts.job_alert_context` (`fetch-polymarket-odds-epl` is one).
   - Keys are compound names (`fetch-oddsportal-<suffix>`), and a key with no history is never alerted ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).
11. Pass `--sport` and the sport's market key (`1x2` or `h2h`) in every manual command and check the first fetch.
    - Several CLI defaults are NBA ([odds-cli](../foundations/odds-cli.md#sport-defaults)).

## Related

- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
