---
type: howto
title: Add a sport
description: "Steps to add a sport: SportKey, suffix, overnight window, league spec, Betfair config, team aliases, cron allowlists, Terraform rules, heartbeat keys."
sources:
  - odds_core.sports.SportKey
  - odds_core.config.DataCollectionConfig
  - odds_core.config.AlertConfig.heartbeat_expectations
  - odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS
  - odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS
  - odds_lambda.betfair.constants.SPORT_CONFIG
  - odds_core.team
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Add a sport

The Odds API bookmakers and markets are config lists (`bookmakers`, `markets` on `odds_core.config.DataCollectionConfig`) and need no code. A new sport needs the steps below; skip a step only when the sport has no feed or job of that kind.

1. Add the sport to `odds_core.sports.SportKey`.
   - It becomes the `sport` argument of jobs, MCP tools and CLI commands ([sport-and-market-keys](sport-and-market-keys.md#sport-keys)).
2. Add a suffix in `odds_lambda.scheduling.jobs._SPORT_SUFFIX_MAP`.
   - The suffix names EventBridge rules, heartbeat keys and `odds_mcp.server.get_scheduled_jobs` filters; without one, compound job names stay bare as for NBA ([job-registry](../scheduling/job-registry.md#compound-job-names)).
3. Add an entry to `odds_lambda.scheduling.decision.OVERNIGHT_WINDOWS`.
   - The decision engine skips runs inside the window ([decision-engine](../scheduling/decision-engine.md#overnight-windows)).
4. If OddsPortal covers it, add a `odds_lambda.jobs.fetch_oddsportal.LeagueSpec` to `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`.
   - The spec carries its own overnight fields, so keep them equal to step 3 ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues)).
5. If Betfair lists it, add an entry to `odds_lambda.betfair.constants.SPORT_CONFIG`.
   - The Betfair cron allowlist in `odds_lambda.scheduling.jobs._JOB_CRON_MAP` is derived from that map, so no cron edit is needed ([betfair-exchange](betfair-exchange.md)).
6. Add team aliases in `odds_core.team`.
   - Event matching compares normalised names; unknown names pass through unchanged and surface as duplicate events or NULL `event_id` ([event-matching](../storage/event-matching.md)).
7. Add the sport to the allowlist of each cron job in `odds_lambda.scheduling.jobs._JOB_CRON_MAP` that should run for it.
   - Allowlists are explicit per job, so a new sport gets no cron by default.
8. Add the sport to `sport_configs` in `deployment/aws/terraform/eventbridge.tf` if the jobs should run on AWS.
   - Rules are created per `sport_configs` entry; MLB is commented out there, so nothing runs on AWS for it ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)).
9. Add heartbeat expectations to `odds_core.config.AlertConfig.heartbeat_expectations` for each job that uses `odds_core.alerts.job_alert_context`.
   - Keys are compound names (`fetch-oddsportal-<suffix>`), and a key with no history is never alerted ([alerting-and-heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).
10. Pass `--sport` and the sport's market key (`1x2` or `h2h`) in every manual command and check the first fetch.
    - Several CLI defaults are NBA ([odds-cli](../foundations/odds-cli.md#sport-defaults)).

## Related

- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Add a scheduler job](../scheduling/howto-add-scheduler-job.md)
