---
type: explanation
title: Odds API quota accounting
description: How quota is spent, recorded in fetch_logs, alerted by the health monitor, and checked with odds status quota, and where those disagree.
sources:
  - odds_core.models.FetchLog
  - odds_core.config.APIConfig
  - odds_core.config.APIConfig.key_for
  - odds_core.config.AlertConfig.quota_warning_threshold
  - odds_core.config.AlertConfig.quota_critical_threshold
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_lambda.ingestion.OddsIngestionService.ingest_sport
  - odds_lambda.jobs.fetch_odds.main
  - odds_lambda.jobs.fetch_odds.CADENCE
  - odds_lambda.jobs.fetch_odds.NBA_CADENCE
  - odds_lambda.jobs.fetch_odds.NBA_LEADS
  - odds_lambda.storage.readers.OddsReader.get_latest_quota_by_key
  - odds_lambda.storage.readers.QUOTA_MAX_AGE
  - odds_lambda.scheduling.jobs._PAID_API_JOBS
  - odds_lambda.health_monitor.HealthMonitor.check_api_quota
  - odds_lambda.storage.readers.OddsReader.get_fetch_logs
  - odds_lambda.storage.readers.OddsReader.get_database_stats
  - odds_cli.commands.status._show_quota
  - odds_analytics.backfill_executor.BackfillExecutor.execute
status: stable
---

# Odds API quota accounting

Quota is a per-key monthly balance that only [the client](odds-api-client.md#cost-per-call) spends. Each sport spends one key, EPL key 0 and NBA key 1 ([keys](odds-api-client.md#retry-and-keys)), so a key's balance is the balance of the sport that spends it, and every view below is per key (two sports given the same index would share one reading). A fetch-odds run (one sport per run) that stores no snapshot while an EPL fixture is within 48 hours, or an NBA game within 14 hours, fails and alerts regardless of quota ([fetch-odds-job](fetch-odds-job.md#flow)). Three places read or act on it, and each reads it differently.

## Where quota is recorded, alerted and checked

| Place | What it reads | Reminded | Recorded in `alert_history` |
|---|---|---|---|
| `odds_core.models.FetchLog.api_quota_remaining` and `odds_core.models.FetchLog.api_quota_used` | the `x-requests-remaining` and `x-requests-used` headers, written once per sport and run by `odds_lambda.ingestion.OddsIngestionService.ingest_sport`, after its listed games are stored (each resolved to its event through the event resolver; [fetch logs](odds-ingestion-service.md#fetch-logs)) | n/a | n/a |
| `odds_lambda.health_monitor.HealthMonitor.check_api_quota` | per key, the newest `fetch_logs` row with a non-null remaining and used count of the sports that spend it, ignoring rows over 31 days old (`odds_lambda.storage.readers.OddsReader.get_latest_quota_by_key`), same fraction | yes | yes |
| `odds status quota` (`odds_cli.commands.status._show_quota`) | per key, the newest 10 `fetch_logs` rows with a non-null remaining quota of its sports, for each key with a reading under 31 days old, headed by the key's plan size, used plus remaining from its newest reading; the per-fetch Used column is the next-older row's remaining minus this row's, `reset` when negative, blank for the oldest | n/a | n/a |

- Quota views skip rows with a null quota, such as Betfair rows ([betfair-exchange](betfair-exchange.md)): `odds_lambda.storage.readers.OddsReader.get_fetch_logs` with `with_quota=True` and `odds_lambda.storage.readers.OddsReader.get_database_stats` both filter in the query; the latter also counts price rows, and snapshots as distinct event and snapshot-time pairs. `fetch_logs` has no source column, so Odds API rows are inferred from the nullable quota.
- `fetch-odds` sends no quota alert; a failed run or a failed booking alerts as in [fetch-odds-job](fetch-odds-job.md).
- The health monitor applies the thresholds as `quota_low:key<N>` and `quota_critical:key<N>` (N is the key's index), sent once per condition and reminded every `stale_reminder_minutes` through `alert_history` ([health-monitor](../operations/health-monitor.md)), so EPL's nearly spent key alerts without NBA's fresh one hiding it, and the alert names the sports that spend the key. A key whose sports recorded no quota in 31 days (an off-season) has no reading and no alert.
- Both thresholds divide a key's balance by its plan size, the used plus remaining counts of its newest reading, so each key is judged against its own plan and no setting holds the size. Rows logged before `api_quota_used` existed carry no plan size and are skipped. `odds status` lists one `API Quota` row per key with the same plan size.

## Scheduled spend

Each `fetch-odds` call names at most 10 bookmakers for its sport (`odds_core.config.DataCollectionConfig.bookmakers`), which the Odds API bills as one region, so a call costs one unit per configured market ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)). The scheduled jobs that spend units are `fetch-odds` and `fetch-scores`, both run on the production host; `fetch-odds` is limited to EPL and NBA and `fetch-scores` to EPL by their `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlists, while Polymarket, Betfair, the ESPN NBA sync and the NBA injury-report job spend none (MLB results come from the free MLB Stats API) ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). `odds scheduler smoke`, which [deploy.sh](../scheduling/howto-run-scheduler-service.md#redeploy) runs, skips both (`odds_lambda.scheduling.jobs._PAID_API_JOBS`) unless passed `--include-paid-api`, so a deploy spends no units ([local-scheduler](../scheduling/local-scheduler.md#cli)).

The vendor bills `/odds` at markets x regions, and every group of 10 bookmakers counts as one region ([usage quota costs](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs)), so a `fetch-odds` call is 3 units and its cadence sets the monthly spend ([cadence](../scheduling/decision-engine.md#cadence)). The closing tier polls every 2 h and the pregame tier every 3 h (`odds_lambda.jobs.fetch_odds.CADENCE`) to fit the 500-unit key, which resets on the 1st of the month ([vendor FAQ](https://the-odds-api.com/frequently-asked-questions/)); Betfair and OddsPortal cover the close. Simulating the cadence over a month of 4 weekend rounds (Friday night to Monday night) with 1 or 3 midweek rounds, against the previous 0.5 h closing and 3 h pregame cadence:

| Cadence | Fetches, 1 midweek round | Units | Fetches, 3 midweek rounds | Units |
|---|---|---|---|---|
| 0.5 h closing, 3 h pregame | 149 | 447 | 179 | 537 |
| 2 h closing, 3 h pregame | 71 | 213 | 88 | 264 |

`fetch-scores` adds 2 units per call (the vendor charges 2 when `daysFrom` is sent); the fetch logs of 17-20 Oct 2026 show about 27 units beyond the odds fetches over one weekend round, so a month adds roughly 110-150 `(inferred)`. The free `/events` sync costs nothing.

On a 500-unit key the quota alerts fire below 100 units (warning) and 50 units (critical) remaining. With 161 units left on EPL's key on 20 Oct, the 23-26 Oct and 31 Oct rounds need about 90 units of `fetch-odds` and about 40 of `fetch-scores` before the reset on 1 Nov, leaving about 30, so that cycle crosses the critical threshold around 31 Oct. In a normal month the thresholds trip only late.

### NBA spend

NBA spends its own key, so its 500 units do not compete with EPL's. One call prices every game of the slate, at 3 units, and `odds_lambda.jobs.fetch_odds.NBA_CADENCE` and `odds_lambda.jobs.fetch_odds.NBA_LEADS` run it three to four times a game day ([fetch-odds-job](fetch-odds-job.md#nba-schedule)): the morning run (10:00 UTC), 3 h and 1 h before the slate's first tip-off, and 1 h before the slate's main tip-off when that is at least 90 minutes after the first. The first-tip runs alone priced 36-45% of games last more than 3 h before their tip-off, because a slate's games tip off over hours; the main-tip run cuts that to 14-21%.

Replaying the schedule over the stored NBA events (30-day windows starting 2025-11-01, 2025-12-01 and 2026-01-10) gives 100, 106 and 117 calls, 300, 318 and 351 units, 60-70% of the key. The 150-200 spare units cover a day with no game (one call), a rescheduled slate, a second slate on a matinee day and manual `odds fetch` runs. `fetch-scores` does not run for NBA, whose results come from ESPN.

## Where the views disagree

- **Failed calls.** A failed `/odds` call writes no `fetch_logs` row, so it leaves no quota trace (#472; [fetch logs](odds-ingestion-service.md#fetch-logs)).
- **Historical calls.** The manifest planner derives a call's cost from the manifest's markets and bookmakers, and the executor records the key's remaining quota (`x-requests-remaining`) in `fetch_logs` after each call, failed calls included, so a backfill's spend shows in `odds status quota`; each call's own cost (`x-requests-last`) is only in the structured log and the run's summed units ([odds-api-backfill](odds-api-backfill.md#executing-calls)).
- **Scores.** `fetch-scores` spends more than the minimum per call because the client always sends `daysFrom` ([results-jobs](results-jobs.md)).

## Related

- [The Odds API client and per-sport keys](odds-api-client.md)
- [Alerts and the alert_history table](../operations/alerts.md)
- [AGENTS.md constraints](../../AGENTS.md#constraints): check `odds status quota` before bulk or historical fetches
