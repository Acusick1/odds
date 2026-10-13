---
type: howto
title: Debug the AWS self-scheduling jobs
description: A read-only workflow for stale data or missing tiers on AWS covering rules, CloudWatch logs, database versus fixtures, tier coverage, common failures and a safe manual invoke.
sources:
  - scripts/check_games_in_database.py
  - scripts/check_game_tier_coverage.py
  - scripts/check_tier_distribution.py
  - scripts/verify-schedules.sh
  - packages/odds-lambda/odds_lambda/scheduling/decision.py
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py
  - packages/odds-lambda/odds_lambda/lambda_handler.py::lambda_handler
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/terraform/eventbridge.tf
  - packages/odds-lambda/odds_lambda/fetch_tier.py
  - packages/odds-lambda/odds_lambda/tier_utils.py
  - packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator
  - packages/odds-lambda/odds_lambda/jobs/fetch_odds.py
  - packages/odds-lambda/odds_lambda/storage/writers.py
status: stable
---

# Debug the AWS self-scheduling jobs

Use when data looks stale or a tier is missing. Background: [Self-scheduling](self-scheduling.md), [Jobs and where each runs](jobs-and-deployment-matrix.md), [Scheduler backends](scheduler-backends.md#aws-eventbridge-backend).

**Safety:** against production use only the read-only `scripts/check_*.py` scripts. Never run migrations, `odds fetch` or any write there ([Databases](../foundations/databases-and-environments.md)). Point a script at production with `DATABASE_URL="<prod_url>"` (take it from `PROD_DATABASE_URL` in `.env`).

## Where things are

Region `eu-west-1`. Scheduler function `odds-scheduler`, log group `/aws/lambda/odds-scheduler`; scraper `odds-scheduler-scraper`. Rules `odds-{job}[-{sport}]`. Live schedule: `odds scheduler list-jobs --backend aws`. The Lambda role lacks `events:ListRules`, so listing works only from a local shell with broader credentials.

## Workflow

1. **Context.** `date -u` and `grep "^DATABASE_URL=" .env`. The daily validation file for the previous day is dated the day it ran: the file dated D covers D-1 ([CI/CD workflows](../operations/ci-cd-workflows.md)).
2. **Last and next run.** The last run should fall within the interval of the nearest game's tier ([Fetch tier](../storage/fetch-tier.md)).
   ```bash
   aws events list-rules --name-prefix odds- | jq -r '.Rules[] | "\(.Name) | \(.State) | \(.ScheduleExpression)"'

   aws logs describe-log-streams --log-group-name /aws/lambda/odds-scheduler \
     --order-by LastEventTime --descending --limit 10 \
     | jq -r '.logStreams[] | "\(.lastEventTimestamp | tonumber / 1000 | strftime("%Y-%m-%d %H:%M:%S UTC")) | \(.logStreamName)"'
   ```
   A self-scheduled rule shows a one-time `cron(... <year>)`; a rule still on `rate(1 day)` was never rescheduled.
3. **Database vs fixtures.** `uv run python scripts/check_games_in_database.py [N]` lists the next scheduled games in the database, not the league's real fixtures. If games are missing, `fetch-odds` has not run; return to step 2. In code, `missing_games` in the tier validator means a completed game the Odds API returns that is absent from `events` (`packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator`). `validate daily` buckets days noon-to-noon UTC and compares against NBA discovery only ([Validation and coverage](../storage/data-validation.md)).
4. **Tier coverage.** `check_game_tier_coverage.py --date D` or `--event ID`; `check_tier_distribution.py [days]`. Tiers should match hours before kickoff and each game should span several ([Fetch tier](../storage/fetch-tier.md)). Four boundary encodings disagree at 3/12/24/72 h, so a game on a boundary may legitimately differ between decision and storage.
5. **Logs.** `aws logs tail /aws/lambda/odds-scheduler --since 2h --format short | grep -i "fetch_odds\|tier"`. Look for `fetch_odds_job_started` (execution began), `fetch_odds_executing` (tier and reason, for example "Game in 7.5h: Team A vs Team B (pregame tier)"), `odds_snapshot_stored` (per-event `fetch_tier`), `fetch_odds_next_scheduled` and `eventbridge_rule_updated` ([Odds API ingestion](../collection/odds-api-ingestion.md)).

## Common issues

- **Missing tier coverage.** Causes: EventBridge did not trigger; a Lambda error stopped the job before it scheduled the next run; a rule was edited by hand. Fix: wait for the next run, or redeploy; `deploy-prod.yml` re-invokes jobs and re-activates chains ([AWS infrastructure](../operations/aws-infrastructure.md)).
- **All events one tier.** The tier is computed per event in `packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter.store_odds_snapshot` (`packages/odds-lambda/odds_lambda/storage/writers.py`); migration `9ff81f073e6b` recalculated older rows ([Odds snapshots and OddsWriter](../storage/odds-storage.md)).
- **Lambda not running.** `aws events describe-rule --name odds-fetch-odds-epl`, `list-targets-by-rule`, then check the rule's invoke permission on the function. Fix by redeploying ([AWS infrastructure](../operations/aws-infrastructure.md)).
- **Dead chain.** Past-year cron expressions are dead rules; `scripts/verify-schedules.sh` finds them.

## Invoke manually

```bash
echo '{"job": "fetch-odds", "sport": "soccer_epl"}' > /tmp/payload.json
aws lambda invoke --function-name odds-scheduler --payload file:///tmp/payload.json /tmp/response.json
cat /tmp/response.json

# Most recent execution logs
aws logs filter-log-events --log-group-name /aws/lambda/odds-scheduler --limit 100 | jq -r '.events[-20:] | .[] | .message'
```

The payload file avoids shell-escaping problems.

Always include `sport`; without it the job runs every sport and reschedules under the bare name, which matches no rule ([Lambda handler](scheduler-backends.md#lambda-handler)). This is a write path (it ingests and reschedules); do not do it against production casually.

## Hazards

- The three `check_*.py` scripts have no sport filter; `check_games_in_database.py` text still talks about NBA.
- Quota or alert failures do not appear in these logs. `fetch-odds` has no heartbeat and its quota alerts are unrate-limited, so only stale-data checks catch a stalled chain ([Alerts and health](../operations/alerts-and-health.md)).
- Deployed Lambda and rule state cannot be verified from the repo; every command above checks the live account.

## Related

- [Add a scheduler job](howto-add-scheduler-job.md)
- [Self-scheduling and the decision engine](self-scheduling.md)
