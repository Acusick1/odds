---
type: howto
title: Debug the AWS self-scheduling jobs
description: A read-only workflow for stale data or missing tiers on AWS covering rules, CloudWatch logs, database versus fixtures, tier coverage, common failures and a safe manual invoke.
sources:
  - scripts/check_games_in_database.py
  - scripts/check_game_tier_coverage.py
  - scripts/check_tier_distribution.py
  - scripts/verify-schedules.sh
  - odds_lambda.scheduling.decision
  - odds_lambda.scheduling.backends.aws
  - odds_lambda.lambda_handler.lambda_handler
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/scraper.tf
  - deployment/aws/terraform/eventbridge.tf
  - odds_lambda.fetch_tier
  - odds_lambda.tier_utils
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.jobs.fetch_odds
  - odds_lambda.storage.writers
status: stable
---

# Debug the AWS self-scheduling jobs

Use when data looks stale or a tier is missing. Background: [Self-scheduling](self-scheduling.md), [Where each job runs](job-deployment-matrix.md), [Scheduler backends](scheduler-backends.md#aws-eventbridge-backend).

Deployed Lambda and rule state cannot be verified from the repo; every command below checks the live account.

**Safety:** against production use only the read-only `scripts/check_*.py` scripts. Never run migrations, `odds fetch` or any write there ([Databases and environments](../foundations/databases-and-environments.md)). Point a script at production with `DATABASE_URL="<prod_url>"` (take it from `PROD_DATABASE_URL` in `.env`).

## Where things are

Region `eu-west-1`. Scheduler function `odds-scheduler`, log group `/aws/lambda/odds-scheduler`; scraper `odds-scheduler-scraper`. Rules `odds-{job}[-{sport}]`. Live schedule: `odds scheduler list-jobs --backend aws`. The Lambda role lacks `events:ListRules`, so listing works only from a local shell with broader credentials (#443).

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
3. **Database vs fixtures.** `uv run python scripts/check_games_in_database.py [N]` lists the next scheduled games in the database, not the league's real fixtures. If games are missing, `fetch-odds` has not run; return to step 2. In code, `missing_games` in the tier validator means a completed game the Odds API returns that is absent from `events` (`odds_lambda.storage.tier_validator.TierCoverageValidator`). `validate daily` buckets days noon-to-noon UTC and compares against NBA discovery only ([Coverage checks](../storage/coverage-checks.md)). The three `check_*.py` scripts have no sport filter, and `check_games_in_database.py` prints NBA text (#465).
4. **Tier coverage.** `check_game_tier_coverage.py --date D` or `--event ID`; `check_tier_distribution.py [days]`. Tiers should match hours before kickoff and each game should span several ([Fetch tier](../storage/fetch-tier.md)). Four boundary encodings disagree at 3/12/24/72 h, so a game on a boundary may legitimately differ between decision and storage.
5. **Logs.** `aws logs tail /aws/lambda/odds-scheduler --since 2h --format short | grep -i "fetch_odds\|tier"`. Look for `fetch_odds_job_started` (execution began), `fetch_odds_executing` (tier and reason, for example "Game in 7.5h: Team A vs Team B (pregame tier)"), `odds_snapshot_stored` (per-event `fetch_tier`), `fetch_odds_next_scheduled` and `eventbridge_rule_updated` ([fetch-odds job](../collection/fetch-odds-job.md)). Quota or alert failures do not appear in these logs: `fetch-odds` writes no heartbeat and its quota alerts are not rate limited, so only stale-data checks catch a stalled chain (#438, [Alerting and heartbeats](../operations/alerting-and-heartbeats.md#heartbeats)).

## Common issues

- **Missing tier coverage.** Causes: EventBridge did not trigger; a Lambda error stopped the job before it scheduled the next run; a rule was edited by hand. Fix: wait for the next run, or redeploy; `deploy-prod.yml` re-invokes jobs and re-activates chains ([AWS infrastructure](../operations/aws-infrastructure.md)).
- **All events one tier.** The tier is computed per event in `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` (`odds_lambda.storage.writers`); migration `9ff81f073e6b` recalculated older rows ([OddsWriter](../storage/odds-writer.md)).
- **Lambda not running.** `aws events describe-rule --name odds-fetch-odds-epl`, `list-targets-by-rule`, then check the rule's invoke permission on the function. Fix by redeploying ([AWS infrastructure](../operations/aws-infrastructure.md)).
- **Dead chain.** Past-year cron expressions are dead rules; `scripts/verify-schedules.sh` finds them.

## Invoke manually

> [!WARNING]
> A manual invoke ingests odds, spends quota and reschedules the live rule. Against production, run it only to restart a dead chain.

```bash
echo '{"job": "fetch-odds", "sport": "soccer_epl"}' > /tmp/payload.json
aws lambda invoke --function-name odds-scheduler --payload file:///tmp/payload.json /tmp/response.json
cat /tmp/response.json

# Most recent execution logs
aws logs filter-log-events --log-group-name /aws/lambda/odds-scheduler --limit 100 | jq -r '.events[-20:] | .[] | .message'
```

The payload file avoids shell-escaping problems.

Always include `sport`; without it the job runs every sport and reschedules under the bare name, which matches no rule ([Lambda handler](scheduler-backends.md#lambda-handler)).

## Related

- [Add a scheduler job](howto-add-scheduler-job.md)
- [Self-scheduling and the decision engine](self-scheduling.md)
