---
type: howto
title: Debug the AWS self-scheduling jobs
summary: "Read-only diagnostic workflow: rules, CloudWatch logs, database vs fixture list, tier coverage, common failures and manual Lambda invoke."
sources:
  - docs/DEBUGGING_SCHEDULER.md#where-to-look
  - docs/DEBUGGING_SCHEDULER.md#diagnostic-workflow
  - docs/DEBUGGING_SCHEDULER.md#common-issues
  - docs/DEBUGGING_SCHEDULER.md#invoking-a-job-manually
  - docs/DEBUGGING_SCHEDULER.md#related-files
  - docs/DEPLOYMENT.md#job-deployment-matrix
  - scripts/check_games_in_database.py
  - scripts/check_game_tier_coverage.py
  - scripts/check_tier_distribution.py
  - packages/odds-lambda/odds_lambda/scheduling/decision.py
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py
  - packages/odds-lambda/odds_lambda/fetch_tier.py
  - packages/odds-lambda/odds_lambda/tier_utils.py
  - packages/odds-lambda/odds_lambda/jobs/fetch_odds.py
  - packages/odds-lambda/odds_lambda/storage/writers.py
verified_at: 7887039
---

# Debug the AWS self-scheduling jobs

Use when data looks stale or a tier is missing. Background: [self-scheduling](self-scheduling.md), [job deployment matrix](job-deployment-matrix.md), [decision engine](scheduling-decision-engine.md).

**Safety:** against production use only the read-only `scripts/check_*.py` scripts. Never run migrations, `odds fetch` or any write there. Point a script at production with `DATABASE_URL="<prod_url>"` (take it from `PROD_DATABASE_URL` in `.env`).

## Where things are

Region `eu-west-1`. Scheduler function `odds-scheduler`, log group `/aws/lambda/odds-scheduler`; scraper `odds-scheduler-scraper`. Rules `odds-{job}[-{sport}]` ([EventBridge backend](aws-eventbridge-backend.md)). Live schedule: `odds scheduler list-jobs --backend aws`.

## Workflow

1. **Context.** `date -u` and `grep "^DATABASE_URL=" .env`. The daily validation file for the previous day is dated the day it ran ([CI/CD workflows](ci-cd-workflows.md)).
2. **Last and next run.** `aws events list-rules --name-prefix odds-` shows state and expression; `aws logs describe-log-streams ... --order-by LastEventTime --descending` shows last runs. A self-scheduled rule shows a one-time `cron(... <year>)`; a rule still on `rate(1 day)` was never rescheduled.
3. **Database vs fixtures.** `uv run python scripts/check_games_in_database.py [N]` lists the next scheduled games in the database, not the league's real fixtures. If games are missing, `fetch-odds` has not run; return to step 2. Games reported as `missing_games` completed with zero snapshots: scheduler down, Lambda errors, or the API never returned them.
4. **Tier coverage.** `check_game_tier_coverage.py --date D` or `--event ID`; `check_tier_distribution.py [days]`. Tiers should match hours before kickoff and each game should span several ([fetch tier](fetch-tier.md), [completeness validation](completeness-validation.md)).
5. **Logs.** `aws logs tail /aws/lambda/odds-scheduler --since 2h --format short | grep -i "fetch_odds\|tier"`. Look for `fetch_odds_executing`, `odds_snapshot_stored` (`fetch_tier`), `fetch_odds_next_scheduled`, `eventbridge_rule_updated` ([fetch-odds](job-fetch-odds.md)).

## Common issues

- **Missing tier coverage.** Causes: EventBridge did not trigger; a Lambda error stopped the job before it scheduled the next run; a rule was edited by hand. Fix: wait for the next run, or redeploy; `deploy-prod.yml` re-invokes jobs and re-activates chains.
- **All events one tier.** The tier is computed per event in `OddsWriter.store_odds_snapshot` (`packages/odds-lambda/odds_lambda/storage/writers.py`); migration `9ff81f073e6b` recalculated older rows ([odds writer](odds-writer.md)).
- **Lambda not running.** `aws events describe-rule --name odds-fetch-odds-epl`, `list-targets-by-rule`, then check the rule's invoke permission on the function. Fix by redeploying ([deploy to AWS](howto-deploy-aws.md)).

## Invoke manually

```bash
echo '{"job": "fetch-odds", "sport": "soccer_epl"}' > /tmp/payload.json
aws lambda invoke --function-name odds-scheduler --payload file:///tmp/payload.json /tmp/response.json
```

Always include `sport`; without it the job runs every sport and reschedules under the bare name, which matches no rule ([Lambda handler](lambda-handler.md)). This is a write path (it ingests and reschedules); do not do it against production casually.

## Gotchas

- The three `check_*.py` scripts have no sport filter; `check_games_in_database.py` text still talks about NBA.
- Past-year cron expressions are dead rules; `scripts/verify-schedules.sh` finds them.
- Quota or alert failures do not appear in these logs; see [health monitor](health-monitor.md).
