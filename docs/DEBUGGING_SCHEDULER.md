# Debugging the AWS Scheduler

How to diagnose and verify the self-scheduling jobs that run on AWS Lambda. Which jobs run
on AWS and which run on the local scheduler is in the
[Job Deployment Matrix](DEPLOYMENT.md#job-deployment-matrix); how scheduling decisions are
made is in [ARCHITECTURE.md](ARCHITECTURE.md#intelligent-scheduling-system).

⚠️ **Production database safety**: against production, use ONLY the read-only
`scripts/check_*.py` scripts. NEVER run migrations, `odds fetch` commands, or any other
write against production.

## Where to look

| Thing | Value |
|-------|-------|
| Region | `eu-west-1` |
| Scheduler Lambda | `odds-scheduler`, log group `/aws/lambda/odds-scheduler` |
| Scraper Lambda | `odds-scheduler-scraper`, log group `/aws/lambda/odds-scheduler-scraper` |
| EventBridge rules | `odds-{job}[-{sport}]`, e.g. `odds-fetch-odds-epl`, `odds-update-status` |
| Live schedule | `odds scheduler list-jobs --backend aws` |

The diagnostic scripts connect to `DATABASE_URL` from `.env`. To check production, prefix
a read-only script with `DATABASE_URL="<prod_url>"`, taking the URL from
`PROD_DATABASE_URL` in your `.env`.

## Diagnostic workflow

### 1. Establish context

```bash
date -u                          # validation files look back one day
grep "^DATABASE_URL=" .env       # which database the scripts will hit
```

The `daily-validation` GitHub workflow runs `odds validate daily` at 09:00 UTC and saves
`validation_YYYYMMDD.json` for the **previous** day: the file dated `20251103` was created
on 4 November and checks games on 3 November.

### 2. Check the last run and the next scheduled run

```bash
aws events list-rules --name-prefix odds- | jq -r '.Rules[] | "\(.Name) | \(.State) | \(.ScheduleExpression)"'

aws logs describe-log-streams \
  --log-group-name /aws/lambda/odds-scheduler \
  --order-by LastEventTime \
  --descending \
  --limit 10 | jq -r '.logStreams[] | "\(.lastEventTimestamp | tonumber / 1000 | strftime("%Y-%m-%d %H:%M:%S UTC")) | \(.logStreamName)"'
```

The last run should fall within the expected interval for the nearest game's tier.
Self-scheduled rules show a specific cron time (e.g. `cron(52 15 4 11 ? 2025)`); a rule
still on its `rate(1 day)` placeholder has never been rescheduled by the Lambda.

### 3. Compare the database with the real fixture list

```bash
uv run python scripts/check_games_in_database.py      # next 10 scheduled games and their tiers
uv run python scripts/check_games_in_database.py 20   # next 20
```

This shows games **in the database**, not the league's published schedule. If the league
has games today that the database lacks, `fetch-odds` has not run within its expected
interval and the database is stale; go back to step 2.

If validation reports `missing_games`, those games completed with **zero** odds snapshots.
Possible causes:

- The scheduler was not running during that period
- Lambda errors prevented collection
- The API never returned those games (rare)

### 4. Check tier coverage

```bash
uv run python scripts/check_game_tier_coverage.py --date 2025-11-03   # all games on a date
uv run python scripts/check_game_tier_coverage.py --event <event_id>  # one game
uv run python scripts/check_tier_distribution.py                      # last 7 days
uv run python scripts/check_tier_distribution.py 14                   # last 14 days
```

When checking a validation file, use the day before the file was created. Each snapshot's
tier should match its hours before commence time (see
[Fetch Tier System](ARCHITECTURE.md#fetch-tier-system)), and each game should have
snapshots across several tiers.

### 5. Read the fetch-odds logs

```bash
aws logs tail /aws/lambda/odds-scheduler --since 2h --format short | grep -i "fetch_odds\|tier" | tail -30
```

Log events to look for:

- `fetch_odds_job_started`: job execution started
- `fetch_odds_executing`: tier and reason (e.g. "Game in 7.5h: Team A vs Team B (pregame tier)")
- `odds_snapshot_stored`: per-event tier assignment (check the `fetch_tier` field)
- `fetch_odds_next_scheduled`: next execution time and tier
- `eventbridge_rule_updated`: the EventBridge rule was updated with the new schedule

## Common issues

### Missing tier coverage

**Symptoms**: validation shows games missing opening/early/sharp/pregame tiers.

**Diagnosis**: check when `fetch-odds` last ran and the rule's schedule (step 2), then
work out the expected run times from the tier intervals.

**Causes**:

- EventBridge failed to trigger (AWS service issue)
- A Lambda error stopped the job before it scheduled its next run
- A rule was disabled or modified by hand

**Fix**: wait for the next scheduled run, or redeploy to production. `deploy-prod.yml`
invokes the scheduler and scraper jobs after `terraform apply`, which re-activates the
self-scheduling chain.

### All events tagged with the same tier

**Symptoms**: snapshots for events with different hours-until-commence share one tier.

**Diagnosis**: the tier is calculated per event from that event's commence time in
`packages/odds-lambda/odds_lambda/storage/writers.py`; check that logic. Migration
`9ff81f073e6b` recalculated tiers for rows stored before per-event calculation.

### Lambda not running

**Symptoms**: no recent log entries in CloudWatch.

**Diagnosis**:

1. Check the rule state: `aws events describe-rule --name odds-fetch-odds-epl`
2. Check the rule targets: `aws events list-targets-by-rule --rule odds-fetch-odds-epl`
3. Check Lambda permissions: the rule must have invoke permission on the Lambda

**Fix**: redeploy via Terraform or the GitHub Actions deploy workflow.

## Invoking a job manually

```bash
# Use a payload file to avoid shell escaping issues
echo '{"job": "fetch-odds", "sport": "soccer_epl"}' > /tmp/payload.json
aws lambda invoke \
  --function-name odds-scheduler \
  --payload file:///tmp/payload.json \
  /tmp/response.json
cat /tmp/response.json

# Most recent execution logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/odds-scheduler \
  --limit 100 | jq -r '.events[-20:] | .[] | .message'
```

Include `sport`: without it, `fetch-odds` runs every configured sport and reschedules under
the bare `fetch-odds` name, which matches no rule.

## Related files

- `packages/odds-lambda/odds_lambda/scheduling/decision.py`: scheduling decision logic
- `packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`: EventBridge integration
- `packages/odds-lambda/odds_lambda/fetch_tier.py`, `tier_utils.py`: tier definitions and calculation
- `packages/odds-lambda/odds_lambda/jobs/fetch_odds.py`: fetch odds job
- `packages/odds-lambda/odds_lambda/storage/writers.py`: tier assignment during storage
