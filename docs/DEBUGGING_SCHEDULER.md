# Debugging the Intelligent Scheduler

**Audience**: This guide is for LLM agents debugging the scheduler system.

**Quick Reference**: How to debug and verify the AWS Lambda intelligent scheduling system.

## Diagnostic Workflow (Start Here)

⚠️ **PRODUCTION DATABASE SAFETY**: If checking production, ONLY use read-only `scripts/check_*.py` commands. NEVER run migrations, CLI fetch commands, or any write operations against production.

When debugging scheduler issues, follow this workflow:

### Step 1: Establish Context

```bash
# What is today's date?
date -u

# Which database am I checking? (local/dev/prod)
grep "^DATABASE_URL=" .env
```

**Key Facts**:

- Validation file `validation_YYYYMMDD.json` is created the NEXT day, looking BACK at previous day
- Example: File dated `20251103` was created on Nov 4, checking Nov 3 games
- Scripts default to local/dev database (check `.env` to confirm)
- Production Lambda uses `PROD_DATABASE_URL`
- **To check production**: Prefix commands with `DATABASE_URL="<prod_url>"` (read-only scripts only)

### Step 2: Quick Health Check

```bash
# Combined health check: last run + next scheduled
aws events list-rules --name-prefix odds-fetch-odds | jq -r '.Rules[] | "\(.Name): \(.ScheduleExpression)"' && \
aws logs describe-log-streams \
  --log-group-name /aws/lambda/odds-scheduler \
  --order-by LastEventTime \
  --descending \
  --limit 1 | jq -r '.logStreams[0] | "Last run: \(.lastEventTimestamp | tonumber / 1000 | strftime("%Y-%m-%d %H:%M:%S UTC"))"'
```

**What to check**: Last run should be recent (within expected tier interval), and next scheduled time should be reasonable based on upcoming games.

### Step 3: Understand the Gap

```bash
# What games does the database have?
uv run python scripts/check_games_in_database.py 5

# What games does NBA actually have?
# Visit: https://www.nba.com/schedule
```

**If there's a mismatch**: Database is stale because fetch-odds hasn't run. Check Step 2 timing.

**If validation shows "missing_games"**: These games completed but have ZERO odds snapshots in database. Possible causes:

- Scheduler wasn't running during that time period
- Lambda errors prevented data collection
- API never returned these games (rare API coverage issue)

### Step 4: Check Tier Coverage (if games exist)

```bash
# For specific date (use PREVIOUS day if checking validation file)
uv run python scripts/check_game_tier_coverage.py --date 2025-11-03
```

## Before You Start: Key Concepts

### 1. Current Date & Time

```bash
date -u
```

**CRITICAL**: Always check the current date first. Validation scripts run the day AFTER looking at the PREVIOUS day's data.

Example: If today is Nov 4, validation file `validation_20251103.json` is checking games from Nov 3.

### 2. Database vs Live Data

- **Production Database**: What the Lambda actually uses (contains games discovered by past fetch-odds runs)
- **Live NBA Schedule**: <https://www.nba.com/schedule> (source of truth for actual games)
- **Gap**: If fetch-odds hasn't run recently, production DB will be missing new games

**Default Connection**: Scripts connect to `DATABASE_URL` from `.env` (usually local or dev database)

**Production Database Access**:

⚠️ **CRITICAL SAFETY RULES**:

- **READ-ONLY OPERATIONS ONLY**: Only use production DB with diagnostic scripts (check_*.py)
- **NEVER run migrations against production**: Always use `LOCAL_DATABASE_URL` or `DEV_DATABASE_URL` for schema changes
- **NEVER run write operations**: No `uv run python -m cli fetch`, no manual data insertion

**To check production (read-only)**:

```bash
# Get production URL
grep PROD_DATABASE_URL .env

# Run read-only diagnostic scripts with production DB
DATABASE_URL="<prod_url>" uv run python scripts/check_games_in_database.py
DATABASE_URL="<prod_url>" uv run python scripts/check_tier_distribution.py
```

**Safe production queries**: Only the `scripts/check_*.py` diagnostic scripts are safe for production

## System Overview

- **Deployment**: AWS Lambda with EventBridge self-scheduling
- **Lambda Function**: `odds-scheduler` (single function, multi-job)
- **Log Group**: `/aws/lambda/odds-scheduler`
- **Region**: `eu-west-1`
- **Production DB**: Neon PostgreSQL (connection string in `.env` as `PROD_DATABASE_URL`)

## Three Scheduled Jobs

The Lambda function handles three jobs based on event payload:

1. **`fetch-odds`**: Fetches current odds for upcoming games
   - Rule: `odds-fetch-odds` (dynamic cron schedule)
   - Bootstrap: `odds-scheduler-fetch-odds-bootstrap` (rate: 1 day)

2. **`fetch-scores`**: Fetches final scores for completed games
   - Rule: `odds-fetch-scores` (dynamic cron schedule)
   - Bootstrap: `odds-scheduler-fetch-scores-bootstrap` (rate: 6 hours)

3. **`update-status`**: Updates event status (scheduled → live → final)
   - Rule: `odds-update-status` (dynamic cron schedule)
   - Bootstrap: `odds-scheduler-update-status-bootstrap` (rate: 1 hour)

## Intelligent Scheduling Logic

### Fetch Tiers (from `packages/odds-lambda/odds_lambda/fetch_tier.py`)

| Tier | Hours Before Game | Fetch Interval | Purpose |
|------|------------------|----------------|---------|
| `closing` | 0-3h | 30 minutes | Critical line movement |
| `pregame` | 3-12h | 3 hours | Active betting period |
| `sharp` | 12-24h | 12 hours | Professional betting |
| `early` | 24-72h (1-3 days) | 24 hours | Line establishment |
| `opening` | >72h (3+ days) | 48 hours | Initial line release |

### Scheduling Algorithm (from `packages/odds-lambda/odds_lambda/scheduling/intelligence.py`)

1. Find closest upcoming game in database
2. Calculate hours until game starts
3. Determine fetch tier based on hours
4. Schedule next execution at `now + tier.interval_hours`
5. Update EventBridge rule with calculated cron expression

**Key Files**:

- `packages/odds-lambda/odds_lambda/scheduling/intelligence.py` - Decision logic
- `packages/odds-lambda/odds_lambda/fetch_tier.py` - Tier definitions and intervals
- `packages/odds-lambda/odds_lambda/tier_utils.py` - Tier calculation functions
- `packages/odds-lambda/odds_lambda/jobs/fetch_odds.py` - Fetch odds job implementation

## Quick Diagnostic Commands

### 1. Check EventBridge Rules Status

```bash
# List all odds-related rules with their schedules
aws events list-rules --name-prefix odds- | jq -r '.Rules[] | "\(.Name) | \(.State) | \(.ScheduleExpression)"'
```

**Expected Output**:

- Dynamic rules show specific cron times (e.g., `cron(52 15 4 11 ? 2025)`)
- Bootstrap rules show rate expressions (e.g., `rate(1 day)`)

### 2. Check Recent Lambda Executions

```bash
# Show last 10 Lambda invocations with timestamps
aws logs describe-log-streams \
  --log-group-name /aws/lambda/odds-scheduler \
  --order-by LastEventTime \
  --descending \
  --limit 10 | jq -r '.logStreams[] | "\(.lastEventTimestamp | tonumber / 1000 | strftime("%Y-%m-%d %H:%M:%S UTC")) | \(.logStreamName)"'
```

### 3. Check Recent fetch-odds Activity

```bash
# Get recent fetch-odds related log entries (last 2 hours)
aws logs tail /aws/lambda/odds-scheduler --since 2h --format short | grep -i "fetch_odds\|tier" | tail -30
```

**Key Log Events to Look For**:

- `fetch_odds_job_started` - Job execution started
- `fetch_odds_executing` - Shows tier and reason (e.g., "Game in 7.5h: Team A vs Team B (pregame tier)")
- `odds_snapshot_stored` - Per-event tier assignment (check `fetch_tier` field)
- `fetch_odds_next_scheduled` - Next execution time and tier
- `eventbridge_rule_updated` - EventBridge rule was updated with new schedule

### 4. Verify Tier Assignment in Database

```bash
# Check recent tier distribution (last 7 days)
uv run python scripts/check_tier_distribution.py

# Check specific number of days back
uv run python scripts/check_tier_distribution.py 14
```

**Expected Tier Ranges** (hours_until_commence):

- closing: 0-3 hours before game
- pregame: 3-12 hours before game
- sharp: 12-24 hours before game
- early: 24-72 hours before game (1-3 days)
- opening: >72 hours before game (3+ days)

### 5. Check Upcoming Games in Database

**IMPORTANT**: This checks what games exist IN THE DATABASE, not the live NBA schedule.

```bash
# List next 10 scheduled games in database with tier info
uv run python scripts/check_games_in_database.py

# List next 20 games
uv run python scripts/check_games_in_database.py 20
```

**To check actual NBA games**: Visit <https://www.nba.com/schedule>

**Common Issue**: If database shows no games but NBA schedule shows games today, the fetch-odds job hasn't run recently and database is stale.

### 6. Check Tier Coverage for Specific Games

```bash
# Check tier coverage for specific event
uv run python scripts/check_game_tier_coverage.py --event <event_id>

# Check all games on specific date
uv run python scripts/check_game_tier_coverage.py --date 2025-11-04
```

## Common Issues

### Issue: Missing Tier Coverage

**Symptoms**: Validation shows games missing opening/early/sharp/pregame tiers

**Diagnosis**:

1. Check when `fetch-odds` last ran (command #2 above)
2. Check EventBridge rule schedule (command #1 above)
3. Calculate expected execution times based on tier intervals

**Causes**:

- EventBridge rule failed to trigger (AWS service issue)
- Lambda execution error prevented next scheduling
- Manual intervention disabled/modified rules

**Fix**:

- Wait for next scheduled execution
- Or manually invoke: Deploy latest code to trigger bootstrap rules

### Issue: All Events Tagged with Same Tier

**Symptoms**: Database shows all events with identical tier despite different hours_until_commence

**Diagnosis**: Check `packages/odds-lambda/odds_lambda/storage/writers.py` tier calculation logic

**Historical Bug**:

- Fixed in commit `5168a36` (Oct 30, 2025)
- Migration `9ff81f073e6b` recalculated historical tiers
- Before fix: Used closest game's tier for all events
- After fix: Calculate tier per-event from timestamps

### Issue: Lambda Not Running

**Symptoms**: No recent log entries in CloudWatch

**Diagnosis**:

1. Check EventBridge rule state: `aws events describe-rule --name odds-fetch-odds`
2. Check rule targets: `aws events list-targets-by-rule --rule odds-fetch-odds`
3. Check Lambda permissions: Rule must have invoke permission on Lambda

**Fix**: Redeploy via Terraform or GitHub Actions workflow

## Manual Testing

### Trigger fetch-odds Manually

```bash
# Via AWS CLI (use file payload to avoid shell escaping issues)
echo '{"job": "fetch-odds"}' > /tmp/payload.json
aws lambda invoke \
  --function-name odds-scheduler \
  --payload file:///tmp/payload.json \
  /tmp/response.json
cat /tmp/response.json
```

### Check Lambda Response

```bash
# Get most recent execution logs
aws logs filter-log-events \
  --log-group-name /aws/lambda/odds-scheduler \
  --limit 100 | jq -r '.events[-20:] | .[] | .message'
```

## Key Metrics to Track

1. **Execution Frequency**: Should match expected tier intervals
2. **Tier Distribution**: Each game should have snapshots across multiple tiers
3. **Scheduling Accuracy**: Next execution should be `now + tier.interval_hours`
4. **EventBridge Consistency**: Dynamic rules should update after each execution

## Related Files

- `packages/odds-lambda/odds_lambda/scheduling/intelligence.py` - Scheduling decision logic
- `packages/odds-lambda/odds_lambda/scheduling/backends/aws.py` - EventBridge integration
- `packages/odds-lambda/odds_lambda/jobs/fetch_odds.py` - Main fetch job
- `packages/odds-lambda/odds_lambda/storage/writers.py` - Tier assignment during storage
- `CLAUDE.md` - Full system architecture documentation
