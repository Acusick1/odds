# Deployment Reference

> Reference documentation for deploying the odds pipeline.

## Deployment Options

The system supports three scheduler backends:

| Backend | Use Case | Cost |
|---------|----------|------|
| AWS Lambda | Production | ~$0.20/month |
| Railway | Alternative cloud | $5-10/month |
| Local | Development | Free |

## Job Deployment Matrix

Where each job *can* run is determined by three things: the Terraform `enable_*` gates
(`eventbridge.tf`), the sports in `local.sport_configs` (`eventbridge.tf`), and the local
scheduler's `bootstrap_jobs` + `_JOB_CRON_MAP` (`config.py`, `jobs.py`). The table below
is the **design**, not a record of what is currently deployed.

**The repo is not a reliable record of the deployed state.** The committed tfvars
(`terraform.tfvars`, `terraform.prod.tfvars`) do not pin the `enable_*` gates, and the
live account has been applied with values that differ from the variable defaults. On top
of that, Terraform uses `ignore_changes = [schedule_expression, state]` on the
self-scheduling rules, so `terraform plan` never shows the real cadences. **Ground truth
for the live deployed schedule is the AWS account itself** — query it with:

```bash
odds scheduler list-jobs --backend aws   # needs AWS_REGION, AWS_LAMBDA_ARN, AWS_RULE_PREFIX + creds
```

| Job | AWS (when enabled) | Local | Schedule | Gated by |
|-----|--------------------|-------|----------|----------|
| `fetch-odds` | Scheduler Lambda (per sport in `sport_configs`) | — | Self-scheduling, tier-based (30 min–48 h) | `sport_configs` |
| `fetch-scores` | Scheduler Lambda (per sport) | — | Self-scheduling, game-activity | `sport_configs` |
| `update-status` | Scheduler Lambda (global) | — | Self-scheduling | always |
| `check-health` | Scheduler Lambda (global) | — | Self-scheduling | always |
| `fetch-polymarket` | Scheduler Lambda (global) | — | Self-scheduling | `enable_polymarket` |
| `backfill-polymarket` | Scheduler Lambda (global) | — | Fixed `rate(3 days)` | `enable_polymarket` |
| `fetch-betfair-exchange` | Scheduler Lambda (per sport) | Cron `*/30 * * * *` | Self-scheduling (AWS) / cron floor (local) | `betfair_enabled` (AWS) |
| `fetch-oddsportal` | Scraper Lambda (per scraper sport) | Bootstrap, self-scheduling | Self-scheduling, hourly | `enable_oddsportal_scraper` (AWS) |
| `fetch-oddsportal-results` | Scraper Lambda (per scraper sport) | Bootstrap, self-scheduling | Self-scheduling, daily ~08:00 | `enable_oddsportal_scraper` (AWS) |
| `score-predictions` | — (inline at end of `fetch-oddsportal`) | — (inline) | n/a | n/a |
| `agent-run` | — | Bootstrap, dynamic | Fixture-proximity tiers | local only |
| `daily-digest` | — | Cron `0 8 * * *` (EPL) | Cron | local only |
| `fetch-espn-fixtures` | — | Cron `0 6 * * *` (EPL) | Cron | local only |
| `fetch-mlb-probables` | — | Cron `0 6 * * *` (MLB) | Cron | local only |

**Always local-only** (no EventBridge rule is ever created for these — they are absent
from both `scheduler_rules_map` and `scraper_rules_map`): `agent-run`, `daily-digest`,
`fetch-espn-fixtures`, `fetch-mlb-probables`, and `score-predictions` (which runs inline).
The agent in particular is local-by-design and in interactive evaluation, not autonomous
production — see [BETTING_AGENT.md](BETTING_AGENT.md).

Everything else (`fetch-odds`, `fetch-scores`, `update-status`, `check-health`,
`fetch-oddsportal[-results]`, `fetch-betfair-exchange`, `fetch-polymarket`) *may* be on AWS,
on the local scheduler, or both, depending on how the account was applied and which jobs
are in the local `bootstrap_jobs`. Run the `list-jobs --backend aws` command above to see
which are actually enabled, and watch for jobs running in **both** places (e.g. the scraper
can be enabled on AWS while also bootstrapped locally) and for stale rules whose next-run
time is in the past (enabled-but-dead leftovers from an earlier apply).

## AWS Lambda (Primary Production)

### Overview

Two Lambda functions deployed in `eu-west-1`:

| Function | Image | Purpose | Memory | Timeout |
|----------|-------|---------|--------|---------|
| **Scheduler** (`odds-scheduler`) | `Dockerfile.lambda` | Odds API fetching, scoring, digest, scheduling | 512 MB | 300s |
| **Scraper** (`odds-scheduler-scraper`) | `Dockerfile.scraper` | OddsPortal headless browser scraping | 2048 MB | 600s |

Both use the same Lambda handler entry point and job registry, but different Docker images with different dependencies.

### EventBridge Rules

See the [Job Deployment Matrix](#job-deployment-matrix) above for the full job/backend
mapping. This section covers the EventBridge mechanics.

**Fixed-schedule rules** (managed by Terraform): `local.fixed_schedule_expressions` is
currently empty — no Lambda-hosted fixed-cron jobs remain. The only fixed rule is
`odds-backfill-polymarket` (`rate(3 days)`), created when `enable_polymarket = true`.
`daily-digest` and `fetch-espn-fixtures` moved to `LocalSchedulerBackend` cron (EPL only)
— see `_JOB_CRON_MAP` in `odds_lambda/scheduling/jobs.py`. `score-predictions` is no longer
a standalone job; it is invoked inline at the end of `fetch-oddsportal`.

**Self-scheduling rules** (pre-created `DISABLED` with a `rate(1 day)` placeholder,
activated and re-scheduled by the Lambda at runtime). One rule per entry in
`local.scheduler_rules_map` / `scraper_rules_map`, e.g. `odds-fetch-odds-epl`,
`odds-fetch-scores-epl`, `odds-update-status`, `odds-check-health`. Scraper rules
(`odds-fetch-oddsportal-epl`, `odds-fetch-oddsportal-results-epl`) are only created when
`enable_oddsportal_scraper = true`.

Terraform uses `ignore_changes = [schedule_expression, state]` on self-scheduling rules so
the Lambda can update them at runtime. Because of this, **the live schedule is not visible
in Terraform state** — `terraform plan` will not show the real cadences. Use
`odds scheduler list-jobs --backend aws` (which lists rules with their raw
`ScheduleExpression` and state) to see what is actually deployed.

### API Key Rotation

Multiple Odds API keys are supported via `ODDS_API_KEYS` (comma-separated). The active key index is tracked in SSM Parameter Store (`/odds/active-api-key-index`), allowing rotation across Lambda invocations without redeployment.

### S3 Model Store

- **Bucket**: `odds-models-{account_id}` (versioned, AES256 encryption, public access blocked)
- **Path**: `{model_name}/latest/model.pkl` + `config.yaml` + `metadata.json`
- **Publishing**: `odds model publish` CLI command uploads to both `latest/` and `{version}/`
- **Loading**: Score-predictions Lambda uses ETag-based caching — only re-downloads when model changes

### Configuration

```bash
SCHEDULER_BACKEND=aws
AWS_REGION=eu-west-1
AWS_LAMBDA_ARN=arn:aws:lambda:eu-west-1:123456789:function:odds-scheduler
AWS_RULE_PREFIX=odds
```

### Key Constraints

- **15-minute max execution** - Design jobs for <5 minutes (scraper: <10 minutes)
- **NullPool required** - No connection reuse between invocations (automatic when `SCHEDULER_BACKEND=aws`)
- **Stateless** - No persistent state between invocations (SSM used for key rotation state)
- **Cold start** - ~1-2 seconds latency (acceptable)
- **EventBridge limit** - 300 rules per region

### Terraform Structure

```
deployment/aws/
├── terraform/              # Main environment infrastructure
│   ├── main.tf             # Provider, backend, default tags
│   ├── variables.tf        # All input variables
│   ├── lambda.tf           # Scheduler Lambda, IAM, CloudWatch
│   ├── scraper.tf          # Scraper Lambda, IAM, CloudWatch, EventBridge
│   ├── eventbridge.tf      # Scheduler EventBridge rules
│   └── outputs.tf
├── bootstrap/              # Shared resources (deploy once)
│   ├── s3_models.tf        # S3 model bucket
│   └── ...
├── Dockerfile.lambda       # Scheduler image (includes xgboost, odds-analytics)
├── Dockerfile.scraper      # Scraper image (includes Playwright, Chromium)
├── build_and_push.sh       # Build + push scheduler image to ECR
└── build_and_push_scraper.sh  # Build + push scraper image to ECR
```

### Key Terraform Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `aws_region` | `eu-west-1` | Deployment region |
| `environment` | `development` | Resource tagging |
| `project_name` | `odds-scheduler-dev` | Lambda function naming (prod tfvars override to `odds-scheduler`) |
| `enable_oddsportal_scraper` | `false` | Deploy scraper Lambda |
| `enable_polymarket` | `false` | Deploy Polymarket rules |
| `odds_api_keys` | `""` | Comma-separated rotation keys |
| `discord_webhook_url` | `""` | Discord alerts/digest webhook |
| `model_name` | `epl-clv-home` | S3 model key prefix |
| `model_bucket_name` | `odds-models-{account_id}` | S3 bucket name |

### Deployment

```bash
# Build and push images
cd deployment/aws
./build_and_push.sh dev eu-west-1 <account-id>
./build_and_push_scraper.sh dev eu-west-1 <account-id>

# Deploy infrastructure
cd terraform
terraform init
terraform plan
terraform apply
```

## Railway (Alternative Cloud)

### Overview

- Continuous deployment from GitHub
- APScheduler with persistent scheduling
- Managed database option
- Auto-restarts on failure

### Configuration

```bash
SCHEDULER_BACKEND=railway
```

### Features

- Simple setup
- Managed PostgreSQL available
- Auto-deploys on git push
- Docker-based

### Cost

- $5-10/month (app + managed PostgreSQL if used)

## Local (Development)

### Overview

- APScheduler with long-running process
- Easy debugging
- No cloud dependencies

### Configuration

```bash
SCHEDULER_BACKEND=local
```

### Usage

```bash
# Start scheduler (runs until Ctrl+C)
uv run odds scheduler start

# Manual fetch (without scheduler)
uv run odds fetch current
```

### Setup

```bash
# Start PostgreSQL
docker-compose up -d

# Verify connection
uv run odds status show
```

## Expected Costs

### The Odds API

| Tier | Cost | Requests/key/month | Notes |
|------|------|--------------------|-------|
| Free | $0/month | 500 | No historical endpoints |
| Basic | $25/month | 20,000 | Historical endpoints available |

Multiple keys can be rotated to multiply effective quota.

### Compute

| Platform | Cost |
|----------|------|
| AWS Lambda (scheduler) | ~$0.20/month (free tier) |
| AWS Lambda (scraper) | ~$0.50/month (2GB memory, hourly) |
| Railway | $5-10/month |

### Storage

| Service | Cost |
|---------|------|
| S3 (models) | < $0.01/month |
| CloudWatch Logs | < $0.10/month (14-day retention) |
| SSM Parameter Store | Free (standard tier) |

### Database

Managed PostgreSQL (Neon/Railway) costs vary. Neon offers generous free tier.

## Operational Notes

### Backups

Managed database providers (Neon/Railway) handle automated backups with point-in-time recovery.

### Recovery

- Data loss addressed via backfill
- API quota managed by intelligent scheduling + key rotation
- Docker containers auto-restart on failure

### Security

- API keys via environment variables and SSM Parameter Store
- SQLModel provides parameterized queries
- S3 bucket: versioned, encrypted, no public access
- Regular database backups via provider

### Discord Alerts

Configured via `DISCORD_WEBHOOK_URL` environment variable. Used by:
- Daily digest (predictions + post-match results)
- Quota warnings (when remaining < 20%)
- Consecutive failure alerts

## Portability

Docker-based architecture allows deployment on any platform:
- DigitalOcean
- Azure
- GCP
- Self-hosted

## Quick Start Checklist

### AWS Lambda

1. Bootstrap shared resources: `cd deployment/aws/bootstrap && terraform apply`
2. Build images: `./build_and_push.sh` and `./build_and_push_scraper.sh`
3. Configure `terraform/terraform.tfvars` with `database_url`, `odds_api_key`, `discord_webhook_url`
4. Set `enable_oddsportal_scraper = true` for EPL scraping
5. Deploy: `cd terraform && terraform apply`
6. Publish model: `uv run odds model publish --name epl-clv-home --path <model_dir>`

### Railway

1. Connect GitHub repository
2. Set environment variables in Railway dashboard
3. Set `SCHEDULER_BACKEND=railway`
4. Deploy (automatic on push)

### Local

1. Start database: `docker-compose up -d`
2. Set `DATABASE_URL=${LOCAL_DATABASE_URL}`
3. Set `SCHEDULER_BACKEND=local`
4. Run: `uv run odds scheduler start`
