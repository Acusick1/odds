# Deployment Reference

> Reference documentation for deploying the odds pipeline.

## Deployment Options

The system supports three scheduler backends:

| Backend | Use Case | Cost |
|---------|----------|------|
| AWS Lambda | Production | ~$0.20/month |
| Railway | Alternative cloud | $5-10/month |
| Local | Development | Free |

## AWS Lambda (Primary Production)

### Overview

Two Lambda functions deployed in `eu-west-1`:

| Function | Image | Purpose | Memory | Timeout |
|----------|-------|---------|--------|---------|
| **Scheduler** (`odds-scheduler-dev`) | `Dockerfile.lambda` | Odds API fetching, scoring, digest, scheduling | 512 MB | 300s |
| **Scraper** (`odds-scheduler-dev-scraper`) | `Dockerfile.scraper` | OddsPortal headless browser scraping | 2048 MB | 600s |

Both use the same Lambda handler entry point and job registry, but different Docker images with different dependencies.

### EventBridge Rules

**Fixed-schedule rules** (managed by Terraform):

| Rule | Schedule | Lambda | Job |
|------|----------|--------|-----|
| `odds-fetch-oddsportal-results` | `cron(0 8 * * ? *)` | Scraper | `fetch-oddsportal-results` |
| `odds-backfill-polymarket` | `rate(3 days)` | Scheduler | `backfill-polymarket` (if enabled) |

`daily-digest` and `fetch-espn-fixtures` run locally via `LocalSchedulerBackend` cron (EPL only) — see `_JOB_CRON_MAP` in `odds_lambda/scheduling/jobs.py`. `score-predictions` no longer exists as a standalone job; it is invoked inline at the end of `fetch-oddsportal`. `fetch-odds` remains on EventBridge (self-scheduling).

**Self-scheduling rules** (pre-created disabled, activated by Lambda at runtime):

| Rule | Job | Scheduling |
|------|-----|------------|
| `odds-fetch-odds` | `fetch-odds` | Tier-based (30min–48h intervals) |
| `odds-fetch-scores` | `fetch-scores` | Game-activity based |
| `odds-update-status` | `update-status` | Dynamic |
| `odds-check-health` | `check-health` | Dynamic |
| `odds-fetch-polymarket` | `fetch-polymarket` | Fixed interval (if enabled) |

Terraform uses `ignore_changes = [schedule_expression, state]` on self-scheduling rules so Lambda can update them at runtime.

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
AWS_LAMBDA_ARN=arn:aws:lambda:eu-west-1:123456789:function:odds-scheduler-dev
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
| `project_name` | `odds-scheduler-dev` | Lambda function naming |
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
