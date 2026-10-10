# Deployment Reference

> Reference documentation for deploying the odds pipeline.

## Deployment targets

Jobs run in two places:

| Target | What runs there |
|--------|-----------------|
| AWS Lambda | Two functions driven by EventBridge rules: the scheduler Lambda and the scraper Lambda |
| Local scheduler | `odds scheduler start` (APScheduler), on a development machine or an always-on host under systemd |

The scheduler backend is selected with `SCHEDULER_BACKEND` (`aws`, `local` or `railway`).
The `railway` backend is a no-op shim, not a deployment target (see
[ARCHITECTURE.md](ARCHITECTURE.md#railway-backend)).

## Job Deployment Matrix

Where each job *can* run is determined by three things: the Terraform `enable_*` and
`betfair_enabled` gates and the sports in `local.sport_configs` (both in
`eventbridge.tf`), and the local scheduler's `scheduler.bootstrap_jobs`
(`packages/odds-core/odds_core/config.py`) plus `_JOB_CRON_MAP`
(`packages/odds-lambda/odds_lambda/scheduling/jobs.py`). The table below is the
**design**, not a record of what is deployed.

**The repo is not a record of the deployed state.** Only `terraform.tfvars.example` is
committed; the tfvars the live account was applied with are not. Terraform also uses
`ignore_changes = [schedule_expression, state]` on the self-scheduling rules, so
`terraform plan` never shows the real cadences. **Ground truth for the live schedule is
the AWS account itself.** Query it with:

```bash
odds scheduler list-jobs --backend aws   # needs AWS_REGION, AWS_LAMBDA_ARN, AWS_RULE_PREFIX + creds
```

| Job | AWS (when enabled) | Local (default `bootstrap_jobs`) | Schedule | AWS gate |
|-----|--------------------|----------------------------------|----------|----------|
| `fetch-odds` | Scheduler Lambda, per sport in `sport_configs` | — | Self-scheduling, fetch tier (30 min–48 h) | `sport_configs` |
| `fetch-scores` | Scheduler Lambda, per sport | — | Self-scheduling, game activity | `sport_configs` |
| `update-status` | Scheduler Lambda, global | — | Self-scheduling | always |
| `check-health` | Scheduler Lambda, global | — | Self-scheduling | always |
| `fetch-polymarket` | Scheduler Lambda, global | — | Self-scheduling | `enable_polymarket` |
| `backfill-polymarket` | Scheduler Lambda, global | — | Fixed `rate(3 days)` | `enable_polymarket` |
| `fetch-betfair-exchange` | Scheduler Lambda, per sport | Bootstrapped, cron `*/30 * * * *` | Self-scheduling (AWS) / cron floor (local) | `betfair_enabled` |
| `fetch-oddsportal` | Scraper Lambda, per scraper sport | Bootstrapped | Self-scheduling by kickoff proximity (30 min–2 h) | `enable_oddsportal_scraper` |
| `fetch-oddsportal-results` | Scraper Lambda, per scraper sport | Bootstrapped | Self-scheduling, daily ~08:00 UTC | `enable_oddsportal_scraper` |
| `score-predictions` | — (inline at end of `fetch-oddsportal`) | — (inline) | n/a | n/a |
| `fetch-espn-fixtures` | — | Bootstrapped, cron `0 6 * * *` (EPL) | Cron | local only |
| `fetch-mlb-probables` | — | Bootstrapped, cron `0 6 * * *` (MLB) | Cron | local only |
| `daily-digest` | — | Opt-in, cron `0 8 * * *` (EPL) | Cron | local only |
| `agent-run` | — | Opt-in | Fixture proximity | local only |

`daily-digest` and `agent-run` are commented out of the default `bootstrap_jobs`; they
run only when `SCHEDULER_BOOTSTRAP_JOBS` lists them.

**Always local-only** (no EventBridge rule is ever created for these; they are absent
from both `scheduler_rules_map` and `scraper_rules_map`): `agent-run`, `daily-digest`,
`fetch-espn-fixtures`, `fetch-mlb-probables`, and `score-predictions` (which runs inline).
The agent is in interactive evaluation, not autonomous production; see
[BETTING_AGENT.md](BETTING_AGENT.md).

Every other job *may* run on AWS, on the local scheduler, or both, depending on how the
account was applied and what `bootstrap_jobs` contains. Run the `list-jobs --backend aws`
command above to see which are enabled. Watch for jobs running in **both** places (for
example the scraper enabled on AWS while also bootstrapped locally) and for stale rules
whose next-run time is in the past (enabled, but nothing reschedules them).

## AWS Lambda

### Functions

Two Lambda functions in `eu-west-1`:

| Function | Image | Jobs | Memory | Timeout |
|----------|-------|------|--------|---------|
| **Scheduler** (`odds-scheduler`) | `Dockerfile.lambda` | Odds API fetching, scores, status updates, health checks, Polymarket, Betfair Exchange | 512 MB | 300 s |
| **Scraper** (`odds-scheduler-scraper`) | `Dockerfile.scraper` | OddsPortal headless browser scraping, with CLV scoring inline | 2048 MB | 900 s |

Both use the same Lambda handler and job registry, built into different Docker images
with different dependencies. Function names come from `project_name` (the scraper adds
`-scraper`).

### EventBridge rules

**Fixed-schedule rules** (managed by Terraform): `local.fixed_schedule_expressions` is
empty, so no Lambda job runs on a fixed cron. The one fixed rule is
`odds-backfill-polymarket` (`rate(3 days)`), created when `enable_polymarket = true`.

**Self-scheduling rules** are pre-created `DISABLED` with a `rate(1 day)` placeholder; the
Lambda activates and reschedules them at runtime. There is one rule per entry in
`local.scheduler_rules_map` / `scraper_rules_map`, named `{rule_prefix}-{job}[-{sport}]`,
for example `odds-fetch-odds-epl`, `odds-fetch-scores-epl`, `odds-update-status`,
`odds-check-health`. Scraper rules (`odds-fetch-oddsportal-epl`,
`odds-fetch-oddsportal-results-epl`) exist only when `enable_oddsportal_scraper = true`.

Because Terraform ignores `schedule_expression` and `state` on these rules, the live
schedule is not visible in Terraform state. `odds scheduler list-jobs --backend aws` lists
the rules with their raw `ScheduleExpression` and state.

### Configuration

Inside Lambda, Terraform sets `DATABASE_URL`, `LAMBDA_ARN` and `RULE_PREFIX`, and the
runtime sets `AWS_REGION`; the AWS backend reads these unprefixed names. A local process
that talks to the AWS backend (such as `odds scheduler list-jobs --backend aws`) reads the
`AWSConfig` settings instead:

```bash
SCHEDULER_BACKEND=aws
AWS_REGION=eu-west-1
AWS_LAMBDA_ARN=arn:aws:lambda:eu-west-1:123456789:function:odds-scheduler
AWS_RULE_PREFIX=odds
```

### API key rotation

Multiple Odds API keys are supported via `ODDS_API_KEYS` (comma-separated). The active key
index is tracked in SSM Parameter Store at `/{project_name}/active-api-key-index`, so
rotation persists across invocations without redeployment.

### S3 model store

- **Bucket**: `model_bucket_name` (versioned, AES256 encryption, public access blocked),
  created by `deployment/aws/bootstrap/s3_models.tf`
- **Path**: `{model_name}/latest/` and `{model_name}/{version}/`, each holding
  `model.pkl`, `config.yaml` and `metadata.json`
- **Publishing**: `odds model publish` uploads to both paths (see
  [CLI.md](CLI.md#publish-model-to-s3))
- **Loading**: `packages/odds-lambda/odds_lambda/model_loader.py`, called by inline
  scoring in `fetch-oddsportal`, sends a HEAD request per invocation and re-downloads only
  when the ETag changes

### Constraints

- **Timeouts**: 300 s for the scheduler and 900 s for the scraper (Lambda's maximum is
  15 minutes)
- **NullPool**: no connection reuse between invocations; applied automatically when the
  `AWS_LAMBDA_FUNCTION_NAME` env var is set, regardless of scheduler backend
- **Stateless**: no state persists between invocations except the key-rotation index in SSM
- **EventBridge**: 300 rules per region

### Terraform layout

```
deployment/aws/
├── terraform/                  # Main environment infrastructure
│   ├── main.tf                 # Provider, default tags
│   ├── backend.tf              # S3 remote state (with backend.dev.hcl / backend.prod.hcl)
│   ├── variables.tf            # All input variables
│   ├── lambda.tf               # Scheduler Lambda, IAM, CloudWatch
│   ├── scraper.tf              # Scraper Lambda, IAM, CloudWatch
│   ├── eventbridge.tf          # EventBridge rules for both Lambdas
│   └── terraform.tfvars.example
├── bootstrap/                  # Shared resources (deploy once): state bucket, ECR, S3 model bucket
├── Dockerfile.lambda           # Scheduler image (includes xgboost, odds-analytics)
├── Dockerfile.scraper          # Scraper image (includes Playwright, Chromium)
├── build_and_push.sh           # Build + push scheduler image to ECR
└── build_and_push_scraper.sh   # Build + push scraper image to ECR
```

Remote-state setup is described in `deployment/aws/bootstrap/README.md`.

### Terraform variables

`deployment/aws/terraform/variables.tf` is authoritative. The main ones:

| Variable | Default | Purpose |
|----------|---------|---------|
| `aws_region` | `eu-west-1` | Deployment region |
| `environment` | `development` | Resource tagging |
| `project_name` | `odds-scheduler-dev` | Lambda, IAM, ECR and SSM naming (production uses `odds-scheduler`) |
| `rule_prefix` | `odds` | EventBridge rule name prefix |
| `lambda_timeout` / `lambda_memory_size` | `300` / `512` | Scheduler Lambda limits |
| `database_url` | required | PostgreSQL connection URL |
| `odds_api_key` / `odds_api_keys` | required / `""` | Single key, or comma-separated rotation keys that override it |
| `enable_oddsportal_scraper` | `false` | Deploy the scraper Lambda and its rules |
| `enable_polymarket` | `false` | Deploy Polymarket rules |
| `betfair_enabled` | `false` | Deploy Betfair Exchange ingestion rules (credentials in `betfair_*` variables) |
| `discord_webhook_url` | `""` | Discord alerts/digest webhook |
| `model_name` | `epl-clv-home` | S3 model key prefix |
| `model_bucket_name` | account-specific | S3 model bucket |

### Deploying

GitHub Actions deploys both environments:

- `.github/workflows/ci-cd.yml` runs tests on every push to `main` that touches code,
  migrations or deployment files, then calls `deploy-dev.yml`, which deploys an ephemeral
  dev stack, runs migrations and smoke checks against it, and destroys it.
- `.github/workflows/deploy-prod.yml` is triggered manually and requires approval on the
  `production` environment.

To deploy by hand:

```bash
# Build and push images
cd deployment/aws
./build_and_push.sh dev eu-west-1 <account-id>
./build_and_push_scraper.sh dev eu-west-1 <account-id>

# Deploy infrastructure
cd terraform
terraform init -backend-config=backend.dev.hcl
terraform plan
terraform apply
```

For a first deployment, apply `deployment/aws/bootstrap/` once before the main
environment, set `database_url`, `odds_api_key` and any `enable_*` gates in your tfvars,
then publish a model with `uv run odds model publish --name epl-clv-home --path <model.pkl>`.

## Local scheduler

Start PostgreSQL and run the scheduler as described in the
[README Quick Start](../README.md#quick-start). The scheduler starts the jobs in
`scheduler.bootstrap_jobs` and runs until Ctrl+C. `SCHEDULER_BACKEND=local` is the default.

### Always-on host (systemd + deploy.sh)

The local scheduler can run as an always-on service on a mini-PC, supervised by systemd
and redeployed with a script that proves the changed jobs run before it restarts the
process.

#### systemd unit

`odds-scheduler.service` (repo root) runs `odds scheduler start` under systemd with
`Restart=always`, so the scheduler comes back after a crash or host reboot. Edit `User`,
`WorkingDirectory`, `EnvironmentFile`, and the absolute `uv` path in `ExecStart` to match
the host, then install it:

```bash
sudo cp odds-scheduler.service /etc/systemd/system/odds-scheduler.service
sudo systemctl daemon-reload
sudo systemctl enable --now odds-scheduler   # start now + on every boot

systemctl status odds-scheduler              # check state
journalctl -u odds-scheduler -f              # follow logs
```

#### Smoke check

`odds scheduler smoke` runs each bootstrapped job once (from `scheduler.bootstrap_jobs`,
expanded per sport, the same source `start` uses) and prints a pass/fail table. It exits
non-zero (exit code = number of failed jobs) if any job fails.

- `agent-run` is skipped by default so a deploy check never places paper bets; pass
  `--include-agent` to run it.
- `daily-digest` posts to Discord; pass `--no-side-effects` to drop it.
- `--only`/`--exclude` accept base or compound job names (repeatable).
- Run with `SCHEDULER_DRY_RUN=true` so self-scheduling is a no-op and the live schedule
  store is left untouched.

Smoke always exercises each job's import, config, decision and schema path, but the full
fetch-and-ingest body only runs when the job's cadence gate (`decision.should_execute`)
says to execute.

#### deploy.sh

`deploy.sh` (repo root) is the redeploy entry point:

```bash
./deploy.sh
```

It runs `git pull` → `uv sync` → `alembic upgrade head` →
`SCHEDULER_DRY_RUN=true odds scheduler smoke` → `sudo systemctl restart odds-scheduler` →
`systemctl status` + `odds scheduler list-jobs`. If migration or smoke fails it aborts
**before** the restart, so the running scheduler keeps serving the old code until the new
code passes smoke.

`alembic upgrade head` runs before the restart, so the still-running old scheduler briefly
sees the new schema. This is safe for additive migrations; for a destructive migration,
stop the service first.

## Expected costs

### The Odds API

| Tier | Cost | Requests/key/month | Notes |
|------|------|--------------------|-------|
| Free | $0/month | 500 | No historical endpoints |
| Basic | $25/month | 20,000 | Historical endpoints available |

Rotating multiple keys multiplies the effective quota.

### AWS

| Service | Cost |
|---------|------|
| Lambda (scheduler) | ~$0.20/month (free tier) |
| S3 (models) | < $0.01/month |
| CloudWatch Logs | < $0.10/month (14-day retention) |
| SSM Parameter Store | Free (standard tier) |

Managed PostgreSQL (Neon) costs vary; Neon has a free tier.

## Operations

- **Backups**: the managed database provider handles automated backups with point-in-time
  recovery.
- **Recovery**: gaps in collected data are filled with backfill commands (see
  [CLI.md](CLI.md#backfill-commands)).
- **Secrets**: API keys come from environment variables and Terraform variables; the
  key-rotation index lives in SSM.
- **Alerts**: Discord alerts are configured with `DISCORD_WEBHOOK_URL`; see
  [REPORTING.md](REPORTING.md) for what is sent.
