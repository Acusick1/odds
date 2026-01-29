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

- Self-scheduling via EventBridge rules
- Serverless with automatic scaling
- Minimal maintenance
- ~$0.20/month compute (within free tier)

### Configuration

```bash
SCHEDULER_BACKEND=aws
AWS_REGION=us-east-1
AWS_LAMBDA_ARN=arn:aws:lambda:us-east-1:123456789:function:odds-fetcher
```

### Key Constraints

- **15-minute max execution** - Design jobs for <5 minutes
- **NullPool required** - No connection reuse between invocations (automatic when `SCHEDULER_BACKEND=aws`)
- **Stateless** - No persistent state between invocations
- **Cold start** - ~1-2 seconds latency (acceptable)
- **EventBridge limit** - 300 rules per region

### Deployment

See `deployment/aws/` for Terraform configuration.

```bash
cd deployment/aws
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

| Tier | Cost | Requests |
|------|------|----------|
| Basic | $25/month | 20,000/month |

**Typical Usage:**
- ~10-15k requests/month during NBA season
- Intelligent scheduling optimizes usage
- Historical endpoint costs 10x (10 units per game)

### Compute

| Platform | Cost |
|----------|------|
| AWS Lambda | ~$0.20/month (free tier) |
| Railway | $5-10/month |

### Database

Managed PostgreSQL (Neon/Railway) costs vary. Neon offers generous free tier.

## Operational Notes

### Backups

Managed database providers (Neon/Railway) handle automated backups with point-in-time recovery.

### Recovery

- Data loss addressed via backfill
- API quota managed by intelligent scheduling
- Docker containers auto-restart on failure

### Security

- API keys in environment variables only
- SQLModel provides parameterized queries
- Regular database backups via provider

## Portability

Docker-based architecture allows deployment on any platform:
- DigitalOcean
- Azure
- GCP
- Self-hosted

## Quick Start Checklist

### AWS Lambda

1. Configure Terraform in `deployment/aws/`
2. Set `SCHEDULER_BACKEND=aws`
3. Set `AWS_REGION` and `AWS_LAMBDA_ARN`
4. Set `DATABASE_URL` to production database
5. Deploy: `terraform apply`

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
