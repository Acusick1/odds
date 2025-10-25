# GitHub Actions Workflows

## Overview

This directory contains CI/CD workflows for the betting odds pipeline:

- **[test.yml](test.yml)** - Run tests and linting
- **[deploy-dev.yml](deploy-dev.yml)** - Deploy to development environment (ephemeral, tears down after testing)
- **[deploy-prod.yml](deploy-prod.yml)** - Deploy to production environment (persistent infrastructure)
- **[ci-cd.yml](ci-cd.yml)** - Orchestrates test → deploy-dev pipeline on pushes to master

## Production Deployment Setup

The production workflow requires a one-time setup in GitHub:

### 1. Create Production Environment

Go to: **Repository Settings → Environments → New Environment**

- Name: `production`
- Click "Configure environment"

### 2. Configure Protection Rules

Under the `production` environment settings:

- ✅ **Required reviewers**: Add yourself (or team members)
  - This forces manual approval before any production deployment
- Optional: **Wait timer** (e.g., 5 minutes) to prevent accidental immediate deploys
- Optional: **Deployment branches**: Limit to `master` branch only

### 3. Add Production Secrets

Under the `production` environment, add these secrets:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `DATABASE_URL` | Neon production database connection string | `postgresql+asyncpg://user:pass@host/db?ssl=require` |
| `ODDS_API_KEY` | The Odds API key | `687a45ba37e41db82d192542ea6d678b` |
| `AWS_ACCESS_KEY_ID` | AWS IAM user access key | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM user secret key | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |

**Important**: These should be PRODUCTION credentials, separate from development.

### 4. Get Production Database URL

From Neon (production branch):

```bash
# Get production connection string
neonctl connection-string main --project-id <your-project-id>

# Format should be:
# postgresql+asyncpg://neondb_owner:PASSWORD@HOST/neondb?ssl=require
```

### 5. Deploy to Production

Once setup is complete:

#### Option A: Manual Deploy (Primary)
1. Go to: **Actions → Deploy to Production**
2. Click **"Run workflow"** button
3. Select branch: `master`
4. Click **"Run workflow"**
5. Workflow will pause and request approval
6. Review the deployment plan
7. Click **"Review deployments"** → **"Approve and deploy"**

#### Option B: Release Deploy (Versioned)
1. Create and push a git tag:
   ```bash
   git tag -a v1.0.0 -m "Production release v1.0.0"
   git push origin v1.0.0
   ```
2. Go to **Releases → Draft a new release**
3. Select your tag: `v1.0.0`
4. Write release notes
5. Click **"Publish release"**
6. Workflow triggers automatically, but still requires approval

## Workflow Differences

### Development Workflow
- **Trigger**: Automatically on push to master (via ci-cd.yml)
- **Environment**: `development` (ephemeral)
- **Config**: Uses `terraform.tfvars.example` with dev overrides
- **Project Name**: `odds-scheduler-dev`
- **Teardown**: ✅ Infrastructure destroyed after tests complete
- **Approval**: ❌ No approval required

### Production Workflow
- **Trigger**: Manual (`workflow_dispatch`) OR Release published
- **Environment**: `production` (persistent)
- **Config**: Uses `terraform.tfvars.example` with production overrides
- **Project Name**: `odds-scheduler` (no -dev suffix)
- **Teardown**: ❌ Infrastructure stays running
- **Approval**: ✅ Requires manual approval via environment protection

## Terraform Variable Strategy

Both workflows use the same base file (`terraform.tfvars.example`) but override key variables:

### Development
```bash
terraform apply \
  -var-file="terraform.tfvars.example" \
  -var="database_url=$DEV_DATABASE_URL" \
  -var="odds_api_key=$ODDS_API_KEY" \
  -auto-approve
```

### Production
```bash
terraform apply \
  -var-file="terraform.tfvars.example" \
  -var="environment=production" \
  -var="project_name=odds-scheduler" \
  -var="database_url=$PROD_DATABASE_URL" \
  -var="odds_api_key=$ODDS_API_KEY"
```

This keeps secrets out of git while allowing environment-specific configurations.

## Security Notes

- ⚠️ **Never commit** `terraform.dev.tfvars` or `terraform.prod.tfvars` (they're in .gitignore)
- ✅ **Only commit** `terraform.tfvars.example` with placeholder values
- 🔒 **Store secrets** in GitHub environment secrets, not in code
- 🔐 **Use separate credentials** for dev and production environments
- ✅ **Require approval** for all production deployments via environment protection

## Troubleshooting

### "Context access might be invalid" warnings in IDE
This is expected - the secrets are defined in GitHub environment settings, not in the workflow file.

### Deployment stuck at "Waiting for approval"
1. Check email for approval notification from GitHub
2. Go to Actions → Click the workflow run
3. Click "Review deployments" → Select `production` → Approve

### Terraform plan shows unexpected changes
1. Review the plan carefully in the workflow logs
2. If unexpected, **reject the deployment**
3. Investigate differences between current state and desired state

### Database migrations fail
1. Check that `DATABASE_URL` secret points to correct production database
2. Ensure database is accessible from GitHub Actions runners
3. Verify migrations are tested in dev environment first

## Monitoring Production Deployments

After deployment completes:

```bash
# Check Lambda function status
aws lambda list-functions --region eu-west-1 --query 'Functions[?contains(FunctionName, `odds-scheduler`)]'

# Check EventBridge rules
aws events list-rules --region eu-west-1 --name-prefix odds-scheduler

# View recent Lambda logs
aws logs tail /aws/lambda/odds-scheduler-fetch-odds --follow

# Check database has new data
# (Run from local machine with production DATABASE_URL)
uv run python -m cli.main status show
```
