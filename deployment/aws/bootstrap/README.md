# Terraform Remote State Bootstrap

Creates S3 bucket and DynamoDB table for storing Terraform state.

## One-Time Setup

```bash
# 1. Create infrastructure (one time)
terraform apply

# 2. In the main terraform project create backend configs
cd ../terraform
cat <<'EOF' > backend.dev.hcl
bucket         = "odds-scheduler-terraform-state"
key            = "dev/terraform.tfstate"
region         = "eu-west-1"
encrypt        = true
dynamodb_table = "odds-scheduler-terraform-locks"
EOF

cat <<'EOF' > backend.prod.hcl
bucket         = "odds-scheduler-terraform-state"
key            = "prod/terraform.tfstate"
region         = "eu-west-1"
encrypt        = true
dynamodb_table = "odds-scheduler-terraform-locks"
EOF

# 3. Initialize each environment when needed (run one command at a time)
terraform init -backend-config=backend.dev.hcl
# or
terraform init -backend-config=backend.prod.hcl

# 4. Commit bootstrap state (safe - no secrets)
cd ../bootstrap
git add terraform.tfstate
git commit -m "Add remote state bootstrap"
```

## What This Creates

- **S3 Bucket** (`odds-scheduler-terraform-state`) - Stores state files
- **DynamoDB Table** (`odds-scheduler-terraform-locks`) - Prevents concurrent modifications

## Why Bootstrap?

Terraform needs somewhere to store state. We use Terraform to create the S3 bucket for state storage, but keep this specific state file local (and commit it). The bootstrap state only contains resource names, no secrets.

## Troubleshooting

**Error acquiring lock:**
```bash
terraform force-unlock <lock-id>
```

**Backend config changed:**
```bash
terraform init -reconfigure
```
