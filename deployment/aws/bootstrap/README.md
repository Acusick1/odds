# Terraform Remote State Bootstrap

Creates S3 bucket and DynamoDB table for storing Terraform state.

## One-Time Setup

```bash
# 1. Create infrastructure
terraform apply

# 2. Migrate main terraform state
cd ../terraform
cp backend.tf.example backend.tf
terraform init
# Answer 'yes' to migrate existing state

# 3. Commit bootstrap state (safe - no secrets)
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
