# Terraform Remote State Backend Configuration
#
# This enables:
# - Shared state between local machine and GitHub Actions
# - State locking (prevents concurrent modifications)
# - State versioning (old versions kept for 30 days)

terraform {
  # Backend configuration is supplied per-environment via `terraform init -backend-config=...`
  # so that development and production maintain isolated state files and locks.
  backend "s3" {}
}
