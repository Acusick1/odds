# Terraform Remote State Backend Configuration
#
# This enables:
# - Shared state between local machine and GitHub Actions
# - State locking (prevents concurrent modifications)
# - State versioning (old versions kept for 30 days)

terraform {
  backend "s3" {
    bucket         = "odds-scheduler-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "eu-west-1"
    encrypt        = true
    dynamodb_table = "odds-scheduler-terraform-locks"
  }
}
