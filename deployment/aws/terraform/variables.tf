variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-1"
}

variable "environment" {
  description = "Environment name (development, production)"
  type        = string
  default     = "development"
}

variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 300
}

variable "lambda_memory_size" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 512
}

variable "database_url" {
  description = "PostgreSQL connection URL"
  type        = string
  sensitive   = true
}

variable "odds_api_key" {
  description = "The Odds API key"
  type        = string
  sensitive   = true
}

variable "project_name" {
  description = "Project name for resource naming (Lambda, IAM, ECR)"
  type        = string
  default     = "odds-scheduler-dev"
}

variable "rule_prefix" {
  description = "Prefix for EventBridge rule names (decoupled from project_name)"
  type        = string
  default     = "odds"
}

variable "enable_polymarket" {
  description = "Deploy Polymarket EventBridge rules (fetch + backfill)"
  type        = bool
  default     = false
}

variable "image_tag" {
  description = "Docker image tag to deploy (e.g., dev-a1b2c3d or dev-latest)"
  type        = string
  default     = "dev-latest"
}

variable "enable_oddsportal_scraper" {
  description = "Deploy OddsPortal scraper Lambda (separate container with Playwright/Chromium)"
  type        = bool
  default     = false
}

variable "scraper_image_tag" {
  description = "Docker image tag for the scraper Lambda (e.g., dev-a1b2c3d or dev-latest)"
  type        = string
  default     = "dev-latest"
}

variable "model_bucket_name" {
  description = "S3 bucket name for trained model artifacts"
  type        = string
  default     = "odds-models-685946576110"
}

variable "model_name" {
  description = "Model name for CLV predictions (S3 key prefix)"
  type        = string
  default     = "epl-clv-home"
}

variable "discord_webhook_url" {
  description = "Discord webhook URL for alerts and digests"
  type        = string
  sensitive   = true
  default     = ""
}
