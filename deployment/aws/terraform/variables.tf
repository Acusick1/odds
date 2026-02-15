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

variable "image_tag" {
  description = "Docker image tag to deploy (e.g., dev-a1b2c3d or dev-latest)"
  type        = string
  default     = "dev-latest"
}
