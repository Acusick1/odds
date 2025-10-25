variable "aws_region" {
  description = "AWS region for state storage resources"
  type        = string
  default     = "eu-west-1"
}

variable "state_bucket_name" {
  description = "Name of the S3 bucket for Terraform state storage"
  type        = string
  default     = "odds-scheduler-terraform-state"
}

variable "lock_table_name" {
  description = "Name of the DynamoDB table for state locking"
  type        = string
  default     = "odds-scheduler-terraform-locks"
}
