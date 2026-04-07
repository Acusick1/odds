# OddsPortal scraper Lambda — separate container image with Playwright + Chromium.
# Gated by var.enable_oddsportal_scraper (default false).

# ECR repository for scraper image is created by build_and_push_scraper.sh,
# not managed by Terraform (same pattern as odds-scheduler ECR).

# Lambda function — scraper
resource "aws_lambda_function" "odds_scraper" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  function_name = "${var.project_name}-scraper"
  role          = aws_iam_role.lambda_exec.arn
  package_type  = "Image"
  image_uri     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/odds-scraper:${var.scraper_image_tag}"
  timeout       = 900
  memory_size   = 2048

  environment {
    variables = {
      SCHEDULER_BACKEND = "aws"
      DATABASE_URL      = var.database_url
      ODDS_API_KEY      = var.odds_api_key
      ODDS_API_KEYS     = var.odds_api_keys
      SSM_API_KEY_INDEX = "/${var.project_name}/active-api-key-index"
      LAMBDA_ARN        = "arn:aws:lambda:${var.aws_region}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}-scraper"
      RULE_PREFIX       = var.rule_prefix
      LOG_LEVEL         = "INFO"
    }
  }
}

# CloudWatch log group
resource "aws_cloudwatch_log_group" "scraper_logs" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  name              = "/aws/lambda/${aws_lambda_function.odds_scraper[0].function_name}"
  retention_in_days = 14
}

# Note: Static EventBridge Scheduler schedules (fetch_oddsportal, fetch_oddsportal_results)
# removed in favour of self-scheduling EventBridge Rules in eventbridge.tf.
# The scraper_scheduler_role IAM role is also removed (no longer needed).

# Outputs
output "scraper_function_arn" {
  description = "ARN of the scraper Lambda function"
  value       = var.enable_oddsportal_scraper ? aws_lambda_function.odds_scraper[0].arn : null
}

output "scraper_ecr_url" {
  description = "ECR repository URL for scraper image"
  value       = var.enable_oddsportal_scraper ? "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/odds-scraper" : null
}
