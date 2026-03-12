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
  timeout       = 600
  memory_size   = 2048

  environment {
    variables = {
      SCHEDULER_BACKEND = "aws"
      DATABASE_URL      = var.database_url
      ODDS_API_KEY      = var.odds_api_key
      ODDS_API_KEYS     = var.odds_api_keys
      SSM_API_KEY_INDEX = "/${var.project_name}/active-api-key-index"
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

# IAM role for EventBridge Scheduler to invoke the scraper Lambda
resource "aws_iam_role" "scraper_scheduler_role" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  name = "${var.project_name}-scraper-scheduler-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "scheduler.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "scraper_scheduler_invoke" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  name = "${var.project_name}-scraper-scheduler-invoke"
  role = aws_iam_role.scraper_scheduler_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "lambda:InvokeFunction"
      Resource = aws_lambda_function.odds_scraper[0].arn
    }]
  })
}

# EventBridge Scheduler — hourly with 15-minute jitter
resource "aws_scheduler_schedule" "fetch_oddsportal" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  name       = "${var.rule_prefix}-fetch-oddsportal"
  group_name = "default"

  schedule_expression = "rate(1 hour)"

  flexible_time_window {
    mode                      = "FLEXIBLE"
    maximum_window_in_minutes = 15
  }

  target {
    arn      = aws_lambda_function.odds_scraper[0].arn
    role_arn = aws_iam_role.scraper_scheduler_role[0].arn
    input    = jsonencode({ job = "fetch-oddsportal" })
  }

  state = "ENABLED"
}

# EventBridge Scheduler — daily results and closing odds collection
resource "aws_scheduler_schedule" "fetch_oddsportal_results" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  name       = "${var.rule_prefix}-fetch-oddsportal-results"
  group_name = "default"

  schedule_expression          = "cron(0 8 * * ? *)"
  schedule_expression_timezone = "UTC"

  flexible_time_window {
    mode                      = "FLEXIBLE"
    maximum_window_in_minutes = 15
  }

  target {
    arn      = aws_lambda_function.odds_scraper[0].arn
    role_arn = aws_iam_role.scraper_scheduler_role[0].arn
    input    = jsonencode({ job = "fetch-oddsportal-results" })
  }

  state = "ENABLED"
}

# Outputs
output "scraper_function_arn" {
  description = "ARN of the scraper Lambda function"
  value       = var.enable_oddsportal_scraper ? aws_lambda_function.odds_scraper[0].arn : null
}

output "scraper_ecr_url" {
  description = "ECR repository URL for scraper image"
  value       = var.enable_oddsportal_scraper ? "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com/odds-scraper" : null
}
