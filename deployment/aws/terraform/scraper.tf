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

# EventBridge rule — hourly fixed schedule
resource "aws_cloudwatch_event_rule" "scraper_schedule" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  name                = "${var.rule_prefix}-fetch-oddsportal"
  description         = "Hourly OddsPortal scrape for upcoming match odds"
  schedule_expression = "rate(1 hour)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "scraper_target" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  rule      = aws_cloudwatch_event_rule.scraper_schedule[0].name
  target_id = "1"
  arn       = aws_lambda_function.odds_scraper[0].arn

  input = jsonencode({
    job = "fetch-oddsportal"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_scraper" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  statement_id  = "AllowEventBridgeScraper-${var.rule_prefix}"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scraper[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scraper_schedule[0].arn
}

# EventBridge rule — daily results and closing odds collection
resource "aws_cloudwatch_event_rule" "scraper_results_schedule" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  name                = "${var.rule_prefix}-fetch-oddsportal-results"
  description         = "Daily EPL results and closing odds collection"
  schedule_expression = "cron(0 8 * * ? *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "scraper_results_target" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  rule      = aws_cloudwatch_event_rule.scraper_results_schedule[0].name
  target_id = "1"
  arn       = aws_lambda_function.odds_scraper[0].arn

  input = jsonencode({
    job = "fetch-oddsportal-results"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_scraper_results" {
  count = var.enable_oddsportal_scraper ? 1 : 0

  statement_id  = "AllowEventBridgeScraperResults-${var.rule_prefix}"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scraper[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scraper_results_schedule[0].arn
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
