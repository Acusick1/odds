# Bootstrap EventBridge rule to trigger initial Lambda execution
# After first execution, Lambda will self-schedule using dynamic rules

resource "aws_cloudwatch_event_rule" "bootstrap_fetch_odds" {
  name                = format("%s-fetch-odds-bootstrap", var.project_name)
  description         = "Bootstrap trigger for odds fetching (will be updated dynamically)"
  schedule_expression = "rate(1 day)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_fetch_odds_target" {
  rule      = aws_cloudwatch_event_rule.bootstrap_fetch_odds.name
  target_id = "lambda"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "fetch-odds"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_fetch_odds" {
  statement_id  = format("AllowExecutionFromEventBridgeFetchOdds-%s", var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_fetch_odds.arn
}

# Bootstrap rule for scores fetching
resource "aws_cloudwatch_event_rule" "bootstrap_fetch_scores" {
  name                = format("%s-fetch-scores-bootstrap", var.project_name)
  description         = "Bootstrap trigger for scores fetching"
  schedule_expression = "rate(6 hours)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_fetch_scores_target" {
  rule      = aws_cloudwatch_event_rule.bootstrap_fetch_scores.name
  target_id = "lambda"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "fetch-scores"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_fetch_scores" {
  statement_id  = format("AllowExecutionFromEventBridgeFetchScores-%s", var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_fetch_scores.arn
}

# Bootstrap rule for status updates
resource "aws_cloudwatch_event_rule" "bootstrap_update_status" {
  name                = format("%s-update-status-bootstrap", var.project_name)
  description         = "Bootstrap trigger for status updates"
  schedule_expression = "rate(1 hour)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_update_status_target" {
  rule      = aws_cloudwatch_event_rule.bootstrap_update_status.name
  target_id = "lambda"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "update-status"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_update_status" {
  statement_id  = format("AllowExecutionFromEventBridgeUpdateStatus-%s", var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_update_status.arn
}

# Bootstrap rule for health checks
resource "aws_cloudwatch_event_rule" "bootstrap_check_health" {
  name                = format("%s-check-health-bootstrap", var.project_name)
  description         = "Bootstrap trigger for health checks"
  schedule_expression = "rate(60 minutes)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_check_health_target" {
  rule      = aws_cloudwatch_event_rule.bootstrap_check_health.name
  target_id = "lambda"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "check-health"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_check_health" {
  statement_id  = format("AllowExecutionFromEventBridgeCheckHealth-%s", var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_check_health.arn
}

# Allow dynamic rules (created by Lambda at runtime) to invoke the function
# This covers rules like odds-fetch-odds, odds-fetch-scores, odds-update-status, odds-check-health
# that are created and updated by the Lambda's self-scheduling logic
resource "aws_lambda_permission" "allow_dynamic_rules" {
  statement_id  = format("AllowExecutionFromDynamicRules-%s", var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = format("arn:aws:events:%s:%s:rule/odds-*", var.aws_region, data.aws_caller_identity.current.account_id)
}

# Outputs
output "bootstrap_rules" {
  description = "Bootstrap EventBridge rules (will be updated by Lambda)"
  value = {
    fetch_odds    = aws_cloudwatch_event_rule.bootstrap_fetch_odds.name
    fetch_scores  = aws_cloudwatch_event_rule.bootstrap_fetch_scores.name
    update_status = aws_cloudwatch_event_rule.bootstrap_update_status.name
    check_health  = aws_cloudwatch_event_rule.bootstrap_check_health.name
  }
}
