# Bootstrap EventBridge rule to trigger initial Lambda execution
# After first execution, Lambda will self-schedule using dynamic rules

resource "aws_cloudwatch_event_rule" "bootstrap_fetch_odds" {
  name                = "odds-fetch-odds-bootstrap"
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
  statement_id  = "AllowExecutionFromEventBridgeFetchOdds"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_fetch_odds.arn
}

# Bootstrap rule for scores fetching
resource "aws_cloudwatch_event_rule" "bootstrap_fetch_scores" {
  name                = "odds-fetch-scores-bootstrap"
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
  statement_id  = "AllowExecutionFromEventBridgeFetchScores"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_fetch_scores.arn
}

# Bootstrap rule for status updates
resource "aws_cloudwatch_event_rule" "bootstrap_update_status" {
  name                = "odds-update-status-bootstrap"
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
  statement_id  = "AllowExecutionFromEventBridgeUpdateStatus"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_update_status.arn
}

# Outputs
output "bootstrap_rules" {
  description = "Bootstrap EventBridge rules (will be updated by Lambda)"
  value = {
    fetch_odds    = aws_cloudwatch_event_rule.bootstrap_fetch_odds.name
    fetch_scores  = aws_cloudwatch_event_rule.bootstrap_fetch_scores.name
    update_status = aws_cloudwatch_event_rule.bootstrap_update_status.name
  }
}
