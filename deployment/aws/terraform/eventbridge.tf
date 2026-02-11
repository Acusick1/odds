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

  depends_on = [
    aws_cloudwatch_event_rule.bootstrap_fetch_odds,
    aws_cloudwatch_event_target.bootstrap_fetch_odds_target
  ]
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

  depends_on = [
    aws_cloudwatch_event_rule.bootstrap_fetch_scores,
    aws_cloudwatch_event_target.bootstrap_fetch_scores_target
  ]
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

  depends_on = [
    aws_cloudwatch_event_rule.bootstrap_update_status,
    aws_cloudwatch_event_target.bootstrap_update_status_target
  ]
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

  depends_on = [
    aws_cloudwatch_event_rule.bootstrap_check_health,
    aws_cloudwatch_event_target.bootstrap_check_health_target
  ]
}

# Bootstrap rule for Polymarket live data collection
# After first execution, job self-schedules at price_poll_interval via dynamic rules
resource "aws_cloudwatch_event_rule" "bootstrap_fetch_polymarket" {
  name                = format("%s-fetch-polymarket-bootstrap", var.project_name)
  description         = "Bootstrap trigger for Polymarket data collection"
  schedule_expression = "rate(5 minutes)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_fetch_polymarket_target" {
  rule      = aws_cloudwatch_event_rule.bootstrap_fetch_polymarket.name
  target_id = "lambda"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "fetch-polymarket"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_fetch_polymarket" {
  statement_id  = format("AllowExecutionFromEventBridgeFetchPolymarket-%s", var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_fetch_polymarket.arn

  depends_on = [
    aws_cloudwatch_event_rule.bootstrap_fetch_polymarket,
    aws_cloudwatch_event_target.bootstrap_fetch_polymarket_target
  ]
}

# Bootstrap rule for Polymarket price history backfill
# Runs on fixed schedule (not self-scheduling) to stay ahead of 30-day CLOB retention window
resource "aws_cloudwatch_event_rule" "bootstrap_backfill_polymarket" {
  name                = format("%s-backfill-polymarket-bootstrap", var.project_name)
  description         = "Recurring Polymarket price history backfill (30-day CLOB retention)"
  schedule_expression = "rate(3 days)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_backfill_polymarket_target" {
  rule      = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket.name
  target_id = "lambda"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "backfill-polymarket"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_backfill_polymarket" {
  statement_id  = format("AllowExecutionFromEventBridgeBackfillPolymarket-%s", var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket.arn

  depends_on = [
    aws_cloudwatch_event_rule.bootstrap_backfill_polymarket,
    aws_cloudwatch_event_target.bootstrap_backfill_polymarket_target
  ]
}

# Self-scheduling rules: pre-created by Terraform, schedule updated by Lambda at runtime.
# Lambda's put_rule() updates the schedule_expression; Terraform ignores those changes
# but owns the lifecycle (create/destroy).

locals {
  self_scheduling_jobs = ["fetch-odds", "fetch-scores", "update-status", "check-health", "fetch-polymarket"]
}

resource "aws_cloudwatch_event_rule" "dynamic" {
  for_each = toset(local.self_scheduling_jobs)

  name                = "odds-${each.key}"
  description         = "Self-scheduling rule for ${each.key} (updated by Lambda)"
  schedule_expression = "rate(1 day)"
  state               = "ENABLED"

  lifecycle {
    ignore_changes = [schedule_expression]
  }
}

resource "aws_cloudwatch_event_target" "dynamic" {
  for_each = toset(local.self_scheduling_jobs)

  rule      = aws_cloudwatch_event_rule.dynamic[each.key].name
  target_id = "lambda"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = each.key
  })
}

resource "aws_lambda_permission" "allow_dynamic" {
  for_each = toset(local.self_scheduling_jobs)

  statement_id  = format("AllowDynamic-%s-%s", each.key, var.project_name)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.dynamic[each.key].arn

  depends_on = [
    aws_cloudwatch_event_rule.dynamic,
    aws_cloudwatch_event_target.dynamic
  ]
}

# Outputs
output "bootstrap_rules" {
  description = "Bootstrap EventBridge rules (initial triggers)"
  value = {
    fetch_odds           = aws_cloudwatch_event_rule.bootstrap_fetch_odds.name
    fetch_scores         = aws_cloudwatch_event_rule.bootstrap_fetch_scores.name
    update_status        = aws_cloudwatch_event_rule.bootstrap_update_status.name
    check_health         = aws_cloudwatch_event_rule.bootstrap_check_health.name
    fetch_polymarket     = aws_cloudwatch_event_rule.bootstrap_fetch_polymarket.name
    backfill_polymarket  = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket.name
  }
}

output "dynamic_rules" {
  description = "Self-scheduling EventBridge rules (schedule updated by Lambda)"
  value = { for k, v in aws_cloudwatch_event_rule.dynamic : k => v.name }
}
