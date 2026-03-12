# Bootstrap rule for Polymarket price history backfill
# Runs on fixed schedule (not self-scheduling) to stay ahead of 30-day CLOB retention window
resource "aws_cloudwatch_event_rule" "bootstrap_backfill_polymarket" {
  count = var.enable_polymarket ? 1 : 0

  name                = format("%s-backfill-polymarket-bootstrap", var.rule_prefix)
  description         = "Recurring Polymarket price history backfill (30-day CLOB retention)"
  schedule_expression = "rate(3 days)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_backfill_polymarket_target" {
  count = var.enable_polymarket ? 1 : 0

  rule      = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket[0].name
  target_id = "1"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "backfill-polymarket"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_backfill_polymarket" {
  count = var.enable_polymarket ? 1 : 0

  statement_id  = format("AllowExecutionFromEventBridgeBackfillPolymarket-%s", var.rule_prefix)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket[0].arn

  depends_on = [
    aws_cloudwatch_event_rule.bootstrap_backfill_polymarket,
    aws_cloudwatch_event_target.bootstrap_backfill_polymarket_target
  ]
}

# Daily digest: sends Discord embed with predictions + post-match results at 08:00 UTC
resource "aws_cloudwatch_event_rule" "daily_digest" {
  name                = format("%s-daily-digest", var.rule_prefix)
  description         = "Daily Discord digest with predictions and post-match results"
  schedule_expression = "cron(0 8 * * ? *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "daily_digest_target" {
  rule      = aws_cloudwatch_event_rule.daily_digest.name
  target_id = "1"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "daily-digest"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_daily_digest" {
  statement_id  = format("AllowExecutionFromEventBridgeDailyDigest-%s", var.rule_prefix)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_digest.arn

  depends_on = [
    aws_cloudwatch_event_rule.daily_digest,
    aws_cloudwatch_event_target.daily_digest_target
  ]
}

# Score predictions: runs CLV model inference after scraper has landed new snapshots.
# Offset 30 min past the hour to account for scraper jitter (up to 15 min) + runtime (up to 10 min).
resource "aws_cloudwatch_event_rule" "score_predictions" {
  name                = format("%s-score-predictions", var.rule_prefix)
  description         = "Hourly CLV model inference on new snapshots"
  schedule_expression = "cron(30 * * * ? *)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "score_predictions_target" {
  rule      = aws_cloudwatch_event_rule.score_predictions.name
  target_id = "1"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "score-predictions"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_score_predictions" {
  statement_id  = format("AllowEventBridgeScorePredictions-%s", var.rule_prefix)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.score_predictions.arn
}

# Self-scheduling rules: pre-created by Terraform, schedule updated by Lambda at runtime.
# Lambda's put_rule() updates the schedule_expression and sets State=ENABLED; Terraform
# ignores those changes but owns the lifecycle (create/destroy).
# Initial state is DISABLED with a placeholder schedule; the post-deploy invocation activates them.

locals {
  core_jobs        = ["fetch-odds", "fetch-scores", "update-status", "check-health"]
  polymarket_jobs  = var.enable_polymarket ? ["fetch-polymarket"] : []
  self_scheduling_jobs = concat(local.core_jobs, local.polymarket_jobs)
}

resource "aws_cloudwatch_event_rule" "dynamic" {
  for_each = toset(local.self_scheduling_jobs)

  name                = "${var.rule_prefix}-${each.key}"
  description         = "Self-scheduling rule for ${each.key} (updated by Lambda)"
  schedule_expression = "rate(1 day)"
  state               = "DISABLED"

  lifecycle {
    ignore_changes = [schedule_expression, state]
  }
}

resource "aws_cloudwatch_event_target" "dynamic" {
  for_each = toset(local.self_scheduling_jobs)

  rule      = aws_cloudwatch_event_rule.dynamic[each.key].name
  target_id = "1"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = each.key
  })
}

resource "aws_lambda_permission" "allow_dynamic" {
  for_each = toset(local.self_scheduling_jobs)

  statement_id  = format("AllowDynamic-%s-%s", each.key, var.rule_prefix)
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
  description = "Bootstrap EventBridge rules (fixed-schedule, not self-scheduling)"
  value = merge(
    { daily_digest = aws_cloudwatch_event_rule.daily_digest.name },
    var.enable_polymarket ? {
      backfill_polymarket = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket[0].name
    } : {}
  )
}

output "dynamic_rules" {
  description = "Self-scheduling EventBridge rules (schedule updated by Lambda)"
  value = { for k, v in aws_cloudwatch_event_rule.dynamic : k => v.name }
}

output "self_scheduling_jobs" {
  description = "Job names that use self-scheduling (CSV for scripts)"
  value       = join(",", local.self_scheduling_jobs)
}
