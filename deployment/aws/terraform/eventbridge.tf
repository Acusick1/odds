# Bootstrap rule for Polymarket price history backfill
# Runs on fixed schedule (not self-scheduling) to stay ahead of 30-day CLOB retention window
resource "aws_cloudwatch_event_rule" "bootstrap_backfill_polymarket" {
  name                = format("%s-backfill-polymarket-bootstrap", var.rule_prefix)
  description         = "Recurring Polymarket price history backfill (30-day CLOB retention)"
  schedule_expression = "rate(3 days)"
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "bootstrap_backfill_polymarket_target" {
  rule      = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket.name
  target_id = "1"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job = "backfill-polymarket"
  })
}

resource "aws_lambda_permission" "allow_eventbridge_backfill_polymarket" {
  statement_id  = format("AllowExecutionFromEventBridgeBackfillPolymarket-%s", var.rule_prefix)
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
# Lambda's put_rule() updates the schedule_expression and sets State=ENABLED; Terraform
# ignores those changes but owns the lifecycle (create/destroy).
# Initial state is DISABLED with a placeholder schedule; the post-deploy invocation activates them.

locals {
  self_scheduling_jobs = ["fetch-odds", "fetch-scores", "update-status", "check-health", "fetch-polymarket"]
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
  value = {
    backfill_polymarket = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket.name
  }
}

output "dynamic_rules" {
  description = "Self-scheduling EventBridge rules (schedule updated by Lambda)"
  value = { for k, v in aws_cloudwatch_event_rule.dynamic : k => v.name }
}
