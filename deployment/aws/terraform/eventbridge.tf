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

# Per-sport fixed-schedule rules (daily-digest and score-predictions).
# Generated from sport_configs so each sport gets independent rules.
resource "aws_cloudwatch_event_rule" "fixed_scheduler" {
  for_each = local.fixed_scheduler_rules_map

  name                = format("%s-%s", var.rule_prefix, each.key)
  description         = "Fixed-schedule rule for ${each.key}"
  schedule_expression = local.fixed_schedule_expressions[each.value.job]
  state               = "ENABLED"
}

resource "aws_cloudwatch_event_target" "fixed_scheduler_target" {
  for_each = local.fixed_scheduler_rules_map

  rule      = aws_cloudwatch_event_rule.fixed_scheduler[each.key].name
  target_id = "1"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode({
    job   = each.value.job
    sport = each.value.sport_key
  })
}

resource "aws_lambda_permission" "allow_fixed_scheduler" {
  for_each = local.fixed_scheduler_rules_map

  statement_id  = format("AllowFixed-%s-%s", each.key, var.rule_prefix)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scheduler.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.fixed_scheduler[each.key].arn

  depends_on = [
    aws_cloudwatch_event_rule.fixed_scheduler,
    aws_cloudwatch_event_target.fixed_scheduler_target
  ]
}

# Self-scheduling rules: pre-created by Terraform, schedule updated by Lambda at runtime.
# Lambda's put_rule() updates the schedule_expression and sets State=ENABLED; Terraform
# ignores those changes but owns the lifecycle (create/destroy).
# Initial state is DISABLED with a placeholder schedule; the post-deploy invocation activates them.

locals {
  # Per-sport configuration. Adding a new sport requires only a new entry here.
  sport_configs = {
    epl = { sport_key = "soccer_epl", scraper = true }
    # mlb = { sport_key = "baseball_mlb", scraper = false }
  }

  # Jobs that run once per sport (scheduler Lambda)
  per_sport_scheduler_jobs = ["fetch-odds", "fetch-scores"]

  # Jobs that run once per sport (scraper Lambda)
  per_sport_scraper_jobs = ["fetch-oddsportal", "fetch-oddsportal-results"]

  # Jobs that run once globally (no sport param)
  global_jobs = ["update-status", "check-health"]

  polymarket_jobs = var.enable_polymarket ? ["fetch-polymarket"] : []

  # Generate per-sport scheduler job names: "fetch-odds-epl", "fetch-scores-epl", etc.
  sport_scheduler_rules = flatten([
    for sport_suffix, cfg in local.sport_configs : [
      for job in local.per_sport_scheduler_jobs : {
        key       = "${job}-${sport_suffix}"
        job       = job
        sport_key = cfg.sport_key
      }
    ]
  ])

  # Generate per-sport scraper job names (only for sports with scraper = true)
  sport_scraper_rules = var.enable_oddsportal_scraper ? flatten([
    for sport_suffix, cfg in local.sport_configs : [
      for job in local.per_sport_scraper_jobs : {
        key       = "${job}-${sport_suffix}"
        job       = job
        sport_key = cfg.sport_key
      }
    ] if cfg.scraper
  ]) : []

  # Maps for for_each (keyed by compound name)
  scheduler_rules_map = merge(
    { for r in local.sport_scheduler_rules : r.key => r },
    { for j in local.global_jobs : j => { key = j, job = j, sport_key = null } },
    { for j in local.polymarket_jobs : j => { key = j, job = j, sport_key = null } },
  )

  scraper_rules_map = { for r in local.sport_scraper_rules : r.key => r }

  # Flat lists for outputs
  self_scheduling_scheduler_jobs = [for k, _ in local.scheduler_rules_map : k]
  self_scheduling_scraper_jobs   = [for k, _ in local.scraper_rules_map : k]

  # Schedule expressions for fixed-schedule jobs (map lookup prevents
  # a new job silently inheriting the wrong schedule via a ternary fallback).
  # score-predictions, daily-digest, and fetch-espn-fixtures now run in the
  # local scheduler so the agent, scraper, and scorer share one DB. Only
  # jobs that still fire from EventBridge belong here.
  fixed_schedule_expressions = {}

  # Per-sport fixed-schedule jobs. Empty until a Lambda-hosted fixed-schedule
  # job is reintroduced; kept as a list comprehension so adding one back is
  # a single-line change.
  sport_fixed_scheduler_rules = []

  fixed_scheduler_rules_map = { for r in local.sport_fixed_scheduler_rules : r.key => r }
}

resource "aws_cloudwatch_event_rule" "dynamic" {
  for_each = local.scheduler_rules_map

  name                = "${var.rule_prefix}-${each.key}"
  description         = "Self-scheduling rule for ${each.key} (updated by Lambda)"
  schedule_expression = "rate(1 day)"
  state               = "DISABLED"

  lifecycle {
    ignore_changes = [schedule_expression, state]
  }
}

resource "aws_cloudwatch_event_target" "dynamic" {
  for_each = local.scheduler_rules_map

  rule      = aws_cloudwatch_event_rule.dynamic[each.key].name
  target_id = "1"
  arn       = aws_lambda_function.odds_scheduler.arn

  input = jsonencode(merge(
    { job = each.value.job },
    each.value.sport_key != null ? { sport = each.value.sport_key } : {}
  ))
}

resource "aws_lambda_permission" "allow_dynamic" {
  for_each = local.scheduler_rules_map

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

# Self-scheduling rules for scraper Lambda (same pattern as scheduler, different target).
resource "aws_cloudwatch_event_rule" "scraper_dynamic" {
  for_each = local.scraper_rules_map

  name                = "${var.rule_prefix}-${each.key}"
  description         = "Self-scheduling rule for ${each.key} (updated by scraper Lambda)"
  schedule_expression = "rate(1 day)"
  state               = "DISABLED"

  lifecycle {
    ignore_changes = [schedule_expression, state]
  }
}

resource "aws_cloudwatch_event_target" "scraper_dynamic" {
  for_each = local.scraper_rules_map

  rule      = aws_cloudwatch_event_rule.scraper_dynamic[each.key].name
  target_id = "1"
  arn       = aws_lambda_function.odds_scraper[0].arn

  input = jsonencode({
    job   = each.value.job
    sport = each.value.sport_key
  })

  retry_policy {
    maximum_retry_attempts         = 0
    maximum_event_age_in_seconds   = 60
  }
}

resource "aws_lambda_permission" "allow_scraper_dynamic" {
  for_each = local.scraper_rules_map

  statement_id  = format("AllowScraperDynamic-%s-%s", each.key, var.rule_prefix)
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.odds_scraper[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scraper_dynamic[each.key].arn

  depends_on = [
    aws_cloudwatch_event_rule.scraper_dynamic,
    aws_cloudwatch_event_target.scraper_dynamic
  ]
}

# Outputs
output "bootstrap_rules" {
  description = "Bootstrap EventBridge rules (fixed-schedule, not self-scheduling)"
  value = merge(
    { for k, v in aws_cloudwatch_event_rule.fixed_scheduler : k => v.name },
    var.enable_polymarket ? {
      backfill_polymarket = aws_cloudwatch_event_rule.bootstrap_backfill_polymarket[0].name
    } : {}
  )
}

output "dynamic_rules" {
  description = "Self-scheduling EventBridge rules (schedule updated by Lambda)"
  value = merge(
    { for k, v in aws_cloudwatch_event_rule.dynamic : k => v.name },
    { for k, v in aws_cloudwatch_event_rule.scraper_dynamic : k => v.name }
  )
}

output "scheduler_jobs" {
  description = "Job names routed to the scheduler Lambda (CSV for scripts)"
  value       = join(",", concat(local.self_scheduling_scheduler_jobs, keys(local.fixed_scheduler_rules_map)))
}

output "scraper_jobs" {
  description = "Job names routed to the scraper Lambda (CSV for scripts)"
  value       = join(",", local.self_scheduling_scraper_jobs)
}

output "self_scheduling_jobs" {
  description = "All self-scheduling job names (CSV for scripts)"
  value       = join(",", concat(local.self_scheduling_scheduler_jobs, local.self_scheduling_scraper_jobs))
}
