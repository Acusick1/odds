---
type: explanation
title: Documentation and issue tracking
description: Where documentation lives (wiki, ADRs, docs/agents), what is authoritative for what, how the wiki lint and Vale keep pages current, and the issue-tracker conventions.
sources:
  - .vale.ini
  - scripts/wiki_lint.py
  - docs/adr/0001-wiki-replaces-docs.md
  - docs/agents/domain.md
  - docs/agents/issue-tracker.md#conventions
  - docs/agents/issue-tracker.md#pull-requests-as-a-triage-surface
  - docs/agents/triage-labels.md
  - deployment/aws/terraform/eventbridge.tf
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
status: stable
---

# Documentation and issue tracking

The wiki is the documentation ([SCHEMA](../SCHEMA.md)); [README.md](../../README.md) is the human entry point and [AGENTS.md](../../AGENTS.md#documentation) tells coding agents to read the wiki first. Hard-to-reverse engineering decisions live in `docs/adr/`; strategy lives on [Goal and strategy](goal-and-strategy.md), which is rewritten as it changes. [ADR-0001](../../docs/adr/0001-wiki-replaces-docs.md) records why a code-citing wiki replaced the old long-form pages. Agent tooling conventions (issue tracker, triage labels, domain docs) live in `docs/agents/`. Code outranks every page: when a page and code disagree, the code is right and the page is stale.

## Authoritative sources

| Topic | Authoritative source |
|---|---|
| Settings, defaults, per-job heartbeat hours (fetch-odds, fetch-scores and each sport's collectors included), stale-data multipliers, the Betfair-close lookback and the default job list | `odds_core.config` ([Configuration](configuration.md#sections)); `.env.example` is a template, not a source of defaults, so a setting removed from `odds_core.config` is removed from it too |
| CLI flags and defaults | each command's `--help` ([The odds CLI](odds-cli.md)) |
| Job and schedule matrix | `odds_core.config.SchedulerConfig.bootstrap_jobs`, `odds_lambda.scheduling.jobs._JOB_CRON_MAP` and `odds_lambda.scheduling.jobs._JOB_SPORTS` for what runs in production (the collectors, Polymarket for each sport it lists, the results jobs and the candidate runner), and `odds_lambda.scheduling.bootstrap.bootstrap_job` for what each scheduler start does to a stored schedule (and `odds_lambda.scheduling.runner.run_chain_job` for the timeout, chain re-booking and recovery alerts every local run gets); `deployment/aws/terraform/eventbridge.tf` for the dormant AWS scheduler rules ([Where each job runs](../scheduling/job-deployment-matrix.md)); deployed AWS state only via `odds scheduler list-jobs --backend aws` |
| Alerting and reporting | `odds_core.alerts` and its `odds_core.alerts.send_job_failure` (once per outage; `odds_core.alerts.send_outage_warning` for warnings, `odds_core.alerts.send_scheduling_failure` for a job that cannot book its next run; the health monitor reminds every `stale_reminder_minutes`; all share `odds_core.alerts.send_recorded_alert`, described in [Alerts](../operations/alerts.md#alert-history)), gated by `odds_core.config.Settings.alerts_deliverable` and `SCHEDULER_DRY_RUN` ([Alerts](../operations/alerts.md#delivery), [Reporting surfaces](../operations/reporting-surfaces.md)) |
| MCP toolset | each `@mcp.tool()` function registered on `odds_mcp.server.mcp`, with its signature, docstring and return shape (the source for a tool's arguments and results) ([odds-mcp server](../agent/mcp-server.md)) |
| Agent behaviour | the prompt files in `packages/odds-mcp/agents/` ([The betting agent](../agent/betting-agent.md)) |
| Schema and storage | the model files, mainly `odds_core.models` (events with their integer key, external ids, and status moves, snapshots (with closed-set source, market, time basis and market status), their prices and archived payloads, and the fetch, data-quality and alert logs; [Storage](../storage/index.md), [snapshots table](../storage/odds-tables.md)) |
| Training data, target and feature config | `odds_analytics.training.config.FeatureConfig` (whose `feature_groups` names, per target, and `target_type` are validated, with the fields a `retail_ev` target does not read, and `retail_ev_arm`, which only a `retail_ev` target may set away from `direct`) and `odds_analytics.feature_groups.prepare_training_data` (which builds the training arrays `X`, `y` and `event_ids` from the groups, over each decision snapshot's price rows and its own source's close, sampling only of the sources whose close has a sharp book); evidence in `experiments/results/` ([experiment pages](../evaluation/index.md)) |
| Agent data sources | [Agent data sources](../agent/agent-data-sources.md) |
| Issue and triage workflow | `docs/agents/` (below) |

## How the docs stay current

The wiki lint (`scripts/wiki_lint.py`) runs as the `wiki-lint` pre-commit hook and in the `Lint` workflow ([Code conventions](code-conventions.md#mechanical-gates)). It checks that cited paths and symbols exist and flags a page as stale when a symbol it cites, or an import that re-exports it, changes after the page does, counting only page edits that survive the branch's squash merge. A comment-only edit inside a cited symbol counts as a change, so it marks every page citing that symbol, except pages that list the symbol under `mentions`, which track only its signature and docstring ([SCHEMA](../SCHEMA.md#staleness)). `--locate` prints the file and line range behind a page's citations or a dotted path. It also flags pages that name the production machine, so they describe its role instead. Rationale with no recorded source is labelled `(inferred)`. [AGENTS.md](../../AGENTS.md) and [README.md](../../README.md) sit at the repo root outside the citation forms, so wiki pages link to them instead of citing them.

Vale rejects change-narrating words in [README.md](../../README.md), [AGENTS.md](../../AGENTS.md), `docs/adr/`, `docs/agents/` and `wiki/`; `.vale.ini` turns both rules off for `docs/adr/*.md` because ADRs are point-in-time records. Every page describes the current state.

`docs/agents/domain.md` tells agents to read [wiki/index.md](../index.md) and the ADRs under `docs/adr/` before exploring and to flag any contradiction of an accepted ADR.

## Issue tracker

Issues live in GitHub Issues on `Acusick1/odds`, added to the `Personal` project board. Claiming an issue (branches `<n>-<slug>` or `issue-<n>-<slug>`) is in `docs/agents/issue-tracker.md`. Open a draft PR whose body contains `Closes #<n>` as soon as the branch has its first commit; linking moves the issue to In Progress and merging moves it to Done. Triage labels, the `gh` command table, and the sub-issue and wayfinding procedures are in `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md`, which are authoritative and loaded through [AGENTS.md](../../AGENTS.md#agent-skills). External PRs are not a triage surface: `docs/agents/issue-tracker.md` sets "PRs as a request surface" to `no`.

## Related

- [Code conventions and lint gates](code-conventions.md)
- [About the system and its pivots](project-overview.md)
