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

The wiki is the documentation ([SCHEMA](../SCHEMA.md)); [README.md](../../README.md) is the human entry point and [AGENTS.md](../../AGENTS.md#documentation) tells coding agents to read the wiki first. Decisions live in `docs/adr/`: [ADR-0001](../../docs/adr/0001-wiki-replaces-docs.md) records why a code-citing wiki replaced the old long-form pages. Agent tooling conventions (issue tracker, triage labels, domain docs) live in `docs/agents/`. Code outranks every page: when a page and code disagree, the code is right and the page is stale.

## Authoritative sources

| Topic | Authoritative source |
|---|---|
| Settings and defaults | `odds_core.config` ([Configuration](configuration.md)); `.env.example` is the template |
| CLI flags and defaults | each command's `--help` ([The odds CLI](odds-cli.md)) |
| Job and schedule matrix | `deployment/aws/terraform/eventbridge.tf` and `odds_lambda.scheduling.jobs._JOB_CRON_MAP` ([Where each job runs](../scheduling/job-deployment-matrix.md)); deployed state only via `odds scheduler list-jobs --backend aws` |
| Alerting and reporting | `odds_core.alerts` ([Alerts](../operations/alerts.md), [Reporting surfaces](../operations/reporting-surfaces.md)) |
| MCP toolset | the `@mcp.tool()` definitions in `odds_mcp.server` ([odds-mcp server](../agent/mcp-server.md)) |
| Agent behaviour | the prompt files in `packages/odds-mcp/agents/` ([The betting agent](../agent/betting-agent.md)) |
| Schema and storage | the model files, chiefly `odds_core.models` ([Storage](../storage/index.md)) |
| Training data, target, features | `odds_analytics.training.config.FeatureConfig` and `odds_analytics.feature_groups`; evidence in `experiments/results/` ([CLV model findings](../evaluation/clv-model-findings.md)) |
| Agent data sources | [Agent data sources](../agent/agent-data-sources.md) |
| Issue and triage workflow | `docs/agents/` (below) |

## How the docs stay current

The wiki lint (`scripts/wiki_lint.py`) runs as the `wiki-lint` pre-commit hook and in the `Lint` workflow ([Code conventions](code-conventions.md#mechanical-gates)). It checks that cited paths and symbols exist and flags a page as stale when a symbol it cites changes after the page does. Rationale with no recorded source is labelled `(inferred)`. `AGENTS.md` and `README.md` sit at the repo root outside the citation forms, so wiki pages link to them instead of citing them.

Vale rejects change-narrating words in `README.md`, `AGENTS.md`, `docs/adr/`, `docs/agents/` and `wiki/`; `.vale.ini` turns both rules off for `docs/adr/*.md` because ADRs are point-in-time records. Pages describe the current state.

`docs/agents/domain.md` tells agents to read `wiki/index.md` and the ADRs under `docs/adr/` before exploring and to flag any contradiction of an ADR.

## Issue tracker

Issues live in GitHub Issues on `Acusick1/odds`, added to the `Personal` project board. Open a draft PR whose body contains `Closes #<n>` as soon as the branch has its first commit; linking moves the issue to In Progress and merging moves it to Done. Triage labels, the `gh` command table, and the sub-issue and wayfinding procedures are in `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md`, which are authoritative and loaded through [AGENTS.md](../../AGENTS.md#agent-skills). External PRs are not a triage surface: `docs/agents/issue-tracker.md` sets "PRs as a request surface" to `no`.

## Related

- [Code conventions and lint gates](code-conventions.md)
- [About the system and its pivots](project-overview.md)
