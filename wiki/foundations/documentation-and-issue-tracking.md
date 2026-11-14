---
type: explanation
title: Documentation and issue tracking
description: Where documentation lives (wiki, ADRs, docs/agents), what is authoritative for what, how the wiki lint and Vale keep pages current, and the issue-tracker conventions.
sources:
  - .vale.ini
  - scripts.wiki_lint.WikiLinter
  - docs/adr/0001-wiki-replaces-docs.md
  - docs/agents/domain.md
  - docs/agents/issue-tracker.md#conventions
  - docs/agents/issue-tracker.md#pull-requests-as-a-triage-surface
  - docs/agents/triage-labels.md
status: stable
---

# Documentation and issue tracking

The wiki is the documentation ([SCHEMA](../SCHEMA.md)); [README.md](../../README.md) is the human entry point and [AGENTS.md](../../AGENTS.md#documentation) tells coding agents to read the wiki first. Hard-to-reverse engineering decisions live in `docs/adr/`; strategy lives on [Goal and strategy](goal-and-strategy.md), which is rewritten as it changes. [ADR-0001](../../docs/adr/0001-wiki-replaces-docs.md) records why a code-citing wiki replaced the old long-form pages. Agent tooling conventions (issue tracker, triage labels, domain docs) live in `docs/agents/`. Code outranks every page: when a page and code disagree, the code is right and the page is stale.

## Authoritative sources

| Topic | Authoritative source |
|---|---|
| Settings and operator overrides | `odds_core.config.Settings`; job defaults (the `cron`, `sports` allowlist and `heartbeats` fields of each job record) come from `odds_core.job_specs.JOB_SPECS`; [Configuration](configuration.md#sections) owns the field map and template rules. |
| CLI flags and defaults | Each command's `--help`; [CLI reference](odds-cli.md#shared-conventions). |
| Jobs and deployment | [Deployment matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs) owns the source map and operational status. |
| Alerting and reporting | [Alerts](../operations/alerts.md#delivery) and [reporting surfaces](../operations/reporting-surfaces.md#reporting-surfaces) cite their implementations. |
| MCP toolset | Registered tool signatures and return shapes; [server catalogue](../agent/mcp-server.md#tool-catalogue-16). |
| Agent behaviour | Prompt files in `packages/odds-mcp/agents/`; [betting agent](../agent/betting-agent.md#prompts). |
| Schema and storage | Model files, mainly `odds_core.models` (the `events` columns and the `odds_core.models.EventPostponement` table are on [events](../storage/events.md#columns), and the `odds_core.backfill_models.OddsApiSnapshotGap` table, the holes a backfill met, on [provider gaps](../collection/odds-api-backfill.md#provider-gaps)); [storage table map](../storage/index.md#table-map). |
| Training data, target and feature config | `odds_analytics.training.config.FeatureConfig` and `odds_analytics.feature_groups.prepare_training_data`; [feature pipeline](../modelling/feature-pipeline.md#feature-pipeline). Evidence lives in `experiments/results/` ([evaluation index](../evaluation/index.md#backtests-and-model-research)). |
| Agent data sources | [Agent data sources](../agent/agent-data-sources.md#agent-data-sources-and-rejected-sources). |
| Issue and triage workflow | `docs/agents/` ([issue tracker](#issue-tracker)). |

## How the docs stay current

The wiki lint (`scripts.wiki_lint.WikiLinter`) runs as the `wiki-lint` pre-commit hook and in the `Lint` workflow ([Code conventions](code-conventions.md#mechanical-gates)). It checks that cited paths and dotted symbols (in the packages and `scripts/`) exist and flags a page as stale when a symbol it cites, or an import that re-exports it, changes after the page does, counting only page edits that survive the branch's squash merge. A comment-only edit inside a cited symbol counts as a change, so it marks every page citing that symbol, except pages that list the symbol under `mentions`, which track only its signature and docstring ([SCHEMA](../SCHEMA.md#staleness)). `--locate` prints the file and line range behind a page's citations or a dotted path. It also flags pages that name the production machine, so they describe its role instead. Rationale with no recorded source is labelled `(inferred)`. [AGENTS.md](../../AGENTS.md) and [README.md](../../README.md) sit at the repo root outside the citation forms, so wiki pages link to them instead of citing them.

Vale rejects change-narrating words in [README.md](../../README.md), [AGENTS.md](../../AGENTS.md), `docs/adr/`, `docs/agents/` and `wiki/`; `.vale.ini` turns both rules off for `docs/adr/*.md` because ADRs are point-in-time records. Every page describes the current state.

`docs/agents/domain.md` tells agents to read the [glossary](glossary.md), [wiki/index.md](../index.md) and the ADRs under `docs/adr/` before exploring, to use the glossary's terms, and to flag any contradiction of an accepted ADR. The glossary gives each domain term one line and links the page that defines it, which stays authoritative.

## Issue tracker

Issues live in GitHub Issues on `Acusick1/odds`, added to the `Personal` project board. Claiming an issue (branches `<n>-<slug>` or `issue-<n>-<slug>`) is in `docs/agents/issue-tracker.md`. Open a draft PR whose body contains `Closes #<n>` as soon as the branch has its first commit; linking moves the issue to In Progress and merging moves it to Done. Triage labels, the `gh` command table, and the sub-issue and wayfinding procedures are in `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md`, which are authoritative and loaded through [AGENTS.md](../../AGENTS.md#agent-skills). External PRs are not a triage surface: `docs/agents/issue-tracker.md` sets "PRs as a request surface" to `no`.

## Related

- [Code conventions and lint gates](code-conventions.md)
- [About the system and its pivots](project-overview.md)
