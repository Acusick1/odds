---
type: concept
title: Documentation map and authority
summary: "Which repo doc is authoritative for what, the doc-truth lints (Vale, lychee), the glossary and ADR convention, and how the wiki relates to the docs."
sources:
  - docs/agents/domain.md
  - docs/DEVELOPMENT.md#code-quality
  - docs/audits/comprehension-audit-2026-06-09.md#a1-doc-truth-pass-over-the-core-doc-set
  - .pre-commit-config.yaml
  - scripts/wiki_lint.py
  - wiki/SCHEMA.md
verified_at: 7887039
---

# Documentation map and authority

The index is the table in [README.md](../README.md#documentation); [AGENTS.md](../AGENTS.md#documentation) points there. Code outranks every doc: when a doc and code disagree, the code is right.

## Authority by topic

| Topic | Authoritative source |
|---|---|
| Settings and defaults | `packages/odds-core/odds_core/config.py` ([Configuration](configuration.md)); `.env.example` is the template |
| CLI flags and defaults | each command's `--help`; `docs/CLI.md` is the reference ([odds CLI](cli-overview.md)) |
| Job and schedule matrix | `docs/DEPLOYMENT.md` ([Job deployment matrix](job-deployment-matrix.md)); deployed state only via `odds scheduler list-jobs --backend aws` |
| Alerting and reporting | `docs/REPORTING.md` ([Reporting surfaces](reporting-surfaces.md)) |
| MCP toolset | the `@mcp.tool()` definitions in `packages/odds-mcp/odds_mcp/server.py` ([odds-mcp server](mcp-server.md)) |
| Agent behaviour | the prompt files in `packages/odds-mcp/agents/`; `docs/BETTING_AGENT.md` summarises them ([Betting agent](betting-agent.md)) |
| Schema and storage | `docs/DATABASE.md` and the model files in `odds_core/` |
| Training data, target, features | `docs/DATA_MODELS.md`; evidence in `docs/MODELING.md` ([CLV model findings](clv-model-findings.md)) |
| Agent data sources | `docs/AGENT_DATA_SOURCES.md` ([Agent data sources](agent-data-sources.md)) |
| Issue and triage workflow | `docs/agents/` ([Use the issue tracker](howto-use-issue-tracker.md)) |

`docs/audits/comprehension-audit-2026-06-09.md` is a dated snapshot from commit `1296b5f`; its line numbers are stale.

## Doc-truth lints

Pre-commit runs [Vale and lychee](code-conventions.md#mechanical-gates): Vale (`Temporal` style) rejects change-narrating words in `README.md`, `AGENTS.md` and `docs/`, and `.vale.ini` turns both rules off for `docs/adr/*.md` because ADRs are point-in-time records; lychee checks Markdown links and anchors offline. Docs describe the current state, with no "previously" or "now".

## Glossary and ADRs

`docs/agents/domain.md` tells agents to read `GLOSSARY.md` and `docs/adr/` before exploring, to use glossary vocabulary, and to flag any contradiction of an ADR explicitly. Both are created lazily by the `domain-modeling` skill. Neither exists in the repo at this commit, and `domain.md` says to proceed silently when they are absent. Wiki pages label rationale with no recorded source `(inferred)`.

## The wiki

`wiki/` is a synthesis layer over code and docs ([schema](SCHEMA.md)). It cites code by `path::symbol`, and `scripts/wiki_lint.py` checks that paths, symbols, links and anchors still resolve. It does not replace the docs above: each page lists the doc sections it absorbed in its `sources`.

## Hazards

- The audit's A1 table of doc/code mismatches is partly out of date and partly live. Re-verify a row against code before relying on it; this wiki records the live ones on the page that owns the topic.
- `AGENTS.md` and `README.md` live at the repo root and are outside the `path::symbol` citation form, so wiki pages link to them instead of citing them in `sources`.

## Related

- [Project overview](project-overview.md)
- [Code conventions](code-conventions.md)
