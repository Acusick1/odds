---
type: concept
title: Code conventions, timezones and documentation
description: The UTC rule and its remaining naive-datetime sites, async-only DB access, the mechanical gates (ruff, Vale, lychee) versus review-only conventions, logging setup, which doc is authoritative for what, and where the issue-tracker conventions live.
sources:
  - packages/odds-core/odds_core/logging_setup.py::configure_logging
  - packages/odds-core/odds_core/time.py::ensure_utc
  - packages/odds-core/odds_core/time.py::parse_api_datetime
  - packages/odds-core/odds_core/time.py::utc_isoformat
  - packages/odds-core/odds_core/time.py::EASTERN
  - packages/odds-core/odds_core/models.py::utc_now
  - packages/odds-lambda/odds_lambda/lambda_handler.py::lambda_handler
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::parse_odds_timestamp
  - packages/odds-lambda/odds_lambda/injury_fetcher.py::_to_naive_et
  - packages/odds-cli/odds_cli/main.py::callback
  - packages/odds-cli/odds_cli/commands/backtest.py::parse_date
  - .pre-commit-config.yaml
  - .vale/styles/Temporal/Narration.yml
  - .github/workflows/lint.yml
  - docs/agents/domain.md
  - docs/agents/issue-tracker.md#conventions
  - docs/agents/issue-tracker.md#pull-requests-as-a-triage-surface
  - docs/agents/triage-labels.md
  - docs/DEVELOPMENT.md#dependency-management
  - docs/DEVELOPMENT.md#code-quality
  - docs/DEVELOPMENT.md#logging
  - docs/DEVELOPMENT.md#issue-workflow
  - docs/audits/comprehension-audit-2026-06-09.md#a1-doc-truth-pass-over-the-core-doc-set
verified: "7887039"
status: stable
---

# Code conventions, timezones and documentation

Rules from [AGENTS.md](../../AGENTS.md#code-style) and [DEVELOPMENT.md](../../docs/DEVELOPMENT.md#code-quality). Mechanical gates come first; the rest is convention checked in review.

## Mechanical gates

Pre-commit (`.pre-commit-config.yaml`) runs on every commit; the `Lint` workflow (`.github/workflows/lint.yml`) runs all hooks on every pushed branch except `main`.

| Hook | Scope | Enforces |
|---|---|---|
| ruff `--fix`, ruff-format | all Python | Rules `E W F I B C4 UP DTZ003 DTZ005`, line length 100, target py312, `E501` ignored (root `pyproject.toml`). `UP` enforces `str | None` and `list[T]` |
| vale | `README.md`, `AGENTS.md`, `docs/*.md` | `Temporal.Narration` (error: "previously", "no longer", "was changed"...) and `Temporal.TimeRelative` (warning: "currently", "as of"...) in `.vale/styles/Temporal/` |
| lychee (Docker) | all `.md` | Links and `#heading` anchors, offline |

## Conventions without tooling

- **Type hints** on every signature. No type checker (mypy, ty, pyright) is configured in pre-commit or CI, so hints are not verified.
- **Async only** for DB work: `AsyncSession` and `async with async_session_maker() as session`, never a sync `Session`, never a session shared across tasks ([Databases](databases-and-environments.md)). The CLI wraps async code in `asyncio.run()`.
- **Model types.** SQLModel for tables, Pydantic `BaseModel` for API responses and configs, `dataclass` for simple containers.
- **Job entry points** are `async def main(ctx: JobContext)` in `odds_lambda/jobs/` ([Jobs](../scheduling/jobs-and-deployment-matrix.md)).
- **Dependencies.** A uv workspace of five packages ([Packages and the odds CLI](packages-and-cli.md)); use `uv add` and `uv sync`, and commit `uv.lock` ([DEVELOPMENT.md](../../docs/DEVELOPMENT.md#dependency-management)).

## Timezones

Rule ([AGENTS.md](../../AGENTS.md#timezone-handling-critical)): use `datetime.now(UTC)` (`from datetime import UTC`), store UTC in the database, convert only for display. Datetime columns are `DateTime(timezone=True)`; the default factory is `packages/odds-core/odds_core/models.py::utc_now`.

Ruff selects only `DTZ003` (`utcnow()`) and `DTZ005` (`now()` without tz). Other naive constructors (`datetime(...)`, `strptime`, `date.today()`, `fromtimestamp`) pass lint, which is how the sites below survive. Review, not tooling, catches them.

Helpers in `packages/odds-core/odds_core/time.py`:

| Helper | Behaviour |
|---|---|
| `ensure_utc` | Naive input is assumed to be UTC (`replace`); aware input is converted with `astimezone` |
| `parse_api_datetime` | Parses The Odds API strings, rewriting a trailing `Z`, then `ensure_utc` |
| `to_eastern`, `EASTERN` | America/New_York for NBA game days and injury reports |
| `utc_isoformat` | ISO 8601 with a trailing `Z` |

Naive datetimes still produced by current code:

- `AWSEventBridgeBackend._parse_schedule_expression` builds `datetime(year=..., ...)` with no tzinfo from a cron expression (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`); the formatter next to it strips tzinfo deliberately after converting to UTC ([EventBridge backend](../scheduling/scheduler-backends.md#aws-eventbridge-backend)).
- `packages/odds-cli/odds_cli/commands/backtest.py::parse_date` returns a naive datetime from `strptime`, so `odds backtest run --start/--end` are naive, while `quality.py` and `discover.py` attach `tzinfo=UTC` after the same call ([Backtesting](../evaluation/backtesting.md)).
- `packages/odds-lambda/odds_lambda/oddsportal_common.py::parse_odds_timestamp` returns a naive datetime by design (OddsPortal odds-history stamps carry no zone and a wrong year); `fix_odds_timestamp` repairs the year ([OddsPortal](../collection/oddsportal.md)).
- `packages/odds-lambda/odds_lambda/injury_fetcher.py::_to_naive_et` converts to naive Eastern time on purpose because the `nbainjuries` API requires it ([NBA feeds](../collection/nba-injuries-and-stats.md)).
- `packages/odds-cli/odds_cli/commands/pbpstats.py` and `nba_stats.py` derive the NBA season from `date.today()` (local date, not UTC).

`Z`-suffix ISO parsing is inlined in `snapshot_utils`, `espn_fixture_fetcher`, `mlb_stats_fetcher`, `fetch_oddsportal_results`, the MCP EPL tools and `sequence_loader` instead of calling `parse_api_datetime`. Time-window correctness for historical analysis is a separate rule: [Decision time and look-ahead bias](../modelling/decision-time-and-look-ahead.md).

## Logging

structlog configured by `packages/odds-core/odds_core/logging_setup.py::configure_logging(settings, json_output)`. Console output is human-readable; the file handler rotates at 10 MB with 5 backups; `json_output=True` renders JSON. An invalid `LOG_LEVEL` falls back to INFO, and an uncreatable log directory falls back to console-only. Config: `LOG_LEVEL`, `LOG_FILE` ([Configuration](configuration.md)).

It is called only from the CLI callback (`packages/odds-cli/odds_cli/main.py::callback`, text) and the Lambda module (`packages/odds-lambda/odds_lambda/lambda_handler.py`, JSON). The MCP server is launched with `fastmcp run` and standalone scripts never call it, so they get structlog defaults and no log file ([odds-mcp server](../agent/mcp-server.md)).

## Documentation authority

The doc index is the table in [README.md](../../README.md#documentation); [AGENTS.md](../../AGENTS.md#documentation) points there. Code outranks every doc: when a doc and code disagree, the code is right.

| Topic | Authoritative source |
|---|---|
| Settings and defaults | `packages/odds-core/odds_core/config.py` ([Configuration](configuration.md)); `.env.example` is the template |
| CLI flags and defaults | each command's `--help`; `docs/CLI.md` is the reference ([odds CLI](packages-and-cli.md#the-odds-cli)) |
| Job and schedule matrix | `docs/DEPLOYMENT.md` ([Jobs](../scheduling/jobs-and-deployment-matrix.md)); deployed state only via `odds scheduler list-jobs --backend aws` |
| Alerting and reporting | `docs/REPORTING.md` ([Reporting](../operations/reporting-and-daily-digest.md)) |
| MCP toolset | the `@mcp.tool()` definitions in `packages/odds-mcp/odds_mcp/server.py` ([odds-mcp server](../agent/mcp-server.md)) |
| Agent behaviour | the prompt files in `packages/odds-mcp/agents/`; `docs/BETTING_AGENT.md` summarises them ([Betting agent](../agent/betting-agent.md)) |
| Schema and storage | `docs/DATABASE.md` and the model files in `odds_core/` |
| Training data, target, features | `docs/DATA_MODELS.md`; evidence in `docs/MODELING.md` ([CLV model findings](../evaluation/clv-model-findings.md)) |
| Agent data sources | `docs/AGENT_DATA_SOURCES.md` ([Agent data sources](../agent/agent-data-sources.md)) |
| Issue and triage workflow | `docs/agents/` (below) |

`docs/audits/comprehension-audit-2026-06-09.md` is a dated snapshot from commit `1296b5f`; its line numbers are stale, and its A1 table of doc/code mismatches is partly out of date and partly live. Re-verify a row against code before relying on it.

Doc-truth lints: Vale rejects change-narrating words in `README.md`, `AGENTS.md` and `docs/`, and `.vale.ini` turns both rules off for `docs/adr/*.md` because ADRs are point-in-time records; lychee checks Markdown links and anchors offline. Docs describe the current state.

`docs/agents/domain.md` tells agents to read `GLOSSARY.md` and `docs/adr/` before exploring and to flag any contradiction of an ADR. Both are created lazily by the `domain-modeling` skill; neither exists at this commit, and `domain.md` says to proceed silently when they are absent.

The wiki is code-cited and linted: pages cite code by `path::symbol`, and the wiki lint script under `scripts/` checks that paths, symbols, links and anchors still resolve ([schema](../SCHEMA.md)). Rationale with no recorded source is labelled `(inferred)`. `AGENTS.md` and `README.md` sit at the repo root outside the `path::symbol` form, so wiki pages link to them instead of citing them.

## Issue tracker

Issues live in GitHub Issues on `Acusick1/odds`, added to the `Personal` project board. Open a draft PR whose body contains `Closes #<n>` as soon as the branch has its first commit; linking moves the issue to In Progress and merging moves it to Done. Triage uses five labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. The `gh` command table, sub-issue and wayfinding procedures are in `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md`, which are authoritative and loaded through [AGENTS.md](../../AGENTS.md#agent-skills). External PRs are not a triage surface: the doc sets "PRs as a request surface" to `no`.

## Hazards

- `.github/workflows/test.yml` and `live-api.yml` pin `setup-python` to 3.11 while every package requires `>=3.12` ([CI/CD workflows](../operations/ci-cd-workflows.md)).
- The Vale hook does not cover `wiki/`; only lychee does.
- Naive-datetime sites above are invisible to ruff; adding a `strptime` or `date.today()` call passes lint silently.

## Related

- [Project overview and pivots](project-overview.md)
- [Configuration](configuration.md)
- [Set up, run and test locally](howto-develop-locally.md)
