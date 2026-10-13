---
type: concept
title: Code conventions, timezones and documentation
description: The UTC rule and its remaining naive-datetime sites, async-only DB access, the mechanical gates (ruff, Vale, lychee, wiki lint) versus review-only conventions, logging setup, what is authoritative for what, and where the issue-tracker conventions live.
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
  - pyproject.toml
  - uv.lock
  - .vale.ini
  - scripts/wiki_lint.py
  - docs/adr/0001-wiki-replaces-docs.md
  - docs/agents/domain.md
  - docs/agents/issue-tracker.md#conventions
  - docs/agents/issue-tracker.md#pull-requests-as-a-triage-surface
  - docs/agents/triage-labels.md
status: stable
---

# Code conventions, timezones and documentation

Mechanical gates come first; the rest is convention checked in review.

## Mechanical gates

Pre-commit (`.pre-commit-config.yaml`) runs on every commit; the `Lint` workflow (`.github/workflows/lint.yml`) runs all hooks on every pushed branch except `main`.

| Hook | Scope | Enforces |
|---|---|---|
| ruff `--fix`, ruff-format | all Python | Rules `E W F I B C4 UP DTZ003 DTZ005`, line length 100, target py312, `E501` ignored (root `pyproject.toml`). `UP` enforces `str | None` and `list[T]` |
| vale | `README.md`, `AGENTS.md`, `wiki/` and the Markdown under `docs/adr/` and `docs/agents/` (hook `files` in `.pre-commit-config.yaml`; rules in `.vale.ini`) | `Temporal.Narration` (error on change narration) and `Temporal.TimeRelative` (warning on moment-in-time words) in `.vale/styles/Temporal/` |
| wiki-lint | `wiki/`, `packages/`, `deployment/` | Frontmatter, links, citations, unqualified symbols, index blocks, coverage and staleness (`scripts/wiki_lint.py`; rules in [SCHEMA](../SCHEMA.md)) |
| lychee (Docker) | all `.md` | Links and `#heading` anchors, offline |

## Conventions without tooling

- **Hooks.** Run `uv run pre-commit install` once to enable the hooks on commit, and `uv run pre-commit run --all-files` to run every hook everywhere (`.pre-commit-config.yaml`).
- **Type hints** on every signature. No type checker (mypy, ty, pyright) is configured in pre-commit or CI, so hints are not verified.
- **Async only** for DB work: `AsyncSession` and `async with async_session_maker() as session`, never a sync `Session`, never a session shared across tasks ([Databases](databases-and-environments.md)). The CLI wraps async code in `asyncio.run()`.
- **Model types.** SQLModel for tables, Pydantic `BaseModel` for API responses and configs, `dataclass` for simple containers.
- **Job entry points** are `async def main(ctx: JobContext)` in `odds_lambda/jobs/` ([Jobs](../scheduling/jobs-and-deployment-matrix.md)).
- **Dependencies.** A uv workspace of five packages ([Packages and the odds CLI](packages-and-cli.md)); use `uv add` and `uv sync`, and commit `uv.lock` (workspace declared in `pyproject.toml`).

## Timezones

Rule: use `datetime.now(UTC)` (`from datetime import UTC`), store UTC in the database, convert only for display. Datetime columns are `DateTime(timezone=True)`; the default factory is `packages/odds-core/odds_core/models.py::utc_now`.

Ruff selects only `DTZ003` (`utcnow()`) and `DTZ005` (`now()` without tz). Other naive constructors (`datetime(...)`, `strptime`, `date.today()`, `fromtimestamp`) pass lint, which is how the sites below survive. Review, not tooling, catches them.

Helpers in `packages/odds-core/odds_core/time.py`:

| Helper | Behaviour |
|---|---|
| `packages/odds-core/odds_core/time.py::ensure_utc` | Naive input is assumed to be UTC (`replace`); aware input is converted with `astimezone` |
| `packages/odds-core/odds_core/time.py::parse_api_datetime` | Parses The Odds API strings, rewriting a trailing `Z`, then `packages/odds-core/odds_core/time.py::ensure_utc` |
| `packages/odds-core/odds_core/time.py::to_eastern`, `EASTERN` | America/New_York for NBA game days and injury reports |
| `packages/odds-core/odds_core/time.py::utc_isoformat` | ISO 8601 with a trailing `Z` |

Naive datetimes still produced by current code:

- `AWSEventBridgeBackend._parse_schedule_expression` builds `datetime(year=..., ...)` with no tzinfo from a cron expression (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`); the formatter next to it strips tzinfo deliberately after converting to UTC ([EventBridge backend](../scheduling/scheduler-backends.md#aws-eventbridge-backend)).
- `packages/odds-cli/odds_cli/commands/backtest.py::parse_date` returns a naive datetime from `strptime`, so `odds backtest run --start/--end` are naive, while `packages/odds-cli/odds_cli/commands/quality.py` and `packages/odds-cli/odds_cli/commands/discover.py` attach `tzinfo=UTC` after the same call ([Backtesting](../evaluation/backtesting.md)).
- `packages/odds-lambda/odds_lambda/oddsportal_common.py::parse_odds_timestamp` returns a naive datetime by design (OddsPortal odds-history stamps carry no zone and a wrong year); `packages/odds-lambda/odds_lambda/oddsportal_common.py::fix_odds_timestamp` repairs the year ([OddsPortal](../collection/oddsportal.md)).
- `packages/odds-lambda/odds_lambda/injury_fetcher.py::_to_naive_et` converts to naive Eastern time on purpose because the `nbainjuries` API requires it ([NBA feeds](../collection/nba-injuries-and-stats.md)).
- `packages/odds-cli/odds_cli/commands/pbpstats.py` and `packages/odds-cli/odds_cli/commands/nba_stats.py` derive the NBA season from `date.today()` (local date, not UTC).

`Z`-suffix ISO parsing is inlined in `snapshot_utils`, `espn_fixture_fetcher`, `mlb_stats_fetcher`, `fetch_oddsportal_results`, the MCP EPL tools and `sequence_loader` instead of calling `packages/odds-core/odds_core/time.py::parse_api_datetime`. Time-window correctness for historical analysis is a separate rule: [Decision time and look-ahead bias](../modelling/decision-time-and-look-ahead.md).

## Logging

structlog configured by `packages/odds-core/odds_core/logging_setup.py::configure_logging(settings, json_output)`. Console output is human-readable; the file handler rotates at 10 MB with 5 backups; `json_output=True` renders JSON. An invalid `LOG_LEVEL` falls back to INFO, and an uncreatable log directory falls back to console-only. Config: `LOG_LEVEL`, `LOG_FILE` ([Configuration](configuration.md)).

It is called only from the CLI callback (`packages/odds-cli/odds_cli/main.py::callback`, text) and the Lambda module (`packages/odds-lambda/odds_lambda/lambda_handler.py`, JSON). The MCP server is launched with `fastmcp run` and standalone scripts never call it, so they get structlog defaults and no log file ([odds-mcp server](../agent/mcp-server.md)).

## Documentation authority

The wiki is the documentation ([SCHEMA](../SCHEMA.md)); [README.md](../../README.md) is the human entry point and [AGENTS.md](../../AGENTS.md#documentation) tells coding agents to read the wiki first. Decisions live in `docs/adr/`: `docs/adr/0001-wiki-replaces-docs.md` records why a code-citing wiki replaced the old long-form pages. Agent tooling conventions (issue tracker, triage labels, domain docs) live in `docs/agents/`. Code outranks every page: when a page and code disagree, the code is right and the page is stale.

| Topic | Authoritative source |
|---|---|
| Settings and defaults | `packages/odds-core/odds_core/config.py` ([Configuration](configuration.md)); `.env.example` is the template |
| CLI flags and defaults | each command's `--help` ([odds CLI](packages-and-cli.md#the-odds-cli)) |
| Job and schedule matrix | `deployment/aws/terraform/eventbridge.tf` and `packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP` ([Jobs](../scheduling/jobs-and-deployment-matrix.md)); deployed state only via `odds scheduler list-jobs --backend aws` |
| Alerting and reporting | `packages/odds-core/odds_core/alerts.py` ([Alerts and health](../operations/alerts-and-health.md), [Reporting](../operations/reporting-and-daily-digest.md)) |
| MCP toolset | the `@mcp.tool()` definitions in `packages/odds-mcp/odds_mcp/server.py` ([odds-mcp server](../agent/mcp-server.md)) |
| Agent behaviour | the prompt files in `packages/odds-mcp/agents/` ([Betting agent](../agent/betting-agent.md)) |
| Schema and storage | the model files, chiefly `packages/odds-core/odds_core/models.py` ([Storage](../storage/index.md)) |
| Training data, target, features | `packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig` and `packages/odds-analytics/odds_analytics/feature_groups.py`; evidence in `experiments/results/` ([CLV model findings](../evaluation/clv-model-findings.md)) |
| Agent data sources | [Agent data sources](../agent/agent-data-sources.md) |
| Issue and triage workflow | `docs/agents/` (below) |

The wiki lint (`scripts/wiki_lint.py`) runs as the `wiki-lint` pre-commit hook and in the `Lint` workflow. It checks that cited paths and symbols exist and flags a page as stale when a symbol it cites changes after the page does. Rationale with no recorded source is labelled `(inferred)`. `AGENTS.md` and `README.md` sit at the repo root outside the `path::symbol` form, so wiki pages link to them instead of citing them.

Vale rejects change-narrating words in `README.md`, `AGENTS.md`, `docs/adr/`, `docs/agents/` and `wiki/`; `.vale.ini` turns both rules off for `docs/adr/*.md` because ADRs are point-in-time records. Pages describe the current state.

`docs/agents/domain.md` tells agents to read `wiki/index.md` and the ADRs under `docs/adr/` before exploring and to flag any contradiction of an ADR.

## Issue tracker

Issues live in GitHub Issues on `Acusick1/odds`, added to the `Personal` project board. Open a draft PR whose body contains `Closes #<n>` as soon as the branch has its first commit; linking moves the issue to In Progress and merging moves it to Done. Triage uses five labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. The `gh` command table, sub-issue and wayfinding procedures are in `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md`, which are authoritative and loaded through [AGENTS.md](../../AGENTS.md#agent-skills). External PRs are not a triage surface: `docs/agents/issue-tracker.md` sets "PRs as a request surface" to `no`.

## Hazards

- `.github/workflows/test.yml` and `live-api.yml` pin `setup-python` to 3.11 while every package requires `>=3.12` ([CI/CD workflows](../operations/ci-cd-workflows.md)).
- Naive-datetime sites above are invisible to ruff; adding a `strptime` or `date.today()` call passes lint silently.

## Related

- [Project overview and pivots](project-overview.md)
- [Configuration](configuration.md)
- [Set up, run and test locally](howto-develop-locally.md)
