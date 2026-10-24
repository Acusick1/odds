---
type: reference
title: Code conventions and lint gates
description: The pre-commit hooks and lint workflow, review-only conventions (type hints, async DB, model types, job entry points), and logging setup.
sources:
  - odds_core.logging_setup.configure_logging
  - odds_lambda.lambda_handler.lambda_handler
  - odds_cli.main.callback
  - .pre-commit-config.yaml
  - .githooks/claim-issue.sh
  - .vale/styles/Temporal/Narration.yml
  - .github/workflows/lint.yml
  - pyproject.toml
  - uv.lock
  - .vale.ini
  - scripts/wiki_lint.py
status: stable
---

# Code conventions and lint gates

Mechanical gates come first; the rest is convention checked in review.

## Mechanical gates

Pre-commit (`.pre-commit-config.yaml`) runs on every commit; the `Lint` workflow (`.github/workflows/lint.yml`) runs all hooks on every pushed branch except main. The Python version CI sets up is on [CI/CD workflows](../operations/ci-cd-workflows.md).

| Hook | Scope | Enforces |
|---|---|---|
| ruff `--fix`, ruff-format | all Python | Rules `E W F I B C4 UP DTZ003 DTZ005`, line length 100, target py312, `E501` ignored (root `pyproject.toml`). `UP` enforces `str | None` and `list[T]` |
| vale | [README.md](../../README.md), [AGENTS.md](../../AGENTS.md), `wiki/` and the Markdown under docs (hook `files` in `.pre-commit-config.yaml`; rules in `.vale.ini`) | `Temporal.Narration` (error on change narration) and `Temporal.TimeRelative` (warning on moment-in-time words) in `.vale/styles/Temporal/` |
| wiki-lint | whole repo, every commit (`always_run`; `--no-write --fail-stale`) | Frontmatter, links, machine-specific terms, citations, unqualified symbols, index blocks, coverage and staleness of cited symbols, re-export imports and files (`scripts/wiki_lint.py`; rules in [SCHEMA](../SCHEMA.md)) |
| lychee (Docker) | all `.md` | Links and `#heading` anchors, offline |

The `claim-issue` hook (`.githooks/claim-issue.sh`) is a plain `post-checkout` git hook outside pre-commit, installed as in [Set up and run locally](howto-develop-locally.md#set-up).

Ruff selects only `DTZ003` and `DTZ005` for timezone checks ([Timezones](timezones.md)).

## Conventions without tooling

- **Type hints** on every signature. No type checker (mypy, ty, pyright) is configured in pre-commit or CI, so hints are not verified.
- **Async only** for DB work: `AsyncSession` and `async with async_session_maker() as session`, never a sync `Session`, never a session shared across tasks ([Databases and environments](databases-and-environments.md)). The CLI wraps async code in `asyncio.run()`.
- **Model types.** SQLModel for tables, Pydantic `BaseModel` for API responses and configs, `dataclass` for simple containers.
- **Job entry points** are `async def main(ctx: JobContext)` in `odds_lambda/jobs/` ([Job registry](../scheduling/job-registry.md)).
- **Dependencies.** A uv workspace of five packages ([Workspace packages](workspace-packages.md)); dependencies are managed with `uv add` and `uv sync`, and the resolved `uv.lock` is committed (workspace declared in `pyproject.toml`). Enabling the hooks is a setup step ([Set up and run locally](howto-develop-locally.md)).

## Logging

structlog configured by `odds_core.logging_setup.configure_logging(settings, json_output)`. Console output is human-readable; the file handler rotates at 10 MB with 5 backups; `json_output=True` renders JSON. An invalid `LOG_LEVEL` falls back to INFO, and an uncreatable log directory falls back to console-only. Config: `LOG_LEVEL`, `LOG_FILE` ([Configuration](configuration.md)).

It is called only from the CLI callback (`odds_cli.main.callback`, text) and the Lambda module (`odds_lambda.lambda_handler`, JSON). The MCP server is launched with `fastmcp run` and standalone scripts never call it, so they get structlog defaults and no log file ([odds-mcp server](../agent/mcp-server.md)).

## Related

- [Timezones and time helpers](timezones.md)
- [Documentation and issue tracking](documentation-and-issue-tracking.md)
- [Set up and run locally](howto-develop-locally.md)
