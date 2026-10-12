---
type: concept
title: Code conventions and quality gates
summary: "Type hints, async-only DB access, model-type choice, the uv workspace, which gates are mechanical (ruff, Vale, lychee) and which are convention, and structlog logging."
sources:
  - packages/odds-core/odds_core/logging_setup.py::configure_logging
  - packages/odds-lambda/odds_lambda/lambda_handler.py::lambda_handler
  - packages/odds-cli/odds_cli/main.py::callback
  - .pre-commit-config.yaml
  - .vale/styles/Temporal/Narration.yml
  - .github/workflows/lint.yml
  - docs/DEVELOPMENT.md#dependency-management
  - docs/DEVELOPMENT.md#code-quality
  - docs/DEVELOPMENT.md#logging
verified_at: 7887039
---

# Code conventions and quality gates

Rules from [AGENTS.md](../AGENTS.md#code-style) and [DEVELOPMENT.md](../docs/DEVELOPMENT.md#code-quality). Mechanical gates come first; the rest is convention checked in review.

## Mechanical gates

Pre-commit (`.pre-commit-config.yaml`) runs on every commit; the `Lint` workflow (`.github/workflows/lint.yml`) runs all hooks on every pushed branch except `main`.

| Hook | Scope | Enforces |
|---|---|---|
| ruff `--fix`, ruff-format | all Python | Rules `E W F I B C4 UP DTZ003 DTZ005`, line length 100, target py312, `E501` ignored (root `pyproject.toml`). `UP` enforces `str | None` and `list[T]` |
| vale | `README.md`, `AGENTS.md`, `docs/*.md` | `Temporal.Narration` (error: "previously", "no longer", "was changed"...) and `Temporal.TimeRelative` (warning: "currently", "as of"...) in `.vale/styles/Temporal/` |
| lychee (Docker) | all `.md` | Links and `#heading` anchors, offline |

## Conventions without tooling

- **Type hints** on every signature. No type checker (mypy, ty, pyright) is configured in pre-commit or CI, so hints are not verified.
- **Async only** for DB work: `AsyncSession` and `async with async_session_maker() as session`, never a sync `Session`, never a session shared across tasks ([Database connections](database-connections.md)). The CLI wraps async code in `asyncio.run()`.
- **Model types.** SQLModel for tables, Pydantic `BaseModel` for API responses and configs, `dataclass` for simple containers.
- **Job entry points** are `async def main(ctx: JobContext)` in `odds_lambda/jobs/` ([Job registry](job-registry.md)).
- **Timestamps** follow [Timezone handling](timezone-handling.md); `DTZ003` and `DTZ005` are the only DTZ rules on.
- **Dependencies.** A uv workspace of five packages; use `uv add` and `uv sync`, and commit `uv.lock` ([DEVELOPMENT.md](../docs/DEVELOPMENT.md#dependency-management)).

## Logging

structlog configured by `packages/odds-core/odds_core/logging_setup.py::configure_logging(settings, json_output)`. Console output is human-readable; the file handler rotates at 10 MB with 5 backups; `json_output=True` renders JSON. An invalid `LOG_LEVEL` falls back to INFO, and an uncreatable log directory falls back to console-only. Config: `LOG_LEVEL`, `LOG_FILE` ([Configuration](configuration.md)).

## Hazards

- `configure_logging` is called only from the CLI callback (`packages/odds-cli/odds_cli/main.py::callback`, text) and the Lambda module (`packages/odds-lambda/odds_lambda/lambda_handler.py`, JSON). The MCP server (`fastmcp run`) and standalone scripts never call it, so they get structlog defaults and no log file.
- `.github/workflows/test.yml` and `live-api.yml` pin `setup-python` to 3.11 while every package requires `>=3.12` ([CI/CD workflows](ci-cd-workflows.md)).
- The Vale hook does not cover `wiki/`; only lychee does.

## Related

- [Documentation map](documentation-map.md) — what the doc lints protect
- [Project overview](project-overview.md) — stack
