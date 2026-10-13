# Betting Odds Pipeline & Agent

Single-user, sport-agnostic betting odds pipeline with per-sport LLM betting agents (EPL and MLB active, in interactive evaluation; NBA code is deprioritized). The edge being tested is the agent's synthesis of odds, exchange prices, team news and context; an XGBoost CLV model is a supplementary signal.

## Documentation

The wiki is the documentation. Before changing a subsystem, read [wiki/index.md](wiki/index.md), then the area index, then the page, then the code it cites. Page rules are in [wiki/SCHEMA.md](wiki/SCHEMA.md); decisions are in `docs/adr/`.

Every code change keeps the wiki current. `uv run python scripts/wiki_lint.py` (also a pre-commit hook and CI check) lists stale pages, with the changed symbols and the page lines that cite them. Edit those pages to match the new code. A new scheduled job, CLI command or MCP tool needs a page that cites it.

## Constraints

- **Never point `DATABASE_URL` at production** from a local machine. That rules out migrations, fetches, test runs and `odds copy from-prod` without an explicit production URL. See [databases](wiki/foundations/databases-and-environments.md).
- **Deployed AWS state is not in the repo.** `odds scheduler list-jobs --backend aws` is the only record of what runs on AWS. See [jobs and deployment](wiki/scheduling/job-deployment-matrix.md).
- **The Odds API costs quota units.** Check `odds status quota` before bulk or historical fetches. See [quota](wiki/collection/odds-api-quota.md).
- **Pass `--sport` and the sport's market key.** Several CLI defaults are NBA. EPL odds use `1x2` and MLB uses `h2h`. See [sources and keys](wiki/collection/sport-and-market-keys.md).
- **Historical analysis uses only data at or before decision time.** `get_odds_at_time` searches both sides of the target time. See [look-ahead](wiki/modelling/look-ahead-bias.md).
- **Training configs set `target_type` and `target_bookmaker` explicitly.** The code defaults are not the production target. See [CLV target](wiki/modelling/clv-target.md).
- **Use one `async with async_session_maker() as session` per task.** Lambda gets `NullPool` automatically. See [databases](wiki/foundations/databases-and-environments.md).

## Tests

- `uv run pytest tests/unit tests/integration` runs the unit and integration tests. `tests/migration` writes to whatever `DATABASE_URL` points at, so only the deploy workflows run it.
- Unit and integration tests run on in-process PostgreSQL (py-pglite) on a fixed port. Run one pytest process at a time.
- Settings need some `DATABASE_URL` to import. Export a placeholder when no `.env` is present.

## Agent skills

### Issue tracker

GitHub Issues on `Acusick1/odds`, added to the `Personal` project board. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Concept pages in the wiki carry the domain vocabulary, and `docs/adr/` carries decisions. See `docs/agents/domain.md`.
