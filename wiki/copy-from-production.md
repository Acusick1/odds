---
type: component
title: Copy from production
summary: "odds copy from-prod: what it copies, its options, and the fallback that treats DATABASE_URL as production."
sources:
  - packages/odds-lambda/odds_lambda/copy_from_production.py::copy_from_prod
  - packages/odds-lambda/odds_lambda/copy_from_production.py::ProductionDataCopier
  - packages/odds-cli/odds_cli/commands/copy_from_prod.py::copy_from_production
  - docs/CLI.md#copy-from-production
  - docs/audits/comprehension-audit-2026-06-09.md#4-flagged-for-owner-decision-explicitly-not-prescribed
verified_at: 7887039
---

# Copy from production

`odds copy from-prod START END` copies events, odds snapshots and predictions for one sport and date range from the production database into the local database, so local work runs on real data ([Set up a local environment](howto-local-setup.md)). The CLI wrapper is `packages/odds-cli/odds_cli/commands/copy_from_prod.py`; the logic lives in `packages/odds-lambda/odds_lambda/copy_from_production.py` (`copy_from_prod`, `ProductionDataCopier`).

## Usage

```bash
odds copy from-prod 2025-08-01 2026-04-10 --sport soccer_epl --dry-run
```

`START` and `END` are positional `YYYY-MM-DD` dates; `END` is inclusive through 23:59:59 UTC. Options (`docs/CLI.md`, confirmed in code): `--sport/-s` (default `soccer_epl`), `--dry-run`, `--skip-existing/--overwrite` (default skip existing), `--prod-url`, `--local-url`.

## Behaviour

- Source URL: `--prod-url`, else `PROD_DATABASE_URL`, else `DATABASE_URL`. Target URL: `--local-url`, else `LOCAL_DATABASE_URL` (required).
- Events come from `OddsReader.get_events_by_date_range` for the sport and window. For each event, with `--skip-existing` an event already present locally is skipped.
- A copied event is upserted with `OddsWriter.upsert_event`; each snapshot not already present locally (within a 1 minute tolerance) is re-stored with `OddsWriter.store_odds_snapshot`, which recomputes the normalized odds and tier ([Odds writer](odds-writer.md)).
- Predictions are copied one by one; duplicates against `uq_prediction_event_snap_model` are skipped.
- `--dry-run` counts events and snapshots it would copy and writes nothing; its prediction count stays 0.
- A per-event failure is logged and counted, and the run continues.

## Hazards

- **Production fallback.** With `PROD_DATABASE_URL` unset, `DATABASE_URL` becomes the production source. In the standard local setup that is the local database, so the command copies it into itself while printing "Production" ([Database connections](database-connections.md)). Set `PROD_DATABASE_URL` explicitly.
- Neither `PROD_DATABASE_URL` nor `LOCAL_DATABASE_URL` is read from `.env`: they are `os.getenv` lookups, so they must be exported in the shell ([Configuration](configuration.md#hazards)).
- Copied predictions keep production's `snapshot_id`, while snapshots are re-created locally with new ids, so a prediction's `snapshot_id` can point at a different local snapshot or fail the foreign key (read from the code, not run).
- A duplicate prediction triggers `local_session.rollback()` on the session that also holds that event's uncommitted upsert and snapshot writes, so those writes can be discarded when `--overwrite` meets existing predictions (read from the code, not run).
- The module lives in odds-lambda but is a Rich-console CLI tool, and its `__main__` block ignores most options ([odds-lambda package](package-odds-lambda.md)).

## Related

- [odds CLI](cli-overview.md)
- [Storage layer](storage-layer.md)
