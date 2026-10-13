---
type: howto
title: Copy production data locally
description: Steps to copy one sport and date range from production into the local database, with the command's options and copy behaviour.
sources:
  - odds_lambda.copy_from_production.copy_from_prod
  - odds_lambda.copy_from_production.ProductionDataCopier
  - odds_cli.commands.copy_from_prod.copy_from_production
  - odds_lambda.storage.readers.OddsReader.get_events_by_date_range
  - odds_lambda.storage.writers.OddsWriter.upsert_event
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
status: stable
---

# Copy production data locally

`odds copy from-prod START END` copies events, odds snapshots and predictions for one sport and date range from the production database into the local database, so local work runs on real data. The CLI wrapper is `odds_cli.commands.copy_from_prod`; the logic is in `odds_lambda.copy_from_production` (`odds_lambda.copy_from_production.copy_from_prod`, `odds_lambda.copy_from_production.ProductionDataCopier`). Environments are described in [Databases and environments](databases-and-environments.md#environments).

## Steps

1. **Export both URLs in the shell.** `PROD_DATABASE_URL` and `LOCAL_DATABASE_URL`; they are `os.getenv` lookups, so `.env` is not read ([variables read outside Settings](configuration.md#variables-read-outside-settings)). With `PROD_DATABASE_URL` unset the command reads `DATABASE_URL` as the source and still prints "Production" (#450); in the standard local setup that copies the local database into itself.
2. **Dry-run first.** Counts what would be copied and writes nothing.

   ```bash
   odds copy from-prod 2025-08-01 2026-04-10 --sport soccer_epl --dry-run
   ```

3. **Run without `--dry-run`.** Add `--overwrite` only to replace events already present locally.
4. **Check the summary.** A per-event failure is logged and counted, and the run continues.

## Command reference

`START` and `END` are positional `YYYY-MM-DD` dates, where `odds backfill` and `odds quality` use `--start`/`--end`; `END` is inclusive through 23:59:59 UTC. Options (`odds_cli.commands.copy_from_prod.copy_from_production`): `--sport/-s` (default `soccer_epl`), `--dry-run`, `--skip-existing/--overwrite` (default skip existing), `--prod-url`, `--local-url`.

- Source URL: `--prod-url`, else `PROD_DATABASE_URL`, else `DATABASE_URL`. Target URL: `--local-url`, else `LOCAL_DATABASE_URL` (required).
- Events come from `odds_lambda.storage.readers.OddsReader.get_events_by_date_range` for the sport and window. With `--skip-existing`, an event already present locally is skipped.
- A copied event is upserted with `odds_lambda.storage.writers.OddsWriter.upsert_event`; each snapshot not already present locally (within a 1 minute tolerance) is re-stored with `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`, which recomputes the normalized odds and tier ([OddsWriter](../storage/odds-writer.md)).
- Predictions are copied one by one; duplicates against `uq_prediction_event_snap_model` are skipped. Predictions keep production's `snapshot_id` while snapshots are re-created with new ids, so a copied prediction can point at a different local snapshot or fail the foreign key (#450).
- A duplicate prediction rolls back the session that holds the event's upsert and snapshots, discarding them (#450).
- `--dry-run` counts events and snapshots it would copy and writes nothing; its prediction count stays 0.

## Related

- [Databases and environments](databases-and-environments.md)
- [The odds CLI](odds-cli.md)
