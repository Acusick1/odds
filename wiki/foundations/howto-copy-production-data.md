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

## Related

- [What the copier does and its options](databases-and-environments.md#copying-production-data)

- [Databases and environments](databases-and-environments.md)
- [The odds CLI](odds-cli.md)
