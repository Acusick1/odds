---
type: howto
title: Copy production data locally
description: Steps to copy one sport and date range from production into the local database, with the command's options and copy behaviour.
sources:
  - odds_lambda.copy_from_production.copy_from_prod
  - odds_lambda.copy_from_production.ProductionDataCopier
  - odds_cli.commands.copy_from_prod.copy_from_production
  - odds_lambda.storage.readers.OddsReader.get_events_by_date_range
status: stable
---

# Copy production data locally

`odds copy from-prod START END` copies events (in kick-off order), odds snapshots and predictions for one sport and date range from the production database into the local database, so local work runs on real data. The CLI wrapper is `odds_cli.commands.copy_from_prod`; the logic is in `odds_lambda.copy_from_production` (`odds_lambda.copy_from_production.copy_from_prod`, `odds_lambda.copy_from_production.ProductionDataCopier`). Environments are described in [Databases and environments](databases-and-environments.md#environments).

## Steps

1. **Export both URLs in the shell.** `PROD_DATABASE_URL` and `LOCAL_DATABASE_URL`; they are `os.getenv` lookups, so `.env` is not read ([variables read outside Settings](configuration.md#variables-read-outside-settings)). On existing machines `PROD_DATABASE_URL` names the Neon archive, which holds no data collected after the move to the mini-PC; to copy current data, point it (or `--prod-url`) at the mini-PC database ([environments](databases-and-environments.md#environments)). With neither `PROD_DATABASE_URL` nor `--prod-url` set the command fails.
2. **Dry-run first.** Counts what would be copied and writes nothing.

   ```bash
   odds copy from-prod 2025-08-01 2026-04-10 --sport soccer_epl --dry-run
   ```

3. **Run without `--dry-run`.** Events already present locally are skipped; add `--overwrite` to replace them, which deletes each replaced event's local snapshots, odds and predictions, including any that exist only locally, and copies production's. Local events for the same match under another id are left untouched, and the summary counts them; merge them with `odds events merge-twins` ([Event merge](../storage/event-merge.md)).
4. **Check the summary.** A failed event is rolled back, listed by id and counted in `events_failed`, and the run continues; the command exits 1 if any event failed, so rerun after fixing the cause.

## Related

- [What the copier does and its options](databases-and-environments.md#copying-production-data)

- [Databases and environments](databases-and-environments.md)
- [The odds CLI](odds-cli.md)
