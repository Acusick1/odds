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

`odds copy from-prod START END` copies events (listed in kick-off order, then event ID, by `odds_lambda.storage.readers.OddsReader.get_events_by_date_range`, and matched to local events by their external ids) with their external ids and health-check acknowledgements, and their odds snapshots and price rows, for one sport and date range from the production database into the local database, so local work runs on real data. Snapshot payloads are not copied. Both databases must be at the same migration, because the copier reads and writes every model column ([what it copies](databases-and-environments.md#copying-production-data)). The CLI wrapper is `odds_cli.commands.copy_from_prod`; the logic is in `odds_lambda.copy_from_production` (`odds_lambda.copy_from_production.copy_from_prod`, `odds_lambda.copy_from_production.ProductionDataCopier`). Environments are described in [Databases and environments](databases-and-environments.md#environments).

## Steps

1. **Set both URLs.** `PROD_DATABASE_URL` and `LOCAL_DATABASE_URL`, in `.env` or the shell, or pass `--prod-url` and `--local-url` ([Configuration](configuration.md#sections)). On existing machines `PROD_DATABASE_URL` names the Neon archive, which holds no data collected after the move to the production host; to copy current data, point it (or `--prod-url`) at the production database ([environments](databases-and-environments.md#environments)). With neither the setting nor the flag for either URL, the command fails.
2. **Dry-run first.** Counts what would be copied and writes nothing.

   ```bash
   odds copy from-prod 2025-08-01 2026-04-10 --sport soccer_epl --dry-run
   ```

3. **Run without `--dry-run`.** Events already present locally, found by their external ids (else by sport, teams and kick-off when the local event holds no different id from the same source), are skipped; add `--overwrite` to replace them, which deletes each replaced event's local snapshots and price rows, including any that exist only locally, and copies production's. An event whose ids several local events hold fails until they are merged. Other local events for the same match are left untouched, and the summary counts them; merge them with `odds events merge-twins` ([Event merge](../storage/event-merge.md)).
4. **Check the summary.** A failed event is rolled back, listed by id and counted in `events_failed`, and the run continues; the command exits 1 if any event failed, so rerun after fixing the cause.

## Related

- [What the copier does and its options](databases-and-environments.md#copying-production-data)

- [Databases and environments](databases-and-environments.md)
- [The odds CLI](odds-cli.md)
