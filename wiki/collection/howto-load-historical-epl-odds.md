---
type: howto
title: Load historical EPL odds
description: Steps to scrape OddsPortal, ingest it, then ingest football-data.co.uk.
sources:
  - scripts/scrape_oddsportal.py
  - scripts/ingest_oddsportal.py
  - scripts/ingest_football_data_uk.py
mentions:
  - odds_core.models.Event
status: stable
---

# Load historical EPL odds

The scripts load `odds_core.models.Event` and `odds_core.models.OddsSnapshot` ([columns and constraints](../storage/odds-tables.md#odds_snapshots)) rows only, so the events they load have no normalized `odds` or `prices` rows and their snapshots carry no source, market or time basis and backtests skip them (#435). What each script reads, matches and writes is on [historical-ingest-scripts](historical-ingest-scripts.md). Run against the local database only: never set `DATABASE_URL` to production ([databases-and-environments](../foundations/databases-and-environments.md#environments)).

1. Scrape OddsPortal history.
   ```bash
   uv run python scripts/scrape_oddsportal.py <league and season options>
   ```
   - `--help` lists the options.
   - It writes `data/external/oddsportal/<prefix>_<season>.json`.
   - A rerun resumes per match, scraping only missing or incomplete ones, so an interrupted run is safe to repeat.
   - Pass `--proxy-url` for books that are geo-restricted.
2. Ingest the OddsPortal files first.
   ```bash
   uv run python scripts/ingest_oddsportal.py --sport soccer --seasons 2023-2024 --dry-run
   ```
   - `--sport soccer` is required: the default sport is basketball.
   - Pass `--seasons` with one or more seasons, or `--all` for the script's full season list.
   - Check the dry-run counts, then run without `--dry-run`.
   - OddsPortal goes first because it creates the `op_<season>_*` events that the FDUK ingest attaches to.
3. Ingest football-data.co.uk.
   ```bash
   uv run python scripts/ingest_football_data_uk.py --seasons 2023-2024 --dry-run
   ```
   - Use the same seasons as step 2, or `--all`. Check the dry-run counts, then run without `--dry-run`.
   - It downloads `E0.csv` per season and writes an opening and a closing snapshot per match, except that the opening snapshot is skipped when its documented deadline is at or after kick-off.
   - Matches that already have an `op_` event of the same sport within 24 hours get no new event; their snapshots are still written.
   - Prices are stored as the CSV decimal (`price_decimal`). Snapshots loaded before the decimal price migration keep the old American `price` until this script is re-run ([Price encodings](../foundations/odds-math.md#price-encodings)).
4. Rerun any step to fill gaps.
   - OddsPortal snapshots are skipped if one with the same `api_request_id` already exists. FDUK snapshots are replaced from the CSV when it changed, so reruns do not duplicate and pick up refreshed files; a rerun also fills the scores of matched events that have none. Each snapshot's `raw_data.timing.basis` is an `odds_core.snapshot_sources.FdukTimeBasis` value.
5. Check the load with the database, not the repo: event and snapshot counts per season are data facts.

## Related

- [Historical ingest scripts](historical-ingest-scripts.md)
- [Backfill Odds API history](howto-backfill-odds-api-history.md), the NBA-only paid route
