---
type: component
title: Historical odds ingestion scripts
summary: One-off scripts that scrape OddsPortal history and load it plus football-data.co.uk CSVs as Event and OddsSnapshot rows in the live raw_data format.
sources:
  - scripts/scrape_oddsportal.py::scrape_season
  - scripts/ingest_oddsportal.py::ingest_season
  - scripts/ingest_football_data_uk.py::ingest_season
  - scripts/team_names.py::normalize_team
  - docs/DATA_MODELS.md#sportsbook-odds
  - docs/MODELING.md#data-used
  - docs/AGENT_DATA_SOURCES.md#scheduled-collection
verified_at: 7887039
---

# Historical odds ingestion scripts

Three manual scripts in `scripts/` produce the history that trains models ([Data sources](data-sources.md), [CLV model findings](clv-model-findings.md)). They are not scheduled and not under the `odds` CLI.

## Pipeline

1. `scripts/scrape_oddsportal.py` runs OddsHarvester `historic` via `uvx` (a fork branch pinned in `FORK_SPEC`, separate from the live jobs' dependency) and writes `data/external/oddsportal/<prefix>_<season>.json`. Resume is per match: it discovers links, then scrapes only missing or incomplete ones in batches of `BATCH_SIZE`, saving after each. Supports `--proxy-url` and `--output-dir` for geo-restricted books.
2. `scripts/ingest_oddsportal.py` loads those JSON files (`--data-dirs` unions bookmakers across directories), matches or creates `Event` rows, and inserts opening and closing `OddsSnapshot` rows via `oddsportal_common.build_raw_data`. `--relink` re-links game logs and injuries (basketball).
3. `scripts/ingest_football_data_uk.py` downloads `E0.csv` for seasons in `SEASONS` (2015-2016 to 2025-2026) and writes two snapshots per match. Team names go through `scripts/team_names.py::normalize_team`, which delegates to `odds_core.team.normalize_team_name` ([Event matching](event-matching.md)).

Both ingesters first call `find_existing_event` (+/-24 h on normalized teams), so a game seen by one source attaches to the other's event. New ids: `op_<season>_<HOME>_<AWAY>_<date>` and `fduk_<season>_<HOME>_<AWAY>_<date>`. All support `--dry-run`. Idempotency: skip if a snapshot with the same `api_request_id` exists (`oddsportal` for h2h, `oddsportal_<market>` otherwise, `football_data_uk`).

## Snapshot shape

- FDUK opening snapshot is stamped 24 h before kickoff and closing 1 h before: synthetic times, not observed ones. Bookmaker set is `BOOKMAKER_PREFIXES` plus `market_max` / `market_avg`; `raw_data.match_stats` carries post-match stats.
- OddsPortal snapshots use real timestamps with the year corrected (`fix_odds_timestamp`) and `raw_data.source = "oddsportal"`.
- `fetch_tier` comes from `hours_to_tier` ([Fetch tier](fetch-tier.md)).

## Hazards

- **Market key differs by source for football**: FDUK writes `h2h`; OddsPortal ingest writes `1x2` for the `1x2` market (`MARKET_CONFIGS`). Readers must accept both ([Odds reader](odds-reader.md)).
- **Tier boundaries differ from live writes**: FDUK's synthetic 24 h opening gets `early` from `hours_to_tier` (strict `<`), where `tier_utils.calculate_tier` would give `sharp`.
- **FDUK events matched to existing `op_` events get no new event**, and its snapshots are skipped if any `football_data_uk` snapshot exists; only `match_stats` is back-filled (`snapshots_updated`).
- Pinnacle gaps after July 2025 and the 1,810 matched-event count in the docs are data properties; check the database, not the repo.

## Related

- [Historical backfill how-to](howto-historical-backfill.md) (Odds API historical endpoint, a different path)
- [OddsPortal scraper](oddsportal-scraper.md)
- [Odds API quota](odds-api-quota.md)
