#!/usr/bin/env python3
"""Fetch all-competition game dates from ESPN Site API.

Downloads game schedules for every EPL team across all relevant competitions
(Premier League, FA Cup, League Cup, Champions League, Europa League, Conference
League) and writes one CSV per season to data/espn_games/ and/or upserts
to the database.

These CSVs are loaded as a pandas DataFrame by the training pipeline to provide
accurate rest-day and congestion features that account for midweek European and
cup matches.

Usage:
    # Fetch all seasons (2015 onwards) — CSV + DB
    uv run python scripts/ingest_espn_games.py

    # Fetch a single season
    uv run python scripts/ingest_espn_games.py --season 2024

    # CSV-only (no DB write)
    uv run python scripts/ingest_espn_games.py --skip-db

Live daily refresh of the current season is handled by the
``fetch-espn-games`` scheduled job — this script is for backfill / manual
multi-season ingest / CSV export.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
from pathlib import Path

from odds_core.database import async_session_maker
from odds_core.epl_data_models import EspnGameRecord
from odds_lambda.espn_game_fetcher import SEASONS, EspnGameFetcher
from odds_lambda.storage.espn_game_writer import EspnGameWriter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_espn_games")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "espn_games"

CSV_COLUMNS = [
    "date",
    "team",
    "opponent",
    "competition",
    "game_round",
    "home_away",
    "score_team",
    "score_opponent",
    "status",
]


def _record_to_csv_row(record: EspnGameRecord) -> dict[str, str]:
    return {
        "date": record.date.isoformat().replace("+00:00", "Z"),
        "team": record.team,
        "opponent": record.opponent,
        "competition": record.competition,
        "game_round": record.game_round,
        "home_away": record.home_away,
        "score_team": record.score_team,
        "score_opponent": record.score_opponent,
        "status": record.status,
    }


def write_csv(records: list[EspnGameRecord], season: int) -> Path:
    """Write games to CSV file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    label = SEASONS[season]
    path = DATA_DIR / f"games_{label}.csv"

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(_record_to_csv_row(r) for r in records)

    return path


async def write_db(records: list[EspnGameRecord]) -> int:
    """Write games to database via EspnGameWriter."""
    async with async_session_maker() as session:
        writer = EspnGameWriter(session)
        count = await writer.upsert_games(records)
        await session.commit()
    return count


async def _run(seasons: list[int], skip_db: bool) -> None:
    total_records = 0

    async with EspnGameFetcher() as fetcher:
        season_records: list[tuple[int, list[EspnGameRecord]]] = []
        for season in seasons:
            label = SEASONS[season]
            log.info(f"[{label}] Fetching games...")
            records = await fetcher.fetch_season(season)
            path = write_csv(records, season)
            total_records += len(records)
            log.info(f"[{label}] Wrote {len(records)} games to {path}")
            season_records.append((season, records))

    if not skip_db:
        for season, records in season_records:
            count = await write_db(records)
            log.info(f"[{SEASONS[season]}] Upserted {count} games to database")

    log.info(f"\nTotal: {total_records} games across {len(seasons)} seasons")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ESPN game schedules for EPL teams")
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help="Single season start year (e.g. 2024 for 2024-25). Default: all seasons.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip writing to database (CSV only).",
    )
    args = parser.parse_args()

    if args.season is not None:
        if args.season not in SEASONS:
            parser.error(f"Unknown season: {args.season}. Available: {sorted(SEASONS)}")
        seasons = [args.season]
    else:
        seasons = sorted(SEASONS.keys())

    asyncio.run(_run(seasons, skip_db=args.skip_db))


if __name__ == "__main__":
    main()
