#!/usr/bin/env python3
"""Ingest football-data.co.uk EPL CSVs into the database.

Downloads CSVs from football-data.co.uk and writes Event + OddsSnapshot records
following the same raw_data format as ingest_oddsportal.py, so the training
pipeline reads them via extract_odds_from_snapshot() without changes.

Usage:
    # Download and ingest all seasons (dry-run first)
    uv run python scripts/ingest_football_data_uk.py --all --dry-run

    # Ingest specific seasons
    uv run python scripts/ingest_football_data_uk.py --seasons 2024-2025

    # Ingest from already-downloaded CSVs
    uv run python scripts/ingest_football_data_uk.py --all --no-download

    # Download only (no database writes)
    uv run python scripts/ingest_football_data_uk.py --all --download-only
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import logging
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.odds_math import PRICE_DECIMAL_KEY
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotSource
from odds_core.team import team_abbrev
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.oddsportal_common import (
    IngestionStats,
    find_existing_event,
    hours_to_tier,
)
from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from team_names import normalize_team as _normalize_team_central

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_football_data_uk")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "football_data_uk"
BASE_URL = "https://www.football-data.co.uk/mmz4281"
SOURCE_TAG = SnapshotSource.FOOTBALL_DATA_UK.value
SPORT_KEY = "soccer_epl"
SPORT_TITLE = "EPL"
EVENT_ID_PREFIX = "fduk"
MARKET_KEY = "1x2"
CLOSING_TIER = FetchTier.CLOSING.value
OPENING_MIN_TIER = FetchTier.PREGAME.value
KICKOFF_MISMATCH_TOLERANCE = timedelta(minutes=15)

# Season label → URL path segment
SEASONS: dict[str, str] = {
    "2015-2016": "1516",
    "2016-2017": "1617",
    "2017-2018": "1718",
    "2018-2019": "1819",
    "2019-2020": "1920",
    "2020-2021": "2021",
    "2021-2022": "2122",
    "2022-2023": "2223",
    "2023-2024": "2324",
    "2024-2025": "2425",
    "2025-2026": "2526",
    "2026-2027": "2627",
}

# Bookmaker column prefixes → pipeline bookmaker key.
# Each entry maps (home_col, draw_col, away_col) derived from prefix + H/D/A.
BOOKMAKER_PREFIXES: dict[str, str] = {
    "PS": "pinnacle",
    "B365": "bet365",
    "BW": "bwin",
    "BFE": "betfair_exchange",
    "BF": "betfair_sportsbook",
    "IW": "interwetten",
    "WH": "williamhill",
    "VC": "betvictor",
    "BV": "betvictor",
    "BFD": "betfred",
    "PP": "paddypower",
    "SKB": "skybet",
    "BMGM": "betmgm",
    "CL": "coral",
    "LB": "ladbrokes",
    "1XB": "onexbet",
}

# Aggregate/market columns — stored as pseudo-bookmakers for analysis.
AGGREGATE_PREFIXES: dict[str, str] = {
    "Max": "market_max",
    "Avg": "market_avg",
    "BbMx": "market_max",
    "BbAv": "market_avg",
}

UK_TZ = ZoneInfo("Europe/London")
DEFAULT_KICKOFF = time(15, 0)


@dataclass(frozen=True)
class CollectionSlot:
    """A weekly deadline by which FDUK has collected its opening odds (UK local time)."""

    weekday: int
    latest: time


# football-data.co.uk/matches.php: weekend fixtures are collected on Friday afternoons
# generally not later than 17:00, midweek fixtures on Tuesdays not later than 13:00.
WEEKEND_SLOT = CollectionSlot(weekday=4, latest=time(17, 0))
MIDWEEK_SLOT = CollectionSlot(weekday=1, latest=time(13, 0))
MIDWEEK_WEEKDAYS = frozenset({1, 2, 3})


@dataclass(frozen=True)
class Kickoff:
    commence_time: datetime
    time_assumed: bool


@dataclass
class FdukStats(IngestionStats):
    openings_withheld: int = 0
    kickoff_mismatches: int = 0
    results_filled: int = 0


def normalize_team(name: str) -> str:
    """Normalize football-data.co.uk team name to pipeline canonical form."""
    return _normalize_team_central(name)


def _season_url(season: str) -> str:
    """Build CSV download URL for a season."""
    code = SEASONS[season]
    return f"{BASE_URL}/{code}/E0.csv"


def _csv_path(season: str) -> Path:
    """Local CSV storage path for a season."""
    code = SEASONS[season]
    return DATA_DIR / f"E0_{code}.csv"


def download_csv(season: str) -> Path:
    """Download a season CSV from football-data.co.uk."""
    url = _season_url(season)
    path = _csv_path(season)
    path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url, timeout=30) as resp:  # noqa: S310
        path.write_bytes(resp.read())

    log.info(f"  Downloaded {url} → {path}")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a football-data.co.uk CSV file."""
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        content = raw.decode("latin-1")
    reader = csv.DictReader(io.StringIO(content))
    return [row for row in reader if row.get("HomeTeam")]


def parse_kickoff(row: dict[str, str]) -> Kickoff:
    """Parse the match date and UK-local kick-off time into a UTC instant.

    Date format is DD/MM/YYYY or DD/MM/YY. Seasons before 2019-20 have no Time
    column; those kick-offs are assumed to be 15:00 and flagged.
    """
    date_str = row["Date"]
    time_str = (row.get("Time") or "").strip()
    fmt = "%d/%m/%Y" if len(date_str.split("/")[-1]) == 4 else "%d/%m/%y"
    parsed_date = datetime.strptime(date_str, fmt).date()
    kickoff_time = datetime.strptime(time_str, "%H:%M").time() if time_str else DEFAULT_KICKOFF
    local = datetime.combine(parsed_date, kickoff_time, tzinfo=UK_TZ)
    return Kickoff(commence_time=local.astimezone(UTC), time_assumed=not time_str)


def _slot_instant(slot: CollectionSlot, on: date) -> datetime:
    return datetime.combine(on, slot.latest, tzinfo=UK_TZ).astimezone(UTC)


def opening_snapshot_time(commence_time: datetime) -> datetime | None:
    """Latest documented collection time of the FDUK opening odds for a match.

    Midweek (Tue-Thu) kick-offs use the Tuesday deadline and all others the Friday
    deadline. Returns None when that deadline is at or after kick-off: the opening
    price then has no documented time before the match, so it is not stamped.
    """
    local = commence_time.astimezone(UK_TZ)
    slot = MIDWEEK_SLOT if local.weekday() in MIDWEEK_WEEKDAYS else WEEKEND_SLOT
    days_back = (local.weekday() - slot.weekday) % 7
    deadline = _slot_instant(slot, local.date() - timedelta(days=days_back))
    return deadline if deadline < commence_time else None


def opening_fetch_tier(hours_before: float) -> str:
    """Tier for an opening snapshot; never the closing tier, which selects closing prices."""
    tier = hours_to_tier(hours_before)
    return OPENING_MIN_TIER if tier == CLOSING_TIER else tier


def _safe_float(value: str | None) -> float | None:
    """Parse a float from CSV, returning None for empty/invalid values."""
    if not value or not value.strip():
        return None
    try:
        f = float(value)
        return f if not math.isnan(f) else None
    except ValueError:
        return None


def _safe_int(value: str | None) -> int | None:
    """Parse an int from CSV, returning None for empty/invalid values."""
    if not value or not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _extract_three_way_odds(
    row: dict[str, str],
    prefix: str,
    bk_key: str,
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    """Extract three-way match odds for a bookmaker from a CSV row."""
    home_odds = _safe_float(row.get(f"{prefix}H"))
    draw_odds = _safe_float(row.get(f"{prefix}D"))
    away_odds = _safe_float(row.get(f"{prefix}A"))

    if home_odds is None or draw_odds is None or away_odds is None:
        return None
    if home_odds <= 1.0 or draw_odds <= 1.0 or away_odds <= 1.0:
        return None

    return {
        "key": bk_key,
        "title": bk_key,
        "last_update": "",  # placeholder, set at snapshot level
        "markets": [
            {
                "key": MARKET_KEY,
                "outcomes": [
                    {"name": home_team, PRICE_DECIMAL_KEY: home_odds},
                    {"name": "Draw", PRICE_DECIMAL_KEY: draw_odds},
                    {"name": away_team, PRICE_DECIMAL_KEY: away_odds},
                ],
            }
        ],
    }


# Match stats columns: (csv_column, json_key) pairs.
# These are post-match stats from the FDUK CSVs stored in raw_data for
# downstream rolling feature computation.
MATCH_STATS_COLUMNS: list[tuple[str, str]] = [
    ("HS", "home_shots"),
    ("AS", "away_shots"),
    ("HST", "home_shots_on_target"),
    ("AST", "away_shots_on_target"),
    ("HC", "home_corners"),
    ("AC", "away_corners"),
    ("HF", "home_fouls"),
    ("AF", "away_fouls"),
    ("HY", "home_yellow_cards"),
    ("AY", "away_yellow_cards"),
    ("HR", "home_red_cards"),
    ("AR", "away_red_cards"),
    ("HTHG", "home_ht_goals"),
    ("HTAG", "away_ht_goals"),
]


def _extract_match_stats(row: dict[str, str]) -> dict[str, Any]:
    """Extract match stats from a CSV row. Returns dict with non-null int stats and referee."""
    stats: dict[str, Any] = {}
    for csv_col, json_key in MATCH_STATS_COLUMNS:
        val = _safe_int(row.get(csv_col))
        if val is not None:
            stats[json_key] = val

    referee = row.get("Referee", "").strip()
    if referee:
        stats["referee"] = referee

    return stats


def _build_snapshot_raw_data(
    row: dict[str, str],
    home_team: str,
    away_team: str,
    *,
    use_closing: bool,
    kickoff_time_assumed: bool = False,
) -> dict[str, Any] | None:
    """Build raw_data dict from a CSV row for one timing (opening or closing).

    Opening odds: columns like PSH, PSD, PSA, B365H, B365D, B365A
    Closing odds: columns like PSCH, PSCD, PSCA, B365CH, B365CD, B365CA
    """
    bookmakers: list[dict[str, Any]] = []

    all_prefixes = {**BOOKMAKER_PREFIXES, **AGGREGATE_PREFIXES}

    seen: set[str] = set()
    for prefix, bk_key in all_prefixes.items():
        if use_closing:
            col_prefix = f"{prefix}C"
        else:
            col_prefix = prefix

        if bk_key in seen:
            continue
        bk_data = _extract_three_way_odds(row, col_prefix, bk_key, home_team, away_team)
        if bk_data:
            seen.add(bk_key)
            bookmakers.append(bk_data)

    if not bookmakers:
        return None

    basis = FdukTimeBasis.KICKOFF if use_closing else FdukTimeBasis.COLLECTION_DEADLINE
    raw_data: dict[str, Any] = {
        "bookmakers": bookmakers,
        "source": SOURCE_TAG,
        "timing": {"basis": basis.value, "kickoff_time_assumed": kickoff_time_assumed},
    }

    match_stats = _extract_match_stats(row)
    if match_stats:
        raw_data["match_stats"] = match_stats

    return raw_data


def build_event_id(
    season: str,
    home_team: str,
    away_team: str,
    commence_time: datetime,
) -> str:
    """Generate deterministic event ID for football-data.co.uk events."""
    home_abbrev = team_abbrev(home_team)
    away_abbrev = team_abbrev(away_team)
    return (
        f"{EVENT_ID_PREFIX}_{season}_{home_abbrev}_{away_abbrev}_{commence_time.date().isoformat()}"
    )


async def ingest_season(
    season: str,
    *,
    dry_run: bool = False,
    no_download: bool = False,
) -> FdukStats:
    """Ingest one season of football-data.co.uk data."""
    stats = FdukStats()

    path = _csv_path(season)
    if not no_download:
        try:
            path = download_csv(season)
        except OSError as e:
            log.error(f"  {season}: download failed — {e}")
            stats.errors.append(f"{season}: download failed — {e}")
            return stats
    elif not path.exists():
        log.error(f"  {season}: CSV not found at {path}")
        stats.errors.append(f"{season}: CSV not found at {path}")
        return stats

    rows = read_csv(path)
    if not rows:
        log.error(f"  {season}: no data rows in CSV")
        return stats

    stats.games_loaded = len(rows)
    log.info(f"  {season}: loaded {len(rows)} matches")

    if dry_run:
        _report_dry_run(season, rows)
        return stats

    async with async_session_maker() as session:
        for row in rows:
            await ingest_row(session, row, season, stats)

        await session.commit()

    return stats


async def ingest_row(
    session: AsyncSession,
    row: dict[str, str],
    season: str,
    stats: FdukStats,
) -> None:
    """Ingest one CSV row in a savepoint so a failure leaves the event's rows untouched."""
    try:
        async with session.begin_nested():
            await _ingest_one_match(session, row, season, stats)
    except Exception as e:
        home = row.get("HomeTeam", "?")
        away = row.get("AwayTeam", "?")
        msg = f"{home} vs {away}: {e}"
        stats.errors.append(msg)
        log.error(f"  Error: {msg}")


def _report_dry_run(season: str, rows: list[dict[str, str]]) -> None:
    """Log dry-run summary for a season."""
    has_pinnacle_opening = sum(1 for r in rows if _safe_float(r.get("PSH")))
    has_pinnacle_closing = sum(1 for r in rows if _safe_float(r.get("PSCH")))
    has_bet365_closing = sum(1 for r in rows if _safe_float(r.get("B365CH")))
    has_betfair = sum(1 for r in rows if _safe_float(r.get("BFEH")))

    log.info(
        f"  {season}: Pinnacle opening={has_pinnacle_opening}, "
        f"Pinnacle closing={has_pinnacle_closing}, "
        f"bet365 closing={has_bet365_closing}, "
        f"Betfair={has_betfair}"
    )


async def _ingest_one_match(
    session: AsyncSession,
    row: dict[str, str],
    season: str,
    stats: FdukStats,
) -> None:
    """Ingest a single CSV row as Event + OddsSnapshots."""
    home_raw = row.get("HomeTeam", "")
    away_raw = row.get("AwayTeam", "")
    if not home_raw or not away_raw:
        stats.games_skipped += 1
        return

    home_team = normalize_team(home_raw)
    away_team = normalize_team(away_raw)
    kickoff = parse_kickoff(row)

    event_id = await find_existing_event(
        session, home_team, away_team, kickoff.commence_time, SPORT_KEY
    )
    if event_id is None:
        event_id = build_event_id(season, home_team, away_team, kickoff.commence_time)
    event = await session.get(Event, event_id)
    if event is None:
        home_score = _safe_int(row.get("FTHG"))
        away_score = _safe_int(row.get("FTAG"))
        is_final = home_score is not None and away_score is not None
        event = Event(
            id=event_id,
            sport_key=SPORT_KEY,
            sport_title=SPORT_TITLE,
            commence_time=kickoff.commence_time,
            home_team=home_team,
            away_team=away_team,
            status=EventStatus.FINAL if is_final else EventStatus.SCHEDULED,
            home_score=home_score,
            away_score=away_score,
            completed_at=kickoff.commence_time + timedelta(hours=2) if is_final else None,
        )
        session.add(event)
        stats.events_created += 1
    else:
        stats.events_matched += 1
        if event_id.startswith(f"{EVENT_ID_PREFIX}_"):
            event.commence_time = kickoff.commence_time
            if event.completed_at is not None:
                event.completed_at = kickoff.commence_time + timedelta(hours=2)
        elif abs(event.commence_time - kickoff.commence_time) > KICKOFF_MISMATCH_TOLERANCE:
            stats.kickoff_mismatches += 1
            log.warning(
                f"  {event_id}: FDUK kick-off {kickoff.commence_time.isoformat()} differs "
                f"from event kick-off {event.commence_time.isoformat()}"
            )
    commence_time = event.commence_time
    if event.home_score is None or event.away_score is None:
        home_score = _safe_int(row.get("FTHG"))
        away_score = _safe_int(row.get("FTAG"))
        if home_score is not None and away_score is not None:
            event.status = EventStatus.FINAL
            event.home_score = home_score
            event.away_score = away_score
            event.completed_at = commence_time + timedelta(hours=2)
            stats.results_filled += 1

    new_snapshots: list[OddsSnapshot] = []
    for use_closing in (False, True):
        raw_data = _build_snapshot_raw_data(
            row,
            home_team,
            away_team,
            use_closing=use_closing,
            kickoff_time_assumed=kickoff.time_assumed,
        )
        if raw_data is None:
            continue
        if use_closing:
            snapshot_time: datetime | None = commence_time
        else:
            snapshot_time = opening_snapshot_time(commence_time)
            if snapshot_time is None:
                stats.openings_withheld += 1
                log.info(f"  {event_id}: opening deadline at or after kick-off, not stamped")
                continue
        hours_before = (commence_time - snapshot_time).total_seconds() / 3600
        for bk in raw_data["bookmakers"]:
            bk["last_update"] = snapshot_time.isoformat().replace("+00:00", "Z")
        new_snapshots.append(
            OddsSnapshot(
                event_id=event_id,
                snapshot_time=snapshot_time,
                raw_data=raw_data,
                bookmaker_count=len(raw_data["bookmakers"]),
                api_request_id=SOURCE_TAG,
                fetch_tier=hours_to_tier(hours_before)
                if use_closing
                else opening_fetch_tier(hours_before),
                hours_until_commence=hours_before,
            )
        )

    existing_snaps = list(
        (
            await session.execute(
                select(OddsSnapshot)
                .where(
                    and_(
                        OddsSnapshot.event_id == event_id,
                        OddsSnapshot.api_request_id == SOURCE_TAG,
                    )
                )
                .order_by(OddsSnapshot.snapshot_time)
            )
        )
        .scalars()
        .all()
    )
    if [_snapshot_content(snap) for snap in existing_snaps] == [
        _snapshot_content(snap) for snap in new_snapshots
    ]:
        return
    if existing_snaps:
        existing_ids = [snap.id for snap in existing_snaps]
        await session.execute(delete(Odds).where(Odds.snapshot_id.in_(existing_ids)))
        await session.execute(delete(OddsSnapshot).where(OddsSnapshot.id.in_(existing_ids)))
        stats.snapshots_updated += len(existing_snaps)
    session.add_all(new_snapshots)
    stats.snapshots_inserted += len(new_snapshots)


def _snapshot_content(snap: OddsSnapshot) -> tuple[Any, ...]:
    return (
        snap.snapshot_time,
        snap.fetch_tier,
        snap.hours_until_commence,
        snap.bookmaker_count,
        snap.raw_data,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest football-data.co.uk EPL data into the database"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to ingest (e.g. 2024-2025)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available seasons (2015-2027)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Download and preview without writing to DB",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Use already-downloaded CSVs (skip HTTP requests)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download CSVs without database ingestion",
    )
    args = parser.parse_args()

    if args.all:
        seasons = list(SEASONS.keys())
    elif args.seasons:
        for s in args.seasons:
            if s not in SEASONS:
                parser.error(f"Unknown season: {s}. Available: {list(SEASONS.keys())}")
        seasons = args.seasons
    else:
        parser.error("Specify --seasons or --all")

    asyncio.run(
        _run(
            seasons,
            dry_run=args.dry_run,
            no_download=args.no_download,
            download_only=args.download_only,
        )
    )


async def _run(
    seasons: list[str],
    *,
    dry_run: bool,
    no_download: bool,
    download_only: bool,
) -> None:
    totals = FdukStats()

    for i, season in enumerate(seasons):
        log.info(f"[{i + 1}/{len(seasons)}] {season}")

        if download_only:
            try:
                download_csv(season)
                totals.seasons_downloaded += 1
            except OSError as e:
                log.error(f"  {season}: download failed — {e}")
                totals.errors.append(f"{season}: {e}")
            continue

        stats = await ingest_season(
            season,
            dry_run=dry_run,
            no_download=no_download,
        )

        totals.games_loaded += stats.games_loaded
        totals.games_skipped += stats.games_skipped
        totals.events_matched += stats.events_matched
        totals.events_created += stats.events_created
        totals.snapshots_inserted += stats.snapshots_inserted
        totals.snapshots_updated += stats.snapshots_updated
        totals.openings_withheld += stats.openings_withheld
        totals.kickoff_mismatches += stats.kickoff_mismatches
        totals.results_filled += stats.results_filled
        totals.errors.extend(stats.errors)

        if not dry_run and not download_only:
            log.info(
                f"  → matched={stats.events_matched}, created={stats.events_created}, "
                f"snapshots={stats.snapshots_inserted}, updated={stats.snapshots_updated}, "
                f"skipped={stats.games_skipped}"
            )

    log.info("")
    log.info("=== Summary ===")
    if download_only:
        log.info(f"Seasons downloaded: {totals.seasons_downloaded}")
    else:
        log.info(f"Matches loaded:     {totals.games_loaded}")
        log.info(f"Matches skipped:    {totals.games_skipped}")
        log.info(f"Events matched:     {totals.events_matched}")
        log.info(f"Events created:     {totals.events_created}")
        log.info(f"Snapshots inserted: {totals.snapshots_inserted}")
        log.info(f"Snapshots replaced: {totals.snapshots_updated}")
        log.info(f"Openings withheld:  {totals.openings_withheld}")
        log.info(f"Kick-off mismatches: {totals.kickoff_mismatches}")
        log.info(f"Results filled:     {totals.results_filled}")

    if totals.errors:
        log.info(f"Errors:             {len(totals.errors)}")
        for err in totals.errors[:10]:
            log.error(f"  {err}")


if __name__ == "__main__":
    main()
