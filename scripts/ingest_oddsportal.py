#!/usr/bin/env python3
"""Ingest OddsPortal historical odds into the existing database.

Converts OddsPortal JSON → Event + OddsSnapshot records. Once ingested,
the existing training pipeline works unchanged — just extend the date range
in the config YAML.

Usage:
    # Ingest all NBA seasons (dry-run first)
    uv run python scripts/ingest_oddsportal.py --all --dry-run

    # Ingest specific seasons
    uv run python scripts/ingest_oddsportal.py --seasons 2024-2025

    # Ingest EPL soccer data
    uv run python scripts/ingest_oddsportal.py --sport soccer --all

    # Merge UK + proxy scrapes (union bookmakers from both directories)
    uv run python scripts/ingest_oddsportal.py --all \\
        --data-dirs data/external/oddsportal data/external/oddsportal_proxy

    # Ingest and re-link game logs + injuries (basketball only)
    uv run python scripts/ingest_oddsportal.py --all --relink
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

# Outcome name for draws — matches sequence_loader.DRAW_OUTCOME
DRAW_OUTCOME = "Draw"

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "oddsportal"

ALL_SEASONS = [
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# OddsPortal bookmaker name → The Odds API-style key
BOOKMAKER_KEY_MAP: dict[str, str] = {
    "10bet": "10bet",
    "bet365": "bet365",
    "BetMGM": "betmgm",
    "bwin": "bwin",
    "Betway": "betway",
    "BetVictor": "betvictor",
    "Betfred": "betfred",
    "BetUK": "betuk",
    "Midnite": "midnite",
    "Unibetuk": "unibet_uk",
    "Betano.uk": "betano",
    "AllBritishCasino": "allbritishcasino",
    "Pinnacle": "pinnacle",
    "DraftKings": "draftkings",
    "FanDuel": "fanduel",
    "Caesars": "williamhill_us",
    "PointsBet": "pointsbetus",
    "BetRivers": "betrivers",
    "Unibet": "unibet",
    "Bovada": "bovada",
    "Marathon Bet": "marathonbet",
    "1xBet": "onexbet",
    "Betfair Exchange": "betfair_exchange",
}

# ---------------------------------------------------------------------------
# Sport configuration
# ---------------------------------------------------------------------------

SPORT_CONFIGS: dict[str, dict[str, Any]] = {
    "basketball": {
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "market_key": "home_away_market",
        "num_outcomes": 2,
        "file_prefix": "nba",
    },
    "soccer": {
        "sport_key": "soccer_epl",
        "sport_title": "EPL",
        "market_key": "1x2_market",
        "num_outcomes": 3,
        "file_prefix": "epl",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_oddsportal")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class IngestionStats:
    games_loaded: int = 0
    games_skipped: int = 0
    events_matched: int = 0
    events_created: int = 0
    snapshots_inserted: int = 0
    game_logs_linked: int = 0
    injuries_linked: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Team normalization
# ---------------------------------------------------------------------------


def _get_nba_normalizer() -> tuple[dict[str, str], Callable[[str], str | None]]:
    """Lazy-import NBA normalization to avoid hard dependency for non-basketball sports."""
    from odds_lambda.polymarket_matching import CANONICAL_TO_ABBREV, normalize_team

    abbrev_to_canonical: dict[str, str] = {v: k for k, v in CANONICAL_TO_ABBREV.items()}
    return abbrev_to_canonical, normalize_team


def _normalize_team_passthrough(name: str) -> str | None:
    """Pass-through normalization — use OddsPortal names as-is."""
    return name.strip() if name and name.strip() else None


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------


def decimal_to_american(d: float) -> int:
    """Convert decimal odds to American odds."""
    if d >= 2.0:
        return round((d - 1) * 100)
    elif d > 1.0:
        return round(-100 / (d - 1))
    return -10000  # edge case: decimal = 1.0


def fix_odds_timestamp(odds_ts: datetime, match_dt: datetime) -> datetime:
    """Fix the year in OddsPortal odds timestamps.

    OddsPortal records timestamps with the scrape year (e.g. 2026) instead
    of the actual game year. Month/day/time are correct.

    Returns a naive datetime (caller adds tzinfo as needed).
    """
    # Strip timezone for comparison — match_dt may be aware
    match_naive = match_dt.replace(tzinfo=None)

    try:
        fixed = odds_ts.replace(year=match_naive.year)
    except ValueError:
        # Feb 29 in scrape year but not in game year
        fixed = odds_ts.replace(year=match_naive.year, day=28)

    # Handle Dec→Jan season boundary: opening odds from Dec for a Jan game
    if (fixed - match_naive).days > 60:
        try:
            fixed = fixed.replace(year=match_naive.year - 1)
        except ValueError:
            fixed = fixed.replace(year=match_naive.year - 1, day=28)
    elif (match_naive - fixed).days > 60:
        try:
            fixed = fixed.replace(year=match_naive.year + 1)
        except ValueError:
            fixed = fixed.replace(year=match_naive.year + 1, day=28)

    return fixed


def hours_to_tier(hours_before: float) -> str:
    """Map hours before game start to fetch tier name."""
    if hours_before < 3:
        return "closing"
    elif hours_before < 12:
        return "pregame"
    elif hours_before < 24:
        return "sharp"
    elif hours_before < 72:
        return "early"
    return "opening"


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


def parse_match_date(date_str: str) -> datetime:
    """Parse OddsPortal match_date string to UTC datetime.

    Format: '2024-10-04 16:00:00 UTC'
    """
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=UTC)


def parse_odds_timestamp(ts_str: str) -> datetime:
    """Parse OddsPortal odds_history timestamp (naive, wrong year)."""
    return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")


def _team_abbrev(name: str) -> str:
    """Derive a short abbreviation from a team name.

    Single-word names get 3 chars (e.g. "Arsenal" → "ARS").
    Multi-word names use first 3 chars of first + last word
    (e.g. "Manchester United" → "MANUNI", "Manchester City" → "MANCIT").
    """
    words = name.split()
    if len(words) == 1:
        return words[0][:3].upper()
    return (words[0][:3] + words[-1][:3]).upper()


def build_event_id(
    season: str,
    home_team: str,
    away_team: str,
    game_date: date,
    canonical_to_abbrev: dict[str, str] | None = None,
) -> str:
    """Generate deterministic event ID for OddsPortal-sourced events."""
    home_abbrev = (
        canonical_to_abbrev.get(home_team) if canonical_to_abbrev else None
    ) or _team_abbrev(home_team)
    away_abbrev = (
        canonical_to_abbrev.get(away_team) if canonical_to_abbrev else None
    ) or _team_abbrev(away_team)
    return f"op_{season}_{home_abbrev}_{away_abbrev}_{game_date.isoformat()}"


def build_raw_data(
    bookmaker_odds: list[dict],
    home_team: str,
    away_team: str,
    *,
    use_opening: bool,
    match_dt: datetime,
    num_outcomes: int = 2,
) -> dict | None:
    """Convert OddsPortal bookmaker data into The Odds API raw_data format.

    Args:
        bookmaker_odds: List of bookmaker dicts from the market data
        home_team: Canonical home team name
        away_team: Canonical away team name
        use_opening: If True, use opening odds; if False, use closing (last in history)
        match_dt: Correct game datetime (for fixing timestamps)
        num_outcomes: Number of outcomes (2 for NBA h2h, 3 for soccer 1x2)

    Returns:
        Dict in The Odds API format, or None if no valid bookmakers found
    """
    bookmakers = []
    snapshot_time: datetime | None = None

    for bk in bookmaker_odds:
        hist = bk.get("odds_history_data")
        if not hist or len(hist) < num_outcomes:
            continue

        bk_name = bk["bookmaker_name"]
        bk_key = BOOKMAKER_KEY_MAP.get(bk_name, _slugify(bk_name))

        home_hist = hist[0]  # Outcome 1 (home)
        away_hist = hist[num_outcomes - 1]  # Last outcome (away)
        draw_hist = hist[1] if num_outcomes >= 3 else None  # Outcome X (draw)

        if use_opening:
            home_entry = home_hist.get("opening_odds")
            away_entry = away_hist.get("opening_odds")
            draw_entry = draw_hist.get("opening_odds") if draw_hist else None
        else:
            # Closing = last entry in odds_history list
            home_entries = home_hist.get("odds_history", [])
            away_entries = away_hist.get("odds_history", [])
            if not home_entries or not away_entries:
                continue
            home_entry = home_entries[-1]
            away_entry = away_entries[-1]

            if draw_hist:
                draw_entries = draw_hist.get("odds_history", [])
                draw_entry = draw_entries[-1] if draw_entries else None
            else:
                draw_entry = None

        if not home_entry or not away_entry:
            continue
        if num_outcomes >= 3 and not draw_entry:
            continue

        home_decimal = home_entry.get("odds")
        away_decimal = away_entry.get("odds")
        if not home_decimal or not away_decimal:
            continue

        # Build outcomes list
        outcomes: list[dict[str, Any]] = [
            {"name": home_team, "price": decimal_to_american(home_decimal)},
        ]
        if draw_entry and num_outcomes >= 3:
            draw_decimal = draw_entry.get("odds")
            if not draw_decimal:
                continue
            outcomes.append({"name": DRAW_OUTCOME, "price": decimal_to_american(draw_decimal)})
        outcomes.append({"name": away_team, "price": decimal_to_american(away_decimal)})

        # Use the home entry's timestamp as the bookmaker last_update
        raw_ts = parse_odds_timestamp(home_entry["timestamp"])
        fixed_ts = fix_odds_timestamp(raw_ts, match_dt)

        # Track the latest timestamp across bookmakers for snapshot_time
        if snapshot_time is None or fixed_ts > snapshot_time:
            snapshot_time = fixed_ts

        bookmakers.append(
            {
                "key": bk_key,
                "title": bk_name,
                "last_update": fixed_ts.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
                "markets": [{"key": "h2h", "outcomes": outcomes}],
            }
        )

    if not bookmakers:
        return None

    return {
        "bookmakers": bookmakers,
        "source": "oddsportal",
        "_snapshot_time": snapshot_time.replace(tzinfo=UTC).isoformat() if snapshot_time else None,
    }


def _slugify(name: str) -> str:
    """Convert bookmaker name to a lowercase slug key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------------
# Multi-directory loading and merging
# ---------------------------------------------------------------------------


def load_season_records(
    season: str, data_dirs: list[Path], *, file_prefix: str, market_key: str
) -> list[dict]:
    """Load and merge records for a season from multiple data directories.

    When the same game appears in multiple directories (matched by match_link),
    bookmaker lists are unioned — each directory may contain different
    bookmakers due to geo-restrictions.
    """
    by_link: dict[str, dict] = {}

    for data_dir in data_dirs:
        path = data_dir / f"{file_prefix}_{season}.json"
        if not path.exists():
            continue

        try:
            records = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning(f"  Failed to read {path}")
            continue

        if not isinstance(records, list):
            continue

        for record in records:
            link = record.get("match_link", "")
            if not link:
                continue

            if link not in by_link:
                by_link[link] = record
            else:
                # Merge bookmaker lists
                by_link[link] = _merge_bookmakers(by_link[link], record, market_key)

    return list(by_link.values())


def _merge_bookmakers(base: dict, other: dict, market_key: str) -> dict:
    """Merge market bookmaker lists from two records of the same game.

    Bookmakers are keyed by name. When both records have the same bookmaker,
    the entry with odds_history_data wins.
    """
    merged = dict(base)
    base_market = {bk["bookmaker_name"]: bk for bk in base.get(market_key, [])}
    other_market = other.get(market_key, [])

    for bk in other_market:
        name = bk["bookmaker_name"]
        if name not in base_market:
            base_market[name] = bk
        elif bk.get("odds_history_data") and not base_market[name].get("odds_history_data"):
            base_market[name] = bk

    merged[market_key] = list(base_market.values())
    return merged


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


async def find_existing_event(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    commence_time: datetime,
) -> str | None:
    """Find an existing Event matching the given game within a ±24h window."""
    window_start = commence_time - timedelta(hours=24)
    window_end = commence_time + timedelta(hours=24)

    query = select(Event.id).where(
        and_(
            Event.commence_time >= window_start,
            Event.commence_time <= window_end,
            Event.home_team == home_team,
            Event.away_team == away_team,
        )
    )
    result = await session.execute(query)
    candidates = list(result.scalars().all())

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        log.warning(
            f"Ambiguous match for {away_team} @ {home_team} on "
            f"{commence_time.date()}: {len(candidates)} candidates"
        )
    return None


async def ingest_season(
    season: str,
    data_dirs: list[Path],
    *,
    sport_config: dict[str, Any],
    normalize_fn: Callable[[str], str | None],
    canonical_to_abbrev: dict[str, str] | None,
    dry_run: bool = False,
) -> IngestionStats:
    """Ingest one season of OddsPortal data into the database."""
    stats = IngestionStats()
    file_prefix: str = sport_config["file_prefix"]
    market_key: str = sport_config["market_key"]

    records = load_season_records(season, data_dirs, file_prefix=file_prefix, market_key=market_key)
    if not records:
        log.error(f"  {season}: no data found in {[str(d) for d in data_dirs]}")
        return stats

    stats.games_loaded = len(records)
    log.info(f"  {season}: loaded {len(records)} games (from {len(data_dirs)} dir(s))")

    if dry_run:
        # Count how many have odds_history
        with_hist = sum(
            1 for r in records if any(bk.get("odds_history_data") for bk in r.get(market_key, []))
        )
        log.info(f"  {season}: {with_hist} games have odds history data")
        return stats

    async with async_session_maker() as session:
        for record in records:
            try:
                await _ingest_one_game(
                    session,
                    record,
                    season,
                    stats,
                    sport_config=sport_config,
                    normalize_fn=normalize_fn,
                    canonical_to_abbrev=canonical_to_abbrev,
                )
            except Exception as e:
                msg = f"{record.get('home_team', '?')} vs {record.get('away_team', '?')}: {e}"
                stats.errors.append(msg)
                log.error(f"  Error: {msg}")

        await session.commit()

    return stats


async def _ingest_one_game(
    session: AsyncSession,
    record: dict,
    season: str,
    stats: IngestionStats,
    *,
    sport_config: dict[str, Any],
    normalize_fn: Callable[[str], str | None],
    canonical_to_abbrev: dict[str, str] | None,
) -> None:
    """Ingest a single OddsPortal game record."""
    market_key: str = sport_config["market_key"]
    num_outcomes: int = sport_config["num_outcomes"]
    sport_key: str = sport_config["sport_key"]
    sport_title: str = sport_config["sport_title"]

    home_raw = record.get("home_team", "")
    away_raw = record.get("away_team", "")
    match_date_str = record.get("match_date", "")
    market_data = record.get(market_key, [])

    if not home_raw or not away_raw or not match_date_str or not market_data:
        stats.games_skipped += 1
        return

    # Normalize team names
    home_team = normalize_fn(home_raw)
    away_team = normalize_fn(away_raw)
    if not home_team or not away_team:
        stats.games_skipped += 1
        log.debug(f"  Unknown team: {home_raw} or {away_raw}")
        return

    # Check at least one bookmaker has odds history
    if not any(bk.get("odds_history_data") for bk in market_data):
        stats.games_skipped += 1
        return

    match_dt = parse_match_date(match_date_str)

    # --- Match or create Event ---
    event_id = await find_existing_event(session, home_team, away_team, match_dt)

    if event_id:
        stats.events_matched += 1
    else:
        event_id = build_event_id(
            season, home_team, away_team, match_dt.date(), canonical_to_abbrev
        )

        # Check if this OddsPortal event already exists (idempotency)
        existing = await session.get(Event, event_id)
        if existing:
            stats.events_matched += 1
        else:
            home_score_str = record.get("home_score", "")
            away_score_str = record.get("away_score", "")
            home_score = int(home_score_str) if home_score_str.isdigit() else None
            away_score = int(away_score_str) if away_score_str.isdigit() else None
            is_final = home_score is not None and away_score is not None

            event = Event(
                id=event_id,
                sport_key=sport_key,
                sport_title=sport_title,
                commence_time=match_dt,
                home_team=home_team,
                away_team=away_team,
                status=EventStatus.FINAL if is_final else EventStatus.SCHEDULED,
                home_score=home_score,
                away_score=away_score,
                completed_at=match_dt if is_final else None,
            )
            session.add(event)
            stats.events_created += 1

    # --- Build and insert OddsSnapshots ---
    # Check for existing snapshots from OddsPortal to avoid duplicates
    existing_check = await session.execute(
        select(OddsSnapshot.id).where(
            and_(
                OddsSnapshot.event_id == event_id,
                OddsSnapshot.api_request_id == "oddsportal",
            )
        )
    )
    if existing_check.scalars().first() is not None:
        return  # Already ingested

    for use_opening in (True, False):
        raw_data = build_raw_data(
            market_data,
            home_team,
            away_team,
            use_opening=use_opening,
            match_dt=match_dt,
            num_outcomes=num_outcomes,
        )
        if not raw_data:
            continue

        snapshot_time_str = raw_data.pop("_snapshot_time", None)
        if not snapshot_time_str:
            continue

        snapshot_time = datetime.fromisoformat(snapshot_time_str.replace("Z", "+00:00"))
        hours_before = (match_dt - snapshot_time).total_seconds() / 3600
        tier = hours_to_tier(hours_before)

        snapshot = OddsSnapshot(
            event_id=event_id,
            snapshot_time=snapshot_time,
            raw_data=raw_data,
            bookmaker_count=len(raw_data["bookmakers"]),
            api_request_id="oddsportal",
            fetch_tier=tier,
            hours_until_commence=max(0.0, hours_before),
        )
        session.add(snapshot)
        stats.snapshots_inserted += 1


# ---------------------------------------------------------------------------
# Re-linking game logs and injuries (basketball only)
# ---------------------------------------------------------------------------


async def relink_game_logs(stats: IngestionStats) -> None:
    """Re-link NbaTeamGameLog records that have NULL event_id."""
    from odds_core.game_log_models import NbaTeamGameLog
    from odds_core.time import EASTERN

    abbrev_to_canonical, _ = _get_nba_normalizer()

    async with async_session_maker() as session:
        # Find unlinked game logs
        query = select(NbaTeamGameLog).where(NbaTeamGameLog.event_id.is_(None))
        result = await session.execute(query)
        unlinked = list(result.scalars().all())

        if not unlinked:
            log.info("  No unlinked game logs found")
            return

        log.info(f"  Found {len(unlinked)} unlinked game log rows")
        linked = 0
        cache: dict[tuple[str, date], str | None] = {}

        for row in unlinked:
            canonical = abbrev_to_canonical.get(row.team_abbreviation)
            if not canonical:
                continue

            cache_key = (canonical, row.game_date)
            if cache_key not in cache:
                cache[cache_key] = await _match_event_for_relink(
                    session, canonical, row.game_date, EASTERN
                )

            event_id = cache[cache_key]
            if event_id:
                row.event_id = event_id
                session.add(row)
                linked += 1

        await session.commit()
        stats.game_logs_linked = linked
        log.info(f"  Linked {linked} game log rows to events")


async def relink_injuries(stats: IngestionStats) -> None:
    """Re-link InjuryReport records that have NULL event_id."""
    from odds_core.injury_models import InjuryReport
    from odds_core.time import EASTERN

    _, normalize_team = _get_nba_normalizer()

    async with async_session_maker() as session:
        query = select(InjuryReport).where(InjuryReport.event_id.is_(None))
        result = await session.execute(query)
        unlinked = list(result.scalars().all())

        if not unlinked:
            log.info("  No unlinked injury reports found")
            return

        log.info(f"  Found {len(unlinked)} unlinked injury report rows")
        linked = 0
        cache: dict[tuple[str, date], str | None] = {}

        for row in unlinked:
            canonical = normalize_team(row.team)
            if not canonical:
                continue

            cache_key = (canonical, row.game_date)
            if cache_key not in cache:
                cache[cache_key] = await _match_event_for_relink(
                    session, canonical, row.game_date, EASTERN
                )

            event_id = cache[cache_key]
            if event_id:
                row.event_id = event_id
                session.add(row)
                linked += 1

        await session.commit()
        stats.injuries_linked = linked
        log.info(f"  Linked {linked} injury report rows to events")


async def _match_event_for_relink(
    session: AsyncSession,
    team_name: str,
    game_date: date,
    eastern: ZoneInfo,
) -> str | None:
    """Match a team + game_date to an Event using the standard ET window."""
    day_start_et = datetime(game_date.year, game_date.month, game_date.day, 10, tzinfo=eastern)
    day_end_et = day_start_et + timedelta(hours=20)
    window_start = day_start_et.astimezone(UTC)
    window_end = day_end_et.astimezone(UTC)

    query = select(Event.id).where(
        and_(
            Event.commence_time >= window_start,
            Event.commence_time <= window_end,
            or_(Event.home_team == team_name, Event.away_team == team_name),
        )
    )
    result = await session.execute(query)
    candidates = list(result.scalars().all())

    if len(candidates) == 1:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest OddsPortal historical odds into the database"
    )
    parser.add_argument(
        "--sport",
        choices=list(SPORT_CONFIGS),
        default="basketball",
        help="Sport to ingest (default: basketball)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help="Seasons to ingest",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all available seasons",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be ingested without writing to DB",
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Data directories to read from (default: data/external/oddsportal). "
            "Multiple dirs are merged — bookmaker lists are unioned per game."
        ),
    )
    parser.add_argument(
        "--relink",
        action="store_true",
        help="Re-link game logs and injuries to newly created events (basketball only)",
    )
    args = parser.parse_args()

    if args.all:
        seasons = ALL_SEASONS
    elif args.seasons:
        seasons = args.seasons
    else:
        parser.error("Specify --seasons or --all")

    sport_config = SPORT_CONFIGS[args.sport]

    # Resolve team normalization and abbreviation map per sport
    if args.sport == "basketball":
        abbrev_to_canonical, normalize_fn = _get_nba_normalizer()
        canonical_to_abbrev: dict[str, str] | None = {v: k for k, v in abbrev_to_canonical.items()}
    else:
        normalize_fn = _normalize_team_passthrough
        canonical_to_abbrev = None

    if args.relink and args.sport != "basketball":
        log.warning("--relink is only supported for basketball — ignoring")

    data_dirs = args.data_dirs or [DEFAULT_DATA_DIR]
    asyncio.run(
        _run(
            seasons,
            data_dirs=data_dirs,
            sport_config=sport_config,
            sport=args.sport,
            normalize_fn=normalize_fn,
            canonical_to_abbrev=canonical_to_abbrev,
            dry_run=args.dry_run,
            relink=args.relink,
        )
    )


async def _run(
    seasons: list[str],
    *,
    data_dirs: list[Path],
    sport_config: dict[str, Any],
    sport: str,
    normalize_fn: Callable[[str], str | None],
    canonical_to_abbrev: dict[str, str] | None,
    dry_run: bool,
    relink: bool,
) -> None:
    totals = IngestionStats()

    for i, season in enumerate(seasons):
        log.info(f"[{i + 1}/{len(seasons)}] {season}")
        stats = await ingest_season(
            season,
            data_dirs,
            sport_config=sport_config,
            normalize_fn=normalize_fn,
            canonical_to_abbrev=canonical_to_abbrev,
            dry_run=dry_run,
        )

        totals.games_loaded += stats.games_loaded
        totals.games_skipped += stats.games_skipped
        totals.events_matched += stats.events_matched
        totals.events_created += stats.events_created
        totals.snapshots_inserted += stats.snapshots_inserted
        totals.errors.extend(stats.errors)

        log.info(
            f"  → matched={stats.events_matched}, created={stats.events_created}, "
            f"snapshots={stats.snapshots_inserted}, skipped={stats.games_skipped}"
        )

    log.info("")
    log.info("=== Summary ===")
    log.info(f"Games loaded:      {totals.games_loaded}")
    log.info(f"Games skipped:     {totals.games_skipped}")
    log.info(f"Events matched:    {totals.events_matched}")
    log.info(f"Events created:    {totals.events_created}")
    log.info(f"Snapshots inserted: {totals.snapshots_inserted}")

    if totals.errors:
        log.info(f"Errors:            {len(totals.errors)}")
        for err in totals.errors[:10]:
            log.error(f"  {err}")

    if relink and sport == "basketball" and not dry_run:
        log.info("")
        log.info("=== Re-linking ===")
        await relink_game_logs(totals)
        await relink_injuries(totals)
        log.info(f"Game logs linked:  {totals.game_logs_linked}")
        log.info(f"Injuries linked:   {totals.injuries_linked}")


if __name__ == "__main__":
    main()
