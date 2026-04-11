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
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.team import team_abbrev
from odds_lambda.oddsportal_common import (
    IngestionStats,
    build_raw_data,
    find_existing_event,
    hours_to_tier,
    parse_match_date,
)
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "oddsportal"

ALL_SEASONS = [
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

# ---------------------------------------------------------------------------
# Sport configuration
# ---------------------------------------------------------------------------

SPORT_CONFIGS: dict[str, dict[str, Any]] = {
    "basketball": {
        "sport_key": "basketball_nba",
        "sport_title": "NBA",
        "default_market": "home_away",
        "file_prefix": "nba",
    },
    "soccer": {
        "sport_key": "soccer_epl",
        "sport_title": "EPL",
        "default_market": "1x2",
        "file_prefix": "epl",
    },
}

# Market name → ingestion config. Determines how OddsHarvester output maps
# to The Odds API format in the database.
MARKET_CONFIGS: dict[str, dict[str, Any]] = {
    "home_away": {
        "num_outcomes": 2,
        "db_market": "h2h",
        "outcome_names": None,  # Use team names
        "line": None,
    },
    "1x2": {
        "num_outcomes": 3,
        "db_market": "h2h",
        "outcome_names": None,  # Home, Draw, Away
        "line": None,
    },
}


def _parse_over_under_market(market: str) -> dict[str, Any] | None:
    """Parse over_under_X_Y market names into config."""
    match = re.match(r"over_under_(\d+)(?:_(\d+))?$", market)
    if not match:
        return None
    whole = int(match.group(1))
    frac = int(match.group(2)) if match.group(2) else 0
    # Convert: over_under_2_5 → 2.5, over_under_3 → 3.0, over_under_1_25 → 1.25
    line = whole + frac / (10 ** len(match.group(2))) if match.group(2) else float(whole)
    return {
        "num_outcomes": 2,
        "db_market": "totals",
        "outcome_names": ("Over", "Under"),
        "line": line,
    }


def get_market_config(market: str) -> dict[str, Any]:
    """Get ingestion config for a market, supporting static and parsed configs."""
    if market in MARKET_CONFIGS:
        return MARKET_CONFIGS[market]
    parsed = _parse_over_under_market(market)
    if parsed:
        return parsed
    msg = f"Unknown market: {market}. Add it to MARKET_CONFIGS or use an over_under_* pattern."
    raise ValueError(msg)


def market_to_key(market: str) -> str:
    """Derive the JSON key OddsHarvester uses for a given market name."""
    return market.replace("-", "_") + "_market"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_oddsportal")


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
    ) or team_abbrev(home_team)
    away_abbrev = (
        canonical_to_abbrev.get(away_team) if canonical_to_abbrev else None
    ) or team_abbrev(away_team)
    return f"op_{season}_{home_abbrev}_{away_abbrev}_{game_date.isoformat()}"


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


async def ingest_season(
    season: str,
    data_dirs: list[Path],
    *,
    sport_config: dict[str, Any],
    market_config: dict[str, Any],
    json_market_key: str,
    file_prefix: str,
    normalize_fn: Callable[[str], str | None],
    canonical_to_abbrev: dict[str, str] | None,
    dry_run: bool = False,
) -> IngestionStats:
    """Ingest one season of OddsPortal data into the database."""
    stats = IngestionStats()
    market_key: str = json_market_key

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
                    market_config=market_config,
                    json_market_key=json_market_key,
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
    market_config: dict[str, Any],
    json_market_key: str,
    normalize_fn: Callable[[str], str | None],
    canonical_to_abbrev: dict[str, str] | None,
) -> None:
    """Ingest a single OddsPortal game record."""
    market_key: str = json_market_key
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
    # Tag with market-specific source for deduplication
    db_market: str = market_config["db_market"]
    source_tag = "oddsportal" if db_market == "h2h" else f"oddsportal_{db_market}"

    # Check for existing snapshots to avoid duplicates
    existing_check = await session.execute(
        select(OddsSnapshot.id).where(
            and_(
                OddsSnapshot.event_id == event_id,
                OddsSnapshot.api_request_id == source_tag,
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
            num_outcomes=market_config["num_outcomes"],
            db_market=market_config["db_market"],
            outcome_names=market_config["outcome_names"],
            line=market_config["line"],
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
            api_request_id=source_tag,
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
        "--market",
        default=None,
        help="OddsHarvester market name (default: per sport). "
        "Examples: 1x2, over_under_2_5, asian_handicap_-0_5",
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

    # Resolve market
    market = args.market or sport_config["default_market"]
    mkt_config = get_market_config(market)
    json_mkt_key = market_to_key(market)
    base_prefix: str = sport_config["file_prefix"]
    file_prefix = (
        f"{base_prefix}_{market}" if market != sport_config["default_market"] else base_prefix
    )

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
            market_config=mkt_config,
            json_market_key=json_mkt_key,
            file_prefix=file_prefix,
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
    market_config: dict[str, Any],
    json_market_key: str,
    file_prefix: str,
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
            market_config=market_config,
            json_market_key=json_market_key,
            file_prefix=file_prefix,
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
