#!/usr/bin/env python3
"""Scrape historical odds from OddsPortal using OddsHarvester.

Resumes at game level — only scrapes matches that are missing or incomplete.

Usage:
    # Scrape recent 3 NBA seasons (closing odds only, fastest)
    uv run python scripts/scrape_oddsportal.py

    # Scrape specific seasons
    uv run python scripts/scrape_oddsportal.py --seasons 2022-2023 2023-2024

    # Include opening/closing odds history per bookmaker (slower)
    uv run python scripts/scrape_oddsportal.py --odds-history

    # Force re-scrape even if data exists
    uv run python scripts/scrape_oddsportal.py --seasons 2022-2023 --force

    # Scrape all available seasons
    uv run python scripts/scrape_oddsportal.py --all

    # Scrape EPL soccer data
    uv run python scripts/scrape_oddsportal.py --sport soccer --league england-premier-league --all

    # Scrape through a proxy to a separate directory (for geo-restricted bookmakers)
    uv run python scripts/scrape_oddsportal.py --all --odds-history \\
        --proxy-url socks5://proxy.example.com:1080 \\
        --output-dir data/external/oddsportal_proxy
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "external" / "oddsportal"

FORK_SPEC = (
    "git+https://github.com/Acusick1/OddsHarvester.git@fix/fractional-odds-and-unknown-bookmaker"
)

ALL_SEASONS = [
    "2021-2022",
    "2022-2023",
    "2023-2024",
    "2024-2025",
    "2025-2026",
]

DEFAULT_SEASONS = ["2023-2024", "2024-2025", "2025-2026"]

ODDS_FORMAT = "Decimal Odds"
BATCH_SIZE = 20
SCRAPE_TIMEOUT = 14400  # 4 hours

# ---------------------------------------------------------------------------
# Sport configuration
# ---------------------------------------------------------------------------

SPORT_CONFIGS: dict[str, dict[str, Any]] = {
    "basketball": {
        "harvester_sport": "basketball",
        "default_league": "nba",
        "market": "home_away",
        "market_key": "home_away_market",
        "file_prefix": "nba",
    },
    "soccer": {
        "harvester_sport": "football",
        "default_league": "england-premier-league",
        "market": "1x2",
        "market_key": "1x2_market",
        "file_prefix": "epl",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scrape_oddsportal")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_existing(path: Path, market_key: str) -> dict[str, dict]:
    """Load a season JSON file and return records indexed by match_link.

    When duplicates exist, the entry with more data wins (non-empty odds
    preferred over empty, odds history preferred over none).
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            return {}
    except (json.JSONDecodeError, OSError):
        return {}

    indexed: dict[str, dict] = {}
    for record in data:
        link = record.get("match_link", "")
        if not link:
            continue
        prev = indexed.get(link)
        if prev is None or _richness(record, market_key) > _richness(prev, market_key):
            indexed[link] = record
    return indexed


def _richness(record: dict, market_key: str) -> int:
    """Score how complete a record is (higher = more data)."""
    score = 0
    market = record.get(market_key, [])
    if market:
        score += 1
        if any(bk.get("odds_history_data") for bk in market):
            score += 1
    return score


def is_complete(record: dict, odds_history: bool, market_key: str) -> bool:
    """Check whether a match record has all the data we need."""
    market = record.get(market_key, [])
    if not market:
        return False
    if odds_history:
        return any(bk.get("odds_history_data") for bk in market)
    return True


def merge_records(
    existing: dict[str, dict], new_records: list[dict], market_key: str
) -> dict[str, dict]:
    """Merge a list of new records into the existing index.

    For each match_link, keep whichever entry is richer.
    """
    merged = dict(existing)
    for record in new_records:
        link = record.get("match_link", "")
        if not link:
            continue
        prev = merged.get(link)
        if prev is None or _richness(record, market_key) > _richness(prev, market_key):
            merged[link] = record
    return merged


def save_season(path: Path, indexed: dict[str, dict]) -> int:
    """Write the deduplicated, merged records to disk. Returns count."""
    records = sorted(indexed.values(), key=lambda r: r.get("match_date", ""))
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    return len(records)


# ---------------------------------------------------------------------------
# Scraper invocation
# ---------------------------------------------------------------------------


def _base_cmd(
    *,
    sport: str,
    league: str,
    market: str,
    proxy_url: str | None = None,
    proxy_user: str | None = None,
    proxy_pass: str | None = None,
) -> list[str]:
    cmd = [
        "uvx",
        "--from",
        FORK_SPEC,
        "oddsharvester",
        "historic",
        "-s",
        sport,
        "-l",
        league,
        "-m",
        market,
        "--headless",
        "--odds-format",
        ODDS_FORMAT,
        "-f",
        "json",
    ]
    if proxy_url:
        cmd.extend(["--proxy-url", proxy_url])
    if proxy_user:
        cmd.extend(["--proxy-user", proxy_user])
    if proxy_pass:
        cmd.extend(["--proxy-pass", proxy_pass])
    return cmd


def run_harvester(
    args: list[str],
    out_path: Path,
    log_path: Path,
    *,
    sport: str,
    league: str,
    market: str,
    timeout: int = SCRAPE_TIMEOUT,
    proxy_url: str | None = None,
    proxy_user: str | None = None,
    proxy_pass: str | None = None,
) -> bool:
    """Run oddsharvester with given extra args. Returns True on success."""
    cmd = (
        _base_cmd(
            sport=sport,
            league=league,
            market=market,
            proxy_url=proxy_url,
            proxy_user=proxy_user,
            proxy_pass=proxy_pass,
        )
        + ["-o", str(out_path)]
        + args
    )
    log.info(f"  Running: {' '.join(cmd[-6:])}")

    with open(log_path, "a") as lf:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=lf,
            timeout=timeout,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    if result.returncode != 0:
        log.error(f"  Scraper failed (exit {result.returncode}), see {log_path}")
        return False
    return True


def discover_season_links(
    season: str,
    log_path: Path,
    *,
    sport: str,
    league: str,
    market: str,
    file_prefix: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout: int = 900,
    force: bool = False,
    proxy_url: str | None = None,
    proxy_user: str | None = None,
    proxy_pass: str | None = None,
) -> list[str]:
    """Discover all match URLs for a season via --links-only.

    Results are cached to ``output_dir/{prefix}_{season}_links.json`` so
    restarts skip re-discovery.  Pass ``force=True`` to re-discover.
    """
    cache_path = output_dir / f"{file_prefix}_{season}_links.json"

    if not force and cache_path.exists():
        cached = load_json_links(cache_path)
        if cached:
            log.info(f"  {season}: loaded {len(cached)} cached links from {cache_path.name}")
            return cached

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        tmp_path = Path(tf.name)

    try:
        ok = run_harvester(
            ["--season", season, "--links-only"],
            tmp_path,
            log_path,
            sport=sport,
            league=league,
            market=market,
            timeout=timeout,
            proxy_url=proxy_url,
            proxy_user=proxy_user,
            proxy_pass=proxy_pass,
        )
        if ok:
            links = load_json_links(tmp_path)
            if links:
                cache_path.write_text(json.dumps(links, indent=2))
                log.info(f"  {season}: cached {len(links)} links to {cache_path.name}")
            return links
    finally:
        tmp_path.unlink(missing_ok=True)
    return []


def load_json(path: Path) -> list[dict]:
    """Load a JSON array of dicts from path, returning [] on failure."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def load_json_links(path: Path) -> list[str]:
    """Load a JSON array of strings from path, returning [] on failure."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Season scraping
# ---------------------------------------------------------------------------


def scrape_season(
    season: str,
    *,
    sport: str,
    league: str,
    market: str,
    market_key: str,
    file_prefix: str,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    odds_history: bool = False,
    force: bool = False,
    proxy_url: str | None = None,
    proxy_user: str | None = None,
    proxy_pass: str | None = None,
) -> int:
    """Scrape a season with game-level resume. Returns final match count.

    All scraping goes through batched --match-link invocations with saves
    after each batch, so progress is never lost on crash.

    Flow:
    1. Discover all match URLs for the season via --links-only (seconds)
    2. Identify which matches are missing or incomplete
    3. Batch scrape via --match-link in groups of BATCH_SIZE, saving after each
    """
    out = output_dir / f"{file_prefix}_{season}.json"
    log_file = output_dir / f"{file_prefix}_{season}.log"
    start = time.time()

    existing = load_existing(out, market_key)
    complete = sum(1 for r in existing.values() if is_complete(r, odds_history, market_key))

    if existing:
        log.info(
            f"  {season}: {len(existing)} existing matches "
            f"({complete} complete, {len(existing) - complete} incomplete)"
        )
    else:
        log.info(f"  {season}: no existing data")

    # --- Step 1: Discover all match links (fast, no odds extraction) ---
    log.info(f"  {season}: discovering match links...")
    all_links = discover_season_links(
        season,
        log_file,
        sport=sport,
        league=league,
        market=market,
        file_prefix=file_prefix,
        output_dir=output_dir,
        proxy_url=proxy_url,
        proxy_user=proxy_user,
        proxy_pass=proxy_pass,
    )
    if not all_links:
        log.error(f"  {season}: link discovery failed or returned 0 links")
        return len(existing)

    log.info(f"  {season}: discovered {len(all_links)} match links")

    # --- Step 2: Identify matches that need scraping ---
    if force:
        links_to_scrape = list(all_links)
    else:
        links_to_scrape = [
            link
            for link in all_links
            if link not in existing or not is_complete(existing[link], odds_history, market_key)
        ]

    if not links_to_scrape:
        elapsed = time.time() - start
        log.info(f"  {season}: all {len(existing)} matches complete ({elapsed / 60:.1f} min)")
        return len(existing)

    log.info(
        f"  {season}: {len(links_to_scrape)}/{len(all_links)} matches need scraping"
        f"{' (with odds history)' if odds_history else ''}"
    )

    # --- Step 3: Batch scrape via --match-link, save after each batch ---
    batches = [
        links_to_scrape[i : i + BATCH_SIZE] for i in range(0, len(links_to_scrape), BATCH_SIZE)
    ]

    for batch_num, batch in enumerate(batches, 1):
        log.info(
            f"  {season}: batch {batch_num}/{len(batches)} "
            f"({len(batch)} matches{', odds history' if odds_history else ''})..."
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            tmp_path = Path(tf.name)
        try:
            extra: list[str] = ["--season", season]
            if odds_history:
                extra.append("--odds-history")
            for link in batch:
                extra.extend(["--match-link", link])
            ok = run_harvester(
                extra,
                tmp_path,
                log_file,
                sport=sport,
                league=league,
                market=market,
                proxy_url=proxy_url,
                proxy_user=proxy_user,
                proxy_pass=proxy_pass,
            )
            if ok:
                batch_records = load_json(tmp_path)
                log.info(f"  {season}: batch {batch_num} returned {len(batch_records)} records")
                existing = merge_records(existing, batch_records, market_key)
                save_season(out, existing)
        finally:
            tmp_path.unlink(missing_ok=True)

    count = len(existing)
    elapsed = time.time() - start
    still_incomplete = sum(
        1 for r in existing.values() if not is_complete(r, odds_history, market_key)
    )
    log.info(
        f"  {season}: {count} matches saved, {still_incomplete} still incomplete "
        f"({elapsed / 60:.1f} min)"
    )
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape OddsPortal historical odds")
    parser.add_argument(
        "--sport",
        choices=list(SPORT_CONFIGS),
        default="basketball",
        help="Sport to scrape (default: basketball)",
    )
    parser.add_argument(
        "--league",
        default=None,
        help="OddsHarvester league name (default: per sport — nba, england-premier-league)",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=None,
        help=f"Seasons to scrape (default: {DEFAULT_SEASONS})",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all available seasons (2021-2026)",
    )
    parser.add_argument(
        "--odds-history",
        action="store_true",
        help="Include opening/closing odds movement per bookmaker (much slower)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing data and re-scrape from scratch",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be scraped without actually scraping",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--proxy-url",
        default=None,
        help="Proxy server URL (e.g. socks5://host:port)",
    )
    parser.add_argument(
        "--proxy-user",
        default=None,
        help="Proxy username (requires --proxy-pass)",
    )
    parser.add_argument(
        "--proxy-pass",
        default=None,
        help="Proxy password (requires --proxy-user)",
    )
    args = parser.parse_args()

    # Resolve sport config
    config = SPORT_CONFIGS[args.sport]
    harvester_sport: str = config["harvester_sport"]
    league: str = args.league or config["default_league"]
    market: str = config["market"]
    market_key: str = config["market_key"]
    file_prefix: str = config["file_prefix"]

    if args.all:
        seasons = ALL_SEASONS
    elif args.seasons:
        seasons = args.seasons
    else:
        seasons = DEFAULT_SEASONS

    output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Sport: {args.sport} (harvester: {harvester_sport})")
    log.info(f"League: {league}")
    log.info(f"Market: {market}")
    log.info(f"Output directory: {output_dir}")
    log.info(f"Seasons: {seasons}")
    log.info(f"Odds history: {'yes' if args.odds_history else 'no'}")
    log.info(f"Force: {'yes' if args.force else 'no'}")
    if args.proxy_url:
        log.info(f"Proxy: {args.proxy_url}")
    log.info("")

    # Preview existing state
    for season in seasons:
        out = output_dir / f"{file_prefix}_{season}.json"
        existing = load_existing(out, market_key)
        if existing:
            complete = sum(
                1 for r in existing.values() if is_complete(r, args.odds_history, market_key)
            )
            log.info(f"  {season}: {len(existing)} matches ({complete} complete)")
        else:
            log.info(f"  {season}: no data yet")

    if args.dry_run:
        log.info("\nDry run — exiting.")
        return

    log.info("")
    total_matches = 0
    total_start = time.time()

    for i, season in enumerate(seasons):
        log.info(f"[{i + 1}/{len(seasons)}] {season}")
        count = scrape_season(
            season,
            sport=harvester_sport,
            league=league,
            market=market,
            market_key=market_key,
            file_prefix=file_prefix,
            output_dir=output_dir,
            odds_history=args.odds_history,
            force=args.force,
            proxy_url=args.proxy_url,
            proxy_user=args.proxy_user,
            proxy_pass=args.proxy_pass,
        )
        total_matches += count

    total_elapsed = time.time() - total_start
    log.info(
        f"\nDone. {total_matches} total matches across {len(seasons)} season(s) "
        f"in {total_elapsed / 60:.1f} min."
    )


if __name__ == "__main__":
    main()
