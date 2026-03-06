"""Shared utilities for OddsPortal data ingestion (historical and live)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any

DRAW_OUTCOME = "Draw"

# OddsPortal bookmaker name → pipeline key.
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


def slugify(name: str) -> str:
    """Convert bookmaker name to a lowercase slug key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def normalize_bookmaker_key(name: str) -> str:
    """Map OddsPortal bookmaker name to pipeline key."""
    return BOOKMAKER_KEY_MAP.get(name, slugify(name))


def decimal_to_american(d: float) -> int:
    """Convert decimal odds to American odds."""
    if d >= 2.0:
        return round((d - 1) * 100)
    elif d > 1.0:
        return round(-100 / (d - 1))
    return -10000


def parse_match_date(date_str: str) -> datetime:
    """Parse OddsPortal date string to UTC datetime.

    Format: '2024-10-04 16:00:00 UTC'
    """
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=UTC)


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


def team_abbrev(name: str) -> str:
    """Derive a short abbreviation from a team name.

    Single-word names get 3 chars (e.g. "Arsenal" -> "ARS").
    Multi-word names use first 3 chars of first + last word
    (e.g. "Manchester United" -> "MANUNI").
    """
    words = name.split()
    if len(words) == 1:
        return words[0][:3].upper()
    return (words[0][:3] + words[-1][:3]).upper()


def parse_odds_timestamp(ts_str: str) -> datetime:
    """Parse OddsPortal odds_history timestamp (naive, wrong year)."""
    return datetime.strptime(ts_str, "%Y-%m-%dT%H:%M:%S")


def fix_odds_timestamp(odds_ts: datetime, match_dt: datetime) -> datetime:
    """Fix the year in OddsPortal odds timestamps.

    OddsPortal records timestamps with the scrape year (e.g. 2026) instead
    of the actual game year. Month/day/time are correct.

    Returns a naive datetime (caller adds tzinfo as needed).
    """
    match_naive = match_dt.replace(tzinfo=None)

    try:
        fixed = odds_ts.replace(year=match_naive.year)
    except ValueError:
        fixed = odds_ts.replace(year=match_naive.year, day=28)

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


def build_raw_data(
    bookmaker_odds: list[dict],
    home_team: str,
    away_team: str,
    *,
    use_opening: bool,
    match_dt: datetime,
    num_outcomes: int = 3,
    db_market: str = "h2h",
    outcome_names: tuple[str, ...] | None = None,
    line: float | None = None,
) -> dict[str, Any] | None:
    """Convert OddsPortal bookmaker data into The Odds API raw_data format.

    Args:
        bookmaker_odds: List of bookmaker dicts from the market data
        home_team: Canonical home team name
        away_team: Canonical away team name
        use_opening: If True, use opening odds; if False, use closing (last in history)
        match_dt: Correct game datetime (for fixing timestamps)
        num_outcomes: Number of outcomes (2 for h2h/totals, 3 for 1x2)
        db_market: Market key for database (e.g. "h2h", "totals")
        outcome_names: Named outcomes (e.g. ("Over", "Under")); None uses team names
        line: Point line for totals markets

    Returns:
        Dict in The Odds API format, or None if no valid bookmakers found
    """
    bookmakers: list[dict[str, Any]] = []
    snapshot_time: datetime | None = None

    for bk in bookmaker_odds:
        hist = bk.get("odds_history_data")
        if not hist or len(hist) < num_outcomes:
            continue

        bk_name = bk["bookmaker_name"]
        bk_key = normalize_bookmaker_key(bk_name)

        entries: list[dict | None] = []
        for i in range(num_outcomes):
            outcome_hist = hist[i]
            if use_opening:
                entries.append(outcome_hist.get("opening_odds"))
            else:
                odds_history = outcome_hist.get("odds_history", [])
                entries.append(odds_history[-1] if odds_history else None)

        if any(e is None for e in entries):
            continue

        decimals = [e.get("odds") for e in entries]  # type: ignore[union-attr]
        if any(d is None for d in decimals):
            continue

        if outcome_names:
            outcomes: list[dict[str, Any]] = []
            for name, decimal in zip(outcome_names, decimals, strict=True):
                outcome: dict[str, Any] = {
                    "name": name,
                    "price": decimal_to_american(decimal),
                }
                if line is not None:
                    outcome["point"] = line
                outcomes.append(outcome)
        else:
            outcomes = [
                {"name": home_team, "price": decimal_to_american(decimals[0])},
            ]
            if num_outcomes >= 3:
                outcomes.append({"name": DRAW_OUTCOME, "price": decimal_to_american(decimals[1])})
            outcomes.append({"name": away_team, "price": decimal_to_american(decimals[-1])})

        raw_ts = parse_odds_timestamp(entries[0]["timestamp"])  # type: ignore[index]
        fixed_ts = fix_odds_timestamp(raw_ts, match_dt)

        if snapshot_time is None or fixed_ts > snapshot_time:
            snapshot_time = fixed_ts

        bookmakers.append(
            {
                "key": bk_key,
                "title": bk_name,
                "last_update": fixed_ts.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z"),
                "markets": [{"key": db_market, "outcomes": outcomes}],
            }
        )

    if not bookmakers:
        return None

    return {
        "bookmakers": bookmakers,
        "source": "oddsportal",
        "_snapshot_time": snapshot_time.replace(tzinfo=UTC).isoformat() if snapshot_time else None,
    }
