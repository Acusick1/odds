"""Convert OddsHarvester 'upcoming' output to pipeline raw_data format.

Pure conversion module — no database access, no subprocess calls.
Produces dicts matching the OddsWriter.store_odds_snapshot() contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

DRAW_OUTCOME = "Draw"

# OddsPortal bookmaker name → pipeline key.
# Copied from ingest_oddsportal.py and extended for upcoming-only bookmakers.
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
    # Upcoming-only bookmakers (not in historical scrapes)
    "7Bet": "7bet",
    "Paddy Power": "paddypower",
    "Skybet": "skybet",
    "Ladbrokes": "ladbrokes",
    "Coral": "coral",
    "William Hill": "williamhill",
    "888sport": "888sport",
    "BoyleSports": "boylesports",
}

# Regex: Betfair format is "FRAC_REPEAT(LIQUIDITY)" e.g. "99/10099/100(300)"
# The fraction is repeated (concatenated), followed by optional (liquidity).
_BETFAIR_RE = re.compile(r"^(\d+/\d+)\1\((\d+)\)$")


@dataclass
class MatchOdds:
    """Converted match ready for ingestion."""

    home_team: str
    away_team: str
    match_date: datetime
    match_link: str
    scraped_date: datetime
    raw_data: dict[str, Any]
    bookmaker_count: int
    league_name: str = ""
    venue: str = ""


def fractional_to_decimal(frac: str) -> float:
    """Convert fractional odds string to decimal odds.

    Handles standard fractions ("9/10"), EVS/evens ("EVS", "1/1"),
    and whole numbers ("3").
    """
    frac = frac.strip()
    if frac.upper() in ("EVS", "EVENS"):
        return 2.0
    if "/" in frac:
        num, den = frac.split("/", 1)
        return float(num) / float(den) + 1.0
    return float(frac) + 1.0


def parse_betfair_odds(raw: str) -> tuple[str, int | None]:
    """Extract fraction and liquidity from Betfair Exchange format.

    Betfair entries look like "99/10099/100(300)" — the fraction repeated
    then "(liquidity)". Normal fractions pass through unchanged.

    Returns:
        (fraction_str, liquidity_or_None)
    """
    raw = raw.strip()
    m = _BETFAIR_RE.match(raw)
    if m:
        return m.group(1), int(m.group(2))
    return raw, None


def normalize_bookmaker_key(name: str) -> str:
    """Map OddsPortal bookmaker name to pipeline key."""
    return BOOKMAKER_KEY_MAP.get(name, _slugify(name))


def _slugify(name: str) -> str:
    """Convert bookmaker name to a lowercase slug key."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def decimal_to_american(d: float) -> int:
    """Convert decimal odds to American odds."""
    if d >= 2.0:
        return round((d - 1) * 100)
    elif d > 1.0:
        return round(-100 / (d - 1))
    return -10000


def parse_match_date(date_str: str) -> datetime:
    """Parse OddsPortal date string to UTC datetime.

    Format: '2026-03-03 19:30:00 UTC'
    """
    return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=UTC)


def convert_upcoming_matches(matches: list[dict[str, Any]], market: str) -> list[MatchOdds]:
    """Convert OddsHarvester upcoming output to pipeline format.

    Args:
        matches: Raw match dicts from OddsHarvester upcoming command.
        market: Market key — "1x2" or "over_under_2_5".

    Returns:
        List of MatchOdds with raw_data matching OddsWriter contract.
    """
    converter = _MARKET_CONVERTERS.get(market)
    if converter is None:
        raise ValueError(f"Unsupported market: {market}")

    results: list[MatchOdds] = []
    json_key = market + "_market"

    for match in matches:
        home = match.get("home_team", "").strip()
        away = match.get("away_team", "").strip()
        date_str = match.get("match_date", "")
        scraped_str = match.get("scraped_date", "")
        market_data = match.get(json_key, [])

        if not home or not away or not date_str or not market_data:
            continue

        match_dt = parse_match_date(date_str)
        scraped_dt = parse_match_date(scraped_str) if scraped_str else datetime.now(UTC)

        raw_data = converter(market_data, home, away)
        if raw_data is None:
            continue

        results.append(
            MatchOdds(
                home_team=home,
                away_team=away,
                match_date=match_dt,
                match_link=match.get("match_link", ""),
                scraped_date=scraped_dt,
                raw_data=raw_data,
                bookmaker_count=len(raw_data["bookmakers"]),
                league_name=match.get("league_name", ""),
                venue=match.get("venue", ""),
            )
        )

    return results


def _convert_1x2_match(
    bookmaker_odds: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    """Convert 1x2 market bookmaker list to raw_data format."""
    bookmakers: list[dict[str, Any]] = []

    for bk in bookmaker_odds:
        bk_name = bk.get("bookmaker_name", "")
        if not bk_name:
            continue

        home_raw = bk.get("1", "")
        draw_raw = bk.get("X", "")
        away_raw = bk.get("2", "")

        if not home_raw or not draw_raw or not away_raw:
            continue

        bk_key = normalize_bookmaker_key(bk_name)
        is_betfair = bk_name == "Betfair Exchange"

        if is_betfair:
            home_frac, _ = parse_betfair_odds(home_raw)
            draw_frac, _ = parse_betfair_odds(draw_raw)
            away_frac, _ = parse_betfair_odds(away_raw)
        else:
            home_frac, draw_frac, away_frac = home_raw, draw_raw, away_raw

        try:
            home_dec = fractional_to_decimal(home_frac)
            draw_dec = fractional_to_decimal(draw_frac)
            away_dec = fractional_to_decimal(away_frac)
        except (ValueError, ZeroDivisionError):
            continue

        outcomes = [
            {"name": home_team, "price": decimal_to_american(home_dec)},
            {"name": DRAW_OUTCOME, "price": decimal_to_american(draw_dec)},
            {"name": away_team, "price": decimal_to_american(away_dec)},
        ]

        bookmakers.append(
            {
                "key": bk_key,
                "title": bk_name,
                "last_update": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "markets": [{"key": "h2h", "outcomes": outcomes}],
            }
        )

    if not bookmakers:
        return None

    return {"bookmakers": bookmakers, "source": "oddsportal_live"}


def _convert_over_under_match(
    bookmaker_odds: list[dict[str, Any]],
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    """Convert over/under 2.5 market bookmaker list to raw_data format."""
    bookmakers: list[dict[str, Any]] = []

    for bk in bookmaker_odds:
        bk_name = bk.get("bookmaker_name", "")
        if not bk_name:
            continue

        over_raw = bk.get("odds_over", "")
        under_raw = bk.get("odds_under", "")

        if not over_raw or not under_raw:
            continue

        bk_key = normalize_bookmaker_key(bk_name)
        is_betfair = bk_name == "Betfair Exchange"

        if is_betfair:
            over_frac, _ = parse_betfair_odds(over_raw)
            under_frac, _ = parse_betfair_odds(under_raw)
        else:
            over_frac, under_frac = over_raw, under_raw

        try:
            over_dec = fractional_to_decimal(over_frac)
            under_dec = fractional_to_decimal(under_frac)
        except (ValueError, ZeroDivisionError):
            continue

        outcomes = [
            {"name": "Over", "price": decimal_to_american(over_dec), "point": 2.5},
            {"name": "Under", "price": decimal_to_american(under_dec), "point": 2.5},
        ]

        bookmakers.append(
            {
                "key": bk_key,
                "title": bk_name,
                "last_update": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "markets": [{"key": "totals", "outcomes": outcomes}],
            }
        )

    if not bookmakers:
        return None

    return {"bookmakers": bookmakers, "source": "oddsportal_live"}


_MARKET_CONVERTERS: dict[str, Any] = {
    "1x2": _convert_1x2_match,
    "over_under_2_5": _convert_over_under_match,
}
