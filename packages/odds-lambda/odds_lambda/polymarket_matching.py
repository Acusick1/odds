"""Polymarket event matching to internal sportsbook Event records."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

import structlog
from odds_core.models import Event
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)

# Canonical team name → set of known aliases
TEAM_ALIASES: dict[str, set[str]] = {
    "Atlanta Hawks": {"Atlanta Hawks", "Hawks", "Atlanta"},
    "Boston Celtics": {"Boston Celtics", "Celtics", "Boston"},
    "Brooklyn Nets": {"Brooklyn Nets", "Nets", "Brooklyn"},
    "Charlotte Hornets": {"Charlotte Hornets", "Hornets", "Charlotte"},
    "Chicago Bulls": {"Chicago Bulls", "Bulls", "Chicago"},
    "Cleveland Cavaliers": {"Cleveland Cavaliers", "Cavaliers", "Cavs", "Cleveland"},
    "Dallas Mavericks": {"Dallas Mavericks", "Mavericks", "Mavs", "Dallas"},
    "Denver Nuggets": {"Denver Nuggets", "Nuggets", "Denver"},
    "Detroit Pistons": {"Detroit Pistons", "Pistons", "Detroit"},
    "Golden State Warriors": {"Golden State Warriors", "Warriors", "Golden State", "GSW"},
    "Houston Rockets": {"Houston Rockets", "Rockets", "Houston"},
    "Indiana Pacers": {"Indiana Pacers", "Pacers", "Indiana"},
    "Los Angeles Clippers": {"Los Angeles Clippers", "Clippers", "LA Clippers"},
    "Los Angeles Lakers": {"Los Angeles Lakers", "Lakers", "LA Lakers"},
    "Memphis Grizzlies": {"Memphis Grizzlies", "Grizzlies", "Memphis"},
    "Miami Heat": {"Miami Heat", "Heat", "Miami"},
    "Milwaukee Bucks": {"Milwaukee Bucks", "Bucks", "Milwaukee"},
    "Minnesota Timberwolves": {"Minnesota Timberwolves", "Timberwolves", "Wolves", "Minnesota"},
    "New Orleans Pelicans": {"New Orleans Pelicans", "Pelicans", "New Orleans"},
    "New York Knicks": {"New York Knicks", "Knicks", "New York"},
    "Oklahoma City Thunder": {"Oklahoma City Thunder", "Thunder", "Oklahoma City", "OKC"},
    "Orlando Magic": {"Orlando Magic", "Magic", "Orlando"},
    "Philadelphia 76ers": {"Philadelphia 76ers", "76ers", "Sixers", "Philadelphia"},
    "Phoenix Suns": {"Phoenix Suns", "Suns", "Phoenix"},
    "Portland Trail Blazers": {"Portland Trail Blazers", "Trail Blazers", "Blazers", "Portland"},
    "Sacramento Kings": {"Sacramento Kings", "Kings", "Sacramento"},
    "San Antonio Spurs": {"San Antonio Spurs", "Spurs", "San Antonio"},
    "Toronto Raptors": {"Toronto Raptors", "Raptors", "Toronto"},
    "Utah Jazz": {"Utah Jazz", "Jazz", "Utah"},
    "Washington Wizards": {"Washington Wizards", "Wizards", "Washington"},
}

# Polymarket ticker abbreviation → canonical team name
NBA_ABBREV_MAP: dict[str, str] = {
    "atl": "Atlanta Hawks",
    "bos": "Boston Celtics",
    "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets",
    "chi": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks",
    "den": "Denver Nuggets",
    "det": "Detroit Pistons",
    "gsw": "Golden State Warriors",
    "hou": "Houston Rockets",
    "ind": "Indiana Pacers",
    "lac": "Los Angeles Clippers",
    "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies",
    "mia": "Miami Heat",
    "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves",
    "nop": "New Orleans Pelicans",
    "nyk": "New York Knicks",
    "okc": "Oklahoma City Thunder",
    "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers",
    "phx": "Phoenix Suns",
    "por": "Portland Trail Blazers",
    "sac": "Sacramento Kings",
    "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors",
    "uta": "Utah Jazz",
    "was": "Washington Wizards",
}

# Pre-computed lowercase alias → canonical name for O(1) lookups
_ALIAS_TO_CANONICAL: dict[str, str] = {
    alias.lower(): canonical for canonical, aliases in TEAM_ALIASES.items() for alias in aliases
}

_TICKER_RE = re.compile(r"^nba-([a-z]+)-([a-z]+)-(\d{4})-(\d{2})-(\d{2})$")


def normalize_team(name: str) -> str | None:
    """Normalize a team name to its canonical form.

    Returns the canonical name (e.g. "Dallas Mavericks") or None if unrecognized.
    """
    return _ALIAS_TO_CANONICAL.get(name.strip().lower())


def parse_ticker(ticker: str) -> tuple[str, str, date] | None:
    """Parse a Polymarket NBA ticker into (away_abbrev, home_abbrev, game_date).

    Format: nba-{away}-{home}-{year}-{month}-{day}
    Returns None if the ticker does not match the expected pattern.
    """
    match = _TICKER_RE.match(ticker.lower())
    if not match:
        return None
    away_abbrev, home_abbrev, year, month, day = match.groups()
    try:
        game_date = date(int(year), int(month), int(day))
    except ValueError:
        return None
    return away_abbrev, home_abbrev, game_date


async def match_polymarket_event(
    session: AsyncSession,
    ticker: str,
    pm_start_date: datetime | None = None,
) -> str | None:
    """Find the matching internal Event record for a Polymarket event.

    Uses the ticker to derive team abbreviations and game date, then queries
    the Event table within a ±24-hour window of the ticker's game date.

    Args:
        session: Async database session
        ticker: Polymarket ticker (e.g. "nba-dal-mil-2026-01-25")
        pm_start_date: Unused, kept for backward compatibility

    Returns event_id if a confident single match is found, None otherwise.
    Never produces false positives — returns None when uncertain.
    """
    parsed = parse_ticker(ticker)
    if not parsed:
        logger.warning("polymarket_ticker_parse_failed", ticker=ticker)
        return None

    away_abbrev, home_abbrev, game_date = parsed

    away_canonical = NBA_ABBREV_MAP.get(away_abbrev)
    home_canonical = NBA_ABBREV_MAP.get(home_abbrev)

    if not away_canonical or not home_canonical:
        logger.warning(
            "polymarket_unknown_team_abbrev",
            ticker=ticker,
            away=away_abbrev,
            home=home_abbrev,
        )
        return None

    # Ticker dates use US local dates but sportsbook stores UTC timestamps.
    # Evening ET games on "Nov 22" have UTC commence times on Nov 23 (00:00-05:00 UTC).
    # Center the window on noon UTC to safely capture all US timezone games.
    game_noon_utc = datetime(game_date.year, game_date.month, game_date.day, 12, tzinfo=UTC)
    window_start = game_noon_utc - timedelta(hours=24)
    window_end = game_noon_utc + timedelta(hours=24)

    query = select(Event).where(
        and_(
            Event.commence_time >= window_start,
            Event.commence_time <= window_end,
            Event.away_team == away_canonical,
            Event.home_team == home_canonical,
        )
    )

    result = await session.execute(query)
    candidates = list(result.scalars().all())

    if len(candidates) == 1:
        event_id = candidates[0].id
        logger.info(
            "polymarket_event_matched",
            ticker=ticker,
            event_id=event_id,
            away=away_canonical,
            home=home_canonical,
        )
        return event_id

    if len(candidates) > 1:
        logger.warning(
            "polymarket_event_ambiguous_match",
            ticker=ticker,
            candidate_count=len(candidates),
        )
        return None

    logger.warning(
        "polymarket_event_no_match",
        ticker=ticker,
        away=away_canonical,
        home=home_canonical,
        window_start=window_start,
        window_end=window_end,
    )
    return None
