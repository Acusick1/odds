"""Fetch NBA team game logs from stats.nba.com via Playwright.

stats.nba.com uses Akamai bot detection that blocks raw HTTP requests.
Playwright launches a headless browser to establish a valid session,
then makes API calls from the browser context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import structlog

logger = structlog.get_logger(__name__)

# LeagueGameFinder column order (from stats.nba.com API response)
_COLUMNS = [
    "SEASON_ID",
    "TEAM_ID",
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "PTS",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
]


@dataclass(slots=True)
class GameLogRecord:
    """Parsed game log record ready for database storage."""

    nba_game_id: str
    team_id: int
    team_abbreviation: str
    team_name: str
    game_date: date
    matchup: str
    wl: str | None
    season: str
    pts: int | None
    fgm: int | None
    fga: int | None
    fg3m: int | None
    fg3a: int | None
    ftm: int | None
    fta: int | None
    oreb: int | None
    dreb: int | None
    reb: int | None
    ast: int | None
    stl: int | None
    blk: int | None
    tov: int | None
    pf: int | None
    plus_minus: int | None


def _safe_int(value: object) -> int | None:
    """Convert a value to int, returning None for None/empty."""
    if value is None or value == "":
        return None
    return int(value)  # type: ignore[arg-type]


def _parse_game_date(date_str: str) -> date:
    """Parse 'APR 13, 2025' or '2025-04-13' format game dates."""
    for fmt in ("%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    msg = f"Cannot parse game date: {date_str!r}"
    raise ValueError(msg)


def _row_to_record(row: list, season: str) -> GameLogRecord:
    """Convert a raw API row (positional array) to a GameLogRecord."""
    col = {name: row[i] for i, name in enumerate(_COLUMNS)}
    return GameLogRecord(
        nba_game_id=str(col["GAME_ID"]),
        team_id=int(col["TEAM_ID"]),
        team_abbreviation=str(col["TEAM_ABBREVIATION"]),
        team_name=str(col["TEAM_NAME"]),
        game_date=_parse_game_date(str(col["GAME_DATE"])),
        matchup=str(col["MATCHUP"]),
        wl=col["WL"] if col["WL"] else None,
        season=season,
        pts=_safe_int(col["PTS"]),
        fgm=_safe_int(col["FGM"]),
        fga=_safe_int(col["FGA"]),
        fg3m=_safe_int(col["FG3M"]),
        fg3a=_safe_int(col["FG3A"]),
        ftm=_safe_int(col["FTM"]),
        fta=_safe_int(col["FTA"]),
        oreb=_safe_int(col["OREB"]),
        dreb=_safe_int(col["DREB"]),
        reb=_safe_int(col["REB"]),
        ast=_safe_int(col["AST"]),
        stl=_safe_int(col["STL"]),
        blk=_safe_int(col["BLK"]),
        tov=_safe_int(col["TOV"]),
        pf=_safe_int(col["PF"]),
        plus_minus=_safe_int(col["PLUS_MINUS"]),
    )


def _parse_response(raw_json: dict, season: str) -> list[GameLogRecord]:
    """Parse LeagueGameFinder JSON response into GameLogRecord instances."""
    result_sets = raw_json.get("resultSets", [])
    if not result_sets:
        return []

    rows = result_sets[0].get("rowSet", [])
    records: list[GameLogRecord] = []
    for row in rows:
        try:
            records.append(_row_to_record(row, season))
        except (ValueError, KeyError, IndexError) as e:
            logger.warning("game_log_parse_error", error=str(e), row=row[:5])
    return records


def fetch_game_logs(season: str) -> list[GameLogRecord]:
    """Fetch all team game logs for a season via Playwright.

    Launches headless Chrome, navigates to nba.com to establish cookies,
    then calls the LeagueGameFinder API from the browser context.

    Args:
        season: NBA season string e.g. '2024-25'.

    Returns:
        List of GameLogRecord instances (~2,460 for a full regular season).
    """
    from playwright.sync_api import sync_playwright

    api_url = (
        "https://stats.nba.com/stats/leaguegamefinder"
        f"?PlayerOrTeam=T&Season={season}&LeagueID=00"
        "&SeasonType=Regular+Season"
    )

    js_fetch = """
    async (url) => {
        const resp = await fetch(url, {
            headers: {
                'Accept': 'application/json',
                'Referer': 'https://www.nba.com/',
                'Origin': 'https://www.nba.com'
            }
        });
        if (!resp.ok) {
            return { error: true, status: resp.status, statusText: resp.statusText };
        }
        return await resp.json();
    }
    """

    with sync_playwright() as p:
        # Firefox avoids HTTP/2 protocol errors that Chromium hits on nba.com in WSL2
        browser = p.firefox.launch(headless=True)
        page = browser.new_page()

        # Navigate to nba.com to establish Akamai session cookies
        logger.info("game_log_establishing_session")
        page.goto(
            "https://www.nba.com/",
            wait_until="domcontentloaded",
            timeout=30000,
        )

        # Make the API call from the browser context
        logger.info("game_log_fetching", season=season, url=api_url)
        result = page.evaluate(js_fetch, api_url)

        browser.close()

    if isinstance(result, dict) and result.get("error"):
        msg = f"stats.nba.com returned {result.get('status')}: {result.get('statusText')}"
        raise RuntimeError(msg)

    records = _parse_response(result, season)
    logger.info("game_log_fetched", season=season, count=len(records))
    return records
