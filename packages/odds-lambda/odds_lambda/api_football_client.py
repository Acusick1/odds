"""API-Football v3 client for confirmed lineup data.

Handles fixture ID lookup (team name + date matching) and lineup fetching.
Free tier: 100 req/day, 10 req/min.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import aiohttp
import structlog
from odds_core.config import get_settings
from odds_core.team import normalize_team_name

logger = structlog.get_logger(__name__)

# API-Football team name -> pipeline canonical name.
# Keys are lowercased for case-insensitive lookup.
_API_FOOTBALL_ALIASES: dict[str, str] = {
    "manchester united": "Manchester Utd",
    "newcastle": "Newcastle",
    "nottingham forest": "Nottingham",
    "tottenham": "Tottenham",
    "wolverhampton": "Wolves",
    "wolverhampton wanderers": "Wolves",
    "west ham": "West Ham",
    "afc bournemouth": "Bournemouth",
    "brighton": "Brighton",
    "leicester": "Leicester",
    "ipswich": "Ipswich",
    "sheffield utd": "Sheffield Utd",
}


def _normalize_api_football_name(name: str) -> str:
    """Normalize an API-Football team name to pipeline canonical form.

    Tries API-Football-specific aliases first, then falls back to the
    standard normalize_team_name.
    """
    cleaned = " ".join(name.split())
    alias = _API_FOOTBALL_ALIASES.get(cleaned.lower())
    if alias is not None:
        return alias
    return normalize_team_name(cleaned)


class ApiFootballClient:
    """Async client for API-Football v3 endpoints."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.api_football.key
        self.base_url = base_url or settings.api_football.base_url
        self.league_id = settings.api_football.epl_league_id

        if not self.api_key:
            raise ValueError("API_FOOTBALL_KEY is not configured")

    async def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        """Make an authenticated GET request to API-Football."""
        url = f"{self.base_url}/{endpoint}"
        headers = {"x-apisports-key": self.api_key}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.error(
                        "api_football_request_failed",
                        endpoint=endpoint,
                        status=resp.status,
                        body=body[:500],
                    )
                    raise RuntimeError(f"API-Football request failed: {resp.status} {body[:200]}")
                data: dict[str, Any] = await resp.json()

        errors = data.get("errors")
        if errors:
            logger.error("api_football_api_error", endpoint=endpoint, errors=errors)
            raise RuntimeError(f"API-Football error: {errors}")

        return data

    async def find_fixture(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
    ) -> dict[str, Any] | None:
        """Find an API-Football fixture by team names and date.

        Fetches all EPL fixtures for the match date and fuzzy-matches
        team names against pipeline canonical names.

        Returns the fixture dict or None if no match found.
        """
        date_str = match_date.strftime("%Y-%m-%d")
        season = _infer_season(match_date)

        data = await self._request(
            "fixtures",
            {
                "league": self.league_id,
                "season": season,
                "date": date_str,
            },
        )

        fixtures = data.get("response", [])
        if not fixtures:
            logger.info(
                "api_football_no_fixtures",
                date=date_str,
                season=season,
            )
            return None

        home_canonical = normalize_team_name(home_team)
        away_canonical = normalize_team_name(away_team)

        for fixture in fixtures:
            teams = fixture.get("teams", {})
            api_home = _normalize_api_football_name(teams.get("home", {}).get("name", ""))
            api_away = _normalize_api_football_name(teams.get("away", {}).get("name", ""))

            if api_home == home_canonical and api_away == away_canonical:
                return fixture

        logger.warning(
            "api_football_fixture_not_matched",
            home_team=home_canonical,
            away_team=away_canonical,
            date=date_str,
            available=[
                f"{f['teams']['home']['name']} vs {f['teams']['away']['name']}" for f in fixtures
            ],
        )
        return None

    async def get_lineups(self, fixture_id: int) -> list[dict[str, Any]]:
        """Fetch confirmed lineups for a fixture.

        Returns a list of team lineup dicts (typically 2, one per team).
        Returns empty list if lineups not yet available.
        """
        data = await self._request("fixtures/lineups", {"fixture": fixture_id})
        return data.get("response", [])


def _infer_season(dt: datetime) -> int:
    """Infer the EPL season year from a match date.

    EPL seasons run Aug-May. A match in Jan 2026 belongs to season 2025.
    """
    if dt.month >= 8:
        return dt.year
    return dt.year - 1


def parse_lineup_response(
    team_data: dict[str, Any],
) -> dict[str, Any]:
    """Parse a single team's lineup from API-Football response into a flat dict.

    Returns dict with: team_name, team_id, formation, coach, start_xi, substitutes.
    """
    team_info = team_data.get("team", {})
    coach_info = team_data.get("coach", {})

    start_xi_raw = team_data.get("startXI", [])
    subs_raw = team_data.get("substitutes", [])

    start_xi = [_parse_player(p) for p in start_xi_raw]
    substitutes = [_parse_player(p) for p in subs_raw]

    coach: dict[str, Any] | None = None
    if coach_info and coach_info.get("id"):
        coach = {
            "id": coach_info["id"],
            "name": coach_info.get("name"),
        }

    return {
        "team_name": team_info.get("name", ""),
        "team_id": team_info.get("id", 0),
        "formation": team_data.get("formation"),
        "coach": coach,
        "start_xi": start_xi,
        "substitutes": substitutes,
    }


def _parse_player(entry: dict[str, Any]) -> dict[str, Any]:
    """Parse a player entry from API-Football lineup response."""
    player = entry.get("player", {})
    return {
        "id": player.get("id"),
        "name": player.get("name"),
        "number": player.get("number"),
        "pos": player.get("pos"),
        "grid": player.get("grid"),
    }


async def fetch_lineups_for_event(
    event_id: str,
    home_team: str,
    away_team: str,
    commence_time: datetime,
) -> dict[str, Any]:
    """High-level: fetch confirmed lineups for a pipeline event.

    Returns a dict with fixture_id, lineups (list of parsed team dicts),
    or an error/not-available message.
    """
    client = ApiFootballClient()

    fixture = await client.find_fixture(home_team, away_team, commence_time)
    if fixture is None:
        return {
            "available": False,
            "message": f"No API-Football fixture found for {home_team} vs {away_team} "
            f"on {commence_time.strftime('%Y-%m-%d')}",
        }

    fixture_id = fixture["fixture"]["id"]

    raw_lineups = await client.get_lineups(fixture_id)
    if not raw_lineups:
        hours_until = (commence_time - datetime.now(UTC)).total_seconds() / 3600
        return {
            "available": False,
            "fixture_id": fixture_id,
            "message": (
                f"Lineups not yet available for fixture {fixture_id}. "
                f"EPL lineups typically appear ~55-75 min pre-KO. "
                f"Kickoff in {hours_until:.1f} hours."
            ),
        }

    parsed = [parse_lineup_response(team) for team in raw_lineups]

    return {
        "available": True,
        "fixture_id": fixture_id,
        "lineups": parsed,
    }
