"""Supported sport keys for the odds pipeline.

Values match The Odds API's sport key convention and are used as the
``sport_key`` on ``Event`` rows and as the ``sport`` argument across
scheduler jobs, MCP tools, and CLI commands.
"""

from datetime import date, datetime, time, timedelta
from typing import Literal, get_args
from zoneinfo import ZoneInfo

SportKey = Literal["soccer_epl", "baseball_mlb", "basketball_nba"]

SUPPORTED_SPORTS: tuple[SportKey, ...] = get_args(SportKey)

LEAGUE_TIMEZONES: dict[SportKey, ZoneInfo] = {
    "soccer_epl": ZoneInfo("Europe/London"),
    "baseball_mlb": ZoneInfo("America/New_York"),
    "basketball_nba": ZoneInfo("America/New_York"),
}


def game_day_window(sport: SportKey, day: date) -> tuple[datetime, datetime]:
    """The half-open span of ``day`` on the league's local calendar."""
    tz = LEAGUE_TIMEZONES[sport]
    return (
        datetime.combine(day, time(), tzinfo=tz),
        datetime.combine(day + timedelta(days=1), time(), tzinfo=tz),
    )
