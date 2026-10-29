"""API response models and conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from odds_core.team import normalize_team_name
from odds_core.time import parse_api_datetime


@dataclass(frozen=True, slots=True)
class ApiEvent:
    """A game as The Odds API lists it; ``odds_api_id`` is the API's id for it, which the
    resolver records as the event's ``odds_api`` external id."""

    odds_api_id: str
    sport_key: str
    sport_title: str
    commence_time: datetime
    home_team: str
    away_team: str


@dataclass(slots=True)
class OddsResponse:
    """Response from get_odds() API call."""

    events: list[ApiEvent]
    raw_events_data: list[dict]  # Original API dicts for snapshot storage
    response_time_ms: int
    quota_remaining: int | None
    timestamp: datetime


@dataclass(slots=True)
class ScoresResponse:
    """Response from get_scores() API call."""

    scores_data: list[dict]  # Keep as dict since we just extract scores
    response_time_ms: int
    quota_remaining: int | None
    timestamp: datetime


@dataclass(slots=True)
class EventsResponse:
    """Response from get_events() API call (free endpoint, no bookmaker data)."""

    events: list[ApiEvent]
    response_time_ms: int
    quota_remaining: int | None
    timestamp: datetime


@dataclass(slots=True)
class HistoricalOddsResponse:
    """Response from get_historical_odds() API call."""

    events: list[ApiEvent]
    raw_events_data: list[dict]
    response_time_ms: int
    quota_remaining: int | None
    timestamp: datetime


def parse_scores_from_api_dict(score_data: dict) -> tuple[int | None, int | None]:
    """
    Extract home and away scores from API scores response.

    Args:
        score_data: Score data from The Odds API scores endpoint
            Expected format:
            {
                "home_team": "Lakers",
                "away_team": "Celtics",
                "scores": [
                    {"name": "Lakers", "score": "108"},
                    {"name": "Celtics", "score": "105"}
                ]
            }

    Returns:
        Tuple of (home_score, away_score), either may be None if not found

    Example:
        >>> score_dict = {
        ...     "home_team": "Lakers",
        ...     "away_team": "Celtics",
        ...     "scores": [
        ...         {"name": "Lakers", "score": "108"},
        ...         {"name": "Celtics", "score": "105"}
        ...     ]
        ... }
        >>> home, away = parse_scores_from_api_dict(score_dict)
        >>> assert home == 108 and away == 105
    """
    home_team = score_data.get("home_team")
    away_team = score_data.get("away_team")
    scores = score_data.get("scores", [])

    home_score = None
    away_score = None

    for score in scores:
        score_name = score.get("name")
        score_value = score.get("score")

        if score_name == home_team and score_value is not None:
            try:
                home_score = int(score_value)
            except (ValueError, TypeError):
                pass  # Invalid score, leave as None

        if score_name == away_team and score_value is not None:
            try:
                away_score = int(score_value)
            except (ValueError, TypeError):
                pass  # Invalid score, leave as None

    return home_score, away_score


def parse_api_event(event_data: dict) -> ApiEvent:
    """The game an Odds API event dict lists, with canonical team names."""
    return ApiEvent(
        odds_api_id=event_data["id"],
        sport_key=event_data["sport_key"],
        sport_title=event_data.get("sport_title", event_data["sport_key"]),
        commence_time=parse_api_datetime(event_data["commence_time"]),
        home_team=normalize_team_name(event_data["home_team"]),
        away_team=normalize_team_name(event_data["away_team"]),
    )
