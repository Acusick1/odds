"""API response models and conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from odds_core.models import Event, EventStatus
from odds_core.time import parse_api_datetime


@dataclass(slots=True)
class OddsResponse:
    """Response from get_odds() API call."""

    events: list[Event]
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
class HistoricalOddsResponse:
    """Response from get_historical_odds() API call."""

    events: list[Event]
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


def create_scheduled_event(event_data: dict) -> Event:
    """
    Create Event instance for scheduled upcoming game (no scores).

    Args:
        event_data: Event data from The Odds API odds endpoint

    Returns:
        Event instance with SCHEDULED status and no scores

    Example:
        >>> event_dict = {
        ...     "id": "abc123",
        ...     "sport_key": "basketball_nba",
        ...     "commence_time": "2024-10-20T00:00:00Z",
        ...     "home_team": "Lakers",
        ...     "away_team": "Celtics"
        ... }
        >>> event = create_scheduled_event(event_dict)
        >>> assert event.status == EventStatus.SCHEDULED
        >>> assert event.home_score is None
    """
    # Parse commence_time as timezone-aware UTC
    commence_time = parse_api_datetime(event_data["commence_time"])

    return Event(
        id=event_data["id"],
        sport_key=event_data["sport_key"],
        sport_title=event_data.get("sport_title", event_data["sport_key"]),
        commence_time=commence_time,
        home_team=event_data["home_team"],
        away_team=event_data["away_team"],
        status=EventStatus.SCHEDULED,
    )


def create_completed_event(event_data: dict) -> Event:
    """
    Create Event instance for completed game with final scores.

    Args:
        event_data: Event data from The Odds API scores endpoint
            Must contain "scores" field with final results

    Returns:
        Event instance with FINAL status and scores populated

    Raises:
        ValueError: If scores are missing or invalid

    Example:
        >>> event_dict = {
        ...     "id": "abc123",
        ...     "sport_key": "basketball_nba",
        ...     "commence_time": "2024-10-20T00:00:00Z",
        ...     "home_team": "Lakers",
        ...     "away_team": "Celtics",
        ...     "scores": [
        ...         {"name": "Lakers", "score": "108"},
        ...         {"name": "Celtics", "score": "105"}
        ...     ]
        ... }
        >>> event = create_completed_event(event_dict)
        >>> assert event.status == EventStatus.FINAL
        >>> assert event.home_score == 108
        >>> assert event.away_score == 105
    """
    # Parse commence_time as timezone-aware UTC
    commence_time = parse_api_datetime(event_data["commence_time"])

    # Extract scores using helper
    home_score, away_score = parse_scores_from_api_dict(event_data)

    # Validate scores are present
    if home_score is None or away_score is None:
        msg = f"Missing or invalid scores for event {event_data.get('id')}"
        raise ValueError(msg)

    return Event(
        id=event_data["id"],
        sport_key=event_data["sport_key"],
        sport_title=event_data.get("sport_title", event_data["sport_key"]),
        commence_time=commence_time,
        home_team=event_data["home_team"],
        away_team=event_data["away_team"],
        status=EventStatus.FINAL,
        home_score=home_score,
        away_score=away_score,
    )


def api_dict_to_event(event_data: dict) -> Event:
    """
    Convert API response dict to Event instance.

    .. deprecated::
        Use create_scheduled_event() or create_completed_event() instead.
        This function is maintained for backward compatibility.

    Args:
        event_data: Event data from The Odds API

    Returns:
        Event instance with parsed data
    """
    return create_scheduled_event(event_data)
