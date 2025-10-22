"""API response models and conversion utilities."""

from dataclasses import dataclass
from datetime import datetime

from core.models import Event, EventStatus


@dataclass
class OddsResponse:
    """Response from get_odds() API call."""

    events: list[Event]
    raw_events_data: list[dict]  # Original API dicts for snapshot storage
    response_time_ms: int
    quota_remaining: int | None
    timestamp: datetime


@dataclass
class ScoresResponse:
    """Response from get_scores() API call."""

    scores_data: list[dict]  # Keep as dict since we just extract scores
    response_time_ms: int
    quota_remaining: int | None
    timestamp: datetime


@dataclass
class HistoricalOddsResponse:
    """Response from get_historical_odds() API call."""

    events: list[Event]
    raw_events_data: list[dict]
    response_time_ms: int
    quota_remaining: int | None
    timestamp: datetime


def api_dict_to_event(event_data: dict) -> Event:
    """
    Convert API response dict to Event instance.

    This is the single source of truth for API → Event conversion.

    Args:
        event_data: Event data from The Odds API

    Returns:
        Event instance with parsed data

    Example:
        >>> event_dict = {
        ...     "id": "abc123",
        ...     "sport_key": "basketball_nba",
        ...     "commence_time": "2024-10-20T00:00:00Z",
        ...     "home_team": "Lakers",
        ...     "away_team": "Celtics"
        ... }
        >>> event = api_dict_to_event(event_dict)
    """
    # Parse commence_time
    commence_time_str = event_data["commence_time"].replace("Z", "+00:00")
    commence_time = datetime.fromisoformat(commence_time_str)

    return Event(
        id=event_data["id"],
        sport_key=event_data["sport_key"],
        sport_title=event_data.get("sport_title", event_data["sport_key"]),
        commence_time=commence_time,
        home_team=event_data["home_team"],
        away_team=event_data["away_team"],
        status=EventStatus.SCHEDULED,
    )
