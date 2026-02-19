"""Time utility helpers for consistent timezone handling."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def ensure_utc(dt: datetime) -> datetime:
    """Return datetime guaranteed to be timezone-aware in UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_api_datetime(value: str) -> datetime:
    """Parse The Odds API datetime strings as UTC-aware datetimes."""
    value = value.strip()
    # Replace trailing Z with explicit UTC offset so fromisoformat works cross-version
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    return ensure_utc(dt)


def to_eastern(dt: datetime) -> datetime:
    """Convert datetime to US Eastern (handles EST/EDT automatically)."""
    return ensure_utc(dt).astimezone(EASTERN)


def utc_isoformat(dt: datetime) -> str:
    """Serialize datetime as ISO 8601 string with trailing Z."""
    return ensure_utc(dt).isoformat().replace("+00:00", "Z")
