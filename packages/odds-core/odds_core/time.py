"""Time utility helpers for consistent timezone handling."""

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
# The month a football season (and the NBA's) starts in.
SEASON_START_MONTH = 8


# Sports whose season runs within one calendar year.
CALENDAR_YEAR_SPORTS = frozenset({"baseball_mlb"})


def season_start_year(commence_time: datetime) -> int:
    """The calendar year the season of a kick-off started in."""
    return (
        commence_time.year if commence_time.month >= SEASON_START_MONTH else commence_time.year - 1
    )


def season_start(now: datetime, sport_key: str) -> datetime:
    """The start of ``now``'s season: 1 January for a calendar-year sport, else 1 August."""
    if sport_key in CALENDAR_YEAR_SPORTS:
        return datetime(now.year, 1, 1, tzinfo=UTC)
    return datetime(season_start_year(now), SEASON_START_MONTH, 1, tzinfo=UTC)


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


def parse_date(value: str, fmt: str) -> date:
    """Parse a calendar date in ``fmt`` without constructing a naive datetime."""
    return datetime.strptime(value, fmt).replace(tzinfo=UTC).date()
