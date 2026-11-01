"""Tests for core.time utilities."""

from datetime import UTC, date, datetime

from odds_core.time import (
    EASTERN,
    ensure_utc,
    parse_api_datetime,
    parse_date,
    to_eastern,
    utc_isoformat,
)


class TestTimeUtils:
    """Validate timezone utility helpers."""

    def test_parse_api_datetime_returns_aware(self):
        """Ensure API datetime strings become UTC-aware datetimes."""
        parsed = parse_api_datetime("2024-01-15T19:00:00Z")
        assert parsed.tzinfo is not None
        assert parsed.tzinfo == UTC
        assert parsed.hour == 19

    def test_ensure_utc_converts_naive(self):
        """Naive datetimes should be promoted to UTC-aware values."""
        naive = datetime(2024, 1, 15, 19, 0, 0)  # noqa: DTZ001
        ensured = ensure_utc(naive)
        assert ensured.tzinfo == UTC
        assert ensured.hour == 19

    def test_utc_isoformat_uses_z_suffix(self):
        """ISO formatter should emit trailing Z for UTC datetimes."""
        dt = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)
        assert utc_isoformat(dt).endswith("Z")
        assert utc_isoformat(dt) == "2024-01-15T19:00:00Z"

    def test_to_eastern_utc_midnight_rollover(self):
        """Saturday 00:00 UTC should convert to Friday 7pm ET (EST)."""
        dt = datetime(2025, 1, 4, 0, 0, 0, tzinfo=UTC)  # Sat UTC
        eastern = to_eastern(dt)
        assert eastern.weekday() == 4  # Friday
        assert eastern.hour == 19
        assert eastern.tzinfo == EASTERN

    def test_to_eastern_dst_handling(self):
        """EST (winter, UTC-5) and EDT (summer, UTC-4) produce correct offsets."""
        winter = datetime(2025, 1, 4, 0, 0, 0, tzinfo=UTC)
        summer = datetime(2025, 7, 4, 0, 0, 0, tzinfo=UTC)

        assert to_eastern(winter).hour == 19  # UTC-5
        assert to_eastern(summer).hour == 20  # UTC-4

    def test_to_eastern_naive_treated_as_utc(self):
        """Naive datetimes should be treated as UTC then converted."""
        naive = datetime(2025, 1, 4, 0, 0, 0)  # noqa: DTZ001
        eastern = to_eastern(naive)
        assert eastern.weekday() == 4  # Friday
        assert eastern.hour == 19


class TestParseDate:
    def test_returns_the_calendar_date(self) -> None:
        assert parse_date("03/15/2025", "%m/%d/%Y") == date(2025, 3, 15)
