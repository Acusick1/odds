"""Tests for core.time utilities."""

from datetime import UTC, datetime

from core.time import ensure_utc, parse_api_datetime, utc_isoformat


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
        naive = datetime(2024, 1, 15, 19, 0, 0)
        ensured = ensure_utc(naive)
        assert ensured.tzinfo == UTC
        assert ensured.hour == 19

    def test_utc_isoformat_uses_z_suffix(self):
        """ISO formatter should emit trailing Z for UTC datetimes."""
        dt = datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC)
        assert utc_isoformat(dt).endswith("Z")
        assert utc_isoformat(dt) == "2024-01-15T19:00:00Z"
