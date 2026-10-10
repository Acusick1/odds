"""Unit tests for tier calculation utilities."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.tier_utils import (
    calculate_hours_until_commence,
    calculate_tier,
    calculate_tier_from_timestamps,
)


class TestCalculateTier:
    """Test tier calculation based on hours until game."""

    def test_closing_tier_lower_bound(self):
        """Test CLOSING tier at lower boundary (0 hours)."""
        assert calculate_tier(0.0) == FetchTier.CLOSING

    def test_closing_tier_upper_bound(self):
        """Test CLOSING tier at upper boundary (3 hours)."""
        assert calculate_tier(3.0) == FetchTier.CLOSING

    def test_closing_tier_middle(self):
        """Test CLOSING tier in middle of range."""
        assert calculate_tier(1.5) == FetchTier.CLOSING

    def test_pregame_tier_lower_bound(self):
        """Test PREGAME tier at lower boundary (just over 3 hours)."""
        assert calculate_tier(3.1) == FetchTier.PREGAME

    def test_pregame_tier_upper_bound(self):
        """Test PREGAME tier at upper boundary (12 hours)."""
        assert calculate_tier(12.0) == FetchTier.PREGAME

    def test_pregame_tier_middle(self):
        """Test PREGAME tier in middle of range."""
        assert calculate_tier(6.0) == FetchTier.PREGAME

    def test_sharp_tier_lower_bound(self):
        """Test SHARP tier at lower boundary (just over 12 hours)."""
        assert calculate_tier(12.1) == FetchTier.SHARP

    def test_sharp_tier_upper_bound(self):
        """Test SHARP tier at upper boundary (24 hours)."""
        assert calculate_tier(24.0) == FetchTier.SHARP

    def test_sharp_tier_middle(self):
        """Test SHARP tier in middle of range."""
        assert calculate_tier(18.0) == FetchTier.SHARP

    def test_early_tier_lower_bound(self):
        """Test EARLY tier at lower boundary (just over 24 hours)."""
        assert calculate_tier(24.1) == FetchTier.EARLY

    def test_early_tier_upper_bound(self):
        """Test EARLY tier at upper boundary (72 hours)."""
        assert calculate_tier(72.0) == FetchTier.EARLY

    def test_early_tier_middle(self):
        """Test EARLY tier in middle of range."""
        assert calculate_tier(48.0) == FetchTier.EARLY

    def test_opening_tier_lower_bound(self):
        """Test OPENING tier at lower boundary (just over 72 hours)."""
        assert calculate_tier(72.1) == FetchTier.OPENING

    def test_opening_tier_large_value(self):
        """Test OPENING tier with large value."""
        assert calculate_tier(200.0) == FetchTier.OPENING

    def test_negative_hours_returns_in_play(self):
        """Test that negative hours (game already started) returns IN_PLAY."""
        assert calculate_tier(-1.0) == FetchTier.IN_PLAY

    def test_in_play_large_negative(self):
        """Test IN_PLAY tier with large negative value (deep into game)."""
        assert calculate_tier(-3.0) == FetchTier.IN_PLAY

    def test_closing_at_zero_hours(self):
        """Test that exactly 0 hours returns CLOSING, not IN_PLAY."""
        assert calculate_tier(0.0) == FetchTier.CLOSING

    def test_tier_intervals_match_enum(self):
        """Test that tier calculation boundaries match FetchTier.interval_hours."""
        # IN_PLAY: negative hours (interval 0.5)
        assert calculate_tier(-1.0) == FetchTier.IN_PLAY
        assert FetchTier.IN_PLAY.interval_hours == 0.5

        # CLOSING: 0-3 hours (interval 0.5)
        assert calculate_tier(0.0) == FetchTier.CLOSING
        assert calculate_tier(3.0) == FetchTier.CLOSING
        assert FetchTier.CLOSING.interval_hours == 0.5

        # PREGAME: 3-12 hours (interval 3.0)
        assert calculate_tier(3.1) == FetchTier.PREGAME
        assert calculate_tier(12.0) == FetchTier.PREGAME
        assert FetchTier.PREGAME.interval_hours == 3.0

        # SHARP: 12-24 hours (interval 6.0)
        assert calculate_tier(12.1) == FetchTier.SHARP
        assert calculate_tier(24.0) == FetchTier.SHARP
        assert FetchTier.SHARP.interval_hours == 6.0

        # EARLY: 24-72 hours (interval 24.0)
        assert calculate_tier(24.1) == FetchTier.EARLY
        assert calculate_tier(72.0) == FetchTier.EARLY
        assert FetchTier.EARLY.interval_hours == 24.0

        # OPENING: 72+ hours (interval 24.0)
        assert calculate_tier(72.1) == FetchTier.OPENING
        assert FetchTier.OPENING.interval_hours == 24.0


class TestCalculateTierFromTimestamps:
    """Test tier calculation from timestamps."""

    def test_closing_tier_from_timestamps(self):
        """Test CLOSING tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=2)

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.CLOSING

    def test_pregame_tier_from_timestamps(self):
        """Test PREGAME tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=8)

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.PREGAME

    def test_sharp_tier_from_timestamps(self):
        """Test SHARP tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=18)

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.SHARP

    def test_early_tier_from_timestamps(self):
        """Test EARLY tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(days=2)  # 48 hours

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.EARLY

    def test_opening_tier_from_timestamps(self):
        """Test OPENING tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(days=5)  # 120 hours

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.OPENING

    def test_past_game_from_timestamps(self):
        """Test tier for game that already started."""
        now = datetime.now(UTC)
        game_time = now - timedelta(hours=1)  # Game started 1 hour ago

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.IN_PLAY

    def test_exact_boundary_3_hours(self):
        """Test exact 3-hour boundary between CLOSING and PREGAME."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=3)

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.CLOSING

    def test_exact_boundary_12_hours(self):
        """Test exact 12-hour boundary between PREGAME and SHARP."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=12)

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.PREGAME

    def test_exact_boundary_24_hours(self):
        """Test exact 24-hour boundary between SHARP and EARLY."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=24)

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.SHARP

    def test_exact_boundary_72_hours(self):
        """Test exact 72-hour boundary between EARLY and OPENING."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=72)

        tier = calculate_tier_from_timestamps(now, game_time)
        assert tier == FetchTier.EARLY


class TestCalculateHoursUntilCommence:
    """Test hours calculation between timestamps."""

    def test_positive_hours(self):
        """Test positive hours until game."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=5)

        hours = calculate_hours_until_commence(now, game_time)
        assert hours == pytest.approx(5.0, rel=1e-3)

    def test_negative_hours(self):
        """Test negative hours (game already started)."""
        now = datetime.now(UTC)
        game_time = now - timedelta(hours=2)

        hours = calculate_hours_until_commence(now, game_time)
        assert hours == pytest.approx(-2.0, rel=1e-3)

    def test_fractional_hours(self):
        """Test fractional hours calculation."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=2, minutes=30)

        hours = calculate_hours_until_commence(now, game_time)
        assert hours == pytest.approx(2.5, rel=1e-3)

    def test_zero_hours(self):
        """Test zero hours (game starting now)."""
        now = datetime.now(UTC)

        hours = calculate_hours_until_commence(now, now)
        assert hours == pytest.approx(0.0, rel=1e-6)

    def test_days_converted_to_hours(self):
        """Test multi-day time differences."""
        now = datetime.now(UTC)
        game_time = now + timedelta(days=3)

        hours = calculate_hours_until_commence(now, game_time)
        assert hours == pytest.approx(72.0, rel=1e-3)

    def test_minutes_converted_to_hours(self):
        """Test minute-level precision."""
        now = datetime.now(UTC)
        game_time = now + timedelta(minutes=90)

        hours = calculate_hours_until_commence(now, game_time)
        assert hours == pytest.approx(1.5, rel=1e-3)

    def test_seconds_converted_to_hours(self):
        """Test second-level precision."""
        now = datetime.now(UTC)
        game_time = now + timedelta(seconds=3600)

        hours = calculate_hours_until_commence(now, game_time)
        assert hours == pytest.approx(1.0, rel=1e-3)


class TestTierUtilsConsistency:
    """Test consistency between tier calculation methods."""

    def test_calculate_tier_matches_from_timestamps(self):
        """Test that both methods produce same result."""
        now = datetime.now(UTC)
        hours_list = [-2.0, -0.5, 0.5, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]

        for hours in hours_list:
            game_time = now + timedelta(hours=hours)

            direct_tier = calculate_tier(hours)
            timestamp_tier = calculate_tier_from_timestamps(now, game_time)

            assert direct_tier == timestamp_tier, (
                f"Mismatch at {hours} hours: {direct_tier} vs {timestamp_tier}"
            )

    def test_calculate_hours_matches_tier_logic(self):
        """Test that hours calculation integrates correctly with tier logic."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=7.5)

        hours = calculate_hours_until_commence(now, game_time)
        tier = calculate_tier(hours)

        # 7.5 hours should be PREGAME (3-12 hours range)
        assert tier == FetchTier.PREGAME
        assert 3 < hours <= 12
