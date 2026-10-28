"""Unit tests for tier calculation utilities."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.tier_utils import (
    calculate_tier,
    snapshot_tier,
    snapshot_tiers,
)

from tests.test_helpers import TIER_HOURS_BEFORE


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

        # SHARP: 12-24 hours (interval 12.0)
        assert calculate_tier(12.1) == FetchTier.SHARP
        assert calculate_tier(24.0) == FetchTier.SHARP
        assert FetchTier.SHARP.interval_hours == 12.0

        # EARLY: 24-72 hours (interval 24.0)
        assert calculate_tier(24.1) == FetchTier.EARLY
        assert calculate_tier(72.0) == FetchTier.EARLY
        assert FetchTier.EARLY.interval_hours == 24.0

        # OPENING: 72+ hours (interval 48.0)
        assert calculate_tier(72.1) == FetchTier.OPENING
        assert FetchTier.OPENING.interval_hours == 48.0


class TestSnapshotTierObserved:
    """An observed time is tiered by calculate_tier."""

    def test_closing_tier_from_timestamps(self):
        """Test CLOSING tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=2)

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.CLOSING

    def test_pregame_tier_from_timestamps(self):
        """Test PREGAME tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=8)

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.PREGAME

    def test_sharp_tier_from_timestamps(self):
        """Test SHARP tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=18)

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.SHARP

    def test_early_tier_from_timestamps(self):
        """Test EARLY tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(days=2)  # 48 hours

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.EARLY

    def test_opening_tier_from_timestamps(self):
        """Test OPENING tier calculated from timestamps."""
        now = datetime.now(UTC)
        game_time = now + timedelta(days=5)  # 120 hours

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.OPENING

    def test_past_game_from_timestamps(self):
        """Test tier for game that already started."""
        now = datetime.now(UTC)
        game_time = now - timedelta(hours=1)  # Game started 1 hour ago

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.IN_PLAY

    def test_exact_boundary_3_hours(self):
        """Test exact 3-hour boundary between CLOSING and PREGAME."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=3)

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.CLOSING

    def test_exact_boundary_12_hours(self):
        """Test exact 12-hour boundary between PREGAME and SHARP."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=12)

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.PREGAME

    def test_exact_boundary_24_hours(self):
        """Test exact 24-hour boundary between SHARP and EARLY."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=24)

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.SHARP

    def test_exact_boundary_72_hours(self):
        """Test exact 72-hour boundary between EARLY and OPENING."""
        now = datetime.now(UTC)
        game_time = now + timedelta(hours=72)

        tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)
        assert tier == FetchTier.EARLY


class TestTierUtilsConsistency:
    """Test consistency between tier calculation methods."""

    def test_calculate_tier_matches_from_timestamps(self):
        """Test that both methods produce same result."""
        now = datetime.now(UTC)
        hours_list = [-2.0, -0.5, 0.5, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0]

        for hours in hours_list:
            game_time = now + timedelta(hours=hours)

            direct_tier = calculate_tier(hours)
            timestamp_tier = snapshot_tier(now, game_time, TimeBasis.OBSERVED)

            assert direct_tier == timestamp_tier, (
                f"Mismatch at {hours} hours: {direct_tier} vs {timestamp_tier}"
            )


KICKOFF = datetime(2026, 10, 24, 15, 0, tzinfo=UTC)


class TestSnapshotTier:
    @pytest.mark.parametrize("tier", list(FetchTier))
    def test_test_helper_hours_lie_in_their_tier(self, tier: FetchTier) -> None:
        assert calculate_tier(TIER_HOURS_BEFORE[tier]) is tier

    def test_a_close_is_the_closing_tier_whatever_its_stamp(self) -> None:
        assert snapshot_tier(KICKOFF + timedelta(hours=2), KICKOFF, TimeBasis.CLOSE) is (
            FetchTier.CLOSING
        )
        assert snapshot_tier(KICKOFF - timedelta(hours=30), KICKOFF, TimeBasis.CLOSE) is (
            FetchTier.CLOSING
        )

    @pytest.mark.parametrize(
        ("hours", "observed", "upper_bound"),
        [
            (3.0, FetchTier.CLOSING, FetchTier.PREGAME),
            (12.0, FetchTier.PREGAME, FetchTier.SHARP),
            (24.0, FetchTier.SHARP, FetchTier.EARLY),
            (72.0, FetchTier.EARLY, FetchTier.OPENING),
            (2.0, FetchTier.CLOSING, FetchTier.CLOSING),
        ],
    )
    def test_an_upper_bound_on_a_boundary_covers_the_farther_tier_only(
        self, hours: float, observed: FetchTier, upper_bound: FetchTier
    ) -> None:
        taken = KICKOFF - timedelta(hours=hours)

        assert snapshot_tier(taken, KICKOFF, TimeBasis.OBSERVED) is observed
        assert snapshot_tier(taken, KICKOFF, TimeBasis.UPPER_BOUND) is upper_bound


class TestSnapshotTiers:
    def test_each_markets_latest_oddsportal_history_stamp_is_its_closing(self) -> None:
        stamped = (Source.ODDSPORTAL, TimeBasis.SOURCE_STAMPED)
        rows = [
            (KICKOFF - timedelta(hours=40), *stamped, "1x2"),
            (KICKOFF + timedelta(minutes=20), *stamped, "1x2"),
            (KICKOFF - timedelta(hours=30), *stamped, "totals"),
            (KICKOFF - timedelta(hours=5), *stamped, "totals"),
            (KICKOFF + timedelta(minutes=30), Source.BETFAIR, TimeBasis.OBSERVED, "1x2"),
        ]

        assert snapshot_tiers(rows, KICKOFF) == [
            FetchTier.EARLY,
            FetchTier.CLOSING,
            FetchTier.EARLY,
            FetchTier.CLOSING,
            FetchTier.IN_PLAY,
        ]

    @pytest.mark.parametrize(
        ("offset", "tier"),
        [(timedelta(hours=-2), FetchTier.CLOSING), (timedelta(hours=30), FetchTier.EARLY)],
    )
    def test_a_lone_history_stamp_is_a_closing_only_after_kick_off(
        self, offset: timedelta, tier: FetchTier
    ) -> None:
        # A lone stamp before kick-off may be an opening whose closing was never scraped.
        rows = [(KICKOFF - offset, Source.ODDSPORTAL, TimeBasis.SOURCE_STAMPED, "1x2")]

        assert snapshot_tiers(rows, KICKOFF) == [tier]

    def test_an_fduk_opening_under_three_hours_is_pregame_beside_its_close(self) -> None:
        fduk = Source.FOOTBALL_DATA_UK
        rows = [
            (KICKOFF - timedelta(hours=2, minutes=30), fduk, TimeBasis.UPPER_BOUND, "1x2"),
            (KICKOFF, fduk, TimeBasis.CLOSE, "1x2"),
            (KICKOFF - timedelta(hours=1), Source.BETFAIR, TimeBasis.OBSERVED, "1x2"),
        ]

        assert snapshot_tiers(rows, KICKOFF) == [
            FetchTier.PREGAME,
            FetchTier.CLOSING,
            FetchTier.CLOSING,
        ]
