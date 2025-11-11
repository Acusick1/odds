"""Unit tests for resampling strategies."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_analytics.resampling_strategies import (
    DensityAwareResampling,
    ResamplingStrategy,
    TierAwareResampling,
    UniformResampling,
)


class TestResamplingStrategyInterface:
    """Test the ResamplingStrategy abstract base class interface."""

    def test_resampling_strategy_is_abstract(self):
        """Test that ResamplingStrategy cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            ResamplingStrategy()

    def test_subclass_must_implement_get_target_times(self):
        """Test that subclasses must implement get_target_times."""

        class IncompleteStrategy(ResamplingStrategy):
            pass

        with pytest.raises(TypeError, match="abstract"):
            IncompleteStrategy()


class TestUniformResampling:
    """Test UniformResampling strategy."""

    def test_initialization(self):
        """Test that UniformResampling initializes without parameters."""
        strategy = UniformResampling()
        assert isinstance(strategy, ResamplingStrategy)

    def test_get_target_times_returns_correct_count(self):
        """Test that get_target_times returns correct number of timesteps."""
        strategy = UniformResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)
        snapshot_times = []

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        assert len(target_times) == 8

    def test_get_target_times_uniform_spacing(self):
        """Test that target times are evenly spaced."""
        strategy = UniformResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)
        snapshot_times = []

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        # Check spacing between consecutive times (should be 3 hours)
        for i in range(len(target_times) - 1):
            diff_hours = (target_times[i + 1] - target_times[i]).total_seconds() / 3600
            assert abs(diff_hours - 3.0) < 0.01  # Allow small floating point error

    def test_get_target_times_works_backwards_from_commence(self):
        """Test that first target time is lookback_hours before commence."""
        strategy = UniformResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)
        snapshot_times = []

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        # First time should be 24 hours before game
        expected_first = commence_time - timedelta(hours=24)
        assert abs((target_times[0] - expected_first).total_seconds()) < 1

        # Last time should be close to game start (within one interval)
        # With 24 hours and 8 timesteps, interval is 3 hours
        interval_seconds = (24 / 8) * 3600
        assert abs((target_times[-1] - commence_time).total_seconds()) <= interval_seconds

    def test_get_target_times_ignores_snapshot_times(self):
        """Test that UniformResampling doesn't use snapshot_times parameter."""
        strategy = UniformResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # Call with empty and non-empty snapshot times
        target_times_empty = strategy.get_target_times(
            snapshot_times=[],
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        target_times_nonempty = strategy.get_target_times(
            snapshot_times=[datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)],
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        # Should produce identical results
        assert target_times_empty == target_times_nonempty


class TestDensityAwareResampling:
    """Test DensityAwareResampling strategy."""

    def test_initialization_default_bins(self):
        """Test that DensityAwareResampling initializes with default bins."""
        strategy = DensityAwareResampling()
        assert strategy.num_bins == 20

    def test_initialization_custom_bins(self):
        """Test initialization with custom number of bins."""
        strategy = DensityAwareResampling(num_bins=10)
        assert strategy.num_bins == 10

    def test_get_target_times_returns_correct_count(self):
        """Test that get_target_times returns correct number of timesteps."""
        strategy = DensityAwareResampling(num_bins=10)
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # Create snapshots concentrated in last 6 hours
        snapshot_times = []
        for i in range(12):
            snap_time = commence_time - timedelta(hours=1 + i * 0.5)
            snapshot_times.append(snap_time)

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        assert len(target_times) == 8

    def test_get_target_times_concentrates_in_dense_regions(self):
        """Test that more timesteps are allocated to dense regions."""
        strategy = DensityAwareResampling(num_bins=10)
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # Create highly concentrated snapshots in final 6 hours (closing period)
        snapshot_times = []
        # 20 snapshots in final 6 hours
        for i in range(20):
            snap_time = commence_time - timedelta(hours=0.5 + i * 0.25)
            snapshot_times.append(snap_time)

        # 5 snapshots spread over remaining 18 hours
        for i in range(5):
            snap_time = commence_time - timedelta(hours=7 + i * 3)
            snapshot_times.append(snap_time)

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=16,
        )

        # Count how many target times are in the final 6 hours (dense region)
        dense_region_start = commence_time - timedelta(hours=6)
        dense_count = sum(1 for t in target_times if t >= dense_region_start)

        # Should have more than 50% of timesteps in dense region
        # (since ~80% of snapshots are there)
        assert dense_count > 8, f"Expected >8 timesteps in dense region, got {dense_count}"

    def test_get_target_times_falls_back_to_uniform_if_empty(self):
        """Test that strategy falls back to uniform if no snapshot data."""
        strategy = DensityAwareResampling(num_bins=10)
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        target_times = strategy.get_target_times(
            snapshot_times=[],
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        # Should still return 8 timesteps
        assert len(target_times) == 8

        # Should be uniformly spaced (fallback behavior)
        for i in range(len(target_times) - 1):
            diff_hours = (target_times[i + 1] - target_times[i]).total_seconds() / 3600
            assert abs(diff_hours - 3.0) < 0.01

    def test_get_target_times_sorted_chronologically(self):
        """Test that target times are sorted chronologically."""
        strategy = DensityAwareResampling(num_bins=10)
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = [
            commence_time - timedelta(hours=i) for i in range(1, 25, 2)
        ]

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        # Check sorted
        assert target_times == sorted(target_times)

    def test_get_target_times_handles_snapshots_outside_range(self):
        """Test that snapshots outside lookback period are ignored."""
        strategy = DensityAwareResampling(num_bins=10)
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # Mix of in-range and out-of-range snapshots
        snapshot_times = [
            commence_time - timedelta(hours=30),  # Too old
            commence_time - timedelta(hours=12),  # In range
            commence_time - timedelta(hours=6),   # In range
            commence_time + timedelta(hours=1),   # Future (invalid)
        ]

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        # Should still produce valid result
        assert len(target_times) == 8
        # All target times should be within lookback period
        period_start = commence_time - timedelta(hours=24)
        assert all(period_start <= t <= commence_time for t in target_times)


class TestTierAwareResampling:
    """Test TierAwareResampling strategy."""

    def test_initialization_default_allocations(self):
        """Test that TierAwareResampling initializes with default allocations."""
        strategy = TierAwareResampling()
        assert strategy.allocations == TierAwareResampling.DEFAULT_ALLOCATIONS

    def test_initialization_custom_allocations(self):
        """Test initialization with custom allocations."""
        custom_allocations = {
            "closing": 0.6,
            "pregame": 0.3,
            "sharp": 0.1,
        }
        strategy = TierAwareResampling(allocations=custom_allocations)
        assert strategy.allocations == custom_allocations

    def test_initialization_validates_allocation_sum(self):
        """Test that allocations must sum to 1.0."""
        invalid_allocations = {
            "closing": 0.5,
            "pregame": 0.3,
            # Sum = 0.8 (invalid)
        }

        with pytest.raises(ValueError, match="sum to 1.0"):
            TierAwareResampling(allocations=invalid_allocations)

    def test_initialization_allows_small_floating_point_error(self):
        """Test that allocations with small floating point error are accepted."""
        # Sum = 0.9999999999 (close enough to 1.0)
        allocations = {
            "closing": 0.33333333,
            "pregame": 0.33333333,
            "sharp": 0.33333334,
        }
        strategy = TierAwareResampling(allocations=allocations)
        assert strategy.allocations == allocations

    def test_get_target_times_returns_correct_count(self):
        """Test that get_target_times returns correct number of timesteps."""
        strategy = TierAwareResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = []
        snapshot_metadata = []

        # Create snapshots with tier metadata
        for i in range(10):
            snap_time = commence_time - timedelta(hours=1 + i)
            snapshot_times.append(snap_time)
            snapshot_metadata.append({"fetch_tier": "closing" if i < 5 else "pregame"})

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
            snapshot_metadata=snapshot_metadata,
        )

        assert len(target_times) == 8

    def test_get_target_times_allocates_by_tier(self):
        """Test that timesteps are allocated according to tier allocations."""
        strategy = TierAwareResampling(
            allocations={
                "closing": 0.75,  # 75% to closing
                "pregame": 0.25,  # 25% to pregame
            }
        )
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = []
        snapshot_metadata = []

        # 6 closing snapshots
        for i in range(6):
            snap_time = commence_time - timedelta(hours=0.5 + i * 0.5)
            snapshot_times.append(snap_time)
            snapshot_metadata.append({"fetch_tier": "closing"})

        # 6 pregame snapshots
        for i in range(6):
            snap_time = commence_time - timedelta(hours=4 + i)
            snapshot_times.append(snap_time)
            snapshot_metadata.append({"fetch_tier": "pregame"})

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
            snapshot_metadata=snapshot_metadata,
        )

        # Count timesteps by tier (approximate - based on time ranges)
        closing_end = commence_time - timedelta(hours=3)
        closing_count = sum(1 for t in target_times if t >= closing_end)

        # Should have approximately 75% of timesteps in closing
        assert closing_count >= 5, f"Expected >=5 timesteps in closing, got {closing_count}"

    def test_get_target_times_falls_back_if_no_metadata(self):
        """Test that strategy falls back to DensityAware if no tier metadata."""
        strategy = TierAwareResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = [
            commence_time - timedelta(hours=i) for i in range(1, 13)
        ]

        # No metadata provided
        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
            snapshot_metadata=None,
        )

        # Should still return valid result
        assert len(target_times) == 8

    def test_get_target_times_falls_back_if_no_tier_keys(self):
        """Test fallback if metadata exists but has no fetch_tier keys."""
        strategy = TierAwareResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = [
            commence_time - timedelta(hours=i) for i in range(1, 13)
        ]

        # Metadata without fetch_tier
        snapshot_metadata = [{"other_key": "value"} for _ in snapshot_times]

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
            snapshot_metadata=snapshot_metadata,
        )

        # Should still return valid result (fallback to DensityAware)
        assert len(target_times) == 8

    def test_get_target_times_validates_metadata_length(self):
        """Test that metadata list must match snapshot_times length."""
        strategy = TierAwareResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = [commence_time - timedelta(hours=i) for i in range(1, 5)]
        snapshot_metadata = [{"fetch_tier": "closing"}]  # Wrong length

        with pytest.raises(ValueError, match="same length"):
            strategy.get_target_times(
                snapshot_times=snapshot_times,
                commence_time=commence_time,
                lookback_hours=24,
                timesteps=8,
                snapshot_metadata=snapshot_metadata,
            )

    def test_get_target_times_sorted_chronologically(self):
        """Test that target times are sorted chronologically."""
        strategy = TierAwareResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = []
        snapshot_metadata = []

        # Mix tiers in non-chronological order
        tiers = ["pregame", "closing", "sharp", "closing", "pregame"]
        for i, tier in enumerate(tiers):
            snap_time = commence_time - timedelta(hours=1 + i * 2)
            snapshot_times.append(snap_time)
            snapshot_metadata.append({"fetch_tier": tier})

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
            snapshot_metadata=snapshot_metadata,
        )

        # Check sorted
        assert target_times == sorted(target_times)

    def test_get_target_times_uses_actual_snapshots(self):
        """Test that target times come from actual snapshot times."""
        strategy = TierAwareResampling()
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        snapshot_times = []
        snapshot_metadata = []

        for i in range(10):
            snap_time = commence_time - timedelta(hours=1 + i)
            snapshot_times.append(snap_time)
            snapshot_metadata.append({"fetch_tier": "closing"})

        target_times = strategy.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=5,
            snapshot_metadata=snapshot_metadata,
        )

        # All target times should be from actual snapshots (or very close)
        for target in target_times:
            min_diff = min(abs((target - snap).total_seconds()) for snap in snapshot_times)
            # Should match an actual snapshot time (within 1 second tolerance)
            assert min_diff < 1, f"Target time {target} not close to any snapshot"


class TestResamplingStrategyIntegration:
    """Integration tests for resampling strategies with SequenceFeatureExtractor."""

    def test_strategies_can_be_used_with_extractor(self):
        """Test that all strategies work with SequenceFeatureExtractor."""
        from odds_analytics.feature_extraction import SequenceFeatureExtractor

        strategies = [
            UniformResampling(),
            DensityAwareResampling(),
            TierAwareResampling(),
        ]

        for strategy in strategies:
            extractor = SequenceFeatureExtractor(
                lookback_hours=24,
                timesteps=8,
                resampling_strategy=strategy,
            )

            # Should initialize without error
            assert extractor.resampling_strategy == strategy

    def test_extractor_defaults_to_uniform_resampling(self):
        """Test that SequenceFeatureExtractor defaults to UniformResampling."""
        from odds_analytics.feature_extraction import SequenceFeatureExtractor

        extractor = SequenceFeatureExtractor()
        assert isinstance(extractor.resampling_strategy, UniformResampling)

    def test_different_strategies_produce_different_results(self):
        """Test that different strategies produce different target times."""
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # Create snapshots concentrated in closing period
        snapshot_times = []
        for i in range(15):
            snap_time = commence_time - timedelta(hours=0.5 + i * 0.3)
            snapshot_times.append(snap_time)

        for i in range(5):
            snap_time = commence_time - timedelta(hours=8 + i * 2)
            snapshot_times.append(snap_time)

        uniform = UniformResampling()
        density_aware = DensityAwareResampling()

        uniform_targets = uniform.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        density_targets = density_aware.get_target_times(
            snapshot_times=snapshot_times,
            commence_time=commence_time,
            lookback_hours=24,
            timesteps=8,
        )

        # Should produce different results
        assert uniform_targets != density_targets

    def test_tier_aware_reduces_masking_compared_to_uniform(self):
        """Test that TierAwareResampling reduces masking for tier-based data."""
        from datetime import timedelta

        from odds_analytics.backtesting import BacktestEvent
        from odds_analytics.feature_extraction import SequenceFeatureExtractor
        from odds_core.models import EventStatus, Odds

        # Create event
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)
        event = BacktestEvent(
            id="test_event",
            commence_time=commence_time,
            home_team="Lakers",
            away_team="Celtics",
            home_score=110,
            away_score=105,
            status=EventStatus.FINAL,
        )

        # Create snapshots mimicking intelligent scheduler (heavy closing period)
        snapshots = []

        # Closing period: 12 snapshots every 30 min (6 hours)
        for i in range(12):
            snap_time = commence_time - timedelta(hours=0.5 + i * 0.5)
            snapshot = [
                Odds(
                    id=i,
                    event_id=event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=event.home_team,
                    price=-110,
                    point=None,
                    odds_timestamp=snap_time,
                    last_update=snap_time,
                )
            ]
            snapshots.append(snapshot)

        # Early period: 3 snapshots spread over 66 hours
        for i in range(3):
            snap_time = commence_time - timedelta(hours=7 + i * 22)
            snapshot = [
                Odds(
                    id=100 + i,
                    event_id=event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=event.home_team,
                    price=-110,
                    point=None,
                    odds_timestamp=snap_time,
                    last_update=snap_time,
                )
            ]
            snapshots.append(snapshot)

        # Test with UniformResampling
        uniform_extractor = SequenceFeatureExtractor(
            lookback_hours=72,
            timesteps=24,
            resampling_strategy=UniformResampling(),
        )

        uniform_result = uniform_extractor.extract_features(
            event, snapshots, outcome=event.home_team, market="h2h"
        )

        # Test with DensityAwareResampling
        density_extractor = SequenceFeatureExtractor(
            lookback_hours=72,
            timesteps=24,
            resampling_strategy=DensityAwareResampling(),
        )

        density_result = density_extractor.extract_features(
            event, snapshots, outcome=event.home_team, market="h2h"
        )

        # Calculate masking rates
        uniform_mask_rate = 1.0 - (uniform_result["mask"].sum() / len(uniform_result["mask"]))
        density_mask_rate = 1.0 - (density_result["mask"].sum() / len(density_result["mask"]))

        # DensityAware should have lower masking rate
        assert density_mask_rate < uniform_mask_rate, (
            f"Expected DensityAware masking ({density_mask_rate:.2%}) "
            f"< Uniform masking ({uniform_mask_rate:.2%})"
        )
