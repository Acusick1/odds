"""
Pluggable resampling strategies for SequenceFeatureExtractor.

This module provides different approaches for resampling irregular time-series
data to fixed timesteps, enabling flexible handling of variable data density.

Architecture:
- ResamplingStrategy: Abstract base class defining the interface
- UniformResampling: Default strategy with evenly-spaced timesteps
- DensityAwareResampling: Concentrates timesteps where snapshots are most dense
- TierAwareResampling: Allocates timesteps based on fetch_tier metadata

Example:
    ```python
    from odds_analytics.resampling_strategies import TierAwareResampling
    from odds_analytics.feature_extraction import SequenceFeatureExtractor

    # Use tier-aware resampling for intelligent data collection
    strategy = TierAwareResampling(
        allocations={
            "closing": 0.50,  # 50% of timesteps in closing period
            "pregame": 0.25,  # 25% in pregame
            "sharp": 0.15,    # 15% in sharp
            "early": 0.08,    # 8% in early
            "opening": 0.02,  # 2% in opening
        }
    )
    extractor = SequenceFeatureExtractor(
        lookback_hours=72,
        timesteps=24,
        resampling_strategy=strategy
    )
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np

__all__ = [
    "ResamplingStrategy",
    "UniformResampling",
    "DensityAwareResampling",
    "TierAwareResampling",
]


class ResamplingStrategy(ABC):
    """
    Abstract base class for resampling strategies.

    Resampling strategies determine how irregular time-series snapshots
    are mapped to fixed timesteps for sequence models. Different strategies
    can optimize for different data characteristics (uniform, dense regions,
    tier-based importance, etc.).

    Subclasses must implement:
    - get_target_times(): Return list of target timestamps for resampling
    """

    @abstractmethod
    def get_target_times(
        self,
        snapshot_times: list[datetime],
        commence_time: datetime,
        lookback_hours: int,
        timesteps: int,
        snapshot_metadata: list[dict[str, Any]] | None = None,
    ) -> list[datetime]:
        """
        Calculate target times for resampling.

        Args:
            snapshot_times: List of available snapshot timestamps
            commence_time: Game start time
            lookback_hours: Hours before game to start sequence
            timesteps: Number of timesteps in sequence
            snapshot_metadata: Optional metadata for each snapshot (e.g., fetch_tier)

        Returns:
            List of target datetimes (length = timesteps)
        """


class UniformResampling(ResamplingStrategy):
    """
    Uniform timestep allocation strategy (default behavior).

    Allocates timesteps evenly across the lookback period, working backwards
    from game start time. This is the simplest strategy and works well when
    data density is relatively uniform.

    Example:
        With lookback_hours=72 and timesteps=24, creates 24 evenly-spaced
        target times at 3-hour intervals (72/24 = 3).

    Characteristics:
    - Simple and predictable
    - Works well for uniformly-sampled data
    - May result in high masking rates for variable-density data
    """

    def get_target_times(
        self,
        snapshot_times: list[datetime],
        commence_time: datetime,
        lookback_hours: int,
        timesteps: int,
        snapshot_metadata: list[dict[str, Any]] | None = None,
    ) -> list[datetime]:
        """
        Calculate uniformly-spaced target times.

        Args:
            snapshot_times: Available snapshot timestamps (unused - uniform spacing)
            commence_time: Game start time
            lookback_hours: Hours before game to start sequence
            timesteps: Number of timesteps in sequence
            snapshot_metadata: Optional metadata (unused)

        Returns:
            List of uniformly-spaced target datetimes
        """
        target_times = []
        for i in range(timesteps):
            hours_before = lookback_hours - (i * lookback_hours / timesteps)
            target_time = commence_time - timedelta(hours=hours_before)
            target_times.append(target_time)

        return target_times


class DensityAwareResampling(ResamplingStrategy):
    """
    Density-aware timestep allocation strategy.

    Concentrates timesteps in regions where snapshots are most dense,
    automatically adapting to data collection patterns without domain knowledge.

    Algorithm:
    1. Divide lookback period into bins
    2. Count snapshots per bin
    3. Allocate timesteps proportionally to snapshot density
    4. Place target times at bin centers

    Example:
        If 70% of snapshots fall in the final 20% of the period (closing),
        approximately 70% of timesteps will be allocated to that region.

    Characteristics:
    - Automatically adapts to data density
    - Domain-agnostic (no need for tier metadata)
    - Reduces masking by sampling where data exists
    - May still have some empty timesteps if data is very sparse

    Args:
        num_bins: Number of bins for density calculation (default: 20)
    """

    def __init__(self, num_bins: int = 20):
        """
        Initialize density-aware resampling strategy.

        Args:
            num_bins: Number of bins to divide lookback period into (default: 20)
        """
        self.num_bins = num_bins

    def get_target_times(
        self,
        snapshot_times: list[datetime],
        commence_time: datetime,
        lookback_hours: int,
        timesteps: int,
        snapshot_metadata: list[dict[str, Any]] | None = None,
    ) -> list[datetime]:
        """
        Calculate density-aware target times.

        Args:
            snapshot_times: Available snapshot timestamps
            commence_time: Game start time
            lookback_hours: Hours before game to start sequence
            timesteps: Number of timesteps in sequence
            snapshot_metadata: Optional metadata (unused)

        Returns:
            List of target datetimes concentrated in dense regions
        """
        if not snapshot_times:
            # Fallback to uniform if no data
            return UniformResampling().get_target_times(
                snapshot_times, commence_time, lookback_hours, timesteps, snapshot_metadata
            )

        # Define the lookback period
        period_start = commence_time - timedelta(hours=lookback_hours)

        # Create bins and count snapshots per bin
        bin_width = lookback_hours / self.num_bins
        bin_counts = [0] * self.num_bins

        for snap_time in snapshot_times:
            if snap_time < period_start or snap_time > commence_time:
                continue

            # Calculate which bin this snapshot belongs to
            hours_from_start = (snap_time - period_start).total_seconds() / 3600
            bin_idx = min(int(hours_from_start / bin_width), self.num_bins - 1)
            bin_counts[bin_idx] += 1

        # Calculate timesteps per bin proportionally to density
        total_snapshots = sum(bin_counts)
        if total_snapshots == 0:
            # Fallback to uniform if no snapshots in range
            return UniformResampling().get_target_times(
                snapshot_times, commence_time, lookback_hours, timesteps, snapshot_metadata
            )

        # Allocate timesteps proportionally to density (minimum 1 per non-empty bin)
        timesteps_per_bin = []
        remaining_timesteps = timesteps

        for count in bin_counts:
            if count > 0 and remaining_timesteps > 0:
                allocated = max(1, int(timesteps * count / total_snapshots))
                timesteps_per_bin.append(allocated)
                remaining_timesteps -= allocated
            else:
                timesteps_per_bin.append(0)

        # Distribute any remaining timesteps to densest bins
        while remaining_timesteps > 0:
            densest_bin = np.argmax(bin_counts)
            timesteps_per_bin[densest_bin] += 1
            remaining_timesteps -= 1
            bin_counts[densest_bin] = -1  # Mark as used to find next densest

        # Generate target times at bin centers
        target_times = []
        for bin_idx, num_timesteps in enumerate(timesteps_per_bin):
            if num_timesteps == 0:
                continue

            # Calculate bin center time
            bin_start_hours = bin_idx * bin_width
            bin_center_hours = bin_start_hours + bin_width / 2
            bin_center_time = period_start + timedelta(hours=bin_center_hours)

            # If multiple timesteps in this bin, spread them evenly within the bin
            if num_timesteps == 1:
                target_times.append(bin_center_time)
            else:
                for i in range(num_timesteps):
                    offset_fraction = (i + 1) / (num_timesteps + 1)
                    offset_hours = (offset_fraction - 0.5) * bin_width
                    target_time = bin_center_time + timedelta(hours=offset_hours)
                    target_times.append(target_time)

        # Sort by time and ensure we have exactly `timesteps` elements
        target_times.sort()
        if len(target_times) < timesteps:
            # Pad with uniform spacing if needed
            deficit = timesteps - len(target_times)
            uniform_times = UniformResampling().get_target_times(
                snapshot_times, commence_time, lookback_hours, deficit, snapshot_metadata
            )
            target_times.extend(uniform_times)
            target_times.sort()

        return target_times[:timesteps]


class TierAwareResampling(ResamplingStrategy):
    """
    Tier-based timestep allocation strategy (sports betting specific).

    Allocates timesteps based on fetch_tier metadata from intelligent scheduler,
    concentrating samples in critical periods (closing, pregame) while maintaining
    coverage of early periods.

    Default allocation (customizable):
    - Closing (0-3h before): 50% of timesteps
    - Pregame (3-12h before): 25% of timesteps
    - Sharp (12-24h before): 15% of timesteps
    - Early (1-3 days before): 8% of timesteps
    - Opening (3+ days before): 2% of timesteps

    Benefits:
    - Domain-aware sampling based on betting market importance
    - Significantly reduces masking (from ~70% to 10-20%)
    - Ensures critical closing line is well-represented
    - Maintains historical context from opening lines

    Args:
        allocations: Dict mapping tier names to allocation fractions (must sum to 1.0)
    """

    DEFAULT_ALLOCATIONS = {
        "closing": 0.50,
        "pregame": 0.25,
        "sharp": 0.15,
        "early": 0.08,
        "opening": 0.02,
    }

    def __init__(self, allocations: dict[str, float] | None = None):
        """
        Initialize tier-aware resampling strategy.

        Args:
            allocations: Dict mapping tier names to allocation fractions (default: DEFAULT_ALLOCATIONS)

        Raises:
            ValueError: If allocations don't sum to approximately 1.0
        """
        self.allocations = allocations or self.DEFAULT_ALLOCATIONS

        # Validate allocations sum to 1.0
        total = sum(self.allocations.values())
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"Allocations must sum to 1.0, got {total}")

    def get_target_times(
        self,
        snapshot_times: list[datetime],
        commence_time: datetime,
        lookback_hours: int,
        timesteps: int,
        snapshot_metadata: list[dict[str, Any]] | None = None,
    ) -> list[datetime]:
        """
        Calculate tier-aware target times.

        Args:
            snapshot_times: Available snapshot timestamps
            commence_time: Game start time
            lookback_hours: Hours before game to start sequence
            timesteps: Number of timesteps in sequence
            snapshot_metadata: List of metadata dicts with 'fetch_tier' key

        Returns:
            List of target datetimes allocated by tier

        Note:
            If snapshot_metadata is None or missing fetch_tier, falls back to DensityAwareResampling.
        """
        # Validate metadata availability
        if not snapshot_metadata or not any("fetch_tier" in m for m in snapshot_metadata):
            # Fallback to density-aware if no tier metadata
            return DensityAwareResampling().get_target_times(
                snapshot_times, commence_time, lookback_hours, timesteps, snapshot_metadata
            )

        if len(snapshot_times) != len(snapshot_metadata):
            raise ValueError("snapshot_times and snapshot_metadata must have same length")

        # Group snapshots by tier
        tier_groups: dict[str, list[datetime]] = defaultdict(list)
        for snap_time, metadata in zip(snapshot_times, snapshot_metadata):
            tier = metadata.get("fetch_tier")
            if tier:
                tier_groups[tier].append(snap_time)

        if not tier_groups:
            # Fallback if no tiers found
            return DensityAwareResampling().get_target_times(
                snapshot_times, commence_time, lookback_hours, timesteps, snapshot_metadata
            )

        # Calculate timesteps per tier based on allocations
        tier_timesteps: dict[str, int] = {}
        remaining_timesteps = timesteps

        for tier, allocation in self.allocations.items():
            if tier in tier_groups:
                allocated = max(1, round(timesteps * allocation))  # At least 1 if tier exists
                tier_timesteps[tier] = min(allocated, remaining_timesteps)
                remaining_timesteps -= tier_timesteps[tier]

        # Distribute any remaining timesteps to largest tiers
        while remaining_timesteps > 0:
            largest_tier = max(tier_groups.keys(), key=lambda t: len(tier_groups[t]))
            tier_timesteps[largest_tier] = tier_timesteps.get(largest_tier, 0) + 1
            remaining_timesteps -= 1

        # Generate target times for each tier
        target_times = []

        for tier, tier_snaps in tier_groups.items():
            num_timesteps = tier_timesteps.get(tier, 0)
            if num_timesteps == 0 or not tier_snaps:
                continue

            # Sort tier snapshots by time
            tier_snaps_sorted = sorted(tier_snaps)

            # Distribute timesteps evenly across tier's snapshots
            if num_timesteps >= len(tier_snaps_sorted):
                # More timesteps than snapshots: use all snapshots
                target_times.extend(tier_snaps_sorted[:num_timesteps])
            else:
                # Fewer timesteps than snapshots: sample evenly
                indices = np.linspace(0, len(tier_snaps_sorted) - 1, num_timesteps, dtype=int)
                target_times.extend([tier_snaps_sorted[i] for i in indices])

        # Sort all target times chronologically
        target_times.sort()

        # Ensure we have exactly `timesteps` elements
        if len(target_times) < timesteps:
            # Pad with uniform spacing if needed
            deficit = timesteps - len(target_times)
            uniform_times = UniformResampling().get_target_times(
                snapshot_times, commence_time, lookback_hours, deficit, snapshot_metadata
            )
            target_times.extend(uniform_times)
            target_times.sort()

        return target_times[:timesteps]
