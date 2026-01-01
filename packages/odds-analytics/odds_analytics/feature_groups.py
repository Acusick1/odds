"""
Composable Feature Group Architecture for ML Training.

This module provides a registry-based system for composing different feature groups
into unified training data preparation. Feature groups are pluggable components
that can be combined via configuration.

Key Components:
- FeatureGroup: Abstract base class defining the feature group interface
- FEATURE_GROUP_REGISTRY: Registry mapping names to feature group classes
- get_feature_groups(): Factory function with validation
- prepare_training_data(): Unified data preparation using composed groups

Example:
    ```python
    from odds_analytics.feature_groups import prepare_training_data

    # Config specifies which groups to compose
    # feature_groups: ["tabular", "trajectory"]

    result = await prepare_training_data(events, session, features_config)
    X_train, X_test = result.X_train, result.X_test
    ```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog
from odds_core.models import Event, EventStatus, Odds
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import (
    SequenceFeatureExtractor,
    TabularFeatureExtractor,
    TrajectoryFeatureExtractor,
)
from odds_analytics.sequence_loader import (
    _extract_odds_from_snapshot,
    calculate_regression_target,
    get_opening_closing_odds_by_tier,
    load_sequences_for_event,
    load_sequences_up_to_tier,
)

if TYPE_CHECKING:
    from odds_analytics.training.config import FeatureConfig

logger = structlog.get_logger()

__all__ = [
    "FeatureGroup",
    "TabularFeatureGroup",
    "TrajectoryFeatureGroup",
    "SequenceFullFeatureGroup",
    "FEATURE_GROUP_REGISTRY",
    "get_feature_groups",
    "prepare_training_data",
    "filter_completed_events",
]


# =============================================================================
# Helper Functions
# =============================================================================


def filter_completed_events(events: list[Event]) -> list[Event]:
    """
    Filter events to only include completed games with final scores.

    Args:
        events: List of Event objects

    Returns:
        List of events with status=FINAL and non-null scores
    """
    return [
        e
        for e in events
        if e.status == EventStatus.FINAL and e.home_score is not None and e.away_score is not None
    ]


def make_backtest_event(event: Event) -> BacktestEvent:
    """Convert Event to BacktestEvent for feature extraction."""
    return BacktestEvent(
        id=event.id,
        commence_time=event.commence_time,
        home_team=event.home_team,
        away_team=event.away_team,
        home_score=event.home_score,
        away_score=event.away_score,
        status=event.status,
    )


# =============================================================================
# Feature Group Abstract Base Class
# =============================================================================


class FeatureGroup(ABC):
    """
    Abstract base class for composable feature groups.

    Feature groups are pluggable components that:
    1. Load required data for a specific type of features
    2. Extract features from that data
    3. Provide feature names for the extracted features

    Subclasses must implement:
    - name: Class attribute identifying the group
    - output_dim: "2d" for tabular models, "3d" for sequence models
    - load_data(): Load required data from database
    - extract(): Extract features from loaded data
    - get_feature_names(): Return ordered list of feature names
    """

    name: str  # e.g., "tabular", "trajectory", "sequence_full"
    output_dim: Literal["2d", "3d"] = "2d"  # For compatibility validation

    def __init__(self, config: FeatureConfig):
        """
        Initialize feature group with configuration.

        Args:
            config: FeatureConfig with extraction parameters
        """
        self.config = config

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Return ordered list of feature names with group prefix.

        Returns:
            List of feature names (e.g., ['tab_is_home_team', 'tab_consensus_prob', ...])
        """
        ...

    @abstractmethod
    async def load_data(
        self,
        event_id: str,
        session: AsyncSession,
    ) -> Any:
        """
        Load required data for this feature group.

        Args:
            event_id: Event identifier
            session: Async database session

        Returns:
            Loaded data (type depends on group), or None if unavailable
        """
        ...

    @abstractmethod
    def extract(
        self,
        data: Any,
        event: BacktestEvent,
        outcome: str,
        market: str,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray] | None:
        """
        Extract features from loaded data.

        Args:
            data: Data returned from load_data()
            event: Event with final scores
            outcome: Target outcome name
            market: Market type (h2h, spreads, totals)

        Returns:
            Feature array, or (features, mask) tuple for 3D groups, or None on failure
        """
        ...

    @classmethod
    def from_config(cls, config: FeatureConfig) -> FeatureGroup:
        """Factory method for consistent instantiation."""
        return cls(config)


# =============================================================================
# Tabular Feature Group
# =============================================================================


class TabularFeatureGroup(FeatureGroup):
    """
    Features from single opening snapshot.

    Extracts point-in-time features from the opening odds snapshot including:
    - Market consensus probabilities
    - Sharp vs retail differentials
    - Market efficiency metrics
    - Best available odds

    Output shape: 2D (n_features,)
    """

    name = "tabular"
    output_dim: Literal["2d", "3d"] = "2d"

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self._extractor = TabularFeatureExtractor.from_config(config)

    def get_feature_names(self) -> list[str]:
        """Return feature names with 'tab_' prefix."""
        from odds_analytics.feature_extraction import TabularFeatures

        return [f"tab_{n}" for n in TabularFeatures.get_feature_names()]

    async def load_data(
        self,
        event_id: str,
        session: AsyncSession,
    ) -> list[Odds] | None:
        """Load opening snapshot for feature extraction."""
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(session)

        # Get first snapshot in opening tier
        snapshot = await reader.get_first_snapshot_in_tier(event_id, self.config.opening_tier)
        if not snapshot:
            return None

        # Extract all odds from snapshot
        odds_list = _extract_odds_from_snapshot(snapshot, event_id)
        return odds_list if odds_list else None

    def extract(
        self,
        data: list[Odds] | None,
        event: BacktestEvent,
        outcome: str,
        market: str,
    ) -> np.ndarray | None:
        """Extract tabular features from opening snapshot."""
        if data is None:
            return None

        try:
            features = self._extractor.extract_features(
                event=event,
                odds_data=data,
                outcome=outcome,
                market=market,
            )
            return features.to_array()
        except Exception as e:
            logger.debug(
                "tabular_extract_failed",
                event_id=event.id,
                error=str(e),
            )
            return None


# =============================================================================
# Trajectory Feature Group
# =============================================================================


class TrajectoryFeatureGroup(FeatureGroup):
    """
    Aggregate features from sequence up to decision tier.

    Extracts trajectory features that capture how the market moved from
    opening to decision time, including:
    - Momentum (direction and rate of change)
    - Volatility (how much the line bounced)
    - Trend (linear regression, reversals)
    - Sharp money indicators

    Output shape: 2D (n_features,)
    """

    name = "trajectory"
    output_dim: Literal["2d", "3d"] = "2d"

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self._extractor = TrajectoryFeatureExtractor.from_config(config)

    def get_feature_names(self) -> list[str]:
        """Return feature names with 'traj_' prefix."""
        from odds_analytics.feature_extraction import TrajectoryFeatures

        return [f"traj_{n}" for n in TrajectoryFeatures.get_feature_names()]

    async def load_data(
        self,
        event_id: str,
        session: AsyncSession,
    ) -> list[list[Odds]] | None:
        """Load sequence filtered to decision tier."""
        sequence = await load_sequences_up_to_tier(
            event_id=event_id,
            session=session,
            decision_tier=self.config.decision_tier,
        )

        # Need at least 2 snapshots for trajectory features
        if not sequence or len(sequence) < 2:
            return None

        return sequence

    def extract(
        self,
        data: list[list[Odds]] | None,
        event: BacktestEvent,
        outcome: str,
        market: str,
    ) -> np.ndarray | None:
        """Extract trajectory features from sequence."""
        if data is None:
            return None

        try:
            features = self._extractor.extract_features(
                event=event,
                odds_data=data,
                outcome=outcome,
                market=market,
            )
            return features.to_array()
        except Exception as e:
            logger.debug(
                "trajectory_extract_failed",
                event_id=event.id,
                error=str(e),
            )
            return None


# =============================================================================
# Sequence Full Feature Group (3D for LSTM)
# =============================================================================


class SequenceFullFeatureGroup(FeatureGroup):
    """
    Full 3D sequence for LSTM/Transformer models.

    Extracts time-series features from historical odds sequences, capturing
    line movement patterns at each timestep.

    Output shape: 3D (timesteps, n_features) with attention mask
    """

    name = "sequence_full"
    output_dim: Literal["2d", "3d"] = "3d"

    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self._extractor = SequenceFeatureExtractor.from_config(config)

    def get_feature_names(self) -> list[str]:
        """Return feature names with 'seq_' prefix."""
        from odds_analytics.feature_extraction import SequenceFeatures

        return [f"seq_{n}" for n in SequenceFeatures.get_feature_names()]

    async def load_data(
        self,
        event_id: str,
        session: AsyncSession,
    ) -> list[list[Odds]] | None:
        """Load full sequence for event."""
        sequence = await load_sequences_for_event(event_id, session)

        if not sequence or all(len(snapshot) == 0 for snapshot in sequence):
            return None

        return sequence

    def extract(
        self,
        data: list[list[Odds]] | None,
        event: BacktestEvent,
        outcome: str,
        market: str,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Extract sequence features.

        Returns:
            Tuple of (sequence, mask) or None on failure
        """
        if data is None:
            return None

        try:
            result = self._extractor.extract_features(
                event=event,
                odds_data=data,
                outcome=outcome,
                market=market,
            )

            sequence = result["sequence"]
            mask = result["mask"]

            # Skip if all timesteps are invalid
            if not mask.any():
                return None

            return sequence, mask
        except Exception as e:
            logger.debug(
                "sequence_extract_failed",
                event_id=event.id,
                error=str(e),
            )
            return None


# =============================================================================
# Feature Group Registry
# =============================================================================


FEATURE_GROUP_REGISTRY: dict[str, type[FeatureGroup]] = {
    "tabular": TabularFeatureGroup,
    "trajectory": TrajectoryFeatureGroup,
    "sequence_full": SequenceFullFeatureGroup,
    # Future groups can be registered here:
    # "historical": HistoricalFeatureGroup,
    # "external": ExternalFeatureGroup,
}


def get_feature_groups(config: FeatureConfig) -> list[FeatureGroup]:
    """
    Instantiate feature groups from config, with validation.

    Args:
        config: FeatureConfig with feature_groups list

    Returns:
        List of instantiated FeatureGroup objects

    Raises:
        ValueError: If unknown group name or incompatible output dimensions
    """
    # Get group names from config
    group_names = getattr(config, "feature_groups", ["tabular"])

    # Validate all group names exist
    for name in group_names:
        if name not in FEATURE_GROUP_REGISTRY:
            raise ValueError(
                f"Unknown feature group '{name}'. "
                f"Available groups: {list(FEATURE_GROUP_REGISTRY.keys())}"
            )

    # Instantiate groups
    groups = [FEATURE_GROUP_REGISTRY[name](config) for name in group_names]

    # Validate compatible output dimensions
    dims = {g.output_dim for g in groups}
    if len(dims) > 1:
        raise ValueError(
            f"Cannot mix feature groups with different output dimensions: {dims}. "
            f"All groups must be either 2D (tabular, trajectory) or 3D (sequence_full)."
        )

    return groups


# =============================================================================
# Unified Training Data Preparation
# =============================================================================


class TrainingDataResult:
    """
    Container for training data preparation results.

    Provides a unified interface for both tabular (XGBoost) and sequence (LSTM) data.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        masks: np.ndarray | None = None,
    ):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.masks = masks

    @property
    def num_samples(self) -> int:
        """Number of samples."""
        return len(self.X)

    @property
    def num_features(self) -> int:
        """Number of features."""
        return len(self.feature_names)


async def prepare_training_data(
    events: list[Event],
    session: AsyncSession,
    config: FeatureConfig,
) -> TrainingDataResult:
    """
    Unified data preparation using composable feature groups.

    This function provides a single entry point for preparing training data
    using any combination of feature groups (tabular, trajectory, sequence_full).

    Args:
        events: List of Event objects with final scores
        session: Async database session
        config: FeatureConfig with feature_groups list

    Returns:
        TrainingDataResult containing features, targets, and metadata

    Raises:
        ValueError: If no valid events or no valid training data

    Example:
        ```python
        config = FeatureConfig(
            feature_groups=["tabular", "trajectory"],
            opening_tier=FetchTier.SHARP,
            closing_tier=FetchTier.CLOSING,
            decision_tier=FetchTier.PREGAME,
        )
        result = await prepare_training_data(events, session, config)
        ```
    """
    # Filter valid events
    valid_events = filter_completed_events(events)

    if not valid_events:
        raise ValueError(f"No valid events found in {len(events)} total events")

    # Get feature groups from config
    groups = get_feature_groups(config)
    output_dim = groups[0].output_dim  # All same dim (validated)

    # Combine feature names from all groups
    all_feature_names: list[str] = []
    for group in groups:
        all_feature_names.extend(group.get_feature_names())

    # Get market from config
    market = config.markets[0] if config.markets else "h2h"

    # Pre-allocate lists for collection
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    masks_list: list[np.ndarray] = []
    skipped_events = 0

    for event in valid_events:
        # Determine target outcome
        outcome = event.home_team if config.outcome == "home" else event.away_team

        # Get opening/closing odds for target calculation
        try:
            opening_odds, closing_odds = await get_opening_closing_odds_by_tier(
                session=session,
                event_id=event.id,
                opening_tier=config.opening_tier,
                closing_tier=config.closing_tier,
                market=market,
                outcome=outcome,
            )
        except Exception as e:
            logger.debug(
                "failed_get_odds",
                event_id=event.id,
                error=str(e),
            )
            skipped_events += 1
            continue

        if opening_odds is None or closing_odds is None:
            skipped_events += 1
            continue

        # Calculate regression target
        target = calculate_regression_target(opening_odds, closing_odds, market)
        if target is None:
            skipped_events += 1
            continue

        # Extract features from each group
        backtest_event = make_backtest_event(event)
        feature_arrays: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        skip = False

        for group in groups:
            # Load data for this group
            data = await group.load_data(event.id, session)

            # Extract features
            result = group.extract(data, backtest_event, outcome, market)

            if result is None:
                skip = True
                break

            # Handle 3D groups that return (features, mask)
            if output_dim == "3d" and isinstance(result, tuple):
                feature_arrays.append(result[0])
                masks.append(result[1])
            else:
                feature_arrays.append(result)

        if skip:
            skipped_events += 1
            continue

        # Concatenate group features
        if output_dim == "2d":
            combined = np.concatenate(feature_arrays)
        else:
            # Concatenate along feature dimension for 3D
            combined = np.concatenate(feature_arrays, axis=-1)
            if masks:
                masks_list.append(masks[0])  # Use first mask (all should be same)

        X_list.append(combined)
        y_list.append(target)

    if not X_list:
        raise ValueError(
            f"No valid training data after processing {len(valid_events)} events "
            f"(skipped {skipped_events})"
        )

    # Convert to arrays
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    # Handle masks for 3D data
    masks_array = np.array(masks_list) if masks_list else None

    logger.info(
        "prepared_training_data",
        num_samples=len(X),
        num_features=len(all_feature_names),
        output_dim=output_dim,
        feature_groups=[g.name for g in groups],
        skipped_events=skipped_events,
        target_mean=float(np.mean(y)),
        target_std=float(np.std(y)),
    )

    return TrainingDataResult(
        X=X,
        y=y,
        feature_names=all_feature_names,
        masks=masks_array,
    )
