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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import structlog
from odds_core.models import Event, EventStatus, Odds
from odds_core.polymarket_models import PolymarketEvent
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import (
    SequenceFeatureExtractor,
    TabularFeatureExtractor,
    TrajectoryFeatureExtractor,
)
from odds_analytics.polymarket_features import (
    CrossSourceFeatureExtractor,
    CrossSourceFeatures,
    PolymarketFeatureExtractor,
    PolymarketTabularFeatures,
    resolve_home_outcome_index,
)
from odds_analytics.sequence_loader import (
    _extract_odds_from_snapshot,
    calculate_devigged_pinnacle_target,
    calculate_regression_target,
    extract_pinnacle_h2h_probs,
    get_opening_closing_odds_by_tier,
    get_snapshots_in_time_range,
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
    "PolymarketFeatureGroup",
    "FEATURE_GROUP_REGISTRY",
    "get_feature_groups",
    "prepare_training_data",
    "prepare_multi_horizon_data",
    "filter_completed_events",
]

# Maps FetchTier to approximate hours before game (midpoint of tier range).
# Used to convert a tier-based decision point into a concrete timestamp for
# Polymarket snapshot queries (PM doesn't use tier-tagged data).
_TIER_DECISION_HOURS: dict[str, float] = {
    "closing": 1.5,  # 0–3 h
    "pregame": 7.5,  # 3–12 h
    "sharp": 18.0,  # 12–24 h
    "early": 48.0,  # 24–72 h
    "opening": 84.0,  # 72 h+
}


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


async def _load_pm_event_context(
    event_id: str,
    session: AsyncSession,
) -> dict | None:
    """Load per-event Polymarket context (PM event, market, home outcome index).

    Returns dict with keys {pm_event, sb_event, market, home_idx, pm_reader}
    or None if any required piece is missing.
    """
    from odds_lambda.storage.polymarket_reader import PolymarketReader

    pm_result = await session.execute(
        select(PolymarketEvent).where(PolymarketEvent.event_id == event_id)
    )
    pm_event = pm_result.scalars().first()
    if pm_event is None:
        return None

    from odds_core.models import Event as SbEvent

    sb_result = await session.execute(select(SbEvent).where(SbEvent.id == event_id))
    sb_event = sb_result.scalar_one_or_none()
    if sb_event is None:
        return None

    pm_reader = PolymarketReader(session)
    market = await pm_reader.get_moneyline_market(pm_event.id)
    if market is None:
        return None

    home_idx = resolve_home_outcome_index(market, sb_event.home_team)
    if home_idx is None:
        logger.debug(
            "pm_home_outcome_unresolved",
            event_id=event_id,
            home_team=sb_event.home_team,
            outcomes=market.outcomes,
        )
        return None

    return {
        "pm_event": pm_event,
        "sb_event": sb_event,
        "market": market,
        "home_idx": home_idx,
        "pm_reader": pm_reader,
    }


async def _load_pm_snapshot_data(
    event_id: str,
    pm_context: dict,
    decision_time: datetime,
    session: AsyncSession,
    velocity_window_hours: float = 2.0,
    tolerance_minutes: int = 30,
) -> dict | None:
    """Load PM price/orderbook/velocity data at a specific decision time.

    Returns the data dict expected by PolymarketFeatureGroup.extract(),
    or None if no PM price snapshot is available within tolerance.
    """
    from datetime import timedelta as td

    from odds_lambda.storage.readers import OddsReader

    pm_reader = pm_context["pm_reader"]
    market = pm_context["market"]
    home_idx = pm_context["home_idx"]

    price_snapshot = await pm_reader.get_price_at_time(
        market.id, decision_time, tolerance_minutes=tolerance_minutes
    )
    if price_snapshot is None:
        return None

    orderbook_snapshot = await pm_reader.get_orderbook_at_time(
        market.id, decision_time, tolerance_minutes=tolerance_minutes
    )

    velocity_start = decision_time - td(hours=velocity_window_hours)
    recent_prices = await pm_reader.get_price_series(market.id, velocity_start, decision_time)

    odds_reader = OddsReader(session)
    sb_odds = await odds_reader.get_odds_at_time(
        event_id, price_snapshot.snapshot_time, tolerance_minutes=tolerance_minutes
    )

    return {
        "price_snapshot": price_snapshot,
        "orderbook_snapshot": orderbook_snapshot,
        "recent_prices": recent_prices,
        "home_outcome_index": home_idx,
        "sb_odds": sb_odds if sb_odds else None,
    }


class PolymarketFeatureGroup(FeatureGroup):
    """
    Point-in-time features from Polymarket price/order book data, plus
    cross-source divergence features comparing PM against sportsbook odds.

    Produces a combined 22-feature vector (14 PM + 8 cross-source).
    Events without a linked Polymarket moneyline market are skipped (load_data
    returns None), so this group naturally filters training data to the
    dual-source subset.

    Output shape: 2D (n_features,)
    """

    name = "polymarket"
    output_dim: Literal["2d", "3d"] = "2d"

    def __init__(self, config: FeatureConfig) -> None:
        super().__init__(config)
        self._pm_extractor = PolymarketFeatureExtractor(
            velocity_window_hours=config.pm_velocity_window_hours,
        )
        self._xsrc_extractor = CrossSourceFeatureExtractor()
        self._sb_extractor = TabularFeatureExtractor.from_config(config)

    def get_feature_names(self) -> list[str]:
        pm_names = [f"pm_{n}" for n in PolymarketTabularFeatures.get_feature_names()]
        xsrc_names = [f"xsrc_{n}" for n in CrossSourceFeatures.get_feature_names()]
        return pm_names + xsrc_names

    async def load_data(self, event_id: str, session: AsyncSession) -> dict | None:
        """
        Load Polymarket and aligned sportsbook data for cross-source feature extraction.

        Returns None (skipping the event) when:
        - No linked PolymarketEvent exists for this event_id
        - No moneyline market found for the PM event
        - Home team outcome cannot be resolved from PM outcome names
        - No PM price snapshot available within tolerance of decision time
        """
        ctx = await _load_pm_event_context(event_id, session)
        if ctx is None:
            return None

        tier_value = self.config.decision_tier.value
        decision_hours = _TIER_DECISION_HOURS.get(tier_value, 7.5)
        decision_time = ctx["sb_event"].commence_time - timedelta(hours=decision_hours)

        return await _load_pm_snapshot_data(
            event_id=event_id,
            pm_context=ctx,
            decision_time=decision_time,
            session=session,
            velocity_window_hours=self.config.pm_velocity_window_hours,
            tolerance_minutes=self.config.pm_price_tolerance_minutes,
        )

    def extract(
        self,
        data: dict | None,
        event: BacktestEvent,
        outcome: str,
        market: str,
    ) -> np.ndarray | None:
        """Extract combined PM + cross-source features."""
        if data is None:
            return None

        try:
            pm_features = self._pm_extractor.extract(
                price_snapshot=data["price_snapshot"],
                orderbook_snapshot=data["orderbook_snapshot"],
                recent_prices=data["recent_prices"],
                home_outcome_index=data["home_outcome_index"],
            )

            sb_features = None
            if data.get("sb_odds"):
                try:
                    sb_features = self._sb_extractor.extract_features(
                        event=event,
                        odds_data=data["sb_odds"],
                        outcome=outcome,
                        market=market,
                    )
                except Exception as e:
                    logger.debug(
                        "polymarket_sb_extract_failed",
                        event_id=event.id,
                        error=str(e),
                    )

            xsrc_features = self._xsrc_extractor.extract(
                pm_features=pm_features,
                sb_features=sb_features,
            )

            return np.concatenate([pm_features.to_array(), xsrc_features.to_array()])

        except Exception as e:
            logger.debug(
                "polymarket_extract_failed",
                event_id=event.id,
                error=str(e),
            )
            return None


FEATURE_GROUP_REGISTRY: dict[str, type[FeatureGroup]] = {
    "tabular": TabularFeatureGroup,
    "trajectory": TrajectoryFeatureGroup,
    "sequence_full": SequenceFullFeatureGroup,
    "polymarket": PolymarketFeatureGroup,
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
        event_ids: np.ndarray | None = None,
    ):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.masks = masks
        self.event_ids = event_ids

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


async def prepare_multi_horizon_data(
    events: list[Event],
    session: AsyncSession,
    config: FeatureConfig,
) -> TrainingDataResult:
    """Prepare multi-horizon training data with devigged Pinnacle target.

    Creates multiple rows per event by sampling snapshots across the configured
    decision_hours_range. Each row's target is the devigged Pinnacle closing
    probability minus the devigged Pinnacle probability at that snapshot.

    Events without Pinnacle closing data are dropped entirely.

    Returns TrainingDataResult with event_ids populated for group-aware CV.
    """
    from odds_lambda.storage.readers import OddsReader

    valid_events = filter_completed_events(events)
    if not valid_events:
        raise ValueError(f"No valid events found in {len(events)} total events")

    reader = OddsReader(session)
    market = config.markets[0] if config.markets else "h2h"
    min_hours, max_hours = config.decision_hours_range
    use_tabular = "tabular" in config.feature_groups
    use_trajectory = "trajectory" in config.feature_groups
    use_pm = "polymarket" in config.feature_groups

    # Build feature names
    feature_names: list[str] = []
    if use_tabular:
        from odds_analytics.feature_extraction import TabularFeatures

        feature_names.extend(f"tab_{n}" for n in TabularFeatures.get_feature_names())
    if use_trajectory:
        from odds_analytics.feature_extraction import TrajectoryFeatures

        feature_names.extend(f"traj_{n}" for n in TrajectoryFeatures.get_feature_names())
    if use_pm:
        feature_names.extend(f"pm_{n}" for n in PolymarketTabularFeatures.get_feature_names())
        feature_names.extend(f"xsrc_{n}" for n in CrossSourceFeatures.get_feature_names())
    feature_names.append("hours_until_event")

    # Instantiate extractors
    tab_extractor = TabularFeatureExtractor.from_config(config) if use_tabular else None
    traj_extractor = TrajectoryFeatureExtractor.from_config(config) if use_trajectory else None
    pm_extractor = (
        PolymarketFeatureExtractor(velocity_window_hours=config.pm_velocity_window_hours)
        if use_pm
        else None
    )
    xsrc_extractor = CrossSourceFeatureExtractor() if use_pm else None
    sb_extractor_for_pm = TabularFeatureExtractor.from_config(config) if use_pm else None

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    event_id_list: list[str] = []
    skipped_events = 0
    total_rows = 0

    for event in valid_events:
        home = event.home_team
        away = event.away_team
        outcome = home if config.outcome == "home" else away
        backtest_event = make_backtest_event(event)

        # Get closing snapshot and check Pinnacle presence
        closing_snapshot = await reader.get_last_snapshot_in_tier(event.id, config.closing_tier)
        if closing_snapshot is None:
            skipped_events += 1
            continue

        closing_odds_all = _extract_odds_from_snapshot(closing_snapshot, event.id, market=market)
        if extract_pinnacle_h2h_probs(closing_odds_all, home, away) is None:
            skipped_events += 1
            continue

        # Get sampled snapshots in decision range
        sampled_snapshots = await get_snapshots_in_time_range(
            session=session,
            event_id=event.id,
            commence_time=event.commence_time,
            min_hours_before=min_hours,
            max_hours_before=max_hours,
            max_samples=config.max_samples_per_event,
        )
        if not sampled_snapshots:
            skipped_events += 1
            continue

        # Load full sequence once for trajectory slicing
        all_sequences: list[list[Odds]] | None = None
        if use_trajectory:
            all_sequences = await load_sequences_for_event(event.id, session)

        # Load PM event context once
        pm_context: dict | None = None
        if use_pm:
            pm_context = await _load_pm_event_context(event.id, session)

        event_had_rows = False
        for snapshot in sampled_snapshots:
            # Extract odds (unfiltered for target, outcome-filtered for features)
            snapshot_odds_all = _extract_odds_from_snapshot(snapshot, event.id, market=market)
            target = calculate_devigged_pinnacle_target(
                snapshot_odds_all, closing_odds_all, home, away
            )
            if target is None:
                continue

            hours_until = (event.commence_time - snapshot.snapshot_time).total_seconds() / 3600

            # Build feature vector
            parts: list[np.ndarray] = []

            # Tabular features
            if use_tabular and tab_extractor is not None:
                snapshot_odds_for_outcome = _extract_odds_from_snapshot(
                    snapshot, event.id, market=market, outcome=outcome
                )
                if not snapshot_odds_for_outcome:
                    continue
                try:
                    tab_feats = tab_extractor.extract_features(
                        event=backtest_event,
                        odds_data=snapshot_odds_for_outcome,
                        outcome=outcome,
                        market=market,
                    )
                    parts.append(tab_feats.to_array())
                except Exception:
                    continue

            # Trajectory features
            if use_trajectory and traj_extractor is not None and all_sequences:
                snapshots_up_to = [
                    s for s in all_sequences if s and s[0].odds_timestamp <= snapshot.snapshot_time
                ]
                if len(snapshots_up_to) >= 2:
                    try:
                        traj_feats = traj_extractor.extract_features(
                            event=backtest_event,
                            odds_data=snapshots_up_to,
                            outcome=outcome,
                            market=market,
                        )
                        parts.append(traj_feats.to_array())
                    except Exception:
                        # Fill with zeros if trajectory extraction fails
                        from odds_analytics.feature_extraction import TrajectoryFeatures

                        parts.append(np.zeros(len(TrajectoryFeatures.get_feature_names())))
                else:
                    from odds_analytics.feature_extraction import TrajectoryFeatures

                    parts.append(np.zeros(len(TrajectoryFeatures.get_feature_names())))

            # PM features
            if use_pm and pm_context is not None:
                pm_data = await _load_pm_snapshot_data(
                    event_id=event.id,
                    pm_context=pm_context,
                    decision_time=snapshot.snapshot_time,
                    session=session,
                    velocity_window_hours=config.pm_velocity_window_hours,
                    tolerance_minutes=config.pm_price_tolerance_minutes,
                )
                if pm_data is not None and pm_extractor and xsrc_extractor:
                    try:
                        pm_feats = pm_extractor.extract(
                            price_snapshot=pm_data["price_snapshot"],
                            orderbook_snapshot=pm_data["orderbook_snapshot"],
                            recent_prices=pm_data["recent_prices"],
                            home_outcome_index=pm_data["home_outcome_index"],
                        )
                        sb_feats = None
                        if pm_data.get("sb_odds") and sb_extractor_for_pm:
                            try:
                                sb_feats = sb_extractor_for_pm.extract_features(
                                    event=backtest_event,
                                    odds_data=pm_data["sb_odds"],
                                    outcome=outcome,
                                    market=market,
                                )
                            except Exception:
                                pass
                        xsrc_feats = xsrc_extractor.extract(
                            pm_features=pm_feats, sb_features=sb_feats
                        )
                        parts.append(np.concatenate([pm_feats.to_array(), xsrc_feats.to_array()]))
                    except Exception:
                        n_pm = len(PolymarketTabularFeatures.get_feature_names())
                        n_xsrc = len(CrossSourceFeatures.get_feature_names())
                        parts.append(np.full(n_pm + n_xsrc, np.nan))
                else:
                    n_pm = len(PolymarketTabularFeatures.get_feature_names())
                    n_xsrc = len(CrossSourceFeatures.get_feature_names())
                    parts.append(np.full(n_pm + n_xsrc, np.nan))

            # hours_until_event feature
            parts.append(np.array([hours_until]))

            X_list.append(np.concatenate(parts))
            y_list.append(target)
            event_id_list.append(event.id)
            event_had_rows = True
            total_rows += 1

        if not event_had_rows:
            skipped_events += 1

    if not X_list:
        raise ValueError(
            f"No valid training data after processing {len(valid_events)} events "
            f"(skipped {skipped_events})"
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    event_ids = np.array(event_id_list)

    n_events = len(set(event_id_list))
    logger.info(
        "prepared_multi_horizon_data",
        num_samples=len(X),
        num_events=n_events,
        num_features=len(feature_names),
        avg_rows_per_event=total_rows / max(n_events, 1),
        skipped_events=skipped_events,
        target_mean=float(np.mean(y)),
        target_std=float(np.std(y)),
        decision_hours_range=config.decision_hours_range,
    )

    return TrainingDataResult(
        X=X,
        y=y,
        feature_names=feature_names,
        event_ids=event_ids,
    )
