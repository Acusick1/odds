"""
5-Layer Feature Pipeline for ML Training.

Separates data collection from model-specific formatting via five layers:
  1. Collection   — EventDataBundle (loads all data once per event)
  2. Sampling     — SnapshotSampler protocol (TierSampler / TimeRangeSampler)
  3. Target       — pure functions from sequence_loader
  4. Adapters     — FeatureAdapter protocol (XGBoostAdapter, LSTMAdapter)
  5. Orchestrator — single prepare_training_data() entry point

Example:
    ```python
    from odds_analytics.feature_groups import prepare_training_data

    result = await prepare_training_data(events, session, config)
    X_train, y_train = result.X, result.y
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import numpy as np
import structlog
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import (
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
    calculate_devigged_pinnacle_target,
    calculate_regression_target,
    extract_odds_from_snapshot,
    extract_pinnacle_h2h_probs,
    load_sequences_for_event,
)
from odds_analytics.training.feature_selection import apply_variance_filter

if TYPE_CHECKING:
    from odds_analytics.training.config import FeatureConfig

logger = structlog.get_logger()

__all__ = [
    "PMEventContext",
    "EventDataBundle",
    "collect_event_data",
    "TierSampler",
    "TimeRangeSampler",
    "AdapterOutput",
    "XGBoostAdapter",
    "LSTMAdapter",
    "PreparedFeatureData",
    "prepare_training_data",
    "filter_completed_events",
]


# =============================================================================
# Layer 1: Collection
# =============================================================================


@dataclass
class PMEventContext:
    """Polymarket context for a sportsbook event."""

    pm_event: PolymarketEvent
    sb_event: Event
    market: PolymarketMarket
    home_idx: int


@dataclass
class EventDataBundle:
    """All pre-loaded data for a single event."""

    event: Event
    snapshots: list[OddsSnapshot]
    closing_snapshot: OddsSnapshot | None
    pm_context: PMEventContext | None
    pm_prices: list[PolymarketPriceSnapshot] = field(default_factory=list)
    pm_orderbooks: list[PolymarketOrderBookSnapshot] = field(default_factory=list)
    sequences: list[list[Odds]] = field(default_factory=list)


async def collect_event_data(
    event: Event,
    session: AsyncSession,
    config: FeatureConfig,
) -> EventDataBundle:
    """Load all data for an event in bulk (minimises per-snapshot DB queries).

    - Loads all OddsSnapshot records once
    - Finds closing snapshot (last in closing_tier)
    - Loads PM context (PM event, moneyline market, home_idx)
    - Bulk-loads all PM prices + orderbooks for the market (2 queries total)
    - Loads sequences via load_sequences_for_event when trajectory features requested
    """
    from odds_lambda.storage.polymarket_reader import PolymarketReader
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)
    all_snapshots = await reader.get_snapshots_for_event(event.id)

    # Derive closing snapshot from already-loaded snapshots (avoid extra DB query)
    closing_tier_value = config.closing_tier.value
    closing_candidates = [s for s in all_snapshots if s.fetch_tier == closing_tier_value]
    closing_snapshot = closing_candidates[-1] if closing_candidates else None

    # PM context (optional)
    pm_context: PMEventContext | None = None
    pm_prices: list[PolymarketPriceSnapshot] = []
    pm_orderbooks: list[PolymarketOrderBookSnapshot] = []

    if "polymarket" in config.feature_groups:
        pm_result = await session.execute(
            select(PolymarketEvent).where(PolymarketEvent.event_id == event.id)
        )
        pm_event = pm_result.scalars().first()

        if pm_event is not None:
            pm_reader = PolymarketReader(session)
            market = await pm_reader.get_moneyline_market(pm_event.id)

            if market is not None:
                home_idx = resolve_home_outcome_index(market, event.home_team)
                if home_idx is not None:
                    pm_context = PMEventContext(
                        pm_event=pm_event,
                        sb_event=event,
                        market=market,
                        home_idx=home_idx,
                    )
                    # Bulk-load all PM data for this market (2 queries)
                    pm_prices = await pm_reader.get_prices_for_market(market.id)
                    pm_orderbooks = await pm_reader.get_orderbooks_for_market(market.id)
                else:
                    logger.debug(
                        "pm_home_outcome_unresolved",
                        event_id=event.id,
                        home_team=event.home_team,
                    )

    # Sequences for trajectory features or LSTM adapter
    sequences: list[list[Odds]] = []
    if "trajectory" in config.feature_groups or config.adapter == "lstm":
        sequences = await load_sequences_for_event(event.id, session)

    return EventDataBundle(
        event=event,
        snapshots=all_snapshots,
        closing_snapshot=closing_snapshot,
        pm_context=pm_context,
        pm_prices=pm_prices,
        pm_orderbooks=pm_orderbooks,
        sequences=sequences,
    )


# =============================================================================
# Layer 2: Sampling
# =============================================================================


class SnapshotSampler(Protocol):
    """Protocol for snapshot sampling strategies."""

    def sample(self, bundle: EventDataBundle) -> list[OddsSnapshot]:
        """Return snapshots to use as decision points for this event."""
        ...


class TierSampler:
    """Returns the latest snapshot no closer to game than the decision tier (single row per event).

    Snapshots in the decision tier or any earlier tier (further from game) are
    candidates. The most recent candidate by wall-clock time is returned.
    """

    def __init__(self, decision_tier: str) -> None:
        from odds_lambda.fetch_tier import FetchTier

        self._decision_tier = FetchTier(decision_tier)

    def sample(self, bundle: EventDataBundle) -> list[OddsSnapshot]:
        from odds_lambda.fetch_tier import FetchTier

        tier_order = FetchTier.get_priority_order()  # CLOSING first (closest)
        decision_idx = tier_order.index(self._decision_tier)

        candidates = []
        for s in bundle.snapshots:
            if s.fetch_tier:
                try:
                    tier = FetchTier(s.fetch_tier)
                    tier_idx = tier_order.index(tier)
                    if tier_idx >= decision_idx:
                        candidates.append(s)
                except ValueError:
                    pass

        if not candidates:
            return []

        return [max(candidates, key=lambda s: s.snapshot_time)]


class TimeRangeSampler:
    """Stratified sampling across [min_hours, max_hours] before game."""

    def __init__(self, min_hours: float, max_hours: float, max_samples: int) -> None:
        self._min_hours = min_hours
        self._max_hours = max_hours
        self._max_samples = max_samples

    def sample(self, bundle: EventDataBundle) -> list[OddsSnapshot]:
        event = bundle.event
        commence = event.commence_time

        range_start = commence - timedelta(hours=self._max_hours)
        range_end = commence - timedelta(hours=self._min_hours)

        in_range = [s for s in bundle.snapshots if range_start <= s.snapshot_time <= range_end]

        if not in_range:
            return []

        if len(in_range) <= self._max_samples:
            return sorted(in_range, key=lambda s: s.snapshot_time)

        # Stratified: divide range into equal bins, pick nearest to each midpoint
        range_seconds = (range_end - range_start).total_seconds()
        bin_size = range_seconds / self._max_samples
        selected: list[OddsSnapshot] = []

        for i in range(self._max_samples):
            bin_mid = range_start + timedelta(seconds=(i + 0.5) * bin_size)
            best: OddsSnapshot | None = None
            best_diff = float("inf")

            for s in in_range:
                diff = abs((s.snapshot_time - bin_mid).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best = s

            if best is not None and best not in selected:
                selected.append(best)

        selected.sort(key=lambda s: s.snapshot_time)
        return selected


def _make_sampler(config: FeatureConfig) -> SnapshotSampler:
    """Instantiate the sampler specified in config.sampling."""
    sc = config.sampling
    if sc.strategy == "tier":
        return TierSampler(sc.decision_tier.value)
    return TimeRangeSampler(sc.min_hours, sc.max_hours, sc.max_samples_per_event)


# =============================================================================
# Helpers
# =============================================================================


def filter_completed_events(events: list[Event]) -> list[Event]:
    """Filter to events with final scores."""
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


def _find_nearest_pm_price(
    prices: list[PolymarketPriceSnapshot],
    target_time: datetime,
    tolerance_minutes: int,
) -> PolymarketPriceSnapshot | None:
    """Return the price snapshot closest to (and not after) target_time within tolerance."""
    time_lower = target_time - timedelta(minutes=tolerance_minutes)
    candidates = [p for p in prices if time_lower <= p.snapshot_time <= target_time]
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs((p.snapshot_time - target_time).total_seconds()))


def _find_nearest_pm_orderbook(
    orderbooks: list[PolymarketOrderBookSnapshot],
    target_time: datetime,
    tolerance_minutes: int,
) -> PolymarketOrderBookSnapshot | None:
    """Return the orderbook snapshot closest to (and not after) target_time within tolerance."""
    time_lower = target_time - timedelta(minutes=tolerance_minutes)
    candidates = [o for o in orderbooks if time_lower <= o.snapshot_time <= target_time]
    if not candidates:
        return None
    return min(candidates, key=lambda o: abs((o.snapshot_time - target_time).total_seconds()))


def _find_velocity_prices(
    prices: list[PolymarketPriceSnapshot],
    target_time: datetime,
    velocity_window_hours: float,
) -> list[PolymarketPriceSnapshot]:
    """Return prices in the velocity window ending at target_time."""
    window_start = target_time - timedelta(hours=velocity_window_hours)
    return [p for p in prices if window_start <= p.snapshot_time <= target_time]


# =============================================================================
# Layer 4: Adapters
# =============================================================================


@dataclass
class AdapterOutput:
    """Structured output from a FeatureAdapter.transform() call."""

    features: np.ndarray  # 1D for XGBoost, 2D (timesteps, features) for LSTM
    mask: np.ndarray | None = None  # (timesteps,) boolean, LSTM only


class FeatureAdapter(Protocol):
    """Protocol for model-specific feature formatting."""

    def feature_names(self, config: FeatureConfig) -> list[str]:
        """Return ordered list of feature names."""
        ...

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: OddsSnapshot,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        """Extract features for one (event, snapshot) pair. Returns None to skip row."""
        ...


class XGBoostAdapter:
    """Tabular feature adapter for XGBoost (and other tree-based models)."""

    def feature_names(self, config: FeatureConfig) -> list[str]:
        names: list[str] = []
        if "tabular" in config.feature_groups:
            from odds_analytics.feature_extraction import TabularFeatures

            names.extend(f"tab_{n}" for n in TabularFeatures.get_feature_names())
        if "trajectory" in config.feature_groups:
            from odds_analytics.feature_extraction import TrajectoryFeatures

            names.extend(f"traj_{n}" for n in TrajectoryFeatures.get_feature_names())
        if "polymarket" in config.feature_groups:
            names.extend(f"pm_{n}" for n in PolymarketTabularFeatures.get_feature_names())
            names.extend(f"xsrc_{n}" for n in CrossSourceFeatures.get_feature_names())
        names.append("hours_until_event")
        return names

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: OddsSnapshot,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        event = bundle.event
        market = config.markets[0] if config.markets else "h2h"
        outcome = event.home_team if config.outcome == "home" else event.away_team
        backtest_event = make_backtest_event(event)
        tab_extractor = TabularFeatureExtractor.from_config(config)

        parts: list[np.ndarray] = []

        # --- Tabular features ---
        if "tabular" in config.feature_groups:
            snap_odds = extract_odds_from_snapshot(snapshot, event.id, market=market)
            if not snap_odds:
                return None
            try:
                tab_feats = tab_extractor.extract_features(
                    event=backtest_event,
                    odds_data=snap_odds,
                    outcome=outcome,
                    market=market,
                )
                parts.append(tab_feats.to_array())
            except Exception:
                logger.debug("tabular_extraction_failed", event_id=event.id)
                return None

        # --- Trajectory features ---
        if "trajectory" in config.feature_groups:
            from odds_analytics.feature_extraction import TrajectoryFeatures

            traj_extractor = TrajectoryFeatureExtractor.from_config(config)
            seqs_up_to = [
                s for s in bundle.sequences if s and s[0].odds_timestamp <= snapshot.snapshot_time
            ]
            if len(seqs_up_to) >= 2:
                try:
                    traj_feats = traj_extractor.extract_features(
                        event=backtest_event,
                        odds_data=seqs_up_to,
                        outcome=outcome,
                        market=market,
                    )
                    parts.append(traj_feats.to_array())
                except Exception:
                    logger.debug("trajectory_extraction_failed", event_id=event.id)
                    parts.append(np.zeros(len(TrajectoryFeatures.get_feature_names())))
            else:
                parts.append(np.zeros(len(TrajectoryFeatures.get_feature_names())))

        # --- Polymarket features (NaN-fill when unavailable to keep row) ---
        if "polymarket" in config.feature_groups:
            n_pm = len(PolymarketTabularFeatures.get_feature_names())
            n_xsrc = len(CrossSourceFeatures.get_feature_names())
            nan_block = np.full(n_pm + n_xsrc, np.nan)

            if bundle.pm_context is None:
                parts.append(nan_block)
            else:
                ctx = bundle.pm_context
                target_time = snapshot.snapshot_time
                price_snap = _find_nearest_pm_price(
                    bundle.pm_prices, target_time, config.pm_price_tolerance_minutes
                )

                if price_snap is None:
                    parts.append(nan_block)
                else:
                    ob_snap = _find_nearest_pm_orderbook(
                        bundle.pm_orderbooks, target_time, config.pm_price_tolerance_minutes
                    )
                    recent = _find_velocity_prices(
                        bundle.pm_prices, target_time, config.pm_velocity_window_hours
                    )

                    try:
                        pm_extractor = PolymarketFeatureExtractor(
                            velocity_window_hours=config.pm_velocity_window_hours
                        )
                        xsrc_extractor = CrossSourceFeatureExtractor()

                        pm_feats = pm_extractor.extract(
                            price_snapshot=price_snap,
                            orderbook_snapshot=ob_snap,
                            recent_prices=recent,
                            home_outcome_index=ctx.home_idx,
                        )

                        # Try to align SB odds for cross-source features
                        sb_feats = None
                        sb_odds_at_time = extract_odds_from_snapshot(
                            snapshot, event.id, market=market
                        )
                        if sb_odds_at_time:
                            try:
                                sb_feats = tab_extractor.extract_features(
                                    event=backtest_event,
                                    odds_data=sb_odds_at_time,
                                    outcome=outcome,
                                    market=market,
                                )
                            except Exception:
                                pass

                        xsrc_feats = xsrc_extractor.extract(
                            pm_features=pm_feats,
                            sb_features=sb_feats,
                        )
                        parts.append(np.concatenate([pm_feats.to_array(), xsrc_feats.to_array()]))
                    except Exception:
                        logger.debug("pm_feature_extraction_failed", event_id=event.id)
                        parts.append(nan_block)

        # --- hours_until_event ---
        hours_until = (event.commence_time - snapshot.snapshot_time).total_seconds() / 3600
        parts.append(np.array([hours_until]))

        return AdapterOutput(features=np.concatenate(parts))


class LSTMAdapter:
    """Sequence feature adapter for LSTM (and other recurrent) models."""

    def feature_names(self, config: FeatureConfig) -> list[str]:
        from odds_analytics.feature_extraction import SequenceFeatures

        return SequenceFeatures.get_feature_names()

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: OddsSnapshot,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        from odds_analytics.feature_extraction import SequenceFeatureExtractor

        event = bundle.event
        market = config.markets[0] if config.markets else "h2h"
        outcome = event.home_team if config.outcome == "home" else event.away_team
        backtest_event = make_backtest_event(event)

        # Filter sequences to those whose first entry is at or before the snapshot time,
        # ensuring no look-ahead bias: snapshot_time is the look-ahead cutoff.
        seqs_up_to = [
            s for s in bundle.sequences if s and s[0].odds_timestamp <= snapshot.snapshot_time
        ]

        extractor = SequenceFeatureExtractor.from_config(config)
        try:
            result = extractor.extract_features(
                event=backtest_event,
                odds_data=seqs_up_to,
                outcome=outcome,
                market=market,
            )
        except Exception as e:
            logger.debug("sequence_extraction_failed", event_id=event.id, error=str(e))
            return None

        return AdapterOutput(features=result["sequence"], mask=result["mask"])


def _make_adapter(config: FeatureConfig) -> FeatureAdapter:
    if config.adapter == "xgboost":
        return XGBoostAdapter()
    if config.adapter == "lstm":
        return LSTMAdapter()
    raise NotImplementedError(f"Adapter '{config.adapter}' is not yet implemented")


# =============================================================================
# Layer 3: Target helpers
# =============================================================================


def _compute_target(
    snapshot: OddsSnapshot,
    closing_snapshot: OddsSnapshot,
    event: Event,
    config: FeatureConfig,
) -> float | None:
    """Compute regression target for one (snapshot, closing) pair."""
    market = config.markets[0] if config.markets else "h2h"

    closing_odds_all = extract_odds_from_snapshot(closing_snapshot, event.id, market=market)

    if config.target_type == "devigged_pinnacle":
        snapshot_odds_all = extract_odds_from_snapshot(snapshot, event.id, market=market)
        return calculate_devigged_pinnacle_target(
            snapshot_odds_all, closing_odds_all, event.home_team, event.away_team
        )
    else:
        # "raw": avg implied prob delta (snapshot → closing)
        outcome = event.home_team if config.outcome == "home" else event.away_team
        snap_odds = extract_odds_from_snapshot(snapshot, event.id, market=market, outcome=outcome)
        closing_odds = extract_odds_from_snapshot(
            closing_snapshot, event.id, market=market, outcome=outcome
        )
        return calculate_regression_target(snap_odds, closing_odds, market)


def _has_pinnacle_closing(closing_snapshot: OddsSnapshot, event: Event) -> bool:
    """Check if closing snapshot has Pinnacle h2h data (required for devigged_pinnacle target)."""
    market = "h2h"
    closing_odds_all = extract_odds_from_snapshot(closing_snapshot, event.id, market=market)
    return (
        extract_pinnacle_h2h_probs(closing_odds_all, event.home_team, event.away_team) is not None
    )


# =============================================================================
# Layer 5: Orchestrator
# =============================================================================


class PreparedFeatureData:
    """Pre-split feature matrix with targets and metadata."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        masks: np.ndarray | None = None,
        event_ids: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.masks = masks
        self.event_ids = event_ids

    @property
    def num_samples(self) -> int:
        return len(self.X)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


async def prepare_training_data(
    events: list[Event],
    session: AsyncSession,
    config: FeatureConfig,
) -> PreparedFeatureData:
    """Unified training data preparation using the 5-layer adapter architecture.

    Sampling strategy and target type are both controlled via config:
      - config.sampling.strategy="tier"        → single row per event
      - config.sampling.strategy="time_range"  → multiple rows per event
      - config.target_type="raw"               → avg implied-prob delta
      - config.target_type="devigged_pinnacle" → devigged Pinnacle delta

    For devigged_pinnacle, events without Pinnacle closing data are dropped.
    Polymarket features NaN-fill when PM data is unavailable, so events are
    kept rather than dropped when PM is missing.

    Args:
        events: List of Event objects (filters to FINAL status internally)
        session: Async database session
        config: FeatureConfig controlling sampling, features, and target

    Returns:
        PreparedFeatureData with X, y, feature_names, masks, and event_ids for CV
    """
    valid_events = filter_completed_events(events)
    if not valid_events:
        raise ValueError(f"No valid events found in {len(events)} total events")

    adapter = _make_adapter(config)
    sampler = _make_sampler(config)
    feature_names = adapter.feature_names(config)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    masks_list: list[np.ndarray] = []
    event_id_list: list[str] = []
    skipped_events = 0
    total_rows = 0

    for event in valid_events:
        # Load all data for this event in bulk
        bundle = await collect_event_data(event, session, config)

        # Closing snapshot is required
        if bundle.closing_snapshot is None:
            skipped_events += 1
            continue

        # For devigged_pinnacle, must have Pinnacle closing data
        if config.target_type == "devigged_pinnacle" and not _has_pinnacle_closing(
            bundle.closing_snapshot, event
        ):
            skipped_events += 1
            continue

        # Get sampled decision snapshots
        decision_snapshots = sampler.sample(bundle)
        if not decision_snapshots:
            skipped_events += 1
            continue

        event_had_rows = False
        for snapshot in decision_snapshots:
            target = _compute_target(snapshot, bundle.closing_snapshot, event, config)
            if target is None:
                continue

            output = adapter.transform(bundle, snapshot, config)
            if output is None:
                continue

            X_list.append(output.features)
            if output.mask is not None:
                masks_list.append(output.mask)
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
    masks = np.array(masks_list, dtype=bool) if masks_list else None

    X, feature_names, _ = apply_variance_filter(X, feature_names, config.variance_threshold)

    n_events = len(set(event_id_list))
    logger.info(
        "prepared_training_data",
        num_samples=len(X),
        num_events=n_events,
        num_features=len(feature_names),
        avg_rows_per_event=total_rows / max(n_events, 1),
        skipped_events=skipped_events,
        sampling_strategy=config.sampling.strategy,
        target_type=config.target_type,
        target_mean=float(np.mean(y)),
        target_std=float(np.std(y)),
    )

    return PreparedFeatureData(
        X=X,
        y=y,
        feature_names=feature_names,
        masks=masks,
        event_ids=event_ids,
    )
