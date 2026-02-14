"""Cross-source Polymarket feature extraction for CLV prediction.

Provides dataclasses and extractors for Polymarket-specific features and
cross-source features that compare Polymarket prices against sportsbook odds.

Feature types:
- PolymarketTabularFeatures: PM price, liquidity, order book, and price velocity
- CrossSourceFeatures: PM vs SB divergence and direction signals
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
from odds_core.polymarket_models import (
    PolymarketMarket,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from odds_lambda.polymarket_matching import normalize_team

from odds_analytics.feature_extraction import TabularFeatures

__all__ = [
    "PolymarketTabularFeatures",
    "CrossSourceFeatures",
    "PolymarketFeatureExtractor",
    "CrossSourceFeatureExtractor",
    "resolve_home_outcome_index",
]


# =============================================================================
# Feature Dataclasses
# =============================================================================


@dataclass
class PolymarketTabularFeatures:
    """
    Point-in-time features from Polymarket price and order book data.

    All fields are optional since PM data may be unavailable or incomplete.
    None values become np.nan in array representation.
    """

    # PM implied probabilities (0.0–1.0)
    pm_home_prob: float | None = None
    pm_away_prob: float | None = None

    # Order book microstructure
    pm_spread: float | None = None
    pm_midpoint: float | None = None
    pm_best_bid: float | None = None
    pm_best_ask: float | None = None

    # Market liquidity signals
    pm_volume: float | None = None
    pm_liquidity: float | None = None

    # Order book depth (populated from PolymarketOrderBookSnapshot when available)
    pm_bid_depth: float | None = None
    pm_ask_depth: float | None = None
    pm_imbalance: float | None = None
    pm_weighted_mid: float | None = None

    # PM price velocity over recent window
    pm_price_velocity: float | None = None  # prob change per hour (linear)
    pm_price_acceleration: float | None = None  # change in velocity (first vs second half)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array. None → np.nan."""
        return np.array(
            [
                getattr(self, f.name) if getattr(self, f.name) is not None else np.nan
                for f in fields(self)
            ],
            dtype=np.float64,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


@dataclass
class CrossSourceFeatures:
    """
    Features comparing Polymarket prices against sportsbook odds.

    Captures divergence, direction, and relative pricing signals between
    the two market types. All fields optional.
    """

    # PM vs SB consensus probability divergence
    pm_sb_prob_divergence: float | None = None  # pm_home_prob - sb_consensus_prob
    pm_sb_divergence_abs: float | None = None  # abs(divergence)
    pm_sb_divergence_direction: float | None = None  # +1 if PM higher, -1 if lower, 0 if equal

    # PM spread vs SB market hold (relative liquidity cost)
    pm_spread_vs_sb_hold: float | None = None  # pm_spread - sb_avg_market_hold

    # PM vs sharp bookmaker divergence
    pm_sharp_divergence: float | None = None  # pm_home_prob - sharp_prob
    pm_sharp_divergence_abs: float | None = None

    # PM midpoint vs SB consensus (using order book midpoint if available)
    pm_mid_vs_sb_consensus: float | None = None  # pm_midpoint - sb_consensus_prob

    # PM-SB convergence rate (change in divergence per hour)
    # None when insufficient historical cross-source data
    pm_sb_convergence_rate: float | None = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array. None → np.nan."""
        return np.array(
            [
                getattr(self, f.name) if getattr(self, f.name) is not None else np.nan
                for f in fields(self)
            ],
            dtype=np.float64,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


# =============================================================================
# Helper
# =============================================================================


def resolve_home_outcome_index(pm_market: PolymarketMarket, home_team: str) -> int | None:
    """
    Determine which outcome index (0 or 1) corresponds to the home team.

    Uses the TEAM_ALIASES lookup from polymarket_matching to normalize PM
    outcome names (e.g. "Lakers") to canonical form (e.g. "Los Angeles Lakers").

    Args:
        pm_market: PolymarketMarket with outcomes list (e.g. ["Lakers", "Celtics"])
        home_team: Canonical sportsbook home team name

    Returns:
        0 or 1, or None if no match found
    """
    for i, outcome_name in enumerate(pm_market.outcomes):
        canonical = normalize_team(outcome_name)
        if canonical == home_team:
            return i
    return None


# =============================================================================
# Polymarket Feature Extractor
# =============================================================================


class PolymarketFeatureExtractor:
    """
    Extracts point-in-time features from PM price and order book snapshots.

    Args:
        velocity_window_hours: Hours of price history used for velocity computation.
    """

    def __init__(self, velocity_window_hours: float = 2.0) -> None:
        self.velocity_window_hours = velocity_window_hours

    def extract(
        self,
        price_snapshot: PolymarketPriceSnapshot | None,
        orderbook_snapshot: PolymarketOrderBookSnapshot | None,
        recent_prices: list[PolymarketPriceSnapshot],
        home_outcome_index: int,
    ) -> PolymarketTabularFeatures:
        """
        Extract PM features from snapshot data.

        Args:
            price_snapshot: Price snapshot at decision time
            orderbook_snapshot: Order book snapshot at decision time (may be None)
            recent_prices: Price series over velocity window (may be empty)
            home_outcome_index: Which outcome (0 or 1) is the home team

        Returns:
            PolymarketTabularFeatures with available fields populated
        """
        if price_snapshot is None:
            return PolymarketTabularFeatures()

        # PM implied probabilities
        if home_outcome_index == 0:
            pm_home_prob = price_snapshot.outcome_0_price
            pm_away_prob = price_snapshot.outcome_1_price
        else:
            pm_home_prob = price_snapshot.outcome_1_price
            pm_away_prob = price_snapshot.outcome_0_price

        features = PolymarketTabularFeatures(
            pm_home_prob=pm_home_prob,
            pm_away_prob=pm_away_prob,
            pm_spread=price_snapshot.spread,
            pm_midpoint=price_snapshot.midpoint,
            pm_best_bid=price_snapshot.best_bid,
            pm_best_ask=price_snapshot.best_ask,
            pm_volume=price_snapshot.volume,
            pm_liquidity=price_snapshot.liquidity,
        )

        # Order book depth (from separate snapshot if available)
        if orderbook_snapshot is not None:
            features.pm_bid_depth = orderbook_snapshot.bid_depth_total
            features.pm_ask_depth = orderbook_snapshot.ask_depth_total
            features.pm_imbalance = orderbook_snapshot.imbalance
            features.pm_weighted_mid = orderbook_snapshot.weighted_mid

        # Price velocity from recent series
        velocity, acceleration = self._compute_velocity(recent_prices, home_outcome_index)
        features.pm_price_velocity = velocity
        features.pm_price_acceleration = acceleration

        return features

    def _compute_velocity(
        self,
        recent_prices: list[PolymarketPriceSnapshot],
        home_outcome_index: int,
    ) -> tuple[float | None, float | None]:
        """Compute price velocity and acceleration from recent price series."""
        if len(recent_prices) < 2:
            return None, None

        probs = [
            s.outcome_0_price if home_outcome_index == 0 else s.outcome_1_price
            for s in recent_prices
        ]
        times = [s.snapshot_time for s in recent_prices]

        total_seconds = (times[-1] - times[0]).total_seconds()
        if total_seconds <= 0:
            return None, None

        total_hours = total_seconds / 3600
        velocity = (probs[-1] - probs[0]) / total_hours

        # Acceleration: slope of second half minus slope of first half
        acceleration: float | None = None
        if len(probs) >= 3:
            mid = len(probs) // 2
            first_hours = (times[mid] - times[0]).total_seconds() / 3600
            second_hours = (times[-1] - times[mid]).total_seconds() / 3600
            if first_hours > 0 and second_hours > 0:
                first_slope = (probs[mid] - probs[0]) / first_hours
                second_slope = (probs[-1] - probs[mid]) / second_hours
                acceleration = second_slope - first_slope

        return velocity, acceleration


# =============================================================================
# Cross-Source Feature Extractor
# =============================================================================


class CrossSourceFeatureExtractor:
    """
    Computes divergence and direction features between PM and SB prices.

    Requires both PM and SB features to be computed first.
    """

    def extract(
        self,
        pm_features: PolymarketTabularFeatures,
        sb_features: TabularFeatures | None,
    ) -> CrossSourceFeatures:
        """
        Compute cross-source features from PM and SB feature objects.

        Args:
            pm_features: Extracted Polymarket features
            sb_features: Extracted sportsbook tabular features (may be None)

        Returns:
            CrossSourceFeatures with available fields populated
        """
        if sb_features is None or pm_features.pm_home_prob is None:
            return CrossSourceFeatures()

        result = CrossSourceFeatures()

        # PM vs SB consensus divergence
        if sb_features.consensus_prob is not None:
            divergence = pm_features.pm_home_prob - sb_features.consensus_prob
            result.pm_sb_prob_divergence = divergence
            result.pm_sb_divergence_abs = abs(divergence)
            result.pm_sb_divergence_direction = (
                1.0 if divergence > 0 else -1.0 if divergence < 0 else 0.0
            )

        # PM spread vs SB market hold
        if pm_features.pm_spread is not None and sb_features.avg_market_hold is not None:
            result.pm_spread_vs_sb_hold = pm_features.pm_spread - sb_features.avg_market_hold

        # PM vs sharp divergence
        if sb_features.sharp_prob is not None:
            sharp_div = pm_features.pm_home_prob - sb_features.sharp_prob
            result.pm_sharp_divergence = sharp_div
            result.pm_sharp_divergence_abs = abs(sharp_div)

        # PM midpoint vs SB consensus
        if pm_features.pm_midpoint is not None and sb_features.consensus_prob is not None:
            result.pm_mid_vs_sb_consensus = pm_features.pm_midpoint - sb_features.consensus_prob

        # pm_sb_convergence_rate: not computed in v1 (requires historical cross-source data)

        return result
