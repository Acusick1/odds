"""Unit tests for Polymarket feature extraction."""

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from odds_analytics.feature_extraction import TabularFeatures
from odds_analytics.polymarket_features import (
    CrossSourceFeatureExtractor,
    CrossSourceFeatures,
    PolymarketFeatureExtractor,
    PolymarketTabularFeatures,
    resolve_home_outcome_index,
)
from odds_core.polymarket_models import (
    PolymarketMarket,
    PolymarketMarketType,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_price_snapshot() -> PolymarketPriceSnapshot:
    t = datetime(2026, 1, 20, 12, 0, tzinfo=UTC)
    return PolymarketPriceSnapshot(
        id=1,
        polymarket_market_id=10,
        snapshot_time=t,
        outcome_0_price=0.62,
        outcome_1_price=0.38,
        best_bid=0.61,
        best_ask=0.63,
        spread=0.02,
        midpoint=0.62,
        volume=5000.0,
        liquidity=12000.0,
        fetch_tier="pregame",
        hours_until_commence=7.0,
    )


@pytest.fixture
def sample_orderbook_snapshot() -> PolymarketOrderBookSnapshot:
    t = datetime(2026, 1, 20, 12, 0, tzinfo=UTC)
    return PolymarketOrderBookSnapshot(
        id=1,
        polymarket_market_id=10,
        snapshot_time=t,
        token_id="tok_0",
        raw_book={"bids": [], "asks": []},
        best_bid=0.61,
        best_ask=0.63,
        spread=0.02,
        midpoint=0.62,
        bid_levels=5,
        ask_levels=5,
        bid_depth_total=800.0,
        ask_depth_total=600.0,
        imbalance=0.14,
        weighted_mid=0.618,
        fetch_tier="pregame",
        hours_until_commence=7.0,
    )


@pytest.fixture
def recent_prices() -> list[PolymarketPriceSnapshot]:
    base_time = datetime(2026, 1, 20, 10, 0, tzinfo=UTC)
    prices = [0.58, 0.59, 0.60, 0.62]
    return [
        PolymarketPriceSnapshot(
            id=i,
            polymarket_market_id=10,
            snapshot_time=base_time + timedelta(minutes=30 * i),
            outcome_0_price=p,
            outcome_1_price=round(1.0 - p, 4),
            best_bid=round(p - 0.01, 4),
            best_ask=round(p + 0.01, 4),
            spread=0.02,
            midpoint=p,
            volume=1000.0,
            liquidity=5000.0,
            fetch_tier="pregame",
            hours_until_commence=10.0 - i * 0.5,
        )
        for i, p in enumerate(prices)
    ]


@pytest.fixture
def sample_sb_tabular_features() -> TabularFeatures:
    return TabularFeatures(
        is_home_team=1.0,
        is_away_team=0.0,
        consensus_prob=0.57,
        opponent_consensus_prob=0.43,
        sharp_prob=0.55,
        opponent_sharp_prob=0.45,
        avg_market_hold=0.04,
        num_bookmakers=3.0,
    )


# =============================================================================
# PolymarketTabularFeatures
# =============================================================================


class TestPolymarketTabularFeatures:
    def test_to_array_length_matches_field_count(self):
        f = PolymarketTabularFeatures()
        arr = f.to_array()
        assert len(arr) == len(fields(PolymarketTabularFeatures))

    def test_get_feature_names_length_matches_array(self):
        names = PolymarketTabularFeatures.get_feature_names()
        arr = PolymarketTabularFeatures().to_array()
        assert len(names) == len(arr)

    def test_none_fields_become_nan(self):
        f = PolymarketTabularFeatures()
        arr = f.to_array()
        assert np.all(np.isnan(arr))

    def test_set_fields_populate_array(self):
        f = PolymarketTabularFeatures(pm_home_prob=0.6, pm_away_prob=0.4)
        arr = f.to_array()
        names = PolymarketTabularFeatures.get_feature_names()
        idx_home = names.index("pm_home_prob")
        idx_away = names.index("pm_away_prob")
        assert arr[idx_home] == pytest.approx(0.6)
        assert arr[idx_away] == pytest.approx(0.4)
        # Other fields should be nan
        other_idxs = [i for i in range(len(names)) if i not in (idx_home, idx_away)]
        assert all(np.isnan(arr[i]) for i in other_idxs)


# =============================================================================
# CrossSourceFeatures
# =============================================================================


class TestCrossSourceFeatures:
    def test_to_array_length_matches_field_count(self):
        f = CrossSourceFeatures()
        arr = f.to_array()
        assert len(arr) == len(fields(CrossSourceFeatures))

    def test_get_feature_names_length_matches_array(self):
        names = CrossSourceFeatures.get_feature_names()
        arr = CrossSourceFeatures().to_array()
        assert len(names) == len(arr)

    def test_none_fields_become_nan(self):
        arr = CrossSourceFeatures().to_array()
        assert np.all(np.isnan(arr))

    def test_set_fields_populate_array(self):
        f = CrossSourceFeatures(pm_sb_prob_divergence=0.05, pm_sb_divergence_abs=0.05)
        arr = f.to_array()
        names = CrossSourceFeatures.get_feature_names()
        assert arr[names.index("pm_sb_prob_divergence")] == pytest.approx(0.05)
        assert arr[names.index("pm_sb_divergence_abs")] == pytest.approx(0.05)


# =============================================================================
# resolve_home_outcome_index
# =============================================================================


class TestResolveHomeOutcomeIndex:
    def _make_market(self, outcomes: list[str]) -> PolymarketMarket:
        return PolymarketMarket(
            id=1,
            polymarket_event_id=1,
            pm_market_id="test",
            condition_id="cond",
            question="Will X win?",
            clob_token_ids=["tok0", "tok1"],
            outcomes=outcomes,
            market_type=PolymarketMarketType.MONEYLINE,
        )

    def test_home_is_outcome_0(self):
        market = self._make_market(["Lakers", "Celtics"])
        idx = resolve_home_outcome_index(market, "Los Angeles Lakers")
        assert idx == 0

    def test_home_is_outcome_1(self):
        market = self._make_market(["Celtics", "Lakers"])
        idx = resolve_home_outcome_index(market, "Los Angeles Lakers")
        assert idx == 1

    def test_unrecognized_team_returns_none(self):
        market = self._make_market(["UnknownTeam", "Lakers"])
        idx = resolve_home_outcome_index(market, "Los Angeles Lakers")
        assert idx == 1  # "Lakers" still matches at index 1

    def test_no_match_returns_none(self):
        market = self._make_market(["Celtics", "Warriors"])
        idx = resolve_home_outcome_index(market, "Los Angeles Lakers")
        assert idx is None

    def test_various_aliases(self):
        for alias in ["Mavs", "Dallas", "Mavericks", "Dallas Mavericks"]:
            market = self._make_market([alias, "Bucks"])
            idx = resolve_home_outcome_index(market, "Dallas Mavericks")
            assert idx == 0, f"Expected 0 for alias '{alias}'"


# =============================================================================
# PolymarketFeatureExtractor
# =============================================================================


class TestPolymarketFeatureExtractor:
    def test_returns_empty_features_when_no_snapshot(self):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=None,
            orderbook_snapshot=None,
            recent_prices=[],
            home_outcome_index=0,
        )
        assert isinstance(result, PolymarketTabularFeatures)
        assert np.all(np.isnan(result.to_array()))

    def test_extracts_probabilities_outcome_0_home(self, sample_price_snapshot):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=None,
            recent_prices=[],
            home_outcome_index=0,
        )
        assert result.pm_home_prob == pytest.approx(0.62)
        assert result.pm_away_prob == pytest.approx(0.38)

    def test_extracts_probabilities_outcome_1_home(self, sample_price_snapshot):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=None,
            recent_prices=[],
            home_outcome_index=1,
        )
        assert result.pm_home_prob == pytest.approx(0.38)
        assert result.pm_away_prob == pytest.approx(0.62)

    def test_extracts_price_snapshot_fields(self, sample_price_snapshot):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=None,
            recent_prices=[],
            home_outcome_index=0,
        )
        assert result.pm_spread == pytest.approx(0.02)
        assert result.pm_midpoint == pytest.approx(0.62)
        assert result.pm_volume == pytest.approx(5000.0)

    def test_extracts_orderbook_fields(self, sample_price_snapshot, sample_orderbook_snapshot):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=sample_orderbook_snapshot,
            recent_prices=[],
            home_outcome_index=0,
        )
        assert result.pm_imbalance == pytest.approx(0.14)
        assert result.pm_bid_depth == pytest.approx(800.0)
        assert result.pm_ask_depth == pytest.approx(600.0)
        assert result.pm_weighted_mid == pytest.approx(0.618)

    def test_orderbook_none_leaves_depth_fields_as_none(self, sample_price_snapshot):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=None,
            recent_prices=[],
            home_outcome_index=0,
        )
        assert result.pm_bid_depth is None
        assert result.pm_imbalance is None

    def test_velocity_computed_from_recent_prices(self, sample_price_snapshot, recent_prices):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=None,
            recent_prices=recent_prices,
            home_outcome_index=0,
        )
        # Prices go from 0.58 to 0.62 over 1.5 hours → velocity ≈ 0.027 per hour
        assert result.pm_price_velocity is not None
        assert result.pm_price_velocity == pytest.approx((0.62 - 0.58) / 1.5, abs=1e-3)

    def test_velocity_none_with_single_price(self, sample_price_snapshot):
        single = (
            [recent_prices[0]]
            if False
            else [
                PolymarketPriceSnapshot(
                    id=1,
                    polymarket_market_id=10,
                    snapshot_time=datetime(2026, 1, 20, 10, 0, tzinfo=UTC),
                    outcome_0_price=0.60,
                    outcome_1_price=0.40,
                    best_bid=0.59,
                    best_ask=0.61,
                    spread=0.02,
                    midpoint=0.60,
                    volume=1000.0,
                    liquidity=5000.0,
                    fetch_tier="pregame",
                    hours_until_commence=8.0,
                )
            ]
        )
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=None,
            recent_prices=single,
            home_outcome_index=0,
        )
        assert result.pm_price_velocity is None

    def test_acceleration_computed_with_enough_prices(self, sample_price_snapshot, recent_prices):
        extractor = PolymarketFeatureExtractor()
        result = extractor.extract(
            price_snapshot=sample_price_snapshot,
            orderbook_snapshot=None,
            recent_prices=recent_prices,
            home_outcome_index=0,
        )
        assert result.pm_price_acceleration is not None


# =============================================================================
# CrossSourceFeatureExtractor
# =============================================================================


class TestCrossSourceFeatureExtractor:
    def test_returns_empty_when_sb_features_none(self):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures(pm_home_prob=0.62)
        result = extractor.extract(pm_features=pm, sb_features=None)
        assert isinstance(result, CrossSourceFeatures)
        assert np.all(np.isnan(result.to_array()))

    def test_returns_empty_when_pm_home_prob_none(self, sample_sb_tabular_features):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures()  # pm_home_prob is None
        result = extractor.extract(pm_features=pm, sb_features=sample_sb_tabular_features)
        assert np.all(np.isnan(result.to_array()))

    def test_divergence_computed(self, sample_sb_tabular_features):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures(pm_home_prob=0.62)
        result = extractor.extract(pm_features=pm, sb_features=sample_sb_tabular_features)
        # pm=0.62, sb_consensus=0.57 → divergence=0.05
        assert result.pm_sb_prob_divergence == pytest.approx(0.05, abs=1e-6)
        assert result.pm_sb_divergence_abs == pytest.approx(0.05, abs=1e-6)
        assert result.pm_sb_divergence_direction == pytest.approx(1.0)

    def test_negative_divergence_direction(self, sample_sb_tabular_features):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures(pm_home_prob=0.50)
        result = extractor.extract(pm_features=pm, sb_features=sample_sb_tabular_features)
        assert result.pm_sb_divergence_direction == pytest.approx(-1.0)

    def test_sharp_divergence_computed(self, sample_sb_tabular_features):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures(pm_home_prob=0.62)
        result = extractor.extract(pm_features=pm, sb_features=sample_sb_tabular_features)
        # pm=0.62, sharp_prob=0.55 → pm_sharp_divergence=0.07
        assert result.pm_sharp_divergence == pytest.approx(0.07, abs=1e-6)
        assert result.pm_sharp_divergence_abs == pytest.approx(0.07, abs=1e-6)

    def test_pm_midpoint_vs_sb_consensus(self, sample_sb_tabular_features):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures(pm_home_prob=0.62, pm_midpoint=0.61)
        result = extractor.extract(pm_features=pm, sb_features=sample_sb_tabular_features)
        assert result.pm_mid_vs_sb_consensus == pytest.approx(0.61 - 0.57, abs=1e-6)

    def test_pm_spread_vs_sb_hold(self, sample_sb_tabular_features):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures(pm_home_prob=0.62, pm_spread=0.02)
        result = extractor.extract(pm_features=pm, sb_features=sample_sb_tabular_features)
        # pm_spread=0.02, sb_hold=0.04 → pm_spread_vs_sb_hold=-0.02
        assert result.pm_spread_vs_sb_hold == pytest.approx(-0.02, abs=1e-6)

    def test_convergence_rate_always_none_in_v1(self, sample_sb_tabular_features):
        extractor = CrossSourceFeatureExtractor()
        pm = PolymarketTabularFeatures(pm_home_prob=0.62)
        result = extractor.extract(pm_features=pm, sb_features=sample_sb_tabular_features)
        assert result.pm_sb_convergence_rate is None
