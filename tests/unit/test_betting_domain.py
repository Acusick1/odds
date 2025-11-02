"""Unit tests for betting domain implementations."""

from datetime import datetime

import pytest

from analytics.betting import BettingEvent, OddsObservation, RuleBasedStrategy
from analytics.core import TabularFeatureSet
from core.models import EventStatus


class MockRuleBasedStrategy(RuleBasedStrategy):
    """Mock rule-based strategy for testing."""

    async def evaluate_opportunity(self, event, observations, config):
        # Simple mock: return empty list
        return []


class TestBettingEvent:
    """Tests for BettingEvent class."""

    def test_creation(self):
        """Test creating a betting event."""
        commence_time = datetime(2024, 1, 15, 19, 0)
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=commence_time,
            home_score=105,
            away_score=100,
            status=EventStatus.FINAL,
        )

        assert event.id == "game1"
        assert event.home_competitor == "Lakers"
        assert event.away_competitor == "Celtics"
        assert event.commence_time == commence_time
        assert event.event_time == commence_time
        assert event.home_score == 105
        assert event.away_score == 100

    def test_outcome_determination_home_win(self):
        """Test that home win outcome is determined correctly."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
            home_score=105,
            away_score=100,
        )

        assert event.get_outcome() == "home_win"
        assert event.get_winner() == "Lakers"
        assert event.get_margin() == 5

    def test_outcome_determination_away_win(self):
        """Test that away win outcome is determined correctly."""
        event = BettingEvent(
            event_id="game2",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
            home_score=95,
            away_score=100,
        )

        assert event.get_outcome() == "away_win"
        assert event.get_winner() == "Celtics"
        assert event.get_margin() == 5

    def test_outcome_determination_draw(self):
        """Test that draw outcome is determined correctly."""
        event = BettingEvent(
            event_id="game3",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
            home_score=100,
            away_score=100,
        )

        assert event.get_outcome() == "draw"
        assert event.get_winner() is None
        assert event.get_margin() == 0

    def test_outcome_none_without_scores(self):
        """Test that outcome is None when scores not available."""
        event = BettingEvent(
            event_id="game4",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
        )

        assert event.get_outcome() is None
        assert event.get_winner() is None
        assert event.get_margin() is None

    def test_has_result(self):
        """Test has_result method."""
        # Event with scores and FINAL status
        event_final = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
            home_score=105,
            away_score=100,
            status=EventStatus.FINAL,
        )
        assert event_final.has_result() is True

        # Event with scores but not FINAL
        event_scheduled = BettingEvent(
            event_id="game2",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
            home_score=105,
            away_score=100,
            status=EventStatus.SCHEDULED,
        )
        assert event_scheduled.has_result() is False

        # Event without scores
        event_no_scores = BettingEvent(
            event_id="game3",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
            status=EventStatus.FINAL,
        )
        assert event_no_scores.has_result() is False

    def test_get_problem_type(self):
        """Test that betting event is a discrete event problem."""
        event = BettingEvent(
            event_id="game1",
            home_competitor="Lakers",
            away_competitor="Celtics",
            commence_time=datetime.now(),
        )

        assert event.get_problem_type() == "discrete_event"


class TestOddsObservation:
    """Tests for OddsObservation class."""

    def test_creation(self):
        """Test creating an odds observation."""
        obs_time = datetime(2024, 1, 15, 18, 0)
        obs = OddsObservation(
            event_id="game1",
            observation_time=obs_time,
            bookmaker="fanduel",
            market="h2h",
            outcome="Lakers",
            odds=-110,
        )

        assert obs.problem_id == "game1"
        assert obs.observation_time == obs_time
        assert obs.bookmaker == "fanduel"
        assert obs.market == "h2h"
        assert obs.outcome == "Lakers"
        assert obs.odds == -110

    def test_get_data(self):
        """Test get_data method returns correct dictionary."""
        obs_time = datetime(2024, 1, 15, 18, 0)
        obs = OddsObservation(
            event_id="game1",
            observation_time=obs_time,
            bookmaker="fanduel",
            market="spreads",
            outcome="Lakers",
            odds=-110,
            line=-5.5,
        )

        data = obs.get_data()
        assert data["bookmaker"] == "fanduel"
        assert data["market"] == "spreads"
        assert data["outcome"] == "Lakers"
        assert data["odds"] == -110
        assert data["line"] == -5.5

    def test_market_type_checks(self):
        """Test market type helper methods."""
        h2h_obs = OddsObservation(
            event_id="game1",
            observation_time=datetime.now(),
            bookmaker="fanduel",
            market="h2h",
            outcome="Lakers",
            odds=-110,
        )
        assert h2h_obs.is_moneyline() is True
        assert h2h_obs.is_spread() is False
        assert h2h_obs.is_total() is False

        spread_obs = OddsObservation(
            event_id="game1",
            observation_time=datetime.now(),
            bookmaker="fanduel",
            market="spreads",
            outcome="Lakers",
            odds=-110,
            line=-5.5,
        )
        assert spread_obs.is_moneyline() is False
        assert spread_obs.is_spread() is True
        assert spread_obs.is_total() is False

        total_obs = OddsObservation(
            event_id="game1",
            observation_time=datetime.now(),
            bookmaker="fanduel",
            market="totals",
            outcome="Over",
            odds=-110,
            line=220.5,
        )
        assert total_obs.is_moneyline() is False
        assert total_obs.is_spread() is False
        assert total_obs.is_total() is True

    def test_implied_probability_negative_odds(self):
        """Test implied probability calculation for negative odds."""
        obs = OddsObservation(
            event_id="game1",
            observation_time=datetime.now(),
            bookmaker="fanduel",
            market="h2h",
            outcome="Lakers",
            odds=-110,
        )

        prob = obs.get_implied_probability()
        # -110 odds -> 110 / (110 + 100) = 0.5238
        assert pytest.approx(prob, rel=0.01) == 0.5238

    def test_implied_probability_positive_odds(self):
        """Test implied probability calculation for positive odds."""
        obs = OddsObservation(
            event_id="game1",
            observation_time=datetime.now(),
            bookmaker="fanduel",
            market="h2h",
            outcome="Lakers",
            odds=150,
        )

        prob = obs.get_implied_probability()
        # +150 odds -> 100 / (150 + 100) = 0.4
        assert pytest.approx(prob, rel=0.01) == 0.4

    def test_is_before_methods(self):
        """Test that observation inherits time comparison methods."""
        obs_time = datetime(2024, 1, 15, 18, 0)
        obs = OddsObservation(
            event_id="game1",
            observation_time=obs_time,
            bookmaker="fanduel",
            market="h2h",
            outcome="Lakers",
            odds=-110,
        )

        later_time = datetime(2024, 1, 15, 19, 0)
        assert obs.is_before(later_time) is True
        assert obs.is_at_or_before(later_time) is True
        assert obs.is_at_or_before(obs_time) is True


class TestOddsSnapshot:
    """Tests for OddsSnapshot class."""

    def test_creation(self):
        """Test creating an odds snapshot."""
        from analytics.betting.observations import OddsSnapshot

        snapshot_time = datetime(2024, 1, 15, 18, 0)
        obs1 = OddsObservation(
            event_id="game1",
            observation_time=snapshot_time,
            bookmaker="fanduel",
            market="h2h",
            outcome="Lakers",
            odds=-110,
        )
        obs2 = OddsObservation(
            event_id="game1",
            observation_time=snapshot_time,
            bookmaker="draftkings",
            market="h2h",
            outcome="Lakers",
            odds=-105,
        )

        snapshot = OddsSnapshot("game1", snapshot_time, [obs1, obs2])

        assert snapshot.event_id == "game1"
        assert snapshot.snapshot_time == snapshot_time
        assert len(snapshot.observations) == 2

    def test_get_observations_for_market(self):
        """Test filtering observations by market."""
        from analytics.betting.observations import OddsSnapshot

        snapshot_time = datetime.now()
        obs1 = OddsObservation("game1", snapshot_time, "fanduel", "h2h", "Lakers", -110)
        obs2 = OddsObservation("game1", snapshot_time, "fanduel", "spreads", "Lakers", -110, -5.5)

        snapshot = OddsSnapshot("game1", snapshot_time, [obs1, obs2])
        h2h_obs = snapshot.get_observations_for_market("h2h")

        assert len(h2h_obs) == 1
        assert h2h_obs[0].market == "h2h"

    def test_get_observations_for_bookmaker(self):
        """Test filtering observations by bookmaker."""
        from analytics.betting.observations import OddsSnapshot

        snapshot_time = datetime.now()
        obs1 = OddsObservation("game1", snapshot_time, "fanduel", "h2h", "Lakers", -110)
        obs2 = OddsObservation("game1", snapshot_time, "draftkings", "h2h", "Lakers", -105)

        snapshot = OddsSnapshot("game1", snapshot_time, [obs1, obs2])
        fanduel_obs = snapshot.get_observations_for_bookmaker("fanduel")

        assert len(fanduel_obs) == 1
        assert fanduel_obs[0].bookmaker == "fanduel"

    def test_get_bookmakers(self):
        """Test getting set of bookmakers."""
        from analytics.betting.observations import OddsSnapshot

        snapshot_time = datetime.now()
        obs1 = OddsObservation("game1", snapshot_time, "fanduel", "h2h", "Lakers", -110)
        obs2 = OddsObservation("game1", snapshot_time, "draftkings", "h2h", "Lakers", -105)
        obs3 = OddsObservation("game1", snapshot_time, "fanduel", "spreads", "Lakers", -110)

        snapshot = OddsSnapshot("game1", snapshot_time, [obs1, obs2, obs3])
        bookmakers = snapshot.get_bookmakers()

        assert len(bookmakers) == 2
        assert "fanduel" in bookmakers
        assert "draftkings" in bookmakers

    def test_get_markets(self):
        """Test getting set of markets."""
        from analytics.betting.observations import OddsSnapshot

        snapshot_time = datetime.now()
        obs1 = OddsObservation("game1", snapshot_time, "fanduel", "h2h", "Lakers", -110)
        obs2 = OddsObservation("game1", snapshot_time, "fanduel", "spreads", "Lakers", -110)

        snapshot = OddsSnapshot("game1", snapshot_time, [obs1, obs2])
        markets = snapshot.get_markets()

        assert len(markets) == 2
        assert "h2h" in markets
        assert "spreads" in markets


class TestBettingStrategy:
    """Tests for BettingStrategy base class."""

    def test_rule_based_strategy_creation(self):
        """Test creating a rule-based strategy."""
        strategy = MockRuleBasedStrategy("TestStrategy", param1="value1")

        assert strategy.get_name() == "TestStrategy"
        assert strategy.get_params() == {"param1": "value1"}
        assert strategy.has_model() is False
        assert strategy.has_feature_extractor() is False
        assert strategy.is_ml_strategy() is False

    def test_strategy_with_feature_extractor(self):
        """Test strategy with feature extractor but no model."""
        from analytics.core import FeatureExtractor

        class MockExtractor(FeatureExtractor):
            def extract(self, problem, observations, decision_time):
                return TabularFeatureSet({"feat": 1.0})

        extractor = MockExtractor()
        strategy = MockRuleBasedStrategy("Test")
        strategy.feature_extractor = extractor

        assert strategy.has_feature_extractor() is True
        assert strategy.has_model() is False
        assert strategy.is_ml_strategy() is False

    def test_strategy_with_model(self):
        """Test strategy with both feature extractor and model."""
        from analytics.core import FeatureExtractor, ModelPredictor, Prediction

        class MockExtractor(FeatureExtractor):
            def extract(self, problem, observations, decision_time):
                return TabularFeatureSet({"feat": 1.0})

        class MockModel(ModelPredictor):
            def predict(self, features):
                return Prediction(predictions={"a": 0.5}, confidence=0.5)

            def get_required_feature_type(self):
                return "tabular"

        extractor = MockExtractor()
        model = MockModel()
        strategy = MockRuleBasedStrategy("Test")
        strategy.feature_extractor = extractor
        strategy.model_predictor = model

        assert strategy.has_feature_extractor() is True
        assert strategy.has_model() is True
        assert strategy.is_ml_strategy() is True
