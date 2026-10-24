"""Unit tests for feature extraction abstraction."""

from datetime import UTC, datetime

import numpy as np
import pytest
from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import (
    FeatureExtractor,
    TabularFeatureExtractor,
    calculate_sharp_retail_metrics,
)
from odds_analytics.training.config import FeatureConfig
from odds_core.models import EventStatus, Odds
from odds_core.odds_math import american_to_decimal, calculate_implied_probability


@pytest.fixture
def sample_event():
    """Create a sample BacktestEvent for testing."""
    return BacktestEvent(
        id="test_event_1",
        commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        home_score=110,
        away_score=105,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def sample_odds_snapshot(sample_event):
    """Create sample odds snapshot for testing."""
    timestamp = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)
    return [
        # Pinnacle (sharp book)
        Odds(
            id=1,
            event_id=sample_event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price_decimal=american_to_decimal(-120),
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=2,
            event_id=sample_event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price_decimal=american_to_decimal(+100),
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        # FanDuel (retail book)
        Odds(
            id=3,
            event_id=sample_event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price_decimal=american_to_decimal(-115),
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=4,
            event_id=sample_event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price_decimal=american_to_decimal(-105),
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        # DraftKings (retail book)
        Odds(
            id=5,
            event_id=sample_event.id,
            bookmaker_key="draftkings",
            bookmaker_title="DraftKings",
            market_key="h2h",
            outcome_name=sample_event.home_team,
            price_decimal=american_to_decimal(-118),
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
        Odds(
            id=6,
            event_id=sample_event.id,
            bookmaker_key="draftkings",
            bookmaker_title="DraftKings",
            market_key="h2h",
            outcome_name=sample_event.away_team,
            price_decimal=american_to_decimal(-102),
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        ),
    ]


class TestFeatureExtractorInterface:
    """Test the FeatureExtractor abstract base class interface."""

    def test_feature_extractor_is_abstract(self):
        """Test that FeatureExtractor cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            FeatureExtractor()

    def test_subclass_must_implement_extract_features(self):
        """Test that subclasses must implement extract_features."""

        class IncompleteExtractor(FeatureExtractor):
            def get_feature_names(self):
                return []

        with pytest.raises(TypeError, match="abstract"):
            IncompleteExtractor()

    def test_subclass_must_implement_get_feature_names(self):
        """Test that subclasses must implement get_feature_names."""

        class IncompleteExtractor(FeatureExtractor):
            def extract_features(self, event, odds_data, outcome=None, **kwargs):
                return {}

        with pytest.raises(TypeError, match="abstract"):
            IncompleteExtractor()

    def test_create_feature_vector_available_to_subclasses(self):
        """Test that base class provides create_feature_vector implementation."""

        class MinimalExtractor(FeatureExtractor):
            def extract_features(self, event, odds_data, outcome=None, **kwargs):
                return {"a": 1.0, "b": 2.0}

            def get_feature_names(self):
                return ["a", "b", "c"]

        extractor = MinimalExtractor()
        features = {"a": 1.0, "b": 2.0}
        vector = extractor.create_feature_vector(features)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        assert vector[0] == 1.0
        assert vector[1] == 2.0
        assert vector[2] == 0.0  # Missing feature filled with 0.0


class TestTabularFeatureExtractor:
    """Test TabularFeatureExtractor functionality."""

    def test_initialization(self):
        """Test that TabularFeatureExtractor initializes correctly."""
        extractor = TabularFeatureExtractor()

        assert extractor.sharp_bookmakers == ["pinnacle"]
        assert extractor.retail_bookmakers == ["fanduel", "draftkings", "betmgm"]

    def test_initialization_with_custom_bookmakers(self):
        """Test initialization with custom bookmaker lists."""
        extractor = TabularFeatureExtractor(
            sharp_bookmakers=["pinnacle", "circa"], retail_bookmakers=["fanduel"]
        )

        assert extractor.sharp_bookmakers == ["pinnacle", "circa"]
        assert extractor.retail_bookmakers == ["fanduel"]

    def test_extract_features_returns_tabular_features(self, sample_event, sample_odds_snapshot):
        """Test that extract_features returns a TabularFeatures instance."""
        from odds_analytics.feature_extraction import TabularFeatures

        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(features, TabularFeatures)
        # Check that we have at least some non-None features
        array = features.to_array()
        assert len(array) > 0

    def test_extract_features_includes_consensus_prob(self, sample_event, sample_odds_snapshot):
        """Test that consensus probability features are calculated."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert features.consensus_prob is not None
        assert 0 < features.consensus_prob < 1

    def test_extract_features_includes_sharp_features(self, sample_event, sample_odds_snapshot):
        """Test that sharp bookmaker features are extracted."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert features.sharp_prob is not None

    def test_extract_features_includes_retail_sharp_diff(self, sample_event, sample_odds_snapshot):
        """Test that retail vs sharp differences are calculated."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert features.retail_sharp_diff is not None

    def test_extract_features_includes_calendar_features(self, sample_event, sample_odds_snapshot):
        """Test that calendar features are derived from commence_time."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # 2024-11-01 is a Friday (weekday=4)
        assert features.day_of_week == 4.0
        assert features.is_weekend == 0.0

    def test_extract_features_weekend_detection(self):
        """Test that is_weekend correctly identifies Saturday and Sunday."""
        extractor = TabularFeatureExtractor()

        # Saturday game
        saturday_event = BacktestEvent(
            id="sat_event",
            commence_time=datetime(2024, 11, 2, 19, 0, 0, tzinfo=UTC),  # Saturday
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            home_score=110,
            away_score=105,
            status=EventStatus.FINAL,
        )
        features = extractor.extract_features(saturday_event, [], market="h2h")
        assert features.day_of_week == 5.0
        assert features.is_weekend == 1.0

        # Sunday game
        sunday_event = BacktestEvent(
            id="sun_event",
            commence_time=datetime(2024, 11, 3, 19, 0, 0, tzinfo=UTC),  # Sunday
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            home_score=110,
            away_score=105,
            status=EventStatus.FINAL,
        )
        features = extractor.extract_features(sunday_event, [], market="h2h")
        assert features.day_of_week == 6.0
        assert features.is_weekend == 1.0

    def test_calendar_friday_evening_stored_as_saturday_utc(self):
        """Friday 8pm ET (Sat 1am UTC) should yield day_of_week=4, is_weekend=0."""
        extractor = TabularFeatureExtractor()
        event = BacktestEvent(
            id="fri_utc_sat",
            commence_time=datetime(2025, 1, 4, 1, 0, 0, tzinfo=UTC),  # Sat 1am UTC = Fri 8pm ET
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            home_score=110,
            away_score=105,
            status=EventStatus.FINAL,
        )
        features = extractor.extract_features(event, [], market="h2h")
        assert features.day_of_week == 4.0
        assert features.is_weekend == 0.0

    def test_calendar_saturday_late_stored_as_sunday_utc(self):
        """Saturday 10pm ET (Sun 3am UTC) should yield day_of_week=5, is_weekend=1."""
        extractor = TabularFeatureExtractor()
        event = BacktestEvent(
            id="sat_utc_sun",
            commence_time=datetime(2025, 1, 5, 3, 0, 0, tzinfo=UTC),  # Sun 3am UTC = Sat 10pm ET
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            home_score=110,
            away_score=105,
            status=EventStatus.FINAL,
        )
        features = extractor.extract_features(event, [], market="h2h")
        assert features.day_of_week == 5.0
        assert features.is_weekend == 1.0

    def test_calendar_sunday_afternoon_no_rollover(self):
        """Sunday 3pm ET (Sun 8pm UTC) — no rollover, sanity check."""
        extractor = TabularFeatureExtractor()
        event = BacktestEvent(
            id="sun_no_roll",
            commence_time=datetime(2025, 1, 5, 20, 0, 0, tzinfo=UTC),  # Sun 8pm UTC = Sun 3pm ET
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            home_score=110,
            away_score=105,
            status=EventStatus.FINAL,
        )
        features = extractor.extract_features(event, [], market="h2h")
        assert features.day_of_week == 6.0
        assert features.is_weekend == 1.0

    def test_extract_features_empty_odds(self, sample_event):
        """Test that extract_features handles empty odds gracefully."""
        from odds_analytics.feature_extraction import TabularFeatures

        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, [], market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(features, TabularFeatures)
        # Odds-derived fields should be None with empty odds
        assert features.consensus_prob is None
        assert features.sharp_prob is None
        # Calendar features are always populated from commence_time
        assert features.is_weekend is not None
        assert features.day_of_week is not None

    def test_get_feature_names_returns_list(self):
        """Test that get_feature_names returns a list of strings."""
        extractor = TabularFeatureExtractor()
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert all(isinstance(name, str) for name in feature_names)

    def test_create_feature_vector_correct_order(self, sample_event, sample_odds_snapshot):
        """Test that feature vector maintains correct order."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # TabularFeatures.to_array() is called automatically
        vector = extractor.create_feature_vector(features)

        assert isinstance(vector, np.ndarray)
        # Should have all features in the defined order
        assert len(vector) == len(features.get_feature_names())

    def test_create_feature_vector_fills_missing_with_nan(self):
        """Test that missing features (None) are converted to np.nan."""
        from odds_analytics.feature_extraction import TabularFeatures

        # Create features with some None values
        features = TabularFeatures(
            consensus_prob=0.55,
            # Other fields default to None
        )

        vector = features.to_array()
        names = TabularFeatures.get_feature_names()

        # Set field should have value
        assert vector[names.index("consensus_prob")] == 0.55

        # Optional fields that are None should be NaN
        assert np.isnan(vector[names.index("sharp_prob")])

    def test_create_feature_vector_uses_get_feature_names_by_default(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that create_feature_vector uses get_feature_names() when feature_names=None."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Call without feature_names argument
        vector = extractor.create_feature_vector(features)

        # Should use all feature names from get_feature_names()
        expected_length = len(extractor.get_feature_names())
        assert len(vector) == expected_length

    def test_feature_extraction_consistency(self, sample_event, sample_odds_snapshot):
        """Test that feature extraction produces consistent results."""
        extractor = TabularFeatureExtractor()

        # Extract features twice
        features1 = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        features2 = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Should be identical (dataclass equality)
        assert features1 == features2

        # Convert to vectors
        vec1 = extractor.create_feature_vector(features1)
        vec2 = extractor.create_feature_vector(features2)

        # Use allclose with equal_nan=True to handle NaN values
        assert np.allclose(vec1, vec2, equal_nan=True)


class TestFeatureExtractorIntegration:
    """Integration tests for feature extractors."""

    def test_tabular_extractor_produces_valid_numpy_arrays(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that TabularFeatureExtractor produces valid numpy arrays for ML models."""
        extractor = TabularFeatureExtractor()

        # Extract features
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Convert to vector
        feature_names = extractor.get_feature_names()
        vector = extractor.create_feature_vector(features, feature_names)

        # Should be valid numpy array with no NaN or Inf
        assert isinstance(vector, np.ndarray)
        assert vector.dtype in [np.float64, np.float32]
        assert np.all(np.isfinite(vector))

    def test_custom_sharp_bookmakers_affects_features(self, sample_event, sample_odds_snapshot):
        """Test that custom sharp bookmakers affect feature extraction."""
        # Default extractor uses Pinnacle
        default_extractor = TabularFeatureExtractor()
        default_features = default_extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Custom extractor with FanDuel as sharp (unusual, but tests customization)
        custom_extractor = TabularFeatureExtractor(sharp_bookmakers=["fanduel"])
        custom_features = custom_extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Sharp probabilities should differ (different bookmaker as baseline)
        if default_features.sharp_prob is not None and custom_features.sharp_prob is not None:
            assert default_features.sharp_prob != custom_features.sharp_prob

    def test_different_extractors_can_be_used_interchangeably(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that different extractors can be used through the same interface."""
        from odds_analytics.feature_extraction import TabularFeatures

        extractors = [
            TabularFeatureExtractor(),
            TabularFeatureExtractor(sharp_bookmakers=["pinnacle"], retail_bookmakers=["fanduel"]),
        ]

        for extractor in extractors:
            # All should implement the same interface
            features = extractor.extract_features(
                sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
            )
            assert isinstance(features, TabularFeatures)

            feature_names = extractor.get_feature_names()
            assert isinstance(feature_names, list)

            vector = extractor.create_feature_vector(features)
            assert isinstance(vector, np.ndarray)


class TestFeatureExtractorFromConfig:
    """Test factory methods for creating extractors from FeatureConfig."""

    def test_tabular_extractor_from_config_default_values(self):
        """Test that TabularFeatureExtractor.from_config uses config defaults correctly."""
        config = FeatureConfig(
            target_type="raw",
        )
        extractor = TabularFeatureExtractor.from_config(config)

        # Should match config defaults
        assert extractor.sharp_bookmakers == ["pinnacle"]
        assert extractor.retail_bookmakers == ["fanduel", "draftkings", "betmgm"]

    def test_tabular_extractor_from_config_custom_values(self):
        """Test that TabularFeatureExtractor.from_config respects custom config values."""
        config = FeatureConfig(
            target_type="raw",
            sharp_bookmakers=["pinnacle", "circa"],
            retail_bookmakers=["fanduel", "betmgm"],
        )
        extractor = TabularFeatureExtractor.from_config(config)

        assert extractor.sharp_bookmakers == ["pinnacle", "circa"]
        assert extractor.retail_bookmakers == ["fanduel", "betmgm"]

    def test_tabular_extractor_from_config_produces_valid_features(
        self, sample_event, sample_odds_snapshot
    ):
        """Test complete pipeline: config → extractor → features for tabular."""
        config = FeatureConfig(
            target_type="raw",
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
        )
        extractor = TabularFeatureExtractor.from_config(config)

        # Extract features using extractor created from config
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Should produce valid features
        assert features.consensus_prob is not None
        assert features.sharp_prob is not None

        # Convert to array
        vector = extractor.create_feature_vector(features)
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(extractor.get_feature_names())

    def test_config_defaults_match_extractor_defaults(self):
        """Test that FeatureConfig defaults align with extractor class defaults."""
        config = FeatureConfig(
            target_type="raw",
        )

        # Tabular: FeatureConfig defaults match TabularFeatureExtractor defaults
        tabular_from_config = TabularFeatureExtractor.from_config(config)
        tabular_default = TabularFeatureExtractor()

        assert tabular_from_config.sharp_bookmakers == tabular_default.sharp_bookmakers
        assert tabular_from_config.retail_bookmakers == tabular_default.retail_bookmakers

    def test_feature_config_normalize_field_exists(self):
        """Test that FeatureConfig has normalize field."""
        config = FeatureConfig(
            target_type="raw",
        )
        assert hasattr(config, "normalize")
        assert config.normalize is False  # Default value

        config_with_normalize = FeatureConfig(target_type="raw", normalize=True)
        assert config_with_normalize.normalize is True

    def test_feature_config_tier_parameters(self):
        """Test that FeatureConfig tier parameters work correctly."""
        from odds_analytics.training import SamplingConfig
        from odds_lambda.fetch_tier import FetchTier

        config = FeatureConfig(
            target_type="raw",
            closing_tier=FetchTier.CLOSING,
            sampling=SamplingConfig(strategy="tier", decision_tier=FetchTier.PREGAME),
        )

        assert config.closing_tier == FetchTier.CLOSING
        assert config.sampling.decision_tier == FetchTier.PREGAME

    def test_feature_config_invalid_hours_range_raises_error(self):
        """Test that invalid time range in SamplingConfig raises validation error."""
        from odds_analytics.training import SamplingConfig

        with pytest.raises(ValueError, match="min_hours.*must be less than max_hours"):
            SamplingConfig(strategy="time_range", min_hours=12.0, max_hours=3.0)

    def test_tabular_extractor_from_config_type_annotation(self):
        """Test that from_config has correct return type."""
        config = FeatureConfig(
            target_type="raw",
        )
        extractor = TabularFeatureExtractor.from_config(config)
        assert isinstance(extractor, TabularFeatureExtractor)


def _make_odds(
    bookmaker_key: str,
    outcome_name: str,
    price: int,
    *,
    event_id: str = "test_event",
    market_key: str = "h2h",
) -> Odds:
    ts = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)
    return Odds(
        id=0,
        event_id=event_id,
        bookmaker_key=bookmaker_key,
        bookmaker_title=bookmaker_key.title(),
        market_key=market_key,
        outcome_name=outcome_name,
        price_decimal=american_to_decimal(price),
        point=None,
        odds_timestamp=ts,
        last_update=ts,
    )


class TestSharpBookmakerPriorityFallback:
    """Test priority-ordered fallback for sharp bookmaker selection."""

    def _odds_with_bookmakers(self, bookmaker_keys: list[str]) -> list[Odds]:
        """Build h2h odds for given bookmakers with distinct prices."""
        prices = {"pinnacle": -120, "betfair_exchange": -125, "bet365": -115}
        odds = []
        for bm in bookmaker_keys:
            odds.append(_make_odds(bm, "Home", prices.get(bm, -110)))
            odds.append(_make_odds(bm, "Away", prices.get(bm, -110) + 10))
        return odds

    def test_prefers_first_bookmaker_when_both_present(self) -> None:
        odds = self._odds_with_bookmakers(["pinnacle", "betfair_exchange", "bet365"])
        result = calculate_sharp_retail_metrics(
            odds,
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["bet365"],
            outcome="Home",
        )
        # sharp_prob should come from pinnacle (-120), not BFE (-125)
        pinnacle_only = calculate_sharp_retail_metrics(
            odds, sharp_bookmakers=["pinnacle"], retail_bookmakers=["bet365"], outcome="Home"
        )
        assert result["sharp_prob"] == pinnacle_only["sharp_prob"]

    def test_falls_back_to_second_bookmaker(self) -> None:
        odds = self._odds_with_bookmakers(["betfair_exchange", "bet365"])
        result = calculate_sharp_retail_metrics(
            odds,
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["bet365"],
            outcome="Home",
        )
        bfe_only = calculate_sharp_retail_metrics(
            odds,
            sharp_bookmakers=["betfair_exchange"],
            retail_bookmakers=["bet365"],
            outcome="Home",
        )
        assert result["sharp_prob"] == bfe_only["sharp_prob"]
        assert result["sharp_prob"] is not None

    def test_returns_none_when_no_sharp_present(self) -> None:
        odds = self._odds_with_bookmakers(["bet365"])
        result = calculate_sharp_retail_metrics(
            odds,
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["bet365"],
            outcome="Home",
        )
        assert result["sharp_prob"] is None
        assert result["diff"] is None

    def test_single_bookmaker_list_unchanged(self) -> None:
        odds = self._odds_with_bookmakers(["pinnacle", "bet365"])
        result = calculate_sharp_retail_metrics(
            odds, sharp_bookmakers=["pinnacle"], retail_bookmakers=["bet365"], outcome="Home"
        )
        assert result["sharp_prob"] is not None


class TestTabularExtractorPriorityFallback:
    """Test priority-ordered fallback in TabularFeatureExtractor."""

    @pytest.fixture
    def event(self) -> BacktestEvent:
        return BacktestEvent(
            id="fallback_test",
            commence_time=datetime(2025, 1, 15, 15, 0, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            home_score=2,
            away_score=1,
            status=EventStatus.FINAL,
        )

    def _odds_for(self, bookmaker_keys: list[str], event: BacktestEvent) -> list[Odds]:
        prices = {"pinnacle": -120, "betfair_exchange": -125, "bet365": -115}
        odds = []
        for bm in bookmaker_keys:
            p = prices.get(bm, -110)
            odds.append(_make_odds(bm, event.home_team, p, event_id=event.id))
            odds.append(_make_odds(bm, event.away_team, p + 10, event_id=event.id))
        return odds

    def test_uses_primary_sharp_when_available(self, event: BacktestEvent) -> None:
        odds = self._odds_for(["pinnacle", "betfair_exchange", "bet365"], event)
        extractor = TabularFeatureExtractor(
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["bet365"],
        )
        features = extractor.extract_features(event, odds, market="h2h", outcome=event.home_team)

        pinnacle_extractor = TabularFeatureExtractor(
            sharp_bookmakers=["pinnacle"], retail_bookmakers=["bet365"]
        )
        pinnacle_features = pinnacle_extractor.extract_features(
            event, odds, market="h2h", outcome=event.home_team
        )
        assert features.sharp_prob == pinnacle_features.sharp_prob

    def test_falls_back_when_primary_absent(self, event: BacktestEvent) -> None:
        odds = self._odds_for(["betfair_exchange", "bet365"], event)
        extractor = TabularFeatureExtractor(
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["bet365"],
        )
        features = extractor.extract_features(event, odds, market="h2h", outcome=event.home_team)

        assert features.sharp_prob is not None
        assert features.retail_sharp_diff is not None

    def test_no_sharp_yields_none(self, event: BacktestEvent) -> None:
        odds = self._odds_for(["bet365"], event)
        extractor = TabularFeatureExtractor(
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["bet365"],
        )
        features = extractor.extract_features(event, odds, market="h2h", outcome=event.home_team)

        assert features.sharp_prob is None
        assert features.retail_sharp_diff is None


class TestTabularExtractorOutcomeAwareSharpFallthrough:
    """The sharp book is the first in priority order that quotes the requested outcome."""

    @pytest.fixture
    def event(self) -> BacktestEvent:
        return BacktestEvent(
            id="fallthrough_test",
            commence_time=datetime(2025, 1, 15, 15, 0, 0, tzinfo=UTC),
            home_team="A",
            away_team="B",
            home_score=2,
            away_score=1,
            status=EventStatus.FINAL,
        )

    def test_skips_sharp_book_missing_requested_outcome(self, event: BacktestEvent) -> None:
        odds = [
            _make_odds("pinnacle", "B", -150, event_id=event.id),
            _make_odds("betfair_exchange", "A", 130, event_id=event.id),
            _make_odds("betfair_exchange", "B", -140, event_id=event.id),
            _make_odds("fanduel", "A", 120, event_id=event.id),
        ]
        extractor = TabularFeatureExtractor(
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["fanduel"],
        )
        features = extractor.extract_features(event, odds, market="h2h", outcome="A")

        expected_sharp = calculate_implied_probability(130)
        assert features.sharp_prob == pytest.approx(expected_sharp)
        assert features.retail_sharp_diff == pytest.approx(
            calculate_implied_probability(120) - expected_sharp
        )

    def test_none_when_no_sharp_book_quotes_requested_outcome(self, event: BacktestEvent) -> None:
        odds = [
            _make_odds("pinnacle", "B", -150, event_id=event.id),
            _make_odds("betfair_exchange", "B", -140, event_id=event.id),
            _make_odds("fanduel", "A", 120, event_id=event.id),
        ]
        extractor = TabularFeatureExtractor(
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            retail_bookmakers=["fanduel"],
        )
        features = extractor.extract_features(event, odds, market="h2h", outcome="A")

        assert features.sharp_prob is None
        assert features.retail_sharp_diff is None
