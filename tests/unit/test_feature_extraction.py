"""Unit tests for feature extraction abstraction."""

from datetime import UTC, datetime

import numpy as np
import pytest
from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import (
    FeatureExtractor,
    SequenceFeatureExtractor,
    TabularFeatureExtractor,
)
from odds_analytics.training.config import FeatureConfig
from odds_core.models import EventStatus, Odds


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
            price=-120,
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
            price=+100,
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
            price=-115,
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
            price=-105,
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
            price=-118,
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
            price=-102,
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
        assert features.opponent_consensus_prob is not None

    def test_extract_features_includes_sharp_features(self, sample_event, sample_odds_snapshot):
        """Test that sharp bookmaker features are extracted."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert features.sharp_prob is not None
        assert features.sharp_market_hold is not None
        assert features.sharp_market_hold > 0  # Should have some vig

    def test_extract_features_includes_retail_sharp_diff(self, sample_event, sample_odds_snapshot):
        """Test that retail vs sharp differences are calculated."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Should have retail-sharp difference features
        assert (
            features.retail_sharp_diff_home is not None
            or features.retail_sharp_diff_away is not None
        )

    def test_extract_features_includes_best_odds(self, sample_event, sample_odds_snapshot):
        """Test that best available odds are found."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert features.best_available_odds is not None
        assert features.best_available_decimal is not None
        assert features.best_available_decimal > 1.0

    def test_extract_features_team_indicators(self, sample_event, sample_odds_snapshot):
        """Test that team indicator features are set correctly."""
        extractor = TabularFeatureExtractor()

        home_features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert home_features.is_home_team == 1.0
        assert home_features.is_away_team == 0.0

        away_features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.away_team
        )

        assert away_features.is_home_team == 0.0
        assert away_features.is_away_team == 1.0

    def test_extract_features_empty_odds(self, sample_event):
        """Test that extract_features handles empty odds gracefully."""
        from odds_analytics.feature_extraction import TabularFeatures

        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, [], market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(features, TabularFeatures)
        # Should have is_home_team and is_away_team set
        assert features.is_home_team == 1.0
        assert features.is_away_team == 0.0

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
            is_home_team=1.0,
            is_away_team=0.0,
            consensus_prob=0.55,
            # Other fields default to None
        )

        vector = features.to_array()

        # Required fields should have values
        assert vector[0] == 1.0  # is_home_team
        assert vector[1] == 0.0  # is_away_team

        # Optional fields that are None should be NaN
        assert np.isnan(vector[2])  # avg_home_odds (None)

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


class TestSequenceFeatureExtractor:
    """Test SequenceFeatureExtractor functionality."""

    def test_initialization(self):
        """Test that SequenceFeatureExtractor initializes with default params."""
        extractor = SequenceFeatureExtractor()

        assert extractor.lookback_hours == 72
        assert extractor.timesteps == 24
        assert extractor.sharp_bookmakers == ["pinnacle"]
        assert extractor.retail_bookmakers == ["fanduel", "draftkings", "betmgm"]

    def test_initialization_with_custom_params(self):
        """Test initialization with custom parameters."""
        extractor = SequenceFeatureExtractor(
            lookback_hours=48,
            timesteps=16,
            sharp_bookmakers=["pinnacle", "circa"],
            retail_bookmakers=["fanduel"],
        )

        assert extractor.lookback_hours == 48
        assert extractor.timesteps == 16
        assert extractor.sharp_bookmakers == ["pinnacle", "circa"]
        assert extractor.retail_bookmakers == ["fanduel"]

    def test_get_feature_names_returns_list(self):
        """Test that get_feature_names returns a list of feature names."""
        extractor = SequenceFeatureExtractor()
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) == 15  # Expected number of features per timestep
        assert "american_odds" in feature_names
        assert "implied_prob" in feature_names
        assert "odds_change_from_prev" in feature_names
        assert "hours_to_game" in feature_names

    def test_extract_features_returns_dict_with_sequence_and_mask(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that extract_features returns dictionary with sequence and mask."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        # Create sequence of snapshots at different times
        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        odds_sequence = []

        for i in range(4):
            snapshot_time = base_time + np.timedelta64(i * 4, "h")
            snapshot = [
                Odds(
                    id=100 + i,
                    event_id=sample_event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-120 - i * 2,
                    point=None,
                    odds_timestamp=snapshot_time,
                    last_update=snapshot_time,
                )
            ]
            odds_sequence.append(snapshot)

        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(result, dict)
        assert "sequence" in result
        assert "mask" in result

    def test_extract_features_sequence_shape(self, sample_event, sample_odds_snapshot):
        """Test that sequence has correct shape."""
        extractor = SequenceFeatureExtractor(lookback_hours=48, timesteps=16)

        # Create simple sequence
        base_time = sample_event.commence_time - np.timedelta64(24, "h")
        odds_sequence = []

        for i in range(6):
            snapshot_time = base_time + np.timedelta64(i * 4, "h")
            snapshot = [
                Odds(
                    id=200 + i,
                    event_id=sample_event.id,
                    bookmaker_key="fanduel",
                    bookmaker_title="FanDuel",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-115,
                    point=None,
                    odds_timestamp=snapshot_time,
                    last_update=snapshot_time,
                )
            ]
            odds_sequence.append(snapshot)

        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        sequence = result["sequence"]
        mask = result["mask"]

        # Check shapes
        assert sequence.shape == (16, 15)  # (timesteps, num_features)
        assert mask.shape == (16,)
        assert isinstance(sequence, np.ndarray)
        assert isinstance(mask, np.ndarray)

    def test_extract_features_mask_indicates_valid_data(self, sample_event):
        """Test that mask correctly indicates which timesteps have valid data."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        # Create sparse sequence (only 3 snapshots)
        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        odds_sequence = []

        for i in range(3):
            snapshot_time = base_time + np.timedelta64(i * 4, "h")
            snapshot = [
                Odds(
                    id=300 + i,
                    event_id=sample_event.id,
                    bookmaker_key="draftkings",
                    bookmaker_title="DraftKings",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-110,
                    point=None,
                    odds_timestamp=snapshot_time,
                    last_update=snapshot_time,
                )
            ]
            odds_sequence.append(snapshot)

        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        mask = result["mask"]

        # Should have some True values (where data exists)
        assert mask.any(), "Mask should have at least some True values"
        # Should have some False values (sparse data)
        assert not mask.all(), "Mask should not be all True for sparse data"

    def test_extract_features_empty_odds_returns_zeros(self, sample_event):
        """Test that empty odds returns zero sequence with all False mask."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        result = extractor.extract_features(
            sample_event, [], market="h2h", outcome=sample_event.home_team
        )

        sequence = result["sequence"]
        mask = result["mask"]

        assert sequence.shape == (8, 15)
        assert np.all(sequence == 0)
        assert np.all(~mask)

    def test_extract_features_includes_line_movement(self, sample_event):
        """Test that line movement features are calculated correctly."""
        extractor = SequenceFeatureExtractor(lookback_hours=12, timesteps=4)

        # Create sequence with changing odds
        base_time = sample_event.commence_time - np.timedelta64(6, "h")
        odds_sequence = []

        for i, price in enumerate([-120, -118, -115, -112]):  # Line moving toward home team
            snapshot_time = base_time + np.timedelta64(i * 2, "h")
            snapshot = [
                Odds(
                    id=400 + i,
                    event_id=sample_event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=price,
                    point=None,
                    odds_timestamp=snapshot_time,
                    last_update=snapshot_time,
                )
            ]
            odds_sequence.append(snapshot)

        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        sequence = result["sequence"]

        # Check that odds_change_from_opening is tracked
        feature_idx = extractor.get_feature_names().index("odds_change_from_opening")
        changes = sequence[:, feature_idx]

        # With valid mask positions, opening change should be 0 initially
        # and increase as line moves
        assert changes[0] == 0.0 or not result["mask"][0]  # Opening or no data

    def test_extract_features_includes_time_encoding(self, sample_event):
        """Test that time features are included."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        odds_sequence = [
            [
                Odds(
                    id=500,
                    event_id=sample_event.id,
                    bookmaker_key="fanduel",
                    bookmaker_title="FanDuel",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-110,
                    point=None,
                    odds_timestamp=base_time,
                    last_update=base_time,
                )
            ]
        ]

        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        feature_names = extractor.get_feature_names()

        # Check time features exist in feature names
        assert "hours_to_game" in feature_names
        assert "time_of_day_sin" in feature_names
        assert "time_of_day_cos" in feature_names

        # Verify time encoding is correctly calculated
        sequence = result["sequence"]
        time_sin_idx = feature_names.index("time_of_day_sin")
        time_cos_idx = feature_names.index("time_of_day_cos")

        # Test should produce valid data - fail if mask is all False
        valid_mask = result["mask"]
        assert valid_mask.any(), "Expected valid data in mask, but all values are False"

        # Cyclical encoding should be in valid range [-1, 1] (ignoring NaN values)
        time_sin_values = sequence[valid_mask, time_sin_idx]
        time_cos_values = sequence[valid_mask, time_cos_idx]
        assert np.all(np.abs(time_sin_values[~np.isnan(time_sin_values)]) <= 1)
        assert np.all(np.abs(time_cos_values[~np.isnan(time_cos_values)]) <= 1)

    def test_extract_features_sharp_vs_retail(self, sample_event):
        """Test that sharp vs retail differential is calculated."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=4)

        base_time = sample_event.commence_time - np.timedelta64(12, "h")

        # Create snapshot with both sharp and retail odds
        snapshot = [
            # Pinnacle (sharp)
            Odds(
                id=600,
                event_id=sample_event.id,
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name=sample_event.home_team,
                price=-120,
                point=None,
                odds_timestamp=base_time,
                last_update=base_time,
            ),
            # FanDuel (retail)
            Odds(
                id=601,
                event_id=sample_event.id,
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name=sample_event.home_team,
                price=-115,
                point=None,
                odds_timestamp=base_time,
                last_update=base_time,
            ),
        ]

        odds_sequence = [snapshot]

        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        feature_names = extractor.get_feature_names()

        # Verify sharp vs retail features exist
        assert "sharp_prob" in feature_names
        assert "retail_sharp_diff" in feature_names

        # Since we have both sharp (Pinnacle -120) and retail (FanDuel -115) books,
        # the differential should be calculated and non-zero
        sequence = result["sequence"]
        retail_sharp_diff_idx = feature_names.index("retail_sharp_diff")

        # Test should produce valid data - fail if mask is all False
        valid_mask = result["mask"]
        assert valid_mask.any(), "Expected valid data in mask, but all values are False"

        # With different odds (-120 vs -115), differential should be non-zero
        retail_sharp_diffs = sequence[valid_mask, retail_sharp_diff_idx]
        assert np.any(retail_sharp_diffs != 0)

    def test_extract_features_handles_missing_outcome_gracefully(self, sample_event):
        """Test that missing outcome data is handled gracefully."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        # Create snapshot with wrong outcome
        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        snapshot = [
            Odds(
                id=700,
                event_id=sample_event.id,
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name=sample_event.away_team,  # Different outcome
                price=-110,
                point=None,
                odds_timestamp=base_time,
                last_update=base_time,
            )
        ]

        odds_sequence = [snapshot]

        # Request home team outcome (not in data)
        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        # Should return zeros
        assert np.all(result["sequence"] == 0)
        assert np.all(~result["mask"])

    def test_extract_features_no_filtering_when_outcome_none(self, sample_event):
        """Test that outcome=None includes all outcomes."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=4)

        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        snapshot = [
            Odds(
                id=800,
                event_id=sample_event.id,
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name=sample_event.home_team,
                price=-120,
                point=None,
                odds_timestamp=base_time,
                last_update=base_time,
            ),
            Odds(
                id=801,
                event_id=sample_event.id,
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name=sample_event.away_team,
                price=+100,
                point=None,
                odds_timestamp=base_time,
                last_update=base_time,
            ),
        ]

        odds_sequence = [snapshot]

        result = extractor.extract_features(
            sample_event,
            odds_sequence,
            market="h2h",
            outcome=None,  # No filtering
        )

        # Should have valid data (both outcomes included)
        assert result["mask"].any()

    def test_sequence_features_produce_valid_values(self, sample_event, sample_odds_snapshot):
        """Test that feature extraction produces valid output structure."""
        from datetime import timedelta

        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        base_time = sample_event.commence_time - timedelta(hours=12)
        odds_sequence = []

        for i in range(4):
            snapshot_time = base_time + timedelta(hours=i * 3)
            snapshot = [
                Odds(
                    id=900 + i,
                    event_id=sample_event.id,
                    bookmaker_key="fanduel",
                    bookmaker_title="FanDuel",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-110,
                    point=None,
                    odds_timestamp=snapshot_time,
                    last_update=snapshot_time,
                )
            ]
            odds_sequence.append(snapshot)

        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        sequence = result["sequence"]
        mask = result["mask"]

        # Should have valid shape
        assert sequence.shape == (8, 15)  # 8 timesteps, 15 features
        assert mask.shape == (8,)

        # Should have some valid data
        assert mask.any(), "Should have at least some valid timesteps"

        # Valid timesteps should have american_odds (core feature)
        feature_names = extractor.get_feature_names()
        american_odds_idx = feature_names.index("american_odds")
        valid_odds = sequence[mask, american_odds_idx]
        # At least one value should be finite and non-zero
        assert np.any(np.isfinite(valid_odds) & (valid_odds != 0))

    def test_extract_features_consistency(self, sample_event):
        """Test that feature extraction is deterministic."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        odds_sequence = [
            [
                Odds(
                    id=1000,
                    event_id=sample_event.id,
                    bookmaker_key="draftkings",
                    bookmaker_title="DraftKings",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-115,
                    point=None,
                    odds_timestamp=base_time,
                    last_update=base_time,
                )
            ]
        ]

        # Extract twice
        result1 = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )
        result2 = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        # Should be identical (use allclose with equal_nan=True for NaN handling)
        assert np.allclose(result1["sequence"], result2["sequence"], equal_nan=True)
        assert np.array_equal(result1["mask"], result2["mask"])


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
        config = FeatureConfig()
        extractor = TabularFeatureExtractor.from_config(config)

        # Should match config defaults
        assert extractor.sharp_bookmakers == ["pinnacle"]
        assert extractor.retail_bookmakers == ["fanduel", "draftkings", "betmgm"]

    def test_tabular_extractor_from_config_custom_values(self):
        """Test that TabularFeatureExtractor.from_config respects custom config values."""
        config = FeatureConfig(
            sharp_bookmakers=["pinnacle", "circa"],
            retail_bookmakers=["fanduel", "betmgm"],
        )
        extractor = TabularFeatureExtractor.from_config(config)

        assert extractor.sharp_bookmakers == ["pinnacle", "circa"]
        assert extractor.retail_bookmakers == ["fanduel", "betmgm"]

    def test_sequence_extractor_from_config_default_values(self):
        """Test that SequenceFeatureExtractor.from_config uses config defaults correctly."""
        config = FeatureConfig()
        extractor = SequenceFeatureExtractor.from_config(config)

        # Should match config defaults
        assert extractor.lookback_hours == 72
        assert extractor.timesteps == 24
        assert extractor.sharp_bookmakers == ["pinnacle"]
        assert extractor.retail_bookmakers == ["fanduel", "draftkings", "betmgm"]

    def test_sequence_extractor_from_config_custom_values(self):
        """Test that SequenceFeatureExtractor.from_config respects custom config values."""
        config = FeatureConfig(
            lookback_hours=48,
            timesteps=16,
            sharp_bookmakers=["pinnacle", "circa"],
            retail_bookmakers=["fanduel"],
        )
        extractor = SequenceFeatureExtractor.from_config(config)

        assert extractor.lookback_hours == 48
        assert extractor.timesteps == 16
        assert extractor.sharp_bookmakers == ["pinnacle", "circa"]
        assert extractor.retail_bookmakers == ["fanduel"]

    def test_tabular_extractor_from_config_produces_valid_features(
        self, sample_event, sample_odds_snapshot
    ):
        """Test complete pipeline: config → extractor → features for tabular."""
        config = FeatureConfig(
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
        )
        extractor = TabularFeatureExtractor.from_config(config)

        # Extract features using extractor created from config
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Should produce valid features
        assert features.is_home_team == 1.0
        assert features.consensus_prob is not None
        assert features.sharp_prob is not None

        # Convert to array
        vector = extractor.create_feature_vector(features)
        assert isinstance(vector, np.ndarray)
        assert len(vector) == len(extractor.get_feature_names())

    def test_sequence_extractor_from_config_produces_valid_features(self, sample_event):
        """Test complete pipeline: config → extractor → features for sequence."""
        config = FeatureConfig(
            lookback_hours=24,
            timesteps=8,
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel"],
        )
        extractor = SequenceFeatureExtractor.from_config(config)

        # Create sequence data
        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        odds_sequence = []

        for i in range(4):
            snapshot_time = base_time + np.timedelta64(i * 3, "h")
            snapshot = [
                Odds(
                    id=1100 + i,
                    event_id=sample_event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=sample_event.home_team,
                    price=-120,
                    point=None,
                    odds_timestamp=snapshot_time,
                    last_update=snapshot_time,
                )
            ]
            odds_sequence.append(snapshot)

        # Extract features using extractor created from config
        result = extractor.extract_features(
            sample_event, odds_sequence, market="h2h", outcome=sample_event.home_team
        )

        # Should produce valid sequence and mask
        assert result["sequence"].shape == (8, 15)  # (timesteps, num_features)
        assert result["mask"].shape == (8,)
        assert result["mask"].any()

    def test_config_defaults_match_extractor_defaults(self):
        """Test that FeatureConfig defaults align with extractor class defaults."""
        config = FeatureConfig()

        # Create extractors both ways
        tabular_from_config = TabularFeatureExtractor.from_config(config)
        tabular_default = TabularFeatureExtractor()

        # Should have identical bookmaker lists
        assert tabular_from_config.sharp_bookmakers == tabular_default.sharp_bookmakers
        assert tabular_from_config.retail_bookmakers == tabular_default.retail_bookmakers

        # Same for sequence extractor
        sequence_from_config = SequenceFeatureExtractor.from_config(config)
        sequence_default = SequenceFeatureExtractor()

        assert sequence_from_config.lookback_hours == sequence_default.lookback_hours
        assert sequence_from_config.timesteps == sequence_default.timesteps
        assert sequence_from_config.sharp_bookmakers == sequence_default.sharp_bookmakers
        assert sequence_from_config.retail_bookmakers == sequence_default.retail_bookmakers

    def test_feature_config_normalize_field_exists(self):
        """Test that FeatureConfig has normalize field."""
        config = FeatureConfig()
        assert hasattr(config, "normalize")
        assert config.normalize is False  # Default value

        config_with_normalize = FeatureConfig(normalize=True)
        assert config_with_normalize.normalize is True

    def test_feature_config_tier_parameters(self):
        """Test that FeatureConfig tier parameters work correctly."""
        from odds_lambda.fetch_tier import FetchTier

        config = FeatureConfig(
            opening_tier=FetchTier.SHARP,
            closing_tier=FetchTier.CLOSING,
            decision_tier=FetchTier.PREGAME,
        )

        assert config.opening_tier == FetchTier.SHARP
        assert config.closing_tier == FetchTier.CLOSING
        assert config.decision_tier == FetchTier.PREGAME

    def test_feature_config_invalid_tier_order_raises_error(self):
        """Test that invalid tier ordering raises validation error."""
        from odds_lambda.fetch_tier import FetchTier

        # Opening must be before closing (chronologically)
        with pytest.raises(ValueError, match="opening_tier.*must be earlier than.*closing_tier"):
            FeatureConfig(
                opening_tier=FetchTier.CLOSING,
                closing_tier=FetchTier.EARLY,
            )

    def test_tabular_extractor_from_config_type_annotation(self):
        """Test that from_config has correct return type."""
        config = FeatureConfig()
        extractor = TabularFeatureExtractor.from_config(config)
        assert isinstance(extractor, TabularFeatureExtractor)

    def test_sequence_extractor_from_config_type_annotation(self):
        """Test that from_config has correct return type."""
        config = FeatureConfig()
        extractor = SequenceFeatureExtractor.from_config(config)
        assert isinstance(extractor, SequenceFeatureExtractor)
