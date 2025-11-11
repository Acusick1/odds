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

    def test_extract_features_returns_dict(self, sample_event, sample_odds_snapshot):
        """Test that extract_features returns a dictionary."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_features_includes_consensus_prob(self, sample_event, sample_odds_snapshot):
        """Test that consensus probability features are calculated."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert "consensus_prob" in features
        assert 0 < features["consensus_prob"] < 1
        assert "opponent_consensus_prob" in features

    def test_extract_features_includes_sharp_features(self, sample_event, sample_odds_snapshot):
        """Test that sharp bookmaker features are extracted."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert "sharp_prob" in features
        assert "sharp_market_hold" in features
        assert features["sharp_market_hold"] > 0  # Should have some vig

    def test_extract_features_includes_retail_sharp_diff(self, sample_event, sample_odds_snapshot):
        """Test that retail vs sharp differences are calculated."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        # Should have retail-sharp difference features
        assert "retail_sharp_diff_home" in features or "retail_sharp_diff_away" in features

    def test_extract_features_includes_best_odds(self, sample_event, sample_odds_snapshot):
        """Test that best available odds are found."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert "best_available_odds" in features
        assert "best_available_decimal" in features
        assert features["best_available_decimal"] > 1.0

    def test_extract_features_team_indicators(self, sample_event, sample_odds_snapshot):
        """Test that team indicator features are set correctly."""
        extractor = TabularFeatureExtractor()

        home_features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
        )

        assert home_features["is_home_team"] == 1.0
        assert home_features["is_away_team"] == 0.0

        away_features = extractor.extract_features(
            sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.away_team
        )

        assert away_features["is_home_team"] == 0.0
        assert away_features["is_away_team"] == 1.0

    def test_extract_features_empty_odds(self, sample_event):
        """Test that extract_features handles empty odds gracefully."""
        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            sample_event, [], market="h2h", outcome=sample_event.home_team
        )

        assert isinstance(features, dict)
        assert len(features) == 0

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

        feature_names = ["is_home_team", "is_away_team", "consensus_prob"]
        vector = extractor.create_feature_vector(features, feature_names)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        assert vector[0] == features.get("is_home_team", 0.0)
        assert vector[1] == features.get("is_away_team", 0.0)
        assert vector[2] == features.get("consensus_prob", 0.0)

    def test_create_feature_vector_fills_missing_with_zero(self):
        """Test that missing features are filled with 0.0."""
        extractor = TabularFeatureExtractor()
        features = {"feature_a": 1.0}
        feature_names = ["feature_a", "feature_b", "feature_c"]

        vector = extractor.create_feature_vector(features, feature_names)

        assert vector[0] == 1.0
        assert vector[1] == 0.0  # Missing feature_b
        assert vector[2] == 0.0  # Missing feature_c

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

        # Should be identical
        assert features1 == features2

        # Convert to vectors
        feature_names = sorted(features1.keys())
        vec1 = extractor.create_feature_vector(features1, feature_names)
        vec2 = extractor.create_feature_vector(features2, feature_names)

        assert np.allclose(vec1, vec2)


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

        # Cyclical encoding should be in valid range [-1, 1]
        assert np.all(np.abs(sequence[valid_mask, time_sin_idx]) <= 1)
        assert np.all(np.abs(sequence[valid_mask, time_cos_idx]) <= 1)

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

    def test_sequence_features_produce_finite_values(self, sample_event, sample_odds_snapshot):
        """Test that all feature values are finite (no NaN or Inf)."""
        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)

        base_time = sample_event.commence_time - np.timedelta64(12, "h")
        odds_sequence = []

        for i in range(4):
            snapshot_time = base_time + np.timedelta64(i * 3, "h")
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

        # All values should be finite
        assert np.all(np.isfinite(sequence))

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

        # Should be identical
        assert np.array_equal(result1["sequence"], result2["sequence"])
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
        if "sharp_prob" in default_features and "sharp_prob" in custom_features:
            assert default_features["sharp_prob"] != custom_features["sharp_prob"]

    def test_different_extractors_can_be_used_interchangeably(
        self, sample_event, sample_odds_snapshot
    ):
        """Test that different extractors can be used through the same interface."""
        extractors = [
            TabularFeatureExtractor(),
            TabularFeatureExtractor(sharp_bookmakers=["pinnacle"], retail_bookmakers=["fanduel"]),
        ]

        for extractor in extractors:
            # All should implement the same interface
            features = extractor.extract_features(
                sample_event, sample_odds_snapshot, market="h2h", outcome=sample_event.home_team
            )
            assert isinstance(features, dict)

            feature_names = extractor.get_feature_names()
            assert isinstance(feature_names, list)

            vector = extractor.create_feature_vector(features)
            assert isinstance(vector, np.ndarray)
