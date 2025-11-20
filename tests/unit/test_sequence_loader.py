"""Unit tests for sequence data loader."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from odds_analytics.sequence_loader import (
    TargetType,
    _calculate_regression_target,
    _extract_opening_closing_odds,
    load_sequences_for_event,
    prepare_lstm_training_data,
)
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot


@pytest.fixture
def sample_event():
    """Create a sample Event for testing."""
    return Event(
        id="test_event_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        status=EventStatus.FINAL,
        home_score=110,
        away_score=105,
    )


@pytest.fixture
def sample_snapshots(sample_event):
    """Create sample OddsSnapshot objects for testing."""
    base_time = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

    snapshots = []
    for i in range(3):
        snapshot_time = base_time + timedelta(hours=i)
        # Simulate raw_data from API
        raw_data = {
            "id": sample_event.id,
            "sport_key": "basketball_nba",
            "commence_time": sample_event.commence_time.isoformat(),
            "home_team": sample_event.home_team,
            "away_team": sample_event.away_team,
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": sample_event.home_team,
                                    "price": -120 - (i * 2),  # Line movement
                                },
                                {
                                    "name": sample_event.away_team,
                                    "price": 100 + (i * 2),
                                },
                            ],
                        }
                    ],
                },
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {
                                    "name": sample_event.home_team,
                                    "price": -115 - (i * 2),
                                },
                                {
                                    "name": sample_event.away_team,
                                    "price": -105 + (i * 2),
                                },
                            ],
                        }
                    ],
                },
            ],
        }

        snapshot = OddsSnapshot(
            id=i + 1,
            event_id=sample_event.id,
            snapshot_time=snapshot_time,
            raw_data=raw_data,
            bookmaker_count=2,
        )
        snapshots.append(snapshot)

    return snapshots


class TestLoadSequencesForEvent:
    """Tests for load_sequences_for_event()."""

    @pytest.mark.asyncio
    async def test_load_sequences_basic(self, sample_event, sample_snapshots):
        """Test basic sequence loading."""
        # Mock session and reader
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.get_snapshots_for_event = AsyncMock(return_value=sample_snapshots)
            mock_reader_class.return_value = mock_reader

            # Load sequences
            sequences = await load_sequences_for_event(sample_event.id, mock_session)

            # Verify results
            assert len(sequences) == 3, "Should have 3 snapshots"
            assert all(isinstance(seq, list) for seq in sequences), "Each snapshot should be a list"

            # First snapshot should have 4 odds (2 bookmakers × 2 outcomes)
            assert len(sequences[0]) == 4, "First snapshot should have 4 odds records"

            # Verify chronological ordering (timestamps should be ascending)
            for i in range(len(sequences) - 1):
                assert sequences[i][0].odds_timestamp < sequences[i + 1][0].odds_timestamp

    @pytest.mark.asyncio
    async def test_load_sequences_empty_snapshots(self):
        """Test handling of events with no snapshots."""
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.get_snapshots_for_event = AsyncMock(return_value=[])
            mock_reader_class.return_value = mock_reader

            sequences = await load_sequences_for_event("no_data_event", mock_session)

            assert sequences == [], "Should return empty list for no snapshots"

    @pytest.mark.asyncio
    async def test_load_sequences_invalid_raw_data(self, sample_event):
        """Test handling of snapshots with invalid raw_data."""
        # Create snapshot with invalid raw_data
        invalid_snapshot = OddsSnapshot(
            id=1,
            event_id=sample_event.id,
            snapshot_time=datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC),
            raw_data={},  # Missing 'bookmakers' key
            bookmaker_count=0,
        )

        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.get_snapshots_for_event = AsyncMock(return_value=[invalid_snapshot])
            mock_reader_class.return_value = mock_reader

            sequences = await load_sequences_for_event(sample_event.id, mock_session)

            assert sequences == [], "Should return empty list for invalid snapshots"

    @pytest.mark.asyncio
    async def test_load_sequences_groups_by_timestamp(self, sample_event, sample_snapshots):
        """Test that odds are correctly grouped by snapshot_time."""
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.get_snapshots_for_event = AsyncMock(return_value=sample_snapshots)
            mock_reader_class.return_value = mock_reader

            sequences = await load_sequences_for_event(sample_event.id, mock_session)

            # All odds in a snapshot should have the same timestamp
            for snapshot in sequences:
                timestamps = {odds.odds_timestamp for odds in snapshot}
                assert len(timestamps) == 1, "All odds in snapshot should have same timestamp"

    @pytest.mark.asyncio
    async def test_load_sequences_preserves_market_data(self, sample_event, sample_snapshots):
        """Test that market data (bookmaker, outcome, price) is preserved."""
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader") as mock_reader_class:
            mock_reader = MagicMock()
            mock_reader.get_snapshots_for_event = AsyncMock(return_value=sample_snapshots)
            mock_reader_class.return_value = mock_reader

            sequences = await load_sequences_for_event(sample_event.id, mock_session)

            # Check first snapshot has expected bookmakers
            first_snapshot = sequences[0]
            bookmakers = {odds.bookmaker_key for odds in first_snapshot}
            assert bookmakers == {"pinnacle", "fanduel"}

            # Check market keys
            markets = {odds.market_key for odds in first_snapshot}
            assert markets == {"h2h"}

            # Check outcomes
            outcomes = {odds.outcome_name for odds in first_snapshot}
            assert sample_event.home_team in outcomes
            assert sample_event.away_team in outcomes


class TestPrepareLSTMTrainingData:
    """Tests for prepare_lstm_training_data()."""

    @pytest.mark.asyncio
    async def test_prepare_basic(self, sample_event, sample_snapshots):
        """Test basic training data preparation."""
        events = [sample_event]
        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            # Create mock sequences (list of lists of Odds)
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            mock_sequences = [
                [
                    Odds(
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
                        event_id=sample_event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name=sample_event.away_team,
                        price=100,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                ]
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=events,
                session=mock_session,
                outcome="home",
                market="h2h",
                timesteps=8,
            )

            # Verify shapes
            assert X.shape[0] == 1, "Should have 1 sample"
            assert X.shape[1] == 8, "Should have 8 timesteps"
            assert X.shape[2] > 0, "Should have features"

            assert y.shape == (1,), "Labels should be 1D array"
            assert masks.shape == (1, 8), "Masks should match (samples, timesteps)"

    @pytest.mark.asyncio
    async def test_prepare_labels_home_win(self, sample_event, sample_snapshots):
        """Test label generation for home team win."""
        # Home team wins (110 > 105)
        events = [sample_event]
        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            mock_sequences = [
                [
                    Odds(
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
                ]
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=events,
                session=mock_session,
                outcome="home",
                timesteps=8,
            )

            assert y[0] == 1, "Home team won, label should be 1"

    @pytest.mark.asyncio
    async def test_prepare_labels_away_win(self):
        """Test label generation for away team win."""
        # Away team wins (105 > 100)
        event = Event(
            id="away_win_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=100,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name=event.away_team,
                        price=-120,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                ]
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                outcome="away",
                timesteps=8,
            )

            assert y[0] == 1, "Away team won, label should be 1"

    @pytest.mark.asyncio
    async def test_prepare_empty_events(self):
        """Test handling of empty event list."""
        mock_session = AsyncMock()

        X, y, masks = await prepare_lstm_training_data(
            events=[],
            session=mock_session,
        )

        assert X.shape == (0,), "Should return empty array"
        assert y.shape == (0,), "Should return empty array"
        assert masks.shape == (0,), "Should return empty array"

    @pytest.mark.asyncio
    async def test_prepare_filters_incomplete_events(self):
        """Test that events without final scores are filtered out."""
        # Event without scores
        incomplete_event = Event(
            id="incomplete",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,  # Not final
            home_score=None,
            away_score=None,
        )

        mock_session = AsyncMock()

        X, y, masks = await prepare_lstm_training_data(
            events=[incomplete_event],
            session=mock_session,
        )

        assert X.shape == (0,), "Should filter out incomplete events"

    @pytest.mark.asyncio
    async def test_prepare_multiple_events(self):
        """Test processing multiple events."""
        events = [
            Event(
                id=f"event_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 11, i, 19, 0, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.FINAL,
                home_score=110,
                away_score=105,
            )
            for i in range(1, 4)
        ]

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            mock_sequences = [
                [
                    Odds(
                        event_id="test",
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name="Lakers",
                        price=-120,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                ]
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=events,
                session=mock_session,
                timesteps=8,
            )

            assert X.shape[0] == 3, "Should have 3 samples"
            assert y.shape[0] == 3, "Should have 3 labels"

    @pytest.mark.asyncio
    async def test_prepare_attention_masks(self):
        """Test that attention masks are properly generated."""
        event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            # Create a sequence with only 2 valid snapshots
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name=event.home_team,
                        price=-120,
                        point=None,
                        odds_timestamp=timestamp + timedelta(hours=i),
                        last_update=timestamp + timedelta(hours=i),
                    ),
                ]
                for i in range(2)
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                timesteps=8,
                lookback_hours=24,
            )

            # Masks should have some True values (for valid timesteps)
            assert masks.dtype == bool, "Masks should be boolean"
            assert masks.shape == (1, 8), "Masks should match (samples, timesteps)"

    @pytest.mark.asyncio
    async def test_prepare_skips_events_without_sequences(self):
        """Test that events without sequences are skipped."""
        event = Event(
            id="no_data_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            # Return empty sequences
            mock_load.return_value = []

            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
            )

            assert X.shape[0] == 0, "Should skip events without sequences"

    @pytest.mark.asyncio
    async def test_prepare_custom_timesteps(self):
        """Test using custom timestep configuration."""
        event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name=event.home_team,
                        price=-120,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                ]
            ]
            mock_load.return_value = mock_sequences

            # Custom timesteps
            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                timesteps=16,  # Custom
                lookback_hours=48,  # Custom
            )

            assert X.shape[1] == 16, "Should respect custom timesteps parameter"


class TestExtractOpeningClosingOdds:
    """Tests for _extract_opening_closing_odds helper function."""

    def test_extract_basic(self):
        """Test basic extraction of opening and closing odds."""
        timestamp1 = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
        timestamp2 = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)

        odds_sequences = [
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Lakers",
                    price=-120,
                    point=None,
                    odds_timestamp=timestamp1,
                    last_update=timestamp1,
                ),
            ],
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Lakers",
                    price=-140,
                    point=None,
                    odds_timestamp=timestamp2,
                    last_update=timestamp2,
                ),
            ],
        ]

        opening, closing = _extract_opening_closing_odds(odds_sequences, "Lakers", "h2h")

        assert opening is not None
        assert closing is not None
        assert opening[0].price == -120
        assert closing[0].price == -140

    def test_extract_empty_sequences(self):
        """Test handling of empty sequences."""
        opening, closing = _extract_opening_closing_odds([], "Lakers", "h2h")
        assert opening is None
        assert closing is None

    def test_extract_no_matching_outcome(self):
        """Test handling of sequences with no matching outcome."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
        odds_sequences = [
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Celtics",  # Different team
                    price=-120,
                    point=None,
                    odds_timestamp=timestamp,
                    last_update=timestamp,
                ),
            ],
        ]

        opening, closing = _extract_opening_closing_odds(odds_sequences, "Lakers", "h2h")
        assert opening is None
        assert closing is None

    def test_extract_spreads_market(self):
        """Test extraction for spreads market with point values."""
        timestamp1 = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
        timestamp2 = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)

        odds_sequences = [
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="spreads",
                    outcome_name="Lakers",
                    price=-110,
                    point=-2.5,
                    odds_timestamp=timestamp1,
                    last_update=timestamp1,
                ),
            ],
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="spreads",
                    outcome_name="Lakers",
                    price=-110,
                    point=-3.5,
                    odds_timestamp=timestamp2,
                    last_update=timestamp2,
                ),
            ],
        ]

        opening, closing = _extract_opening_closing_odds(odds_sequences, "Lakers", "spreads")

        assert opening is not None
        assert closing is not None
        assert opening[0].point == -2.5
        assert closing[0].point == -3.5


class TestCalculateRegressionTarget:
    """Tests for _calculate_regression_target helper function."""

    def test_h2h_probability_delta(self):
        """Test h2h market uses implied probability delta."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price=-120,  # ~54.5% implied prob
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price=-150,  # ~60% implied prob
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = _calculate_regression_target(opening_odds, closing_odds, "h2h")

        assert result is not None
        # Probability should increase (line moved in favor of Lakers)
        assert result > 0
        # Approximately 60% - 54.5% = 5.5%
        assert 0.04 < result < 0.07

    def test_spreads_point_delta(self):
        """Test spreads market uses point delta."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price=-110,
                point=-2.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price=-110,
                point=-3.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = _calculate_regression_target(opening_odds, closing_odds, "spreads")

        assert result is not None
        # Point delta: -3.5 - (-2.5) = -1.0
        assert result == -1.0

    def test_totals_point_delta(self):
        """Test totals market uses point delta."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="totals",
                outcome_name="Over",
                price=-110,
                point=215.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="totals",
                outcome_name="Over",
                price=-110,
                point=218.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = _calculate_regression_target(opening_odds, closing_odds, "totals")

        assert result is not None
        # Point delta: 218.5 - 215.5 = 3.0
        assert result == 3.0

    def test_missing_opening_odds(self):
        """Test handling of missing opening odds."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price=-150,
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = _calculate_regression_target(None, closing_odds, "h2h")
        assert result is None

    def test_missing_closing_odds(self):
        """Test handling of missing closing odds."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name="Lakers",
                price=-120,
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = _calculate_regression_target(opening_odds, None, "h2h")
        assert result is None

    def test_unknown_market(self):
        """Test handling of unknown market type."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="unknown",
                outcome_name="Lakers",
                price=-120,
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="unknown",
                outcome_name="Lakers",
                price=-150,
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = _calculate_regression_target(opening_odds, closing_odds, "unknown")
        assert result is None

    def test_multiple_bookmakers_average(self):
        """Test averaging across multiple bookmakers."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        opening_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price=-110,
                point=-2.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
            Odds(
                event_id="test",
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="spreads",
                outcome_name="Lakers",
                price=-110,
                point=-3.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        closing_odds = [
            Odds(
                event_id="test",
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="spreads",
                outcome_name="Lakers",
                price=-110,
                point=-4.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
            Odds(
                event_id="test",
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="spreads",
                outcome_name="Lakers",
                price=-110,
                point=-5.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]

        result = _calculate_regression_target(opening_odds, closing_odds, "spreads")

        assert result is not None
        # Opening avg: (-2.5 + -3.5) / 2 = -3.0
        # Closing avg: (-4.5 + -5.5) / 2 = -5.0
        # Delta: -5.0 - (-3.0) = -2.0
        assert result == -2.0


class TestPrepareLSTMRegressionMode:
    """Tests for prepare_lstm_training_data in regression mode."""

    @pytest.mark.asyncio
    async def test_regression_mode_h2h(self):
        """Test regression mode with h2h market."""
        event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp1 = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            timestamp2 = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)

            # Create sequences with different odds to produce a regression target
            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name="Lakers",
                        price=-120,  # Opening
                        point=None,
                        odds_timestamp=timestamp1,
                        last_update=timestamp1,
                    ),
                ],
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name="Lakers",
                        price=-150,  # Closing
                        point=None,
                        odds_timestamp=timestamp2,
                        last_update=timestamp2,
                    ),
                ],
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                outcome="home",
                market="h2h",
                timesteps=8,
                target_type=TargetType.REGRESSION,
            )

            assert X.shape[0] == 1, "Should have 1 sample"
            assert y.dtype == np.float32, "Regression targets should be float32"
            # Y should be the probability delta (positive since line moved in favor)
            assert y[0] > 0, "Target should be positive (line moved in favor)"

    @pytest.mark.asyncio
    async def test_regression_mode_spreads(self):
        """Test regression mode with spreads market."""
        event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp1 = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            timestamp2 = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)

            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="spreads",
                        outcome_name="Lakers",
                        price=-110,
                        point=-2.5,  # Opening
                        odds_timestamp=timestamp1,
                        last_update=timestamp1,
                    ),
                ],
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="spreads",
                        outcome_name="Lakers",
                        price=-110,
                        point=-4.5,  # Closing
                        odds_timestamp=timestamp2,
                        last_update=timestamp2,
                    ),
                ],
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                outcome="home",
                market="spreads",
                timesteps=8,
                target_type="regression",  # Test string conversion
            )

            assert X.shape[0] == 1, "Should have 1 sample"
            assert y.dtype == np.float32, "Regression targets should be float32"
            # Point delta: -4.5 - (-2.5) = -2.0
            assert y[0] == pytest.approx(-2.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_regression_skips_missing_data(self):
        """Test that regression mode skips events without opening/closing data."""
        event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

            # Sequences with non-matching outcome (no Lakers odds)
            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name="Celtics",  # Different team
                        price=-120,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                ],
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                outcome="home",
                market="h2h",
                timesteps=8,
                target_type=TargetType.REGRESSION,
            )

            # Should skip the event due to missing regression target
            assert X.shape[0] == 0, "Should skip events without regression target"

    @pytest.mark.asyncio
    async def test_classification_backward_compatibility(self):
        """Test that classification mode still works (backward compatibility)."""
        event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="h2h",
                        outcome_name="Lakers",
                        price=-120,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                ]
            ]
            mock_load.return_value = mock_sequences

            # Default target_type should be classification
            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                outcome="home",
                timesteps=8,
            )

            assert y.dtype == np.int32, "Classification targets should be int32"
            assert y[0] == 1, "Home team won, label should be 1"

    @pytest.mark.asyncio
    async def test_regression_totals_market(self):
        """Test regression mode with totals market."""
        event = Event(
            id="test_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        mock_session = AsyncMock()

        with patch("odds_analytics.sequence_loader.load_sequences_for_event") as mock_load:
            timestamp1 = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
            timestamp2 = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)

            mock_sequences = [
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="totals",
                        outcome_name="Over",
                        price=-110,
                        point=215.5,  # Opening
                        odds_timestamp=timestamp1,
                        last_update=timestamp1,
                    ),
                ],
                [
                    Odds(
                        event_id=event.id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key="totals",
                        outcome_name="Over",
                        price=-110,
                        point=220.5,  # Closing
                        odds_timestamp=timestamp2,
                        last_update=timestamp2,
                    ),
                ],
            ]
            mock_load.return_value = mock_sequences

            X, y, masks = await prepare_lstm_training_data(
                events=[event],
                session=mock_session,
                outcome="Over",  # For totals, use Over/Under
                market="totals",
                timesteps=8,
                target_type=TargetType.REGRESSION,
            )

            assert X.shape[0] == 1, "Should have 1 sample"
            # Point delta: 220.5 - 215.5 = 5.0
            assert y[0] == pytest.approx(5.0, abs=0.01)
