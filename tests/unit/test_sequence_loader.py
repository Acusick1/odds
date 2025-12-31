"""Unit tests for sequence data loader."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_analytics.sequence_loader import (
    calculate_regression_target,
    extract_opening_closing_odds,
    load_sequences_for_event,
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


class TestExtractOpeningClosingOdds:
    """Tests for extract_opening_closing_odds helper function."""

    def test_extract_basic_legacy_mode(self):
        """Test basic extraction without commence_time (legacy first/last mode)."""
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

        # Legacy mode (no commence_time)
        opening, closing = extract_opening_closing_odds(odds_sequences, "Lakers", "h2h")

        assert opening is not None
        assert closing is not None
        assert opening[0].price == -120
        assert closing[0].price == -140

    def test_extract_with_time_windows(self):
        """Test extraction with configurable time windows."""
        # Game at 19:00
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # Snapshots at various times before game
        timestamp_48h = datetime(2024, 10, 30, 19, 0, 0, tzinfo=UTC)  # 48h before
        timestamp_24h = datetime(2024, 10, 31, 19, 0, 0, tzinfo=UTC)  # 24h before
        timestamp_1h = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)  # 1h before
        timestamp_30m = datetime(2024, 11, 1, 18, 30, 0, tzinfo=UTC)  # 30min before

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
                    odds_timestamp=timestamp_48h,
                    last_update=timestamp_48h,
                ),
            ],
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Lakers",
                    price=-130,
                    point=None,
                    odds_timestamp=timestamp_24h,
                    last_update=timestamp_24h,
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
                    odds_timestamp=timestamp_1h,
                    last_update=timestamp_1h,
                ),
            ],
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Lakers",
                    price=-150,
                    point=None,
                    odds_timestamp=timestamp_30m,
                    last_update=timestamp_30m,
                ),
            ],
        ]

        # Find opening at 48h before, closing at 0.5h before
        opening, closing = extract_opening_closing_odds(
            odds_sequences,
            "Lakers",
            "h2h",
            commence_time=commence_time,
            opening_hours_before=48.0,
            closing_hours_before=0.5,
        )

        assert opening is not None
        assert closing is not None
        assert opening[0].price == -120  # 48h before snapshot
        assert closing[0].price == -150  # 30min before snapshot

    def test_extract_finds_closest_snapshot(self):
        """Test that extraction finds snapshot closest to target time."""
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # No snapshot exactly at 48h, but one at 47h and one at 50h
        timestamp_50h = datetime(2024, 10, 30, 17, 0, 0, tzinfo=UTC)  # 50h before
        timestamp_47h = datetime(2024, 10, 30, 20, 0, 0, tzinfo=UTC)  # 47h before
        timestamp_1h = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)  # 1h before

        odds_sequences = [
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Lakers",
                    price=-110,
                    point=None,
                    odds_timestamp=timestamp_50h,
                    last_update=timestamp_50h,
                ),
            ],
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Lakers",
                    price=-120,
                    point=None,
                    odds_timestamp=timestamp_47h,
                    last_update=timestamp_47h,
                ),
            ],
            [
                Odds(
                    event_id="test",
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name="Lakers",
                    price=-150,
                    point=None,
                    odds_timestamp=timestamp_1h,
                    last_update=timestamp_1h,
                ),
            ],
        ]

        # 47h is closer to 48h target than 50h
        opening, closing = extract_opening_closing_odds(
            odds_sequences,
            "Lakers",
            "h2h",
            commence_time=commence_time,
            opening_hours_before=48.0,
            closing_hours_before=1.0,
        )

        assert opening is not None
        assert closing is not None
        assert opening[0].price == -120  # 47h before (closest to 48h target)
        assert closing[0].price == -150  # 1h before

    def test_extract_same_snapshot_returns_none(self):
        """Test that same snapshot for opening/closing returns None."""
        commence_time = datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC)

        # Only one snapshot - both opening and closing resolve to it
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

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
                    odds_timestamp=timestamp,
                    last_update=timestamp,
                ),
            ],
        ]

        opening, closing = extract_opening_closing_odds(
            odds_sequences,
            "Lakers",
            "h2h",
            commence_time=commence_time,
            opening_hours_before=48.0,
            closing_hours_before=0.5,
        )

        # Should return None because both resolve to same snapshot
        assert opening is None
        assert closing is None

    def test_extract_empty_sequences(self):
        """Test handling of empty sequences."""
        opening, closing = extract_opening_closing_odds([], "Lakers", "h2h")
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

        opening, closing = extract_opening_closing_odds(odds_sequences, "Lakers", "h2h")
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

        opening, closing = extract_opening_closing_odds(odds_sequences, "Lakers", "spreads")

        assert opening is not None
        assert closing is not None
        assert opening[0].point == -2.5
        assert closing[0].point == -3.5

    def test_extract_single_snapshot_legacy_mode_returns_none(self):
        """Test that single snapshot in legacy mode returns None."""
        timestamp = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

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
                    odds_timestamp=timestamp,
                    last_update=timestamp,
                ),
            ],
        ]

        # Legacy mode with single snapshot should return None
        opening, closing = extract_opening_closing_odds(odds_sequences, "Lakers", "h2h")
        assert opening is None
        assert closing is None


class TestCalculateRegressionTarget:
    """Tests for calculate_regression_target helper function."""

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

        result = calculate_regression_target(opening_odds, closing_odds, "h2h")

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

        result = calculate_regression_target(opening_odds, closing_odds, "spreads")

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

        result = calculate_regression_target(opening_odds, closing_odds, "totals")

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

        result = calculate_regression_target(None, closing_odds, "h2h")
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

        result = calculate_regression_target(opening_odds, None, "h2h")
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

        result = calculate_regression_target(opening_odds, closing_odds, "unknown")
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

        result = calculate_regression_target(opening_odds, closing_odds, "spreads")

        assert result is not None
        # Opening avg: (-2.5 + -3.5) / 2 = -3.0
        # Closing avg: (-4.5 + -5.5) / 2 = -5.0
        # Delta: -5.0 - (-3.0) = -2.0
        assert result == -2.0
