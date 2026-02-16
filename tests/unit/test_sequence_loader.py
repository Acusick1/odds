"""Unit tests for sequence data loader."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_analytics.sequence_loader import (
    calculate_devigged_pinnacle_target,
    calculate_regression_target,
    extract_pinnacle_h2h_probs,
    load_sequences_for_event,
)
from odds_analytics.utils import devig_probabilities
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


class TestDevigProbabilities:
    """Tests for devig_probabilities utility."""

    def test_symmetric_overround(self):
        """Equal vig on both sides normalizes to 0.5/0.5."""
        home, away = devig_probabilities(0.524, 0.524)
        assert home == pytest.approx(0.5)
        assert away == pytest.approx(0.5)

    def test_no_vig(self):
        """Probabilities already summing to 1 are unchanged."""
        home, away = devig_probabilities(0.6, 0.4)
        assert home == pytest.approx(0.6)
        assert away == pytest.approx(0.4)

    def test_zero_total(self):
        """Zero total returns 0.5/0.5 fallback."""
        home, away = devig_probabilities(0.0, 0.0)
        assert home == 0.5
        assert away == 0.5

    def test_typical_nba_vig(self):
        """Typical -110/-110 line: both ~0.524, devig to 0.5."""
        # -110 American = 1/1.909 ≈ 0.5238
        home_raw = 110 / 210  # 0.5238
        away_raw = 110 / 210
        home, away = devig_probabilities(home_raw, away_raw)
        assert home == pytest.approx(0.5, abs=0.001)
        assert away == pytest.approx(0.5, abs=0.001)
        assert home + away == pytest.approx(1.0)

    def test_asymmetric_line(self):
        """Asymmetric line preserves ratio after devigging."""
        # -150/+130: home=150/250=0.60, away=100/230≈0.4348
        home_raw = 0.60
        away_raw = 0.4348
        home, away = devig_probabilities(home_raw, away_raw)
        assert home + away == pytest.approx(1.0)
        assert home > away
        # Ratio should be preserved: home/away ≈ 0.60/0.4348
        assert home / away == pytest.approx(home_raw / away_raw, rel=1e-4)


class TestExtractPinnacleH2hProbs:
    """Tests for extract_pinnacle_h2h_probs."""

    @pytest.fixture
    def timestamp(self):
        return datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

    def _make_odds(
        self,
        bookmaker: str,
        outcome: str,
        price: int,
        timestamp: datetime,
        market: str = "h2h",
    ) -> Odds:
        return Odds(
            event_id="test",
            bookmaker_key=bookmaker,
            bookmaker_title=bookmaker.title(),
            market_key=market,
            outcome_name=outcome,
            price=price,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        )

    def test_extracts_and_devigs_pinnacle(self, timestamp):
        """Extracts Pinnacle h2h, ignores other bookmakers, devigs."""
        odds = [
            self._make_odds("pinnacle", "Lakers", -150, timestamp),
            self._make_odds("pinnacle", "Celtics", 130, timestamp),
            self._make_odds("fanduel", "Lakers", -160, timestamp),
            self._make_odds("fanduel", "Celtics", 140, timestamp),
        ]
        result = extract_pinnacle_h2h_probs(odds, "Lakers", "Celtics")
        assert result is not None
        home, away = result
        assert home + away == pytest.approx(1.0)
        assert home > away  # -150 favorite
        # Raw: home=150/250=0.6, away=100/230≈0.4348, total≈1.0348
        # Devigged: home≈0.60/1.0348≈0.5800, away≈0.4200
        assert home == pytest.approx(0.60 / (0.60 + 100 / 230), rel=1e-3)

    def test_no_pinnacle_returns_none(self, timestamp):
        """No Pinnacle odds returns None."""
        odds = [
            self._make_odds("fanduel", "Lakers", -160, timestamp),
            self._make_odds("fanduel", "Celtics", 140, timestamp),
        ]
        assert extract_pinnacle_h2h_probs(odds, "Lakers", "Celtics") is None

    def test_pinnacle_missing_one_side(self, timestamp):
        """Pinnacle present but only one side → None."""
        odds = [
            self._make_odds("pinnacle", "Lakers", -150, timestamp),
        ]
        assert extract_pinnacle_h2h_probs(odds, "Lakers", "Celtics") is None

    def test_pinnacle_spreads_ignored(self, timestamp):
        """Pinnacle spreads not treated as h2h."""
        odds = [
            self._make_odds("pinnacle", "Lakers", -110, timestamp, market="spreads"),
            self._make_odds("pinnacle", "Celtics", -110, timestamp, market="spreads"),
        ]
        assert extract_pinnacle_h2h_probs(odds, "Lakers", "Celtics") is None

    def test_empty_odds_list(self, timestamp):
        assert extract_pinnacle_h2h_probs([], "Lakers", "Celtics") is None


class TestCalculateDeviggedPinnacleTarget:
    """Tests for calculate_devigged_pinnacle_target."""

    @pytest.fixture
    def timestamp(self):
        return datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

    def _make_odds(self, outcome: str, price: int, timestamp: datetime) -> Odds:
        return Odds(
            event_id="test",
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=outcome,
            price=price,
            point=None,
            odds_timestamp=timestamp,
            last_update=timestamp,
        )

    def test_correct_delta(self, timestamp):
        """Target is fair_close_home - fair_snapshot_home."""
        snapshot_odds = [
            self._make_odds("Lakers", -150, timestamp),
            self._make_odds("Celtics", 130, timestamp),
        ]
        closing_odds = [
            self._make_odds("Lakers", -200, timestamp),
            self._make_odds("Celtics", 170, timestamp),
        ]
        result = calculate_devigged_pinnacle_target(
            snapshot_odds, closing_odds, "Lakers", "Celtics"
        )
        assert result is not None

        # Verify manually
        snap_probs = extract_pinnacle_h2h_probs(snapshot_odds, "Lakers", "Celtics")
        close_probs = extract_pinnacle_h2h_probs(closing_odds, "Lakers", "Celtics")
        expected = close_probs[0] - snap_probs[0]
        assert result == pytest.approx(expected)
        # Line moved toward Lakers (more negative = bigger favorite)
        assert result > 0

    def test_no_movement(self, timestamp):
        """Same odds → target is 0."""
        odds = [
            self._make_odds("Lakers", -150, timestamp),
            self._make_odds("Celtics", 130, timestamp),
        ]
        result = calculate_devigged_pinnacle_target(odds, odds, "Lakers", "Celtics")
        assert result == pytest.approx(0.0)

    def test_missing_snapshot_pinnacle(self, timestamp):
        """No Pinnacle in snapshot → None."""
        snapshot_odds = [
            Odds(
                event_id="test",
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name="Lakers",
                price=-150,
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]
        closing_odds = [
            self._make_odds("Lakers", -200, timestamp),
            self._make_odds("Celtics", 170, timestamp),
        ]
        assert (
            calculate_devigged_pinnacle_target(snapshot_odds, closing_odds, "Lakers", "Celtics")
            is None
        )

    def test_missing_closing_pinnacle(self, timestamp):
        """No Pinnacle in closing → None."""
        snapshot_odds = [
            self._make_odds("Lakers", -150, timestamp),
            self._make_odds("Celtics", 130, timestamp),
        ]
        closing_odds = []
        assert (
            calculate_devigged_pinnacle_target(snapshot_odds, closing_odds, "Lakers", "Celtics")
            is None
        )
