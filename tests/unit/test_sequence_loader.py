"""Unit tests for sequence data loader."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_analytics.sequence_loader import (
    DeviggedProbs,
    DeviggedTotalsProbs,
    calculate_devigged_bookmaker_target,
    calculate_devigged_totals_target,
    calculate_regression_target,
    extract_devigged_h2h_probs,
    extract_devigged_totals_probs,
    load_sequences_for_event,
)
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.odds_math import devig_probabilities


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

    def test_three_way_sums_to_one(self):
        """3-way devigging (soccer h2h) returns 3 values summing to 1."""
        result = devig_probabilities(0.45, 0.30, 0.40)
        assert len(result) == 3
        assert sum(result) == pytest.approx(1.0)

    def test_three_way_preserves_ratios(self):
        """3-way devigging preserves pairwise ratios."""
        home_raw, draw_raw, away_raw = 0.45, 0.30, 0.40
        home, draw, away = devig_probabilities(home_raw, draw_raw, away_raw)
        assert home / draw == pytest.approx(home_raw / draw_raw, rel=1e-4)
        assert home / away == pytest.approx(home_raw / away_raw, rel=1e-4)

    def test_three_way_zero_total(self):
        """3-way with zero total returns uniform fallback."""
        result = devig_probabilities(0.0, 0.0, 0.0)
        assert len(result) == 3
        assert all(p == pytest.approx(1 / 3) for p in result)

    def test_empty_input(self):
        """No arguments returns empty tuple."""
        assert devig_probabilities() == ()

    def test_single_outcome(self):
        """Single outcome normalizes to 1.0."""
        result = devig_probabilities(0.7)
        assert result == (pytest.approx(1.0),)


class TestExtractDeviggedH2hProbs:
    """Tests for extract_devigged_h2h_probs."""

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
        result = extract_devigged_h2h_probs(odds, "Lakers", "Celtics")
        assert result is not None
        assert isinstance(result, DeviggedProbs)
        assert result.home + result.away == pytest.approx(1.0)
        assert result.draw is None
        assert result.home > result.away  # -150 favorite
        # Raw: home=150/250=0.6, away=100/230≈0.4348, total≈1.0348
        # Devigged: home≈0.60/1.0348≈0.5800, away≈0.4200
        assert result.home == pytest.approx(0.60 / (0.60 + 100 / 230), rel=1e-3)

    def test_no_pinnacle_returns_none(self, timestamp):
        """No Pinnacle odds returns None."""
        odds = [
            self._make_odds("fanduel", "Lakers", -160, timestamp),
            self._make_odds("fanduel", "Celtics", 140, timestamp),
        ]
        assert extract_devigged_h2h_probs(odds, "Lakers", "Celtics") is None

    def test_pinnacle_missing_one_side(self, timestamp):
        """Pinnacle present but only one side → None."""
        odds = [
            self._make_odds("pinnacle", "Lakers", -150, timestamp),
        ]
        assert extract_devigged_h2h_probs(odds, "Lakers", "Celtics") is None

    def test_pinnacle_spreads_ignored(self, timestamp):
        """Pinnacle spreads not treated as h2h."""
        odds = [
            self._make_odds("pinnacle", "Lakers", -110, timestamp, market="spreads"),
            self._make_odds("pinnacle", "Celtics", -110, timestamp, market="spreads"),
        ]
        assert extract_devigged_h2h_probs(odds, "Lakers", "Celtics") is None

    def test_empty_odds_list(self, timestamp):
        assert extract_devigged_h2h_probs([], "Lakers", "Celtics") is None

    def test_three_way_soccer_market(self, timestamp):
        """3-way soccer 1x2 returns DeviggedProbs with draw populated."""
        odds = [
            self._make_odds("pinnacle", "Arsenal", -120, timestamp, market="1x2"),
            self._make_odds("pinnacle", "Draw", 250, timestamp, market="1x2"),
            self._make_odds("pinnacle", "Chelsea", 300, timestamp, market="1x2"),
        ]
        result = extract_devigged_h2h_probs(odds, "Arsenal", "Chelsea", market_key="1x2")
        assert result is not None
        assert result.draw is not None
        assert result.home + result.draw + result.away == pytest.approx(1.0)
        assert result.home > result.draw  # Arsenal favored
        assert result.home > result.away

    def test_two_way_has_draw_none(self, timestamp):
        """2-way market returns DeviggedProbs with draw=None."""
        odds = [
            self._make_odds("pinnacle", "Arsenal", -120, timestamp),
            self._make_odds("pinnacle", "Chelsea", 100, timestamp),
        ]
        result = extract_devigged_h2h_probs(odds, "Arsenal", "Chelsea")
        assert result is not None
        assert result.draw is None
        assert result.home + result.away == pytest.approx(1.0)

    def test_three_way_missing_home_returns_none(self, timestamp):
        """3-way market missing home team returns None."""
        odds = [
            self._make_odds("pinnacle", "Draw", 250, timestamp, market="1x2"),
            self._make_odds("pinnacle", "Chelsea", 300, timestamp, market="1x2"),
        ]
        assert extract_devigged_h2h_probs(odds, "Arsenal", "Chelsea", market_key="1x2") is None

    def test_three_way_missing_away_returns_none(self, timestamp):
        """3-way market missing away team returns None."""
        odds = [
            self._make_odds("pinnacle", "Arsenal", -120, timestamp, market="1x2"),
            self._make_odds("pinnacle", "Draw", 250, timestamp, market="1x2"),
        ]
        assert extract_devigged_h2h_probs(odds, "Arsenal", "Chelsea", market_key="1x2") is None


class TestCalculateDeviggedBookmakerTarget:
    """Tests for calculate_devigged_bookmaker_target."""

    @pytest.fixture
    def timestamp(self):
        return datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

    def _make_odds(
        self, outcome: str, price: int, timestamp: datetime, market: str = "h2h"
    ) -> Odds:
        return Odds(
            event_id="test",
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key=market,
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
        result = calculate_devigged_bookmaker_target(
            snapshot_odds, closing_odds, "Lakers", "Celtics"
        )
        assert result is not None

        # Verify manually
        snap_probs = extract_devigged_h2h_probs(snapshot_odds, "Lakers", "Celtics")
        close_probs = extract_devigged_h2h_probs(closing_odds, "Lakers", "Celtics")
        assert snap_probs is not None and close_probs is not None
        expected = close_probs.home - snap_probs.home
        assert result == pytest.approx(expected)
        # Line moved toward Lakers (more negative = bigger favorite)
        assert result > 0

    def test_no_movement(self, timestamp):
        """Same odds → target is 0."""
        odds = [
            self._make_odds("Lakers", -150, timestamp),
            self._make_odds("Celtics", 130, timestamp),
        ]
        result = calculate_devigged_bookmaker_target(odds, odds, "Lakers", "Celtics")
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
            calculate_devigged_bookmaker_target(snapshot_odds, closing_odds, "Lakers", "Celtics")
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
            calculate_devigged_bookmaker_target(snapshot_odds, closing_odds, "Lakers", "Celtics")
            is None
        )

    def test_three_way_target_uses_home_prob(self, timestamp):
        """3-way market target is close.home - snapshot.home."""
        snapshot_odds = [
            self._make_odds("Arsenal", -120, timestamp, market="1x2"),
            self._make_odds("Draw", 250, timestamp, market="1x2"),
            self._make_odds("Chelsea", 300, timestamp, market="1x2"),
        ]
        closing_odds = [
            self._make_odds("Arsenal", -150, timestamp, market="1x2"),
            self._make_odds("Draw", 280, timestamp, market="1x2"),
            self._make_odds("Chelsea", 350, timestamp, market="1x2"),
        ]
        result = calculate_devigged_bookmaker_target(
            snapshot_odds, closing_odds, "Arsenal", "Chelsea", market_key="1x2"
        )
        assert result is not None
        snap = extract_devigged_h2h_probs(snapshot_odds, "Arsenal", "Chelsea", market_key="1x2")
        close = extract_devigged_h2h_probs(closing_odds, "Arsenal", "Chelsea", market_key="1x2")
        assert snap is not None and close is not None
        assert snap.draw is not None
        assert close.draw is not None
        assert result == pytest.approx(close.home - snap.home)


class TestExtractDeviggedTotalsProbs:
    """Tests for extract_devigged_totals_probs."""

    @pytest.fixture
    def timestamp(self):
        return datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

    def _make_odds(
        self,
        bookmaker: str,
        outcome: str,
        price: int,
        timestamp: datetime,
    ) -> Odds:
        return Odds(
            event_id="test",
            bookmaker_key=bookmaker,
            bookmaker_title=bookmaker.title(),
            market_key="totals",
            outcome_name=outcome,
            price=price,
            point=2.5,
            odds_timestamp=timestamp,
            last_update=timestamp,
        )

    def test_extracts_and_devigs(self, timestamp):
        """Extracts bookmaker totals and devigs Over/Under."""
        odds = [
            self._make_odds("bet365", "Over", -110, timestamp),
            self._make_odds("bet365", "Under", -110, timestamp),
        ]
        result = extract_devigged_totals_probs(odds, "bet365")
        assert result is not None
        assert isinstance(result, DeviggedTotalsProbs)
        assert result.over + result.under == pytest.approx(1.0)
        # Symmetric line devigs to 0.5/0.5
        assert result.over == pytest.approx(0.5, abs=0.01)

    def test_asymmetric_line(self, timestamp):
        """Asymmetric Over/Under preserves ratio."""
        odds = [
            self._make_odds("bet365", "Over", 120, timestamp),
            self._make_odds("bet365", "Under", -150, timestamp),
        ]
        result = extract_devigged_totals_probs(odds, "bet365")
        assert result is not None
        assert result.over + result.under == pytest.approx(1.0)
        assert result.under > result.over  # Under is favorite

    def test_filters_by_bookmaker(self, timestamp):
        """Only uses the requested bookmaker."""
        odds = [
            self._make_odds("bet365", "Over", -110, timestamp),
            self._make_odds("bet365", "Under", -110, timestamp),
            self._make_odds("betway", "Over", -120, timestamp),
            self._make_odds("betway", "Under", 100, timestamp),
        ]
        result = extract_devigged_totals_probs(odds, "bet365")
        assert result is not None
        assert result.over == pytest.approx(0.5, abs=0.01)

    def test_missing_bookmaker_returns_none(self, timestamp):
        odds = [
            self._make_odds("betway", "Over", -110, timestamp),
            self._make_odds("betway", "Under", -110, timestamp),
        ]
        assert extract_devigged_totals_probs(odds, "bet365") is None

    def test_missing_over_returns_none(self, timestamp):
        odds = [
            self._make_odds("bet365", "Under", -110, timestamp),
        ]
        assert extract_devigged_totals_probs(odds, "bet365") is None

    def test_missing_under_returns_none(self, timestamp):
        odds = [
            self._make_odds("bet365", "Over", -110, timestamp),
        ]
        assert extract_devigged_totals_probs(odds, "bet365") is None

    def test_empty_odds_list(self):
        assert extract_devigged_totals_probs([], "bet365") is None

    def test_h2h_market_ignored(self, timestamp):
        """h2h odds are not treated as totals."""
        odds = [
            Odds(
                event_id="test",
                bookmaker_key="bet365",
                bookmaker_title="Bet365",
                market_key="h2h",
                outcome_name="Over",
                price=-110,
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
            Odds(
                event_id="test",
                bookmaker_key="bet365",
                bookmaker_title="Bet365",
                market_key="h2h",
                outcome_name="Under",
                price=-110,
                point=None,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]
        assert extract_devigged_totals_probs(odds, "bet365") is None


class TestCalculateDeviggedTotalsTarget:
    """Tests for calculate_devigged_totals_target."""

    @pytest.fixture
    def timestamp(self):
        return datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

    def _make_odds(self, outcome: str, price: int, timestamp: datetime) -> Odds:
        return Odds(
            event_id="test",
            bookmaker_key="bet365",
            bookmaker_title="Bet365",
            market_key="totals",
            outcome_name=outcome,
            price=price,
            point=2.5,
            odds_timestamp=timestamp,
            last_update=timestamp,
        )

    def test_over_target_correct_delta(self, timestamp):
        """Target is fair_close_over - fair_snapshot_over."""
        snapshot_odds = [
            self._make_odds("Over", -110, timestamp),
            self._make_odds("Under", -110, timestamp),
        ]
        closing_odds = [
            self._make_odds("Over", -130, timestamp),
            self._make_odds("Under", 110, timestamp),
        ]
        result = calculate_devigged_totals_target(snapshot_odds, closing_odds, "bet365", "Over")
        assert result is not None

        snap = extract_devigged_totals_probs(snapshot_odds, "bet365")
        close = extract_devigged_totals_probs(closing_odds, "bet365")
        assert snap is not None and close is not None
        assert result == pytest.approx(close.over - snap.over)
        # Over became more likely (line moved toward Over)
        assert result > 0

    def test_under_target_correct_delta(self, timestamp):
        """Under target is fair_close_under - fair_snapshot_under."""
        snapshot_odds = [
            self._make_odds("Over", -110, timestamp),
            self._make_odds("Under", -110, timestamp),
        ]
        closing_odds = [
            self._make_odds("Over", -130, timestamp),
            self._make_odds("Under", 110, timestamp),
        ]
        result = calculate_devigged_totals_target(snapshot_odds, closing_odds, "bet365", "Under")
        assert result is not None

        snap = extract_devigged_totals_probs(snapshot_odds, "bet365")
        close = extract_devigged_totals_probs(closing_odds, "bet365")
        assert snap is not None and close is not None
        assert result == pytest.approx(close.under - snap.under)
        # Under became less likely (opposite of Over movement)
        assert result < 0

    def test_over_and_under_sum_to_zero(self, timestamp):
        """Over delta + Under delta = 0 (zero-sum market)."""
        snapshot_odds = [
            self._make_odds("Over", -110, timestamp),
            self._make_odds("Under", -110, timestamp),
        ]
        closing_odds = [
            self._make_odds("Over", -130, timestamp),
            self._make_odds("Under", 110, timestamp),
        ]
        over_delta = calculate_devigged_totals_target(snapshot_odds, closing_odds, "bet365", "Over")
        under_delta = calculate_devigged_totals_target(
            snapshot_odds, closing_odds, "bet365", "Under"
        )
        assert over_delta is not None and under_delta is not None
        assert over_delta + under_delta == pytest.approx(0.0)

    def test_no_movement(self, timestamp):
        """Same odds → target is 0."""
        odds = [
            self._make_odds("Over", -110, timestamp),
            self._make_odds("Under", -110, timestamp),
        ]
        result = calculate_devigged_totals_target(odds, odds, "bet365", "Over")
        assert result == pytest.approx(0.0)

    def test_missing_snapshot_bookmaker(self, timestamp):
        """No target bookmaker in snapshot → None."""
        snapshot_odds = [
            Odds(
                event_id="test",
                bookmaker_key="betway",
                bookmaker_title="Betway",
                market_key="totals",
                outcome_name="Over",
                price=-110,
                point=2.5,
                odds_timestamp=timestamp,
                last_update=timestamp,
            ),
        ]
        closing_odds = [
            self._make_odds("Over", -130, timestamp),
            self._make_odds("Under", 110, timestamp),
        ]
        assert (
            calculate_devigged_totals_target(snapshot_odds, closing_odds, "bet365", "Over") is None
        )

    def test_missing_closing_bookmaker(self, timestamp):
        """No target bookmaker in closing → None."""
        snapshot_odds = [
            self._make_odds("Over", -110, timestamp),
            self._make_odds("Under", -110, timestamp),
        ]
        assert calculate_devigged_totals_target(snapshot_odds, [], "bet365", "Over") is None
