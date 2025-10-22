"""Unit tests for data validators."""

from datetime import UTC, datetime, timedelta

from storage.validators import OddsValidator


class TestOddsValidator:
    """Tests for OddsValidator class."""

    def test_american_to_probability(self):
        """Test converting American odds to probability."""
        # Negative odds (favorite)
        prob = OddsValidator.american_to_probability(-110)
        assert 0.52 < prob < 0.53  # ~52.4%

        prob = OddsValidator.american_to_probability(-200)
        assert 0.66 < prob < 0.67  # ~66.7%

        # Positive odds (underdog)
        prob = OddsValidator.american_to_probability(150)
        assert 0.39 < prob < 0.41  # ~40%

        prob = OddsValidator.american_to_probability(200)
        assert 0.33 < prob < 0.34  # ~33.3%

    def test_validate_odds_value_valid(self):
        """Test validation of valid odds values."""
        warnings = OddsValidator.validate_odds_value(-110, "fanduel", "h2h", "Lakers")
        assert len(warnings) == 0

        warnings = OddsValidator.validate_odds_value(150, "draftkings", "h2h", "Celtics")
        assert len(warnings) == 0

    def test_validate_odds_value_out_of_range(self):
        """Test validation of odds outside valid range."""
        warnings = OddsValidator.validate_odds_value(50000, "fanduel", "h2h", "Lakers")
        assert len(warnings) > 0
        assert "out of valid range" in warnings[0]

        warnings = OddsValidator.validate_odds_value(-50000, "fanduel", "h2h", "Lakers")
        assert len(warnings) > 0

    def test_validate_odds_value_suspicious(self):
        """Test validation of suspicious odds near even."""
        warnings = OddsValidator.validate_odds_value(101, "fanduel", "h2h", "Lakers")
        assert len(warnings) > 0
        assert "Suspicious odds" in warnings[0]

    def test_validate_vig_normal(self):
        """Test vig validation for normal two-way market."""
        outcomes = [
            {"price": -110},
            {"price": -110},
        ]
        warnings = OddsValidator.validate_vig(outcomes, "fanduel", "h2h")
        assert len(warnings) == 0  # Normal vig (~4.5%)

    def test_validate_vig_too_low(self):
        """Test vig validation for suspiciously low vig (arbitrage)."""
        outcomes = [
            {"price": 110},
            {"price": 110},
        ]
        warnings = OddsValidator.validate_vig(outcomes, "fanduel", "h2h")
        assert len(warnings) > 0
        assert "too low" in warnings[0]

    def test_validate_vig_too_high(self):
        """Test vig validation for unreasonably high vig."""
        outcomes = [
            {"price": -200},
            {"price": -200},
        ]
        warnings = OddsValidator.validate_vig(outcomes, "fanduel", "h2h")
        assert len(warnings) > 0
        assert "too high" in warnings[0]

    def test_validate_vig_wrong_number_of_outcomes(self):
        """Test vig validation skips non-two-way markets."""
        outcomes = [
            {"price": -110},
        ]
        warnings = OddsValidator.validate_vig(outcomes, "fanduel", "h2h")
        assert len(warnings) == 0  # Should skip validation

        outcomes = [
            {"price": -110},
            {"price": -110},
            {"price": -110},
        ]
        warnings = OddsValidator.validate_vig(outcomes, "fanduel", "h2h")
        assert len(warnings) == 0  # Should skip validation

    def test_validate_line_movement_spreads(self):
        """Test spread line movement validation."""
        # Normal movement
        warnings = OddsValidator.validate_line_movement(-2.5, -3.0, "spreads", "fanduel")
        assert len(warnings) == 0

        # Excessive movement
        warnings = OddsValidator.validate_line_movement(-2.5, -15.0, "spreads", "fanduel")
        assert len(warnings) > 0
        assert "Excessive" in warnings[0]

    def test_validate_line_movement_totals(self):
        """Test total line movement validation."""
        # Normal movement
        warnings = OddsValidator.validate_line_movement(218.5, 220.0, "totals", "fanduel")
        assert len(warnings) == 0

        # Excessive movement
        warnings = OddsValidator.validate_line_movement(218.5, 245.0, "totals", "fanduel")
        assert len(warnings) > 0
        assert "Excessive" in warnings[0]

    def test_validate_line_movement_none_values(self):
        """Test line movement validation with None values."""
        warnings = OddsValidator.validate_line_movement(None, -3.0, "spreads", "fanduel")
        assert len(warnings) == 0  # Should skip

        warnings = OddsValidator.validate_line_movement(-2.5, None, "spreads", "fanduel")
        assert len(warnings) == 0  # Should skip

    def test_validate_event_data_valid(self):
        """Test validation of valid event data."""
        event_data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "commence_time": (datetime.now(UTC) + timedelta(hours=24)).isoformat(),
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        is_valid, warnings = OddsValidator.validate_event_data(event_data)
        assert is_valid
        assert len(warnings) == 0

    def test_validate_event_data_missing_fields(self):
        """Test validation of event data with missing fields."""
        event_data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            # Missing commence_time, home_team, away_team
        }

        is_valid, warnings = OddsValidator.validate_event_data(event_data)
        assert not is_valid
        assert len(warnings) >= 3  # Multiple missing fields

    def test_validate_event_data_old_commence_time(self):
        """Test validation of event with old commence time."""
        event_data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "commence_time": (datetime.now(UTC) - timedelta(days=2)).isoformat(),
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        is_valid, warnings = OddsValidator.validate_event_data(event_data)
        assert not is_valid
        assert any("past" in w for w in warnings)

    def test_validate_odds_snapshot_valid(self, sample_odds_data):
        """Test validation of valid odds snapshot."""
        is_valid, warnings = OddsValidator.validate_odds_snapshot(
            sample_odds_data, sample_odds_data["id"]
        )
        assert is_valid
        assert len(warnings) == 0

    def test_validate_odds_snapshot_empty_data(self):
        """Test validation of empty odds data."""
        is_valid, warnings = OddsValidator.validate_odds_snapshot([], "test123")
        assert not is_valid
        assert any("Empty" in w for w in warnings)

    def test_validate_odds_snapshot_no_bookmakers(self):
        """Test validation of snapshot with no bookmakers."""
        data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "commence_time": datetime.now(UTC).isoformat(),
            "home_team": "Lakers",
            "away_team": "Celtics",
            "bookmakers": [],
        }

        is_valid, warnings = OddsValidator.validate_odds_snapshot(data, "test123")
        assert not is_valid
        assert any("No bookmakers" in w for w in warnings)

    def test_validate_odds_snapshot_future_timestamp(self):
        """Test validation catches future timestamps."""
        data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "commence_time": datetime.now(UTC).isoformat(),
            "home_team": "Lakers",
            "away_team": "Celtics",
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Lakers", "price": -110},
                                {"name": "Celtics", "price": -110},
                            ],
                        }
                    ],
                }
            ],
        }

        is_valid, warnings = OddsValidator.validate_odds_snapshot(data, "test123")
        assert not is_valid
        assert any("Future timestamp" in w for w in warnings)
