"""Unit tests for PBPStats fetcher."""

from unittest.mock import MagicMock, patch

import pytest
from odds_lambda.pbpstats_fetcher import (
    PlayerSeasonRecord,
    _parse_player,
    _safe_float,
    _safe_int,
    convert_name,
    fetch_player_season_stats,
)


class TestConvertName:
    def test_simple_name(self):
        assert convert_name("Mikal Bridges") == "Bridges, Mikal"

    def test_suffix_jr(self):
        assert convert_name("Michael Porter Jr.") == "Porter Jr., Michael"

    def test_suffix_iii(self):
        assert convert_name("Trey Murphy III") == "Murphy III, Trey"

    def test_suffix_ii(self):
        assert convert_name("Derrick White II") == "White II, Derrick"

    def test_suffix_iv(self):
        assert convert_name("Some Player IV") == "Player IV, Some"

    def test_single_word_name(self):
        assert convert_name("Nene") == "Nene"

    def test_hyphenated_first_name(self):
        assert convert_name("Karl-Anthony Towns") == "Towns, Karl-Anthony"

    def test_hyphenated_last_name(self):
        assert convert_name("Shai Gilgeous-Alexander") == "Gilgeous-Alexander, Shai"

    def test_suffix_sr(self):
        assert convert_name("Gary Trent Sr.") == "Trent Sr., Gary"

    def test_empty_string(self):
        assert convert_name("") == ""

    def test_two_word_with_suffix(self):
        """Two-word name where second word is suffix — treat as regular name."""
        assert convert_name("A Jr.") == "Jr., A"


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int_coerced(self):
        assert _safe_float(42) == 42.0

    def test_string_number(self):
        assert _safe_float("1.5") == 1.5

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_invalid_string_returns_none(self):
        assert _safe_float("abc") is None


class TestSafeInt:
    def test_normal_int(self):
        assert _safe_int(42) == 42

    def test_float_truncates(self):
        assert _safe_int(3.7) == 3

    def test_none_returns_none(self):
        assert _safe_int(None) is None

    def test_string_number(self):
        assert _safe_int("10") == 10

    def test_invalid_string_returns_none(self):
        assert _safe_int("abc") is None


class TestParsePlayer:
    @pytest.fixture()
    def sample_player(self) -> dict:
        return {
            "EntityId": "1628969",
            "Name": "Mikal Bridges",
            "TeamId": "1610612752",
            "TeamAbbreviation": "NYK",
            "Minutes": 3036,
            "GamesPlayed": 82,
            "OnOffRtg": 120.558,
            "OnDefRtg": 114.547,
            "Usage": 19.714,
            "TsPct": 0.585,
            "EfgPct": 0.570,
            "Assists": 306,
            "Turnovers": 132,
            "Rebounds": 259,
            "Steals": 78,
            "Blocks": 43,
            "Points": 1444,
            "PlusMinus": 334,
        }

    def test_basic_identity(self, sample_player: dict):
        record = _parse_player(sample_player, "2024-25")
        assert record.player_id == 1628969
        assert record.player_name == "Bridges, Mikal"
        assert record.team_id == 1610612752
        assert record.team_abbreviation == "NYK"
        assert record.season == "2024-25"

    def test_entity_id_string_to_int(self, sample_player: dict):
        record = _parse_player(sample_player, "2024-25")
        assert isinstance(record.player_id, int)
        assert isinstance(record.team_id, int)

    def test_stat_fields(self, sample_player: dict):
        record = _parse_player(sample_player, "2024-25")
        assert record.minutes == 3036.0
        assert record.games_played == 82
        assert record.on_off_rtg == 120.558
        assert record.on_def_rtg == 114.547
        assert record.usage == 19.714
        assert record.ts_pct == 0.585
        assert record.efg_pct == 0.570
        assert record.assists == 306
        assert record.turnovers == 132
        assert record.rebounds == 259
        assert record.steals == 78
        assert record.blocks == 43
        assert record.points == 1444
        assert record.plus_minus == 334.0

    def test_missing_advanced_stat(self, sample_player: dict):
        del sample_player["OnOffRtg"]
        del sample_player["Usage"]
        record = _parse_player(sample_player, "2024-25")
        assert record.on_off_rtg is None
        assert record.usage is None

    def test_name_with_suffix(self, sample_player: dict):
        sample_player["Name"] = "Michael Porter Jr."
        record = _parse_player(sample_player, "2024-25")
        assert record.player_name == "Porter Jr., Michael"


class TestFetchPlayerSeasonStats:
    @patch("odds_lambda.pbpstats_fetcher.requests.get")
    def test_successful_fetch(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "multi_row_table_data": [
                {
                    "EntityId": "1",
                    "Name": "Test Player",
                    "TeamId": "100",
                    "TeamAbbreviation": "TST",
                    "Minutes": 100,
                    "GamesPlayed": 5,
                    "OnOffRtg": 110.0,
                    "OnDefRtg": 105.0,
                    "Usage": 20.0,
                    "TsPct": 0.55,
                    "EfgPct": 0.50,
                    "Assists": 10,
                    "Turnovers": 5,
                    "Rebounds": 20,
                    "Steals": 3,
                    "Blocks": 2,
                    "Points": 50,
                    "PlusMinus": 10,
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        records = fetch_player_season_stats("2024-25")
        assert len(records) == 1
        assert records[0].player_name == "Player, Test"
        assert records[0].season == "2024-25"

    @patch("odds_lambda.pbpstats_fetcher.requests.get")
    def test_empty_response(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {"multi_row_table_data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        records = fetch_player_season_stats("2024-25")
        assert records == []

    @patch("odds_lambda.pbpstats_fetcher.requests.get")
    def test_http_error_raises(self, mock_get: MagicMock):
        from requests.exceptions import HTTPError

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        with pytest.raises(HTTPError):
            fetch_player_season_stats("2024-25")

    @patch("odds_lambda.pbpstats_fetcher.requests.get")
    def test_malformed_row_skipped(self, mock_get: MagicMock):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "multi_row_table_data": [
                {"Name": "Bad Player"},  # Missing required fields
                {
                    "EntityId": "1",
                    "Name": "Good Player",
                    "TeamId": "100",
                    "TeamAbbreviation": "TST",
                    "Minutes": 100,
                    "GamesPlayed": 5,
                    "Assists": 10,
                    "Turnovers": 5,
                    "Rebounds": 20,
                    "Steals": 3,
                    "Blocks": 2,
                    "Points": 50,
                    "PlusMinus": 10,
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        records = fetch_player_season_stats("2024-25")
        assert len(records) == 1
        assert records[0].player_name == "Player, Good"


class TestPlayerSeasonRecord:
    def test_slots(self):
        assert hasattr(PlayerSeasonRecord, "__slots__")
