"""Unit tests for game log fetcher."""

from datetime import date

import pytest
from odds_lambda.game_log_fetcher import (
    GameLogRecord,
    _parse_game_date,
    _row_to_record,
    _safe_int,
)


class TestSafeInt:
    def test_normal_int(self):
        assert _safe_int(42) == 42

    def test_float_truncates(self):
        assert _safe_int(3.0) == 3

    def test_none_returns_none(self):
        assert _safe_int(None) is None

    def test_empty_string_returns_none(self):
        assert _safe_int("") is None

    def test_string_number(self):
        assert _safe_int("10") == 10


class TestParseGameDate:
    def test_month_day_year_format(self):
        assert _parse_game_date("APR 13, 2025") == date(2025, 4, 13)

    def test_iso_format(self):
        assert _parse_game_date("2025-04-13") == date(2025, 4, 13)

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            _parse_game_date("13/04/2025")


class TestRowToRecord:
    """Test conversion of raw API row to GameLogRecord."""

    @pytest.fixture()
    def sample_row(self) -> list:
        return [
            "22024",  # SEASON_ID
            1610612738,  # TEAM_ID
            "BOS",  # TEAM_ABBREVIATION
            "Boston Celtics",  # TEAM_NAME
            "0022400061",  # GAME_ID
            "OCT 22, 2024",  # GAME_DATE
            "BOS vs. NYK",  # MATCHUP
            "W",  # WL
            240,  # MIN
            132,  # PTS
            48,  # FGM
            95,  # FGA
            0.505,  # FG_PCT
            29,  # FG3M
            61,  # FG3A
            0.475,  # FG3_PCT
            7,  # FTM
            8,  # FTA
            0.875,  # FT_PCT
            11,  # OREB
            29,  # DREB
            40,  # REB
            33,  # AST
            6,  # STL
            3,  # BLK
            3,  # TOV
            15,  # PF
            20,  # PLUS_MINUS
        ]

    def test_basic_conversion(self, sample_row: list):
        record = _row_to_record(sample_row, "2024-25")
        assert record.nba_game_id == "0022400061"
        assert record.team_id == 1610612738
        assert record.team_abbreviation == "BOS"
        assert record.team_name == "Boston Celtics"
        assert record.game_date == date(2024, 10, 22)
        assert record.matchup == "BOS vs. NYK"
        assert record.wl == "W"
        assert record.season == "2024-25"

    def test_box_score_stats(self, sample_row: list):
        record = _row_to_record(sample_row, "2024-25")
        assert record.pts == 132
        assert record.fgm == 48
        assert record.fga == 95
        assert record.fg3m == 29
        assert record.fg3a == 61
        assert record.ftm == 7
        assert record.fta == 8
        assert record.oreb == 11
        assert record.dreb == 29
        assert record.reb == 40
        assert record.ast == 33
        assert record.stl == 6
        assert record.blk == 3
        assert record.tov == 3
        assert record.pf == 15
        assert record.plus_minus == 20

    def test_nullable_wl(self, sample_row: list):
        sample_row[7] = None  # WL
        record = _row_to_record(sample_row, "2024-25")
        assert record.wl is None

    def test_nullable_stats(self, sample_row: list):
        sample_row[9] = None  # PTS
        record = _row_to_record(sample_row, "2024-25")
        assert record.pts is None

    def test_matchup_home(self, sample_row: list):
        """'vs.' in matchup indicates home team."""
        record = _row_to_record(sample_row, "2024-25")
        assert "vs." in record.matchup

    def test_matchup_away(self, sample_row: list):
        sample_row[6] = "BOS @ NYK"
        record = _row_to_record(sample_row, "2024-25")
        assert "@" in record.matchup


class TestGameLogRecord:
    def test_slots(self):
        """Verify slots optimization is active."""
        assert hasattr(GameLogRecord, "__slots__")
