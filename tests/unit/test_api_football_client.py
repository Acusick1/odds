"""Tests for API-Football client: name normalization, response parsing, season inference."""

from datetime import UTC, datetime

from odds_lambda.api_football_client import (
    _infer_season,
    _normalize_api_football_name,
    parse_lineup_response,
)


class TestNormalizeApiFootballName:
    def test_direct_alias(self) -> None:
        assert _normalize_api_football_name("Manchester United") == "Manchester Utd"

    def test_falls_through_to_team_module(self) -> None:
        assert _normalize_api_football_name("Wolverhampton Wanderers") == "Wolves"

    def test_already_canonical(self) -> None:
        assert _normalize_api_football_name("Arsenal") == "Arsenal"

    def test_whitespace_cleanup(self) -> None:
        assert _normalize_api_football_name("  Newcastle  ") == "Newcastle"


class TestInferSeason:
    def test_august_onwards(self) -> None:
        assert _infer_season(datetime(2025, 8, 16, tzinfo=UTC)) == 2025

    def test_january(self) -> None:
        assert _infer_season(datetime(2026, 1, 15, tzinfo=UTC)) == 2025

    def test_may(self) -> None:
        assert _infer_season(datetime(2026, 5, 25, tzinfo=UTC)) == 2025


class TestParseLineupResponse:
    def test_parses_full_response(self) -> None:
        raw = {
            "team": {"id": 42, "name": "Arsenal"},
            "coach": {"id": 1, "name": "Mikel Arteta"},
            "formation": "4-3-3",
            "startXI": [
                {"player": {"id": 100, "name": "Saka", "number": 7, "pos": "M", "grid": "3:4"}},
                {"player": {"id": 101, "name": "Rice", "number": 41, "pos": "M", "grid": "3:2"}},
            ],
            "substitutes": [
                {"player": {"id": 200, "name": "Nketiah", "number": 14, "pos": "F", "grid": None}},
            ],
        }
        result = parse_lineup_response(raw)

        assert result["team_name"] == "Arsenal"
        assert result["team_id"] == 42
        assert result["formation"] == "4-3-3"
        assert result["coach"] == {"id": 1, "name": "Mikel Arteta"}
        assert len(result["start_xi"]) == 2
        assert result["start_xi"][0]["name"] == "Saka"
        assert result["start_xi"][0]["pos"] == "M"
        assert len(result["substitutes"]) == 1

    def test_handles_missing_coach(self) -> None:
        raw = {
            "team": {"id": 42, "name": "Arsenal"},
            "coach": {},
            "formation": "4-4-2",
            "startXI": [],
            "substitutes": [],
        }
        result = parse_lineup_response(raw)
        assert result["coach"] is None
