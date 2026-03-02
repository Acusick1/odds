"""Tests for OddsPortal ingestion script — build_raw_data and build_event_id."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

# Import directly from script (it's on sys.path via the repo root)
from scripts.ingest_oddsportal import (
    _team_abbrev,
    build_event_id,
    build_raw_data,
)


def _make_bookmaker(
    name: str,
    *,
    num_outcomes: int = 2,
    home_opening: float = 2.0,
    away_opening: float = 1.8,
    draw_opening: float = 3.5,
    home_closing: float = 1.9,
    away_closing: float = 1.85,
    draw_closing: float = 3.4,
) -> dict:
    """Build a bookmaker dict matching OddsHarvester's output format."""
    histories = []
    openings = (
        [home_opening, draw_opening, away_opening]
        if num_outcomes == 3
        else [home_opening, away_opening]
    )
    closings = (
        [home_closing, draw_closing, away_closing]
        if num_outcomes == 3
        else [home_closing, away_closing]
    )

    for open_odds, close_odds in zip(openings, closings, strict=True):
        histories.append(
            {
                "opening_odds": {"timestamp": "2026-01-14T10:00:00", "odds": open_odds},
                "odds_history": [{"timestamp": "2026-01-15T14:30:00", "odds": close_odds}],
            }
        )

    return {"bookmaker_name": name, "1": "Home", "2": "Away", "odds_history_data": histories}


MATCH_DT = datetime(2026, 1, 15, 15, 0, 0, tzinfo=UTC)


class TestBuildRawData2Way:
    def test_produces_two_outcomes(self) -> None:
        bk = _make_bookmaker("bet365", num_outcomes=2)
        result = build_raw_data(
            [bk], "Team A", "Team B", use_opening=False, match_dt=MATCH_DT, num_outcomes=2
        )

        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        assert len(outcomes) == 2
        assert outcomes[0]["name"] == "Team A"
        assert outcomes[1]["name"] == "Team B"

    def test_opening_odds_used_when_requested(self) -> None:
        bk = _make_bookmaker("bet365", num_outcomes=2, home_opening=2.5, away_opening=1.5)
        result = build_raw_data(
            [bk], "Home", "Away", use_opening=True, match_dt=MATCH_DT, num_outcomes=2
        )

        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        # 2.5 decimal → (2.5-1)*100 = 150 American
        assert outcomes[0]["price"] == 150
        # 1.5 decimal → -100/(1.5-1) = -200 American
        assert outcomes[1]["price"] == -200

    def test_skips_bookmaker_without_history(self) -> None:
        bk = {"bookmaker_name": "broken", "odds_history_data": None}
        result = build_raw_data(
            [bk], "A", "B", use_opening=False, match_dt=MATCH_DT, num_outcomes=2
        )
        assert result is None

    def test_skips_bookmaker_with_too_few_entries(self) -> None:
        bk = {
            "bookmaker_name": "broken",
            "odds_history_data": [
                {
                    "opening_odds": {"timestamp": "2026-01-14T10:00:00", "odds": 2.0},
                    "odds_history": [],
                }
            ],
        }
        result = build_raw_data(
            [bk], "A", "B", use_opening=False, match_dt=MATCH_DT, num_outcomes=2
        )
        assert result is None


class TestBuildRawData3Way:
    def test_produces_three_outcomes_home_draw_away(self) -> None:
        bk = _make_bookmaker("Pinnacle", num_outcomes=3)
        result = build_raw_data(
            [bk], "Arsenal", "Chelsea", use_opening=False, match_dt=MATCH_DT, num_outcomes=3
        )

        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        assert len(outcomes) == 3
        assert outcomes[0]["name"] == "Arsenal"
        assert outcomes[1]["name"] == "Draw"
        assert outcomes[2]["name"] == "Chelsea"

    def test_opening_odds_3way(self) -> None:
        bk = _make_bookmaker(
            "Pinnacle",
            num_outcomes=3,
            home_opening=2.0,
            draw_opening=3.5,
            away_opening=4.0,
        )
        result = build_raw_data(
            [bk], "Arsenal", "Chelsea", use_opening=True, match_dt=MATCH_DT, num_outcomes=3
        )

        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        assert outcomes[0]["name"] == "Arsenal"
        assert outcomes[0]["price"] == 100  # 2.0 → +100
        assert outcomes[1]["name"] == "Draw"
        assert outcomes[1]["price"] == 250  # 3.5 → +250
        assert outcomes[2]["name"] == "Chelsea"
        assert outcomes[2]["price"] == 300  # 4.0 → +300

    def test_skips_bookmaker_with_only_2_entries_for_3way(self) -> None:
        """If odds_history_data has only 2 entries but num_outcomes=3, skip."""
        bk = _make_bookmaker("bet365", num_outcomes=2)  # Only 2 entries
        result = build_raw_data(
            [bk], "Arsenal", "Chelsea", use_opening=False, match_dt=MATCH_DT, num_outcomes=3
        )
        assert result is None

    def test_skips_when_draw_closing_missing(self) -> None:
        """If draw outcome has empty odds_history, bookmaker is skipped."""
        bk = _make_bookmaker("Pinnacle", num_outcomes=3)
        # Remove draw closing odds
        bk["odds_history_data"][1]["odds_history"] = []
        result = build_raw_data(
            [bk], "Arsenal", "Chelsea", use_opening=False, match_dt=MATCH_DT, num_outcomes=3
        )
        assert result is None

    def test_skips_when_draw_opening_missing(self) -> None:
        """If draw outcome has no opening_odds, bookmaker is skipped."""
        bk = _make_bookmaker("Pinnacle", num_outcomes=3)
        bk["odds_history_data"][1]["opening_odds"] = None
        result = build_raw_data(
            [bk], "Arsenal", "Chelsea", use_opening=True, match_dt=MATCH_DT, num_outcomes=3
        )
        assert result is None

    def test_market_key_is_h2h(self) -> None:
        bk = _make_bookmaker("Pinnacle", num_outcomes=3)
        result = build_raw_data(
            [bk], "Arsenal", "Chelsea", use_opening=False, match_dt=MATCH_DT, num_outcomes=3
        )
        assert result is not None
        assert result["bookmakers"][0]["markets"][0]["key"] == "h2h"

    def test_multiple_bookmakers(self) -> None:
        bk1 = _make_bookmaker("bet365", num_outcomes=3, home_closing=1.8)
        bk2 = _make_bookmaker("Pinnacle", num_outcomes=3, home_closing=1.9)
        result = build_raw_data(
            [bk1, bk2], "Arsenal", "Chelsea", use_opening=False, match_dt=MATCH_DT, num_outcomes=3
        )
        assert result is not None
        assert len(result["bookmakers"]) == 2
        assert result["bookmakers"][0]["key"] == "bet365"
        assert result["bookmakers"][1]["key"] == "pinnacle"


class TestTeamAbbrev:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Arsenal", "ARS"),
            ("Manchester United", "MANUNI"),
            ("Manchester City", "MANCIT"),
            ("Tottenham Hotspur", "TOTHOT"),
            ("Nottingham Forest", "NOTFOR"),
            ("Crystal Palace", "CRYPAL"),
            ("West Ham United", "WESUNI"),
            ("Brighton and Hove Albion", "BRIALB"),
        ],
    )
    def test_team_abbrev(self, name: str, expected: str) -> None:
        assert _team_abbrev(name) == expected

    def test_manchester_derby_unique(self) -> None:
        assert _team_abbrev("Manchester United") != _team_abbrev("Manchester City")


class TestBuildEventId:
    def test_with_canonical_to_abbrev(self) -> None:
        mapping = {"Boston Celtics": "BOS", "Los Angeles Lakers": "LAL"}
        result = build_event_id(
            "2024-2025",
            "Boston Celtics",
            "Los Angeles Lakers",
            datetime(2025, 1, 15).date(),
            canonical_to_abbrev=mapping,
        )
        assert result == "op_2024-2025_BOS_LAL_2025-01-15"

    def test_without_mapping_uses_team_abbrev(self) -> None:
        result = build_event_id(
            "2024-2025",
            "Manchester United",
            "Manchester City",
            datetime(2025, 1, 15).date(),
        )
        assert result == "op_2024-2025_MANUNI_MANCIT_2025-01-15"

    def test_fallback_when_team_not_in_mapping(self) -> None:
        mapping = {"Boston Celtics": "BOS"}
        result = build_event_id(
            "2024-2025",
            "Boston Celtics",
            "Arsenal",
            datetime(2025, 1, 15).date(),
            canonical_to_abbrev=mapping,
        )
        assert result == "op_2024-2025_BOS_ARS_2025-01-15"
