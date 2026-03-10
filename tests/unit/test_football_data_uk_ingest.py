"""Tests for football-data.co.uk ingestion script."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scripts.ingest_football_data_uk import (
    _build_snapshot_raw_data,
    _extract_h2h_odds,
    build_event_id,
    normalize_team,
)


class TestNormalizeTeam:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Man United", "Manchester United"),
            ("Man City", "Manchester City"),
            ("Nott'm Forest", "Nottingham Forest"),
            ("Wolves", "Wolverhampton Wanderers"),
            ("Tottenham", "Tottenham Hotspur"),
            ("West Ham", "West Ham United"),
            ("Brighton", "Brighton and Hove Albion"),
            ("Newcastle", "Newcastle United"),
            ("Leicester", "Leicester City"),
            ("Leeds", "Leeds United"),
            ("Sheffield United", "Sheffield Utd"),
            ("West Brom", "West Bromwich Albion"),
        ],
    )
    def test_mapped_teams(self, raw: str, expected: str) -> None:
        assert normalize_team(raw) == expected

    @pytest.mark.parametrize(
        "name",
        ["Arsenal", "Chelsea", "Liverpool", "Everton", "Crystal Palace", "Brentford"],
    )
    def test_passthrough_teams(self, name: str) -> None:
        assert normalize_team(name) == name


class TestExtractH2hOdds:
    def test_extracts_three_way_odds(self) -> None:
        row = {"PSH": "2.50", "PSD": "3.40", "PSA": "2.80"}
        result = _extract_h2h_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")

        assert result is not None
        assert result["key"] == "pinnacle"
        assert result["markets"][0]["key"] == "h2h"
        outcomes = result["markets"][0]["outcomes"]
        assert len(outcomes) == 3
        assert outcomes[0]["name"] == "Arsenal"
        assert outcomes[0]["price"] == 150  # 2.50 decimal → +150
        assert outcomes[1]["name"] == "Draw"
        assert outcomes[1]["price"] == 240  # 3.40 decimal → +240
        assert outcomes[2]["name"] == "Chelsea"
        assert outcomes[2]["price"] == 180  # 2.80 decimal → +180

    def test_returns_none_when_column_missing(self) -> None:
        row = {"PSH": "2.50", "PSD": "3.40"}  # PSA missing
        result = _extract_h2h_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")
        assert result is None

    def test_returns_none_when_value_empty(self) -> None:
        row = {"PSH": "2.50", "PSD": "", "PSA": "2.80"}
        result = _extract_h2h_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")
        assert result is None

    def test_returns_none_when_odds_at_or_below_one(self) -> None:
        row = {"PSH": "1.00", "PSD": "3.40", "PSA": "2.80"}
        result = _extract_h2h_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")
        assert result is None

    def test_favourite_american_odds(self) -> None:
        row = {"B365H": "1.50", "B365D": "4.00", "B365A": "6.00"}
        result = _extract_h2h_odds(row, "B365", "bet365", "Liverpool", "Norwich")

        assert result is not None
        outcomes = result["markets"][0]["outcomes"]
        assert outcomes[0]["price"] == -200  # 1.50 decimal → -200
        assert outcomes[1]["price"] == 300  # 4.00 → +300
        assert outcomes[2]["price"] == 500  # 6.00 → +500

    def test_closing_prefix(self) -> None:
        row = {"PSCH": "2.10", "PSCD": "3.50", "PSCA": "3.20"}
        result = _extract_h2h_odds(row, "PSC", "pinnacle", "Arsenal", "Chelsea")

        assert result is not None
        assert result["key"] == "pinnacle"
        assert len(result["markets"][0]["outcomes"]) == 3


class TestBuildSnapshotRawData:
    def _make_row(self) -> dict[str, str]:
        return {
            # Opening odds
            "PSH": "2.50",
            "PSD": "3.40",
            "PSA": "2.80",
            "B365H": "2.40",
            "B365D": "3.30",
            "B365A": "2.90",
            # Closing odds
            "PSCH": "2.60",
            "PSCD": "3.30",
            "PSCA": "2.70",
            "B365CH": "2.45",
            "B365CD": "3.25",
            "B365CA": "2.85",
        }

    def test_opening_uses_non_c_columns(self) -> None:
        row = self._make_row()
        result = _build_snapshot_raw_data(row, "Arsenal", "Chelsea", use_closing=False)

        assert result is not None
        assert result["source"] == "football_data_uk"
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "pinnacle" in keys
        assert "bet365" in keys

        pinnacle = next(bk for bk in result["bookmakers"] if bk["key"] == "pinnacle")
        home_price = pinnacle["markets"][0]["outcomes"][0]["price"]
        assert home_price == 150  # 2.50 → +150

    def test_closing_uses_c_columns(self) -> None:
        row = self._make_row()
        result = _build_snapshot_raw_data(row, "Arsenal", "Chelsea", use_closing=True)

        assert result is not None
        pinnacle = next(bk for bk in result["bookmakers"] if bk["key"] == "pinnacle")
        home_price = pinnacle["markets"][0]["outcomes"][0]["price"]
        assert home_price == 160  # 2.60 → +160

    def test_returns_none_when_no_bookmakers(self) -> None:
        row = {"HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}
        result = _build_snapshot_raw_data(row, "Arsenal", "Chelsea", use_closing=False)
        assert result is None

    def test_includes_aggregates(self) -> None:
        row = {
            "MaxH": "2.60",
            "MaxD": "3.50",
            "MaxA": "3.00",
            "AvgH": "2.45",
            "AvgD": "3.30",
            "AvgA": "2.85",
        }
        result = _build_snapshot_raw_data(row, "Arsenal", "Chelsea", use_closing=False)

        assert result is not None
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "market_max" in keys
        assert "market_avg" in keys

    def test_closing_aggregates(self) -> None:
        row = {
            "MaxCH": "2.60",
            "MaxCD": "3.50",
            "MaxCA": "3.00",
            "AvgCH": "2.45",
            "AvgCD": "3.30",
            "AvgCA": "2.85",
        }
        result = _build_snapshot_raw_data(row, "Arsenal", "Chelsea", use_closing=True)

        assert result is not None
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "market_max" in keys
        assert "market_avg" in keys

    def test_betfair_exchange_prefix(self) -> None:
        row = {"BFEH": "2.50", "BFED": "3.40", "BFEA": "2.80"}
        result = _build_snapshot_raw_data(row, "Arsenal", "Chelsea", use_closing=False)

        assert result is not None
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "betfair_exchange" in keys

    def test_betfair_sportsbook_bf_prefix(self) -> None:
        row = {"BFH": "2.50", "BFD": "3.40", "BFA": "2.80"}
        result = _build_snapshot_raw_data(row, "Arsenal", "Chelsea", use_closing=False)

        assert result is not None
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "betfair_sportsbook" in keys


class TestBuildEventId:
    def test_deterministic_format(self) -> None:
        dt = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
        result = build_event_id("2024-2025", "Manchester United", "Manchester City", dt)
        assert result == "fduk_2024-2025_MANUNI_MANCIT_2025-01-15"

    def test_single_word_team(self) -> None:
        dt = datetime(2025, 3, 8, 15, 0, tzinfo=UTC)
        result = build_event_id("2024-2025", "Arsenal", "Chelsea", dt)
        assert result == "fduk_2024-2025_ARS_CHE_2025-03-08"

    def test_uses_fduk_prefix(self) -> None:
        dt = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
        result = build_event_id("2024-2025", "Arsenal", "Chelsea", dt)
        assert result.startswith("fduk_")

    def test_different_dates_produce_different_ids(self) -> None:
        dt1 = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
        dt2 = datetime(2025, 3, 15, 15, 0, tzinfo=UTC)
        id1 = build_event_id("2024-2025", "Arsenal", "Chelsea", dt1)
        id2 = build_event_id("2024-2025", "Arsenal", "Chelsea", dt2)
        assert id1 != id2
