"""Tests for football-data.co.uk ingestion script."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from scripts.ingest_football_data_uk import (
    SEASONS,
    UK_TZ,
    _build_envelope,
    _extract_three_way_odds,
    build_event_id,
    normalize_team,
    opening_snapshot_time,
    parse_kickoff,
)


class TestNormalizeTeam:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Man United", "Manchester Utd"),
            ("Man City", "Manchester City"),
            ("Nott'm Forest", "Nottingham"),
            ("Sheffield United", "Sheffield Utd"),
        ],
    )
    def test_mapped_teams(self, raw: str, expected: str) -> None:
        assert normalize_team(raw) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "Arsenal",
            "Chelsea",
            "Liverpool",
            "Everton",
            "Crystal Palace",
            "Brentford",
            "Wolves",
            "Tottenham",
            "West Ham",
            "Brighton",
            "Newcastle",
            "Leicester",
            "Leeds",
            "West Brom",
        ],
    )
    def test_passthrough_teams(self, name: str) -> None:
        assert normalize_team(name) == name


class TestExtractH2hOdds:
    def test_extracts_three_way_odds(self) -> None:
        row = {"PSH": "2.50", "PSD": "3.40", "PSA": "2.80"}
        result = _extract_three_way_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")

        assert result is not None
        assert result["key"] == "pinnacle"
        assert result["markets"][0]["key"] == "1x2"
        outcomes = result["markets"][0]["outcomes"]
        assert len(outcomes) == 3
        assert outcomes[0]["name"] == "Arsenal"
        assert outcomes[0]["price_decimal"] == 2.5
        assert outcomes[1]["name"] == "Draw"
        assert outcomes[1]["price_decimal"] == 3.4
        assert outcomes[2]["name"] == "Chelsea"
        assert outcomes[2]["price_decimal"] == 2.8

    def test_returns_none_when_column_missing(self) -> None:
        row = {"PSH": "2.50", "PSD": "3.40"}  # PSA missing
        result = _extract_three_way_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")
        assert result is None

    def test_returns_none_when_value_empty(self) -> None:
        row = {"PSH": "2.50", "PSD": "", "PSA": "2.80"}
        result = _extract_three_way_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")
        assert result is None

    def test_returns_none_when_odds_at_or_below_one(self) -> None:
        row = {"PSH": "1.00", "PSD": "3.40", "PSA": "2.80"}
        result = _extract_three_way_odds(row, "PS", "pinnacle", "Arsenal", "Chelsea")
        assert result is None

    def test_favourite_decimal_odds(self) -> None:
        row = {"B365H": "1.50", "B365D": "4.00", "B365A": "6.00"}
        result = _extract_three_way_odds(row, "B365", "bet365", "Liverpool", "Norwich")

        assert result is not None
        outcomes = result["markets"][0]["outcomes"]
        assert outcomes[0]["price_decimal"] == 1.5
        assert outcomes[1]["price_decimal"] == 4.0
        assert outcomes[2]["price_decimal"] == 6.0

    def test_closing_prefix(self) -> None:
        row = {"PSCH": "2.10", "PSCD": "3.50", "PSCA": "3.20"}
        result = _extract_three_way_odds(row, "PSC", "pinnacle", "Arsenal", "Chelsea")

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
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)

        assert result is not None
        assert result["source"] == "football_data_uk"
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "pinnacle" in keys
        assert "bet365" in keys

        pinnacle = next(bk for bk in result["bookmakers"] if bk["key"] == "pinnacle")
        home_price = pinnacle["markets"][0]["outcomes"][0]["price_decimal"]
        assert home_price == 2.5

    def test_closing_uses_c_columns(self) -> None:
        row = self._make_row()
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=True)

        assert result is not None
        pinnacle = next(bk for bk in result["bookmakers"] if bk["key"] == "pinnacle")
        home_price = pinnacle["markets"][0]["outcomes"][0]["price_decimal"]
        assert home_price == 2.6

    def test_returns_none_when_no_bookmakers(self) -> None:
        row = {"HomeTeam": "Arsenal", "AwayTeam": "Chelsea"}
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)
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
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)

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
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=True)

        assert result is not None
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "market_max" in keys
        assert "market_avg" in keys

    def test_betfair_exchange_prefix(self) -> None:
        row = {"BFEH": "2.50", "BFED": "3.40", "BFEA": "2.80"}
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)

        assert result is not None
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "betfair_exchange" in keys

    def test_mixed_bookmaker_availability(self) -> None:
        """Bookmakers with opening but no closing columns (e.g. IW, LB, VC in older seasons)."""
        row = {
            # PS has both opening and closing
            "PSH": "2.50",
            "PSD": "3.40",
            "PSA": "2.80",
            "PSCH": "2.60",
            "PSCD": "3.30",
            "PSCA": "2.70",
            # IW has opening only (no closing columns in older CSVs)
            "IWH": "2.45",
            "IWD": "3.35",
            "IWA": "2.85",
            # VC has opening only
            "VCH": "2.40",
            "VCD": "3.30",
            "VCA": "2.90",
        }
        opening = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)
        assert opening is not None
        opening_keys = {bk["key"] for bk in opening["bookmakers"]}
        assert "pinnacle" in opening_keys
        assert "interwetten" in opening_keys
        assert "betvictor" in opening_keys

        closing = _build_envelope(row, "Arsenal", "Chelsea", use_closing=True)
        assert closing is not None
        closing_keys = {bk["key"] for bk in closing["bookmakers"]}
        assert "pinnacle" in closing_keys
        assert "interwetten" not in closing_keys
        assert "betvictor" not in closing_keys

    def test_betfair_sportsbook_bf_prefix(self) -> None:
        row = {"BFH": "2.50", "BFD": "3.40", "BFA": "2.80"}
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)

        assert result is not None
        keys = {bk["key"] for bk in result["bookmakers"]}
        assert "betfair_sportsbook" in keys


class TestParseKickoff:
    def test_winter_kickoff_is_gmt(self) -> None:
        kickoff = parse_kickoff({"Date": "01/01/2025", "Time": "15:00"})
        assert kickoff.commence_time == datetime(2025, 1, 1, 15, 0, tzinfo=UTC)
        assert not kickoff.time_assumed

    def test_summer_kickoff_converts_bst_to_utc(self) -> None:
        kickoff = parse_kickoff({"Date": "16/08/2024", "Time": "20:00"})
        assert kickoff.commence_time == datetime(2024, 8, 16, 19, 0, tzinfo=UTC)

    def test_two_digit_year(self) -> None:
        kickoff = parse_kickoff({"Date": "13/08/16", "Time": "12:30"})
        assert kickoff.commence_time == datetime(2016, 8, 13, 11, 30, tzinfo=UTC)

    @pytest.mark.parametrize("row", [{"Date": "17/08/2024"}, {"Date": "17/08/2024", "Time": ""}])
    def test_missing_time_assumes_1500_local(self, row: dict[str, str]) -> None:
        kickoff = parse_kickoff(row)
        assert kickoff.commence_time == datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
        assert kickoff.time_assumed

    def test_result_is_utc(self) -> None:
        assert parse_kickoff({"Date": "01/01/2025", "Time": "15:00"}).commence_time.tzinfo == UTC


class TestOpeningSnapshotTime:
    @pytest.mark.parametrize(
        ("kickoff_local", "expected_utc"),
        [
            # Saturday 12:30 BST: Friday 17:00 BST deadline
            ("2024-08-17 12:30", datetime(2024, 8, 16, 16, 0, tzinfo=UTC)),
            # Sunday and Monday kick-offs also use the Friday deadline (GMT)
            ("2025-01-12 16:30", datetime(2025, 1, 10, 17, 0, tzinfo=UTC)),
            ("2025-01-13 20:00", datetime(2025, 1, 10, 17, 0, tzinfo=UTC)),
            # Wednesday evening: Tuesday 13:00 GMT deadline
            ("2025-01-15 19:45", datetime(2025, 1, 14, 13, 0, tzinfo=UTC)),
            # Tuesday evening: same-day Tuesday deadline
            ("2025-01-14 20:00", datetime(2025, 1, 14, 13, 0, tzinfo=UTC)),
            # Friday evening: same-day Friday deadline
            ("2025-01-10 20:00", datetime(2025, 1, 10, 17, 0, tzinfo=UTC)),
        ],
    )
    def test_deadline_by_kickoff_weekday(self, kickoff_local: str, expected_utc: datetime) -> None:
        local = datetime.strptime(kickoff_local, "%Y-%m-%d %H:%M").replace(tzinfo=UK_TZ)
        assert opening_snapshot_time(local.astimezone(UTC)) == expected_utc

    def test_deadline_at_or_after_kickoff_has_no_opening_time(self) -> None:
        friday_1500 = datetime(2025, 1, 10, 15, 0, tzinfo=UK_TZ).astimezone(UTC)
        boxing_day = datetime(2023, 12, 26, 12, 30, tzinfo=UK_TZ).astimezone(UTC)
        friday_1700 = datetime(2025, 1, 10, 17, 0, tzinfo=UK_TZ).astimezone(UTC)
        assert opening_snapshot_time(friday_1500) is None
        assert opening_snapshot_time(boxing_day) is None
        assert opening_snapshot_time(friday_1700) is None

    def test_deadline_follows_uk_clock_across_march_change(self) -> None:
        # Saturday 2025-03-29 15:00 GMT; Friday 17:00 GMT
        before = datetime(2025, 3, 29, 15, 0, tzinfo=UK_TZ).astimezone(UTC)
        assert opening_snapshot_time(before) == datetime(2025, 3, 28, 17, 0, tzinfo=UTC)
        # Sunday 2025-03-30 (clocks went forward at 01:00) 16:30 BST; Friday still 17:00 GMT
        after = datetime(2025, 3, 30, 16, 30, tzinfo=UK_TZ).astimezone(UTC)
        assert opening_snapshot_time(after) == datetime(2025, 3, 28, 17, 0, tzinfo=UTC)
        # Saturday 2025-04-05 15:00 BST; Friday 17:00 BST is 16:00Z
        bst = datetime(2025, 4, 5, 15, 0, tzinfo=UK_TZ).astimezone(UTC)
        assert opening_snapshot_time(bst) == datetime(2025, 4, 4, 16, 0, tzinfo=UTC)

    def test_opening_precedes_kickoff(self) -> None:
        start = datetime(2024, 8, 1, tzinfo=UTC)
        for hours in range(0, 24 * 120, 5):
            kickoff = start + timedelta(hours=hours)
            opening = opening_snapshot_time(kickoff)
            assert opening is None or opening < kickoff


class TestSnapshotTimingMetadata:
    def test_opening_and_closing_record_basis(self) -> None:
        row = {"B365H": "2.4", "B365D": "3.3", "B365A": "2.9"}
        opening = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)
        assert opening is not None
        assert opening["timing"] == {
            "basis": "fduk_collection_deadline",
            "kickoff_time_assumed": False,
        }
        closing_row = {"B365CH": "2.4", "B365CD": "3.3", "B365CA": "2.9"}
        closing = _build_envelope(
            closing_row, "Arsenal", "Chelsea", use_closing=True, kickoff_time_assumed=True
        )
        assert closing is not None
        assert closing["timing"] == {"basis": "kickoff", "kickoff_time_assumed": True}


class TestRetailBookmakerPrefixes:
    @pytest.mark.parametrize(
        ("prefix", "key"),
        [
            ("BV", "betvictor"),
            ("BFD", "betfred"),
            ("PP", "paddypower"),
            ("SKB", "skybet"),
            ("BMGM", "betmgm"),
            ("CL", "coral"),
        ],
    )
    def test_opening_and_closing(self, prefix: str, key: str) -> None:
        row = {
            f"{prefix}H": "2.1",
            f"{prefix}D": "3.4",
            f"{prefix}A": "3.6",
            f"{prefix}CH": "2.0",
            f"{prefix}CD": "3.5",
            f"{prefix}CA": "3.7",
        }
        for use_closing in (False, True):
            result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=use_closing)
            assert result is not None
            assert [bk["key"] for bk in result["bookmakers"]] == [key]


class TestDuplicateBookmakerKeys:
    def test_betvictor_columns_collapse_to_one_bookmaker(self) -> None:
        row = {
            "VCH": "2.0", "VCD": "3.0", "VCA": "4.0",
            "BVH": "2.1", "BVD": "3.1", "BVA": "4.1",
            "MaxH": "2.5", "MaxD": "3.5", "MaxA": "4.5",
            "BbMxH": "2.6", "BbMxD": "3.6", "BbMxA": "4.6",
        }  # fmt: skip
        result = _build_envelope(row, "Arsenal", "Chelsea", use_closing=False)
        assert result is not None
        keys = [bk["key"] for bk in result["bookmakers"]]
        assert sorted(keys) == ["betvictor", "market_max"]


class TestSeasons:
    def test_2026_2027_registered(self) -> None:
        assert SEASONS["2026-2027"] == "2627"


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
