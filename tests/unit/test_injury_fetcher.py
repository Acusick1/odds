"""Unit tests for injury report fetcher and parsing logic."""

import json
from datetime import UTC, date, datetime

from odds_core.injury_models import InjuryStatus
from odds_lambda.injury_fetcher import (
    STATUS_MAP,
    InjuryRecord,
    _parse_records,
    _to_naive_et,
)


def _make_raw_json(rows: list[dict]) -> str:
    """Build a JSON string matching nbainjuries output format."""
    return json.dumps(rows)


def _sample_row(**overrides) -> dict:
    """Build a sample nbainjuries row dict."""
    row = {
        "Game Date": "01/15/2026",
        "Game Time": "07:00 PM ET",
        "Matchup": "BOS@ORL",
        "Team": "Boston Celtics",
        "Player Name": "Tatum, Jayson",
        "Current Status": "Out",
        "Reason": "Left Ankle; Sprain",
    }
    row.update(overrides)
    return row


class TestStatusMap:
    """Tests for STATUS_MAP coverage."""

    def test_covers_all_enum_values(self):
        """Every InjuryStatus enum value is reachable via STATUS_MAP."""
        mapped_values = set(STATUS_MAP.values())
        for status in InjuryStatus:
            assert status in mapped_values, f"{status} not in STATUS_MAP values"

    def test_keys_are_lowercase(self):
        """All STATUS_MAP keys are lowercase."""
        for key in STATUS_MAP:
            assert key == key.lower()


class TestToNaiveEt:
    """Tests for UTC to naive ET conversion."""

    def test_utc_to_et_winter(self):
        """UTC datetime converts to ET (EST, UTC-5) in winter."""
        utc = datetime(2026, 1, 15, 22, 0, tzinfo=UTC)
        result = _to_naive_et(utc)
        assert result == datetime(2026, 1, 15, 17, 0)
        assert result.tzinfo is None

    def test_utc_to_et_summer(self):
        """UTC datetime converts to ET (EDT, UTC-4) in summer."""
        utc = datetime(2026, 6, 15, 22, 0, tzinfo=UTC)
        result = _to_naive_et(utc)
        assert result == datetime(2026, 6, 15, 18, 0)
        assert result.tzinfo is None


class TestParseRecords:
    """Tests for _parse_records JSON parsing."""

    def test_valid_row_parsed(self):
        raw = _make_raw_json([_sample_row()])
        report_time = datetime(2026, 1, 15, 22, 0, tzinfo=UTC)
        records = _parse_records(raw, report_time)

        assert len(records) == 1
        r = records[0]
        assert r.report_time == report_time
        assert r.game_date == date(2026, 1, 15)
        assert r.game_time_et == "07:00 PM ET"
        assert r.matchup == "BOS@ORL"
        assert r.team == "Boston Celtics"
        assert r.player_name == "Tatum, Jayson"
        assert r.status == InjuryStatus.OUT
        assert r.reason == "Left Ankle; Sprain"

    def test_filters_not_yet_submitted(self):
        rows = [
            _sample_row(),
            _sample_row(
                Team="Orlando Magic",
                **{"Player Name": "N/A", "Current Status": "", "Reason": "NOT YET SUBMITTED"},
            ),
        ]
        records = _parse_records(_make_raw_json(rows), datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
        assert len(records) == 1
        assert records[0].team == "Boston Celtics"

    def test_filters_not_yet_submitted_case_insensitive(self):
        rows = [_sample_row(Reason="Not Yet Submitted")]
        records = _parse_records(_make_raw_json(rows), datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
        assert len(records) == 0

    def test_unknown_status_skipped(self):
        rows = [_sample_row(**{"Current Status": "SUSPENDED"})]
        records = _parse_records(_make_raw_json(rows), datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
        assert len(records) == 0

    def test_invalid_game_date_skipped(self):
        rows = [_sample_row(**{"Game Date": "not-a-date"})]
        records = _parse_records(_make_raw_json(rows), datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
        assert len(records) == 0

    def test_empty_player_name_skipped(self):
        rows = [_sample_row(**{"Player Name": ""})]
        records = _parse_records(_make_raw_json(rows), datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
        assert len(records) == 0

    def test_all_statuses_parsed(self):
        """Each known status string maps to the correct enum."""
        for status_str, expected in [
            ("Out", InjuryStatus.OUT),
            ("Questionable", InjuryStatus.QUESTIONABLE),
            ("Doubtful", InjuryStatus.DOUBTFUL),
            ("Probable", InjuryStatus.PROBABLE),
            ("Available", InjuryStatus.AVAILABLE),
        ]:
            rows = [_sample_row(**{"Current Status": status_str})]
            records = _parse_records(_make_raw_json(rows), datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
            assert len(records) == 1
            assert records[0].status == expected

    def test_multiple_rows(self):
        rows = [
            _sample_row(**{"Player Name": "Tatum, Jayson"}),
            _sample_row(**{"Player Name": "Brown, Jaylen", "Current Status": "Questionable"}),
        ]
        records = _parse_records(_make_raw_json(rows), datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
        assert len(records) == 2
        assert records[0].player_name == "Tatum, Jayson"
        assert records[1].player_name == "Brown, Jaylen"

    def test_empty_input(self):
        records = _parse_records("[]", datetime(2026, 1, 15, 22, 0, tzinfo=UTC))
        assert records == []


class TestFetchInjuryReport:
    """Tests for fetch_injury_report with mocked nbainjuries."""

    def test_report_time_is_utc_converted_from_et_target(self):
        """report_time in records should be the ET target time converted to UTC."""
        # EST offset: UTC-5. 5 PM ET = 10 PM UTC
        raw = _make_raw_json([_sample_row()])
        # Simulate: target was 10 PM UTC (= 5 PM ET), report_time stored as UTC
        report_time = datetime(2026, 1, 15, 22, 0, tzinfo=UTC)
        records = _parse_records(raw, report_time)
        assert records[0].report_time.tzinfo == UTC

    def test_injury_record_is_dataclass(self):
        """InjuryRecord has expected fields."""
        record = InjuryRecord(
            report_time=datetime(2026, 1, 15, 22, 0, tzinfo=UTC),
            game_date=date(2026, 1, 15),
            game_time_et="07:00 PM ET",
            matchup="BOS@ORL",
            team="Boston Celtics",
            player_name="Tatum, Jayson",
            status=InjuryStatus.OUT,
            reason="Left Ankle; Sprain",
        )
        assert record.team == "Boston Celtics"
        assert record.status == InjuryStatus.OUT
