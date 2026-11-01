"""Unit tests for injury report fetcher and parsing logic."""

import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from odds_core.injury_models import InjuryStatus
from odds_lambda.injury_fetcher import (
    STATUS_MAP,
    InjuryRecord,
    _decode,
    _encode,
    _parse_records,
    _to_naive_et,
    fetch_latest_injury_report,
    fetch_latest_injury_report_isolated,
    latest_report_slot,
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
        assert result == datetime(2026, 1, 15, 17, 0)  # noqa: DTZ001
        assert result.tzinfo is None

    def test_utc_to_et_summer(self):
        """UTC datetime converts to ET (EDT, UTC-4) in summer."""
        utc = datetime(2026, 6, 15, 22, 0, tzinfo=UTC)
        result = _to_naive_et(utc)
        assert result == datetime(2026, 6, 15, 18, 0)  # noqa: DTZ001
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


def _sample_record() -> InjuryRecord:
    return InjuryRecord(
        report_time=datetime(2026, 10, 31, 14, 45, tzinfo=UTC),
        game_date=date(2026, 10, 31),
        game_time_et="07:00 (ET)",
        matchup="BOS@ORL",
        team="Boston Celtics",
        player_name="Tatum, Jayson",
        status=InjuryStatus.OUT,
        reason="Ankle",
    )


class TestLatestReportSlot:
    def test_rounds_down_to_the_quarter_hour(self) -> None:
        now = datetime(2026, 10, 31, 15, 14, 59, tzinfo=UTC)

        assert latest_report_slot(now) == datetime(2026, 10, 31, 15, 0, tzinfo=UTC)

    def test_a_slot_time_is_its_own_slot(self) -> None:
        slot = datetime(2026, 10, 31, 15, 45, tzinfo=UTC)

        assert latest_report_slot(slot) == slot


class TestFetchLatestInjuryReport:
    def test_steps_back_to_the_newest_published_slot(self, monkeypatch) -> None:
        asked: list[datetime] = []
        record = InjuryRecord(
            report_time=datetime(2026, 10, 31, 14, 45, tzinfo=UTC),
            game_date=date(2026, 10, 31),
            game_time_et="07:00 (ET)",
            matchup="BOS@ORL",
            team="Boston Celtics",
            player_name="Tatum, Jayson",
            status=InjuryStatus.OUT,
            reason="Ankle",
        )

        def fake(target: datetime) -> list[InjuryRecord]:
            asked.append(target)
            return [record] if target.minute == 45 else []

        monkeypatch.setattr("odds_lambda.injury_fetcher.fetch_injury_report", fake)

        found = fetch_latest_injury_report(
            datetime(2026, 10, 31, 15, 4, tzinfo=UTC), max_slots_back=4
        )

        assert found == (datetime(2026, 10, 31, 14, 45, tzinfo=UTC), [record])
        assert [t.strftime("%H:%M") for t in asked] == ["15:00", "14:45"]

    def test_steps_over_a_slot_that_fails_to_parse(self, monkeypatch) -> None:
        record = _sample_record()

        def fake(target: datetime) -> list[InjuryRecord]:
            if target.minute == 0:
                raise ValueError("PDF still uploading")
            return [record]

        monkeypatch.setattr("odds_lambda.injury_fetcher.fetch_injury_report", fake)

        found = fetch_latest_injury_report(
            datetime(2026, 10, 31, 15, 4, tzinfo=UTC), max_slots_back=4
        )

        assert found is not None
        assert found[0].minute == 45

    def test_raises_when_every_slot_fails_to_parse(self, monkeypatch) -> None:
        def fake(target: datetime) -> list[InjuryRecord]:
            raise ValueError("unreadable")

        monkeypatch.setattr("odds_lambda.injury_fetcher.fetch_injury_report", fake)

        with pytest.raises(ValueError, match="unreadable"):
            fetch_latest_injury_report(datetime(2026, 10, 31, 15, 4, tzinfo=UTC), max_slots_back=2)

    def test_none_when_no_slot_in_reach_has_a_report(self, monkeypatch) -> None:
        asked: list[datetime] = []
        monkeypatch.setattr(
            "odds_lambda.injury_fetcher.fetch_injury_report",
            lambda target: asked.append(target) or [],
        )

        found = fetch_latest_injury_report(
            datetime(2026, 10, 31, 15, 4, tzinfo=UTC), max_slots_back=2
        )

        assert found is None
        assert len(asked) == 3


class TestChildProcessCodec:
    def test_report_round_trips(self) -> None:
        slot = datetime(2026, 10, 31, 15, 0, tzinfo=UTC)
        records = _decode(
            _encode(
                slot,
                [
                    InjuryRecord(
                        report_time=slot,
                        game_date=date(2026, 10, 31),
                        game_time_et="07:00 (ET)",
                        matchup="BOS@ORL",
                        team="Boston Celtics",
                        player_name="Tatum, Jayson",
                        status=InjuryStatus.QUESTIONABLE,
                        reason="Ankle; Sprain",
                    )
                ],
            )
        )

        assert records[0] == slot
        assert records[1][0].status is InjuryStatus.QUESTIONABLE
        assert records[1][0].game_date == date(2026, 10, 31)
        assert records[1][0].report_time == slot


class _FakeChild:
    def __init__(self, returncode: int, stderr: bytes = b"", hang: bool = False) -> None:
        self.returncode: int | None = None if hang else returncode
        self._stderr = stderr
        self._hang = hang
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._hang:
            await asyncio.sleep(10)
        return b"", self._stderr

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return -9


class TestFetchLatestInjuryReportIsolated:
    NOW = datetime(2026, 10, 31, 15, 4, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_child_failure_raises_with_its_stderr(self, monkeypatch) -> None:
        child = _FakeChild(1, b"JVMNotFoundException")
        monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=child))

        with pytest.raises(RuntimeError, match="JVMNotFoundException"):
            await fetch_latest_injury_report_isolated(self.NOW, max_slots_back=4)

    @pytest.mark.asyncio
    async def test_hung_child_is_killed(self, monkeypatch) -> None:
        child = _FakeChild(0, hang=True)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=child))
        monkeypatch.setattr("odds_lambda.injury_fetcher.SUBPROCESS_TIMEOUT_SECONDS", 0.01)

        with pytest.raises(RuntimeError, match="killed"):
            await fetch_latest_injury_report_isolated(self.NOW, max_slots_back=4)

        assert child.killed

    @pytest.mark.asyncio
    async def test_cancelled_run_kills_the_child(self, monkeypatch) -> None:
        child = _FakeChild(0, hang=True)
        monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=child))

        task = asyncio.create_task(fetch_latest_injury_report_isolated(self.NOW, max_slots_back=4))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert child.killed

    @pytest.mark.asyncio
    async def test_no_report_is_none(self, monkeypatch) -> None:
        async def spawn(*args: str, **_: object) -> _FakeChild:
            Path(args[args.index("--out") + 1]).write_text("")
            return _FakeChild(0)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

        assert await fetch_latest_injury_report_isolated(self.NOW, max_slots_back=4) is None
