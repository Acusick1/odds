"""Tests for the agent-run summary viewer in experiments/scripts."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "show_agent_summaries.py"


@pytest.fixture(scope="module")
def viewer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("show_agent_summaries", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve string annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _session(start: str, end: str, result: str) -> list[dict[str, Any]]:
    return [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "timestamp": start},
        {"type": "user", "timestamp": end},
        {"type": "result", "subtype": "success", "result": result},
    ]


def _write_trace(log_dir: Path, name: str, entries: list[dict[str, Any]]) -> Path:
    path = log_dir / name
    path.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return path


class TestLoadSummariesSince:
    def test_keeps_later_session_of_file_started_before_since(
        self, viewer: ModuleType, tmp_path: Path
    ) -> None:
        _write_trace(
            tmp_path,
            "soccer_epl_20261001T100000Z_42.jsonl",
            _session("2026-10-01T10:00:05Z", "2026-10-01T10:08:00Z", "first")
            + _session("2026-10-03T09:00:05Z", "2026-10-03T09:07:00Z", "second"),
        )

        summaries = viewer.load_summaries(
            sport_filter=None,
            since=datetime(2026, 10, 2, tzinfo=UTC),
            log_dir=tmp_path,
            explicit_file=None,
        )

        assert [s.result for s in summaries] == ["second"]
        assert summaries[0].session_idx == 1
        assert summaries[0].session_start == datetime(2026, 10, 3, 9, 0, 5, tzinfo=UTC)

    def test_keeps_session_that_ends_after_since(self, viewer: ModuleType, tmp_path: Path) -> None:
        _write_trace(
            tmp_path,
            "baseball_mlb_20261001T235500Z_7.jsonl",
            _session("2026-10-01T23:55:05Z", "2026-10-02T00:05:00Z", "straddles"),
        )

        summaries = viewer.load_summaries(
            sport_filter=None,
            since=datetime(2026, 10, 2, tzinfo=UTC),
            log_dir=tmp_path,
            explicit_file=None,
        )

        assert [s.result for s in summaries] == ["straddles"]

    def test_drops_sessions_that_end_before_since(self, viewer: ModuleType, tmp_path: Path) -> None:
        _write_trace(
            tmp_path,
            "soccer_epl_20261001T100000Z_42.jsonl",
            _session("2026-10-01T10:00:05Z", "2026-10-01T10:08:00Z", "old"),
        )

        summaries = viewer.load_summaries(
            sport_filter=None,
            since=datetime(2026, 10, 2, tzinfo=UTC),
            log_dir=tmp_path,
            explicit_file=None,
        )

        assert summaries == []

    def test_session_without_timestamps_falls_back_to_file_start(
        self, viewer: ModuleType, tmp_path: Path
    ) -> None:
        _write_trace(
            tmp_path,
            "soccer_epl_20261003T100000Z_42.jsonl",
            [{"type": "result", "subtype": "success", "result": "untimed"}],
        )

        summaries = viewer.load_summaries(
            sport_filter=None,
            since=datetime(2026, 10, 2, tzinfo=UTC),
            log_dir=tmp_path,
            explicit_file=None,
        )

        assert [s.result for s in summaries] == ["untimed"]
        assert summaries[0].session_start == datetime(2026, 10, 3, 10, tzinfo=UTC)

    def test_orders_newest_session_first(self, viewer: ModuleType, tmp_path: Path) -> None:
        _write_trace(
            tmp_path,
            "soccer_epl_20261001T100000Z_42.jsonl",
            _session("2026-10-01T10:00:05Z", "2026-10-01T10:08:00Z", "a")
            + _session("2026-10-04T09:00:05Z", "2026-10-04T09:07:00Z", "c"),
        )
        _write_trace(
            tmp_path,
            "soccer_epl_20261002T100000Z_43.jsonl",
            _session("2026-10-02T10:00:05Z", "2026-10-02T10:08:00Z", "b"),
        )

        summaries = viewer.load_summaries(
            sport_filter=None, since=None, log_dir=tmp_path, explicit_file=None
        )

        assert [s.result for s in summaries] == ["c", "b", "a"]
