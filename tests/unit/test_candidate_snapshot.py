"""The registry snapshot: run-candidates runs the registry and rule hashes its process first
loaded, whatever the checkout holds later."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from odds_lambda.candidate_rules.pinning import rule_sha256
from odds_lambda.candidate_rules.snapshot import load_snapshot, registry_snapshot
from odds_lambda.jobs import run_candidates
from odds_lambda.jobs.run_candidates import runnable_candidates

from tests.unit.conftest import SourceEdit

EPL = "soccer_epl"
NOW = datetime(2026, 11, 20, 12, tzinfo=UTC)
NAME = "gap"


def write(directory: Path, name: str = NAME, **fields: object) -> None:
    data = {
        "name": name,
        "description": "d",
        "kind": "rule",
        "market": "1x2",
        "books": "uk_retail",
        "leagues": [EPL],
        "version": 1,
        "issue": 1,
        "start_date": "2026-11-01",
        "status": "active",
        "rule": "exchange_gap",
        "rule_sha256": rule_sha256("exchange_gap"),
        "params": {"threshold": 0.01, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
        **fields,
    }
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(data))


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("odds_lambda.candidate_rules.snapshot.candidates_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def alert(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    mock = AsyncMock()
    monkeypatch.setattr(run_candidates, "send_recorded_alert", mock)
    return mock


class TestRunCandidatesKeepsItsFirstRegistry:
    async def test_changed_files_and_rule_source_change_nothing(
        self, registry: Path, edit_source: SourceEdit, alert: AsyncMock
    ) -> None:
        write(registry)
        [first] = await runnable_candidates(EPL, NOW)

        # A pull that changes the rule's code and records its new hash in a new version, and
        # registers another candidate.
        edit_source("odds_lambda.candidate_rules.gap", lambda source: source + "\nCHANGED = 1\n")
        changed = rule_sha256("exchange_gap")
        assert changed != first.candidate.rule_sha256
        write(registry, version=2, rule_sha256=changed)
        write(registry, "other")

        assert await runnable_candidates(EPL, NOW) == [first]
        alert.assert_not_awaited()

    async def test_a_registry_broken_after_the_first_load_is_not_read(
        self, registry: Path, alert: AsyncMock
    ) -> None:
        write(registry)
        [first] = await runnable_candidates(EPL, NOW)

        (registry / "broken.yaml").write_text("name: [")

        assert await runnable_candidates(EPL, NOW) == [first]
        alert.assert_not_awaited()

    async def test_a_candidate_invalid_at_the_first_load_stays_skipped(
        self, registry: Path, alert: AsyncMock
    ) -> None:
        write(registry, rule_sha256="0" * 64)
        assert await runnable_candidates(EPL, NOW) == []

        write(registry)

        assert await runnable_candidates(EPL, NOW) == []
        assert [call.args[0] for call in alert.await_args_list] == [f"candidate_invalid:{NAME}"] * 2


class TestLoadSnapshot:
    def test_an_unreadable_registry_raises(self, tmp_path: Path) -> None:
        (tmp_path / "broken.yaml").write_text("name: [")

        with pytest.raises(ValueError, match="broken.yaml"):
            load_snapshot(tmp_path)

    def test_an_invalid_candidate_is_recorded_not_raised(self, tmp_path: Path) -> None:
        write(tmp_path)
        write(tmp_path, "stale", rule_sha256="0" * 64)

        snapshot = load_snapshot(tmp_path)

        assert [c.name for c in snapshot.candidates] == [NAME, "stale"]
        assert list(snapshot.invalid) == ["stale"]
        assert "the code of rule 'exchange_gap' changed" in str(snapshot.invalid["stale"])

    def test_an_unreadable_registry_is_not_cached(self, registry: Path) -> None:
        (registry / "broken.yaml").write_text("name: [")
        with pytest.raises(ValueError):
            registry_snapshot()

        (registry / "broken.yaml").unlink()
        write(registry)

        assert registry_snapshot().get(NAME) is not None
