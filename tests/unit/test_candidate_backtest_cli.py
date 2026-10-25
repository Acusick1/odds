"""The backtest and parity commands: argument handling, output and exit codes."""

from __future__ import annotations

import dataclasses
import json
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from odds_analytics.candidate_backtest import Coverage, HashStatus, score
from odds_analytics.candidate_parity import DecisionParity, ParityReport, ParityStatus
from odds_analytics.candidate_report import Verdict
from odds_cli.main import app
from odds_core.candidates import MechanicalCandidate
from typer.testing import CliRunner

from tests.unit.test_candidates import MECHANICAL, write

N_BOOT = 100
NOW = datetime(2026, 9, 1, tzinfo=UTC)
RUNNER = CliRunner(env={"COLUMNS": "250"})


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "candidates"
    write(directory, MECHANICAL)
    monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: directory)

    @asynccontextmanager
    async def session():
        yield None

    async def no_decisions(_session: Any) -> dict[tuple[str, int], datetime]:
        return {}

    monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
    monkeypatch.setattr("odds_cli.commands.candidates.load_decision_starts", no_decisions)
    return directory


def fake_replay(monkeypatch: pytest.MonkeyPatch) -> list[MechanicalCandidate]:
    seen: list[MechanicalCandidate] = []

    async def replay(_session: Any, candidate: MechanicalCandidate, start: date, end: date, k: int):
        seen.append(candidate)
        coverage = Coverage(
            events=10, with_exchange=8, with_retail=9, with_bet=0, bets=0, bets_with_close=0
        )
        return score(candidate, start, end, [], coverage, k, NOW, N_BOOT, 0)

    monkeypatch.setattr("odds_cli.commands.candidates.replay", replay)
    return seen


class TestBacktestCommand:
    def test_registered_name_prints_coverage_and_writes_json(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        out = tmp_path / "result.json"

        result = RUNNER.invoke(
            app,
            [
                "candidates",
                "backtest",
                "back-longest",
                "--from",
                "2025-08-01",
                "--to",
                "2025-12-31",
                "--json",
                str(out),
            ],
        )

        assert result.exit_code == 0, result.stdout
        assert seen[0].name == "back-longest"
        assert "With exchange" in result.stdout and "not_yet" in result.stdout
        assert "promote" not in result.stdout
        summary = json.loads(out.read_text())
        assert summary["coverage"]["events"] == 10
        assert summary["hash_status"] == HashStatus.MATCH
        assert summary["start"] == "2025-08-01"

    def test_started_sibling_raises_k_and_is_shown(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_replay(monkeypatch)
        write(registry, {**MECHANICAL, "name": "sibling", "start_date": "2026-01-01"})
        out = tmp_path / "result.json"

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", "back-longest", "--from", "2025-08-01", "--to", "2025-12-31"]
            + ["--json", str(out)],
        )

        assert result.exit_code == 0, result.stdout
        assert "k=2" in result.stdout
        assert json.loads(out.read_text())["summary"]["family_versions"] == 2

    def test_unstarted_registered_candidate_counts_itself_once(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_replay(monkeypatch)
        result = RUNNER.invoke(
            app,
            [
                "candidates",
                "backtest",
                "back-longest",
                "--from",
                "2025-08-01",
                "--to",
                "2025-12-31",
            ],
        )
        assert "k=1" in result.stdout

    def test_draft_backtest_survives_a_rule_hash_mismatch_in_the_registry(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_replay(monkeypatch)
        write(registry, {**MECHANICAL, "name": "stale", "rule_sha256": "0" * 64})
        draft = tmp_path / "draft.yaml"
        draft.write_text(yaml.safe_dump({**MECHANICAL, "name": "draft"}))

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", str(draft), "--from", "2025-08-01", "--to", "2025-12-31"],
        )

        assert result.exit_code == 0, result.stdout
        assert "k=" in result.stdout

    def test_draft_path_without_hash_is_reported(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        draft = tmp_path / "draft.yaml"
        draft.write_text(
            yaml.safe_dump({k: v for k, v in MECHANICAL.items() if k != "rule_sha256"})
        )

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", str(draft), "--from", "2025-08-01", "--to", "2025-12-31"],
        )

        assert result.exit_code == 0, result.stdout
        assert seen[0].rule_sha256 is None
        assert "rule_sha256 absent" in result.stdout

    @pytest.mark.parametrize(
        "args",
        [
            ["nope", "--from", "2025-08-01", "--to", "2025-12-31"],
            ["back-longest", "--from", "2025-12-31", "--to", "2025-08-01"],
            ["back-longest", "--from", "August", "--to", "2025-12-31"],
        ],
    )
    def test_bad_arguments_exit_1(self, registry: Path, args: list[str]) -> None:
        assert RUNNER.invoke(app, ["candidates", "backtest", *args]).exit_code == 1


def parity_report(*statuses: ParityStatus, db_replayed: tuple = (("home", "bet365", 2.2, 1.0),)):
    return ParityReport(
        "back-longest",
        1,
        [
            DecisionParity(
                i, f"e{i}", NOW, status, (), (("home", "bet365", 2.2, 1.0),), db_replayed
            )
            for i, status in enumerate(statuses)
        ],
    )


class TestParityCommand:
    def invoke(self, monkeypatch: pytest.MonkeyPatch, report: ParityReport):
        async def check(_session: Any, candidate: MechanicalCandidate, version: int | None):
            return report

        monkeypatch.setattr("odds_cli.commands.candidates.check_parity", check)
        return RUNNER.invoke(app, ["candidates", "parity", "back-longest"])

    def test_clean_report_exits_0(self, registry: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self.invoke(monkeypatch, parity_report(ParityStatus.MATCH, ParityStatus.MATCH))

        assert result.exit_code == 0, result.stdout
        assert "2 decisions checked, 2 matched, 0 mismatches, 0 data differences" in result.stdout

    def test_data_difference_is_informational(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.invoke(monkeypatch, parity_report(ParityStatus.MATCH, db_replayed=()))

        assert result.exit_code == 0, result.stdout
        assert (
            "1 data differences" in result.stdout and "data difference decision 0" in result.stdout
        )

    def test_mismatch_exits_1_with_details(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.invoke(monkeypatch, parity_report(ParityStatus.MISMATCH))

        assert result.exit_code == 1
        assert "mismatch decision 0" in result.stdout and "replayed" in result.stdout

    def test_no_decisions_warns_and_exits_0(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.invoke(monkeypatch, parity_report())

        assert result.exit_code == 0, result.stdout
        assert "no decisions to check" in result.stdout

    def test_unknown_candidate_exits_1(self, registry: Path) -> None:
        assert RUNNER.invoke(app, ["candidates", "parity", "nope"]).exit_code == 1


class TestBacktestVerdictWording:
    def tables(self, verdict: Verdict) -> str:
        from io import StringIO

        from odds_cli.commands.candidates import _backtest_tables
        from rich.console import Console

        cov = Coverage(
            events=1, with_exchange=1, with_retail=1, with_bet=0, bets=0, bets_with_close=0
        )
        result = score(
            MechanicalCandidate.model_validate(MECHANICAL),
            date(2025, 8, 1),
            date(2025, 9, 1),
            [],
            cov,
            1,
            NOW,
            N_BOOT,
            0,
        )
        result = dataclasses.replace(
            result, summary=dataclasses.replace(result.summary, verdict=verdict)
        )
        out = StringIO()
        for table in _backtest_tables(result):
            Console(file=out, width=250).print(table)
        return out.getvalue()

    def test_promote_bar_is_not_a_promotion(self) -> None:
        text = self.tables(Verdict.PROMOTE)
        assert "meets_promote_bar (development data, not a promotion)" in text
        assert "promote " not in text.replace("promote bar", "")

    @pytest.mark.parametrize("verdict", [Verdict.KILL, Verdict.CONTINUE, Verdict.NOT_YET])
    def test_other_verdicts_are_shown_as_is(self, verdict: Verdict) -> None:
        assert verdict.value in self.tables(verdict)
