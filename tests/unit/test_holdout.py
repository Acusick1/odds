"""The sealed holdout: which ranges it covers, what opens it, the one-shot ledger and the gate."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path

import pytest
from odds_analytics import holdout
from odds_analytics.candidate_report import Checkpoint
from odds_cli.main import app
from odds_core.candidates import MechanicalCandidate
from typer.testing import CliRunner

from tests.unit.conftest import HOLDOUT_CELL
from tests.unit.test_candidate_backtest_cli import fake_replay
from tests.unit.test_candidates import MECHANICAL, write

RUNNER = CliRunner(env={"COLUMNS": "250"})
SEALED = ("2025-09-01", "2025-09-30")


def draft(**changes: object) -> MechanicalCandidate:
    data = {
        **{k: v for k, v in MECHANICAL.items() if k != "rule_sha256"},
        "sport": "soccer_spain_la_liga",
        "params": dict(HOLDOUT_CELL),
        **changes,
    }
    return MechanicalCandidate.model_validate(data)


def checkpoint(n: int, mean_ev: float, low: float, high: float) -> Checkpoint:
    return Checkpoint(
        n=n,
        mean_ev=mean_ev,
        ci95=(low, high),
        family_versions=1,
        promote_level=0.99,
        promote_ci=(low, high),
    )


class TestWindows:
    @pytest.mark.parametrize(
        ("sport", "start", "end", "sealed"),
        [
            ("soccer_spain_la_liga", date(2025, 9, 1), date(2025, 9, 30), True),
            ("soccer_efl_champ", date(2026, 7, 31), date(2026, 8, 31), True),
            ("soccer_spain_la_liga", date(2024, 8, 1), date(2025, 7, 31), False),
            ("soccer_epl", date(2025, 9, 1), date(2025, 9, 30), False),
            ("soccer_epl", date(2023, 1, 1), date(2023, 1, 2), True),
            ("soccer_epl", date(2022, 7, 1), date(2022, 7, 31), False),
        ],
    )
    def test_intersection_by_sport_and_days(
        self, sport: str, start: date, end: date, sealed: bool
    ) -> None:
        assert bool(holdout.intersecting(sport, start, end)) is sealed

    def test_epl_2022_24_follows_the_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(holdout, "EPL_2022_24_SEALED", False)
        assert not holdout.intersecting("soccer_epl", date(2023, 1, 1), date(2023, 1, 2))

    def test_the_switch_defaults_to_sealed(self) -> None:
        assert holdout.EPL_2022_24_SEALED is True


class TestAdmit:
    def test_a_range_outside_the_holdout_needs_nothing(self, tmp_path: Path) -> None:
        assert holdout.admit(draft(), date(2024, 9, 1), date(2024, 9, 30), tmp_path) is None

    def test_refused_without_a_preregistration(self, holdout_repo: Path) -> None:
        with pytest.raises(holdout.HoldoutRefused, match="no pre-registration"):
            holdout.admit(draft(issue=611), date(2025, 9, 1), date(2025, 9, 30), holdout_repo)

    def test_admitted_with_a_committed_preregistration(self, holdout_repo: Path) -> None:
        prereg = holdout.admit(draft(), date(2025, 9, 1), date(2025, 9, 30), holdout_repo)
        assert prereg is not None and prereg.cell == HOLDOUT_CELL

    def test_refused_when_the_preregistration_is_uncommitted(self, holdout_repo: Path) -> None:
        (holdout_repo / "preregistrations" / "611.yaml").write_text(
            (holdout_repo / "preregistrations" / "610.yaml").read_text().replace("610", "611")
        )
        with pytest.raises(holdout.HoldoutRefused, match="not committed"):
            holdout.admit(draft(issue=611), date(2025, 9, 1), date(2025, 9, 30), holdout_repo)

    def test_refused_when_the_preregistration_was_edited_since_commit(
        self, holdout_repo: Path
    ) -> None:
        path = holdout_repo / "preregistrations" / "610.yaml"
        path.write_text(path.read_text().replace("0.02", "0.05"))
        with pytest.raises(holdout.HoldoutRefused, match="differs from its committed"):
            holdout.admit(draft(), date(2025, 9, 1), date(2025, 9, 30), holdout_repo)

    def test_refused_when_the_params_are_not_the_selected_cell(self, holdout_repo: Path) -> None:
        other = {**HOLDOUT_CELL, "threshold": 0.05}
        with pytest.raises(holdout.HoldoutRefused, match="not the selected cell"):
            holdout.admit(draft(params=other), date(2025, 9, 1), date(2025, 9, 30), holdout_repo)

    def test_refused_when_the_grid_commit_is_not_in_history(self, holdout_repo: Path) -> None:
        path = holdout_repo / "preregistrations" / "610.yaml"
        grid = path.read_text().split("grid_sha: ")[1].split()[0]
        path.write_text(path.read_text().replace(grid, "f" * 40))
        subprocess.run(["git", "commit", "-qam", "edit"], cwd=holdout_repo, check=True)
        with pytest.raises(holdout.HoldoutRefused, match="not an ancestor"):
            holdout.admit(draft(), date(2025, 9, 1), date(2025, 9, 30), holdout_repo)

    def test_refused_when_tracked_files_are_dirty(self, holdout_repo: Path) -> None:
        (holdout_repo / "other.yaml").write_text("a: 1\n")
        subprocess.run(["git", "add", "other.yaml"], cwd=holdout_repo, check=True)
        subprocess.run(["git", "commit", "-qm", "x"], cwd=holdout_repo, check=True)
        (holdout_repo / "other.yaml").write_text("a: 2\n")
        with pytest.raises(holdout.HoldoutRefused, match="uncommitted"):
            holdout.admit(draft(), date(2025, 9, 1), date(2025, 9, 30), holdout_repo)


class TestGate:
    def test_positive_and_not_killed_passes(self) -> None:
        g = holdout.gate(checkpoint(80, 0.03, -0.01, 0.07))
        assert g.verdict is holdout.GateVerdict.PASS and not g.low_bets

    def test_non_positive_mean_fails(self) -> None:
        assert holdout.gate(checkpoint(80, -0.01, -0.05, 0.03)).verdict is holdout.GateVerdict.FAIL

    def test_killed_fails_whatever_the_mean(self) -> None:
        g = holdout.gate(checkpoint(80, 0.01, -0.05, 0.0))
        assert g.killed and g.verdict is holdout.GateVerdict.FAIL

    def test_under_30_bets_not_killed_alone_passes_flagged(self) -> None:
        g = holdout.gate(checkpoint(12, -0.04, -0.2, 0.1))
        assert g.verdict is holdout.GateVerdict.PASS and g.low_bets

    def test_under_30_bets_killed_fails(self) -> None:
        assert holdout.gate(checkpoint(12, -0.1, -0.3, -0.01)).verdict is holdout.GateVerdict.FAIL

    def test_no_bets_fails_flagged(self) -> None:
        g = holdout.gate(None)
        assert g.verdict is holdout.GateVerdict.FAIL and g.low_bets


class TestBacktestCommand:
    @pytest.fixture
    def registry(self, holdout_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        @asynccontextmanager
        async def session() -> AsyncIterator[None]:
            yield None

        async def no_decisions(_session: object) -> dict[tuple[str, int], datetime]:
            return {}

        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: holdout_repo)
        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
        monkeypatch.setattr("odds_cli.commands.candidates.load_decision_starts", no_decisions)
        return holdout_repo

    def write_draft(self, tmp_path: Path, **changes: object) -> Path:
        path = tmp_path / "draft.yaml"
        data = {
            **{k: v for k, v in MECHANICAL.items() if k != "rule_sha256"},
            "sport": "soccer_spain_la_liga",
            "params": dict(HOLDOUT_CELL),
            **changes,
        }
        write(tmp_path, data, stem="draft")
        return path

    def run(self, path: Path) -> object:
        return RUNNER.invoke(
            app, ["candidates", "backtest", str(path), "--from", SEALED[0], "--to", SEALED[1]]
        )

    def test_refused_without_a_preregistration_and_nothing_replays(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        result = self.run(self.write_draft(tmp_path, issue=999))
        assert result.exit_code == 1  # type: ignore[attr-defined]
        assert "sealed holdout" in result.stdout  # type: ignore[attr-defined]
        assert seen == [] and holdout.load_runs(registry) == []

    def test_a_preregistered_run_records_the_ledger_and_prints_the_gate(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_replay(monkeypatch)
        result = self.run(self.write_draft(tmp_path))
        assert result.exit_code == 0, result.stdout  # type: ignore[attr-defined]
        assert "Holdout gate: fail" in result.stdout  # type: ignore[attr-defined]
        (run,) = holdout.load_runs(registry)
        assert (run.issue, run.version, run.cell) == (610, 1, HOLDOUT_CELL)
        assert (run.start, run.end) == (date(2025, 9, 1), date(2025, 9, 30))
        assert len(run.git_sha) == 40 and run.grid_sha != run.git_sha

    def test_a_second_run_of_the_idea_and_version_is_refused(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        path = self.write_draft(tmp_path)
        assert self.run(path).exit_code == 0  # type: ignore[attr-defined]
        subprocess.run(["git", "commit", "-qam", "ledger"], cwd=registry, check=True)

        again = self.run(path)

        assert again.exit_code == 1  # type: ignore[attr-defined]
        assert "already ran" in again.stdout  # type: ignore[attr-defined]
        assert len(seen) == 1 and len(holdout.load_runs(registry)) == 1

    def test_compare_is_refused_on_the_holdout(
        self, registry: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        seen = fake_replay(monkeypatch)
        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", str(self.write_draft(tmp_path)), "--from", SEALED[0]]
            + ["--to", SEALED[1], "--compare", "back-longest"],
        )
        assert result.exit_code == 1 and seen == []


class TestCommittedLedger:
    def test_every_row_validates(self) -> None:
        root = Path(__file__).resolve().parents[2] / "candidates"
        holdout.load_runs(root)
        assert (root / holdout.HOLDOUT_LEDGER).exists()
        for path in (root / holdout.PREREGISTRATION_DIR).glob("*.yaml"):
            prereg = holdout.Preregistration.model_validate(
                __import__("yaml").safe_load(path.read_text())
            )
            assert path.stem == str(prereg.issue)
