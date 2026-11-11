"""Unit tests for scripts/check_candidate_versions.py."""

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from scripts import check_candidate_versions as hook
from tests.unit.conftest import INVALID_LEDGER_ROWS, NEW, OLD, ledger_row, ledger_row_candidate

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_candidate_versions.py"

BASE_YAML = """\
name: back-longest
description: Back the longest price
version: 2
status: active
params:
  min_edge: 0.02
"""


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


class TestCheckCandidateVersions:
    @pytest.fixture
    def repo(self, candidates_repo: Path) -> Path:
        (candidates_repo / "candidates" / "back-longest.yaml").write_text(BASE_YAML)
        git(candidates_repo, "add", "candidates")
        git(candidates_repo, "commit", "-q", "-m", "base")
        return candidates_repo

    def run(self, repo: Path, base: str | None = "HEAD") -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *(["--base", base] if base else [])],
            cwd=repo,
            capture_output=True,
            text=True,
        )

    def edit(self, repo: Path, old: str, new: str) -> None:
        path = repo / "candidates" / "back-longest.yaml"
        path.write_text(path.read_text().replace(old, new))

    def test_unchanged_passes(self, repo: Path) -> None:
        result = self.run(repo)
        assert result.returncode == 0 and result.stderr == ""

    def test_status_only_change_passes(self, repo: Path) -> None:
        self.edit(repo, "status: active", "status: killed")
        assert self.run(repo).returncode == 0

    def test_description_only_change_passes(self, repo: Path) -> None:
        self.edit(repo, "Back the longest price", "Back the longest UK retail price")
        assert self.run(repo).returncode == 0

    def test_params_change_without_bump_fails(self, repo: Path) -> None:
        self.edit(repo, "0.02", "0.05")
        result = self.run(repo)
        assert result.returncode == 1
        assert "back-longest.yaml" in result.stderr and "params" in result.stderr

    def test_params_change_with_bump_passes(self, repo: Path) -> None:
        self.edit(repo, "0.02", "0.05")
        self.edit(repo, "version: 2", "version: 3")
        assert self.run(repo).returncode == 0

    def test_version_decrease_fails(self, repo: Path) -> None:
        self.edit(repo, "version: 2", "version: 1")
        result = self.run(repo)
        assert result.returncode == 1 and "decreased" in result.stderr

    def test_deletion_fails(self, repo: Path) -> None:
        (repo / "candidates" / "back-longest.yaml").unlink()
        result = self.run(repo)
        assert result.returncode == 1 and "status: killed" in result.stderr

    def test_new_file_passes(self, repo: Path) -> None:
        (repo / "candidates" / "fresh.yaml").write_text(BASE_YAML.replace("back-longest", "fresh"))
        assert self.run(repo).returncode == 0

    def test_default_base_is_merge_base_with_main(self, repo: Path) -> None:
        git(repo, "checkout", "-q", "-b", "feature")
        self.edit(repo, "0.02", "0.05")
        self.edit(repo, "version: 2", "version: 3")
        git(repo, "commit", "-qam", "bump")
        self.edit(repo, "0.05", "0.07")
        git(repo, "commit", "-qam", "tweak")

        assert self.run(repo, base=None).returncode == 0
        assert self.run(repo, base="HEAD~1").returncode == 1

    def test_rename_fails(self, repo: Path) -> None:
        git(repo, "mv", "candidates/back-longest.yaml", "candidates/renamed.yaml")
        result = self.run(repo)
        assert result.returncode == 1 and "back-longest.yaml: deleted" in result.stderr

    def test_malformed_yaml_fails(self, repo: Path) -> None:
        (repo / "candidates" / "back-longest.yaml").write_text("name: [unclosed\n")
        result = self.run(repo)
        assert result.returncode == 1
        assert "back-longest.yaml: invalid YAML" in result.stderr

    def test_non_integer_version_fails(self, repo: Path) -> None:
        self.edit(repo, "version: 2", 'version: "3"')
        result = self.run(repo)
        assert result.returncode == 1 and "non-integer" in result.stderr


class TestLeagueField:
    """A one-league `sport` reads as `sports: [it]`: the rename alone is no change."""

    @pytest.fixture
    def repo(self, candidates_repo: Path) -> Path:
        (candidates_repo / "candidates" / "back-longest.yaml").write_text(
            BASE_YAML + "sport: soccer_epl\n"
        )
        git(candidates_repo, "add", "candidates")
        git(candidates_repo, "commit", "-q", "-m", "base")
        return candidates_repo

    def write(self, repo: Path, sports: str, version: int = 2) -> None:
        path = repo / "candidates" / "back-longest.yaml"
        path.write_text(
            BASE_YAML.replace("version: 2", f"version: {version}") + f"sports: {sports}\n"
        )

    def errors(self, repo: Path) -> list[str]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--base", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        return result.stderr.splitlines()

    def test_renaming_sport_to_a_one_league_list_passes(self, repo: Path) -> None:
        self.write(repo, "[soccer_epl]")
        assert self.errors(repo) == []

    def test_renaming_to_another_league_fails(self, repo: Path) -> None:
        self.write(repo, "[soccer_italy_serie_a]")
        (error,) = self.errors(repo)
        assert "sports changed without a version bump" in error

    def test_adding_a_league_without_a_bump_fails(self, repo: Path) -> None:
        self.write(repo, "[soccer_epl, soccer_italy_serie_a]")
        (error,) = self.errors(repo)
        assert "sports changed without a version bump" in error

    def test_adding_a_league_with_a_bump_passes(self, repo: Path) -> None:
        self.write(repo, "[soccer_epl, soccer_italy_serie_a]", version=3)
        assert self.errors(repo) == []

    def test_dropping_a_league_without_a_bump_fails(self, repo: Path) -> None:
        self.write(repo, "[soccer_epl, soccer_italy_serie_a]", version=3)
        git(repo, "commit", "-qam", "pooled")
        self.write(repo, "[soccer_epl]", version=3)
        (error,) = self.errors(repo)
        assert "sports changed without a version bump" in error


class TestIdentityKeys:
    """A name's rule and market are fixed: another rule or market is another candidate."""

    YAML = BASE_YAML + "rule: exchange_gap\nmarket: 1x2\nsports: [soccer_epl]\n"

    @pytest.fixture
    def repo(self, candidates_repo: Path) -> Path:
        (candidates_repo / "candidates" / "back-longest.yaml").write_text(self.YAML)
        git(candidates_repo, "add", "candidates")
        git(candidates_repo, "commit", "-q", "-m", "base")
        return candidates_repo

    def errors(self, repo: Path, new: str) -> list[str]:
        (repo / "candidates" / "back-longest.yaml").write_text(new)
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--base", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        return result.stderr.splitlines()

    def bumped(self, *replacements: tuple[str, str]) -> str:
        text = self.YAML.replace("version: 2", "version: 3")
        for old, new in replacements:
            text = text.replace(old, new)
        return text

    def test_a_rule_change_is_refused_even_with_a_bump(self, repo: Path) -> None:
        (error,) = self.errors(repo, self.bumped(("exchange_gap", "polymarket_gap")))
        assert "rule changed" in error and "new name" in error

    def test_a_market_change_is_refused_even_with_a_bump(self, repo: Path) -> None:
        (error,) = self.errors(repo, self.bumped(("1x2", "totals")))
        assert "market changed" in error and "new name" in error

    def test_a_league_change_with_a_bump_passes(self, repo: Path) -> None:
        added = self.bumped(("[soccer_epl]", "[soccer_epl, soccer_italy_serie_a]"))
        assert self.errors(repo, added) == []


class TestUnstartedCandidates:
    @pytest.fixture
    def repo(self, candidates_repo: Path) -> Path:
        return candidates_repo

    def commit_base(self, repo: Path, start_offset_days: int) -> Path:
        start = datetime.now(UTC).date() + timedelta(days=start_offset_days)
        path = repo / "candidates" / "back-longest.yaml"
        path.write_text(BASE_YAML + f"start_date: {start.isoformat()}\n")
        git(repo, "add", "candidates")
        git(repo, "commit", "-q", "-m", "base")
        return path

    def run(self, repo: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--base", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
        )

    def test_unstarted_base_allows_in_place_change(self, repo: Path) -> None:
        path = self.commit_base(repo, start_offset_days=7)
        old = path.read_text()
        later = (datetime.now(UTC).date() + timedelta(days=14)).isoformat()
        start_line = old.splitlines()[-1]
        path.write_text(old.replace("0.02", "0.05").replace(start_line, f"start_date: {later}"))
        result = self.run(repo)
        assert result.returncode == 0 and result.stderr == ""

    @pytest.mark.parametrize("offset", [0, -3])
    def test_base_started_today_or_earlier_still_fails(self, repo: Path, offset: int) -> None:
        path = self.commit_base(repo, start_offset_days=offset)
        path.write_text(path.read_text().replace("0.02", "0.05"))
        result = self.run(repo)
        assert result.returncode == 1 and "params" in result.stderr

    def test_unstarted_version_decrease_still_fails(self, repo: Path) -> None:
        path = self.commit_base(repo, start_offset_days=7)
        path.write_text(path.read_text().replace("version: 2", "version: 1"))
        result = self.run(repo)
        assert result.returncode == 1 and "decreased" in result.stderr

    def test_unstarted_deletion_still_fails(self, repo: Path) -> None:
        path = self.commit_base(repo, start_offset_days=7)
        path.unlink()
        result = self.run(repo)
        assert result.returncode == 1 and "status: killed" in result.stderr

    def test_string_start_date_does_not_apply_exception(self, repo: Path) -> None:
        start = (datetime.now(UTC).date() + timedelta(days=7)).isoformat()
        path = repo / "candidates" / "back-longest.yaml"
        path.write_text(BASE_YAML + f'start_date: "{start}"\n')
        git(repo, "add", "candidates")
        git(repo, "commit", "-q", "-m", "base")
        path.write_text(path.read_text().replace("0.02", "0.05"))
        assert self.run(repo).returncode == 1

    def test_missing_start_date_does_not_apply_exception(self, repo: Path) -> None:
        path = repo / "candidates" / "back-longest.yaml"
        path.write_text(BASE_YAML)
        git(repo, "add", "candidates")
        git(repo, "commit", "-q", "-m", "base")
        path.write_text(BASE_YAML.replace("0.02", "0.05"))
        assert self.run(repo).returncode == 1


class TestRehashLedger:
    @pytest.fixture
    def path(self, candidates_repo: Path) -> Path:
        path = candidates_repo / "candidates" / "back-longest.yaml"
        path.write_text(BASE_YAML + f"start_date: 2020-01-01\nrule: gap\nrule_sha256: {OLD}\n")
        git(candidates_repo, "add", "candidates")
        git(candidates_repo, "commit", "-q", "-m", "base")
        path.write_text(path.read_text().replace(OLD, NEW))
        return path

    def ledger(self, path: Path, *rows: Any) -> None:
        (path.parent / "rehashes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def errors(self, path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        monkeypatch.chdir(path.parents[1])
        return hook.check("HEAD")

    def test_a_complete_ledger_row_passes_at_the_same_version(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.ledger(path, ledger_row())
        assert self.errors(path, monkeypatch) == []

    def test_a_candidate_without_recorded_decisions_passes(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.ledger(
            path, ledger_row(candidates=ledger_row_candidate(parity_matched=0, parity_total=0))
        )
        assert self.errors(path, monkeypatch) == []

    def test_without_a_ledger_a_hash_change_fails(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        errors = self.errors(path, monkeypatch)
        assert len(errors) == 1 and "rule_sha256" in errors[0]

    @pytest.mark.parametrize(
        "row",
        [ledger_row(rule="other"), ledger_row(old="c" * 64), ledger_row(new="c" * 64)],
        ids=["another rule", "another old hash", "another new hash"],
    )
    def test_a_row_for_another_rule_or_hash_fails(
        self, path: Path, monkeypatch: pytest.MonkeyPatch, row: dict[str, Any]
    ) -> None:
        self.ledger(path, row)
        errors = self.errors(path, monkeypatch)
        assert len(errors) == 1 and "rule_sha256" in errors[0]

    @pytest.mark.parametrize("case", list(INVALID_LEDGER_ROWS))
    def test_an_incomplete_row_is_an_error_and_accepts_nothing(
        self, path: Path, monkeypatch: pytest.MonkeyPatch, case: str
    ) -> None:
        row = INVALID_LEDGER_ROWS[case]
        (path.parent / "rehashes.jsonl").write_text(
            (row if isinstance(row, str) else json.dumps(row)) + "\n"
        )
        errors = self.errors(path, monkeypatch)
        assert len(errors) == 2
        assert "rehashes.jsonl:1" in errors[0] and "rule_sha256" in errors[1]

    def test_a_rehash_with_another_change_fails(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.ledger(path, ledger_row())
        path.write_text(path.read_text().replace("0.02", "0.05"))
        errors = self.errors(path, monkeypatch)
        assert len(errors) == 1 and "params" in errors[0] and "rule_sha256" not in errors[0]


class TestHoldoutLedgerHook:
    ROW = json.dumps(
        {"kind": "reserved", "issue": 1, "market": "1x2", "version": 1, "git_sha": "a" * 40}
    )
    OTHER = json.dumps(
        {"kind": "reserved", "issue": 2, "market": "1x2", "version": 1, "git_sha": "a" * 40}
    )

    @pytest.fixture
    def repo(self, candidates_repo: Path) -> Path:
        directory = candidates_repo / "candidates"
        (directory / "holdout.jsonl").write_text(self.ROW + "\n")
        for folder in ("preregistrations", "selections"):
            (directory / folder).mkdir()
            (directory / folder / "1-1x2-v1.yaml").write_text("issue: 1\n")
        git(candidates_repo, "add", "candidates")
        git(candidates_repo, "commit", "-q", "-m", "base")
        return candidates_repo

    def errors(self, repo: Path) -> list[str]:
        return hook.check("HEAD")

    def test_appending_passes(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(repo)
        with (repo / "candidates" / "holdout.jsonl").open("a") as f:
            f.write(self.OTHER + "\n")
        assert self.errors(repo) == []

    def test_removing_a_committed_row_fails(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(repo)
        (repo / "candidates" / "holdout.jsonl").write_text("")
        assert any("append-only" in e for e in self.errors(repo))

    def test_editing_a_committed_row_fails(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(repo)
        (repo / "candidates" / "holdout.jsonl").write_text(
            self.ROW.replace('"version": 1', '"version": 7') + "\n"
        )
        assert any("append-only" in e for e in self.errors(repo))

    def test_reserving_a_version_twice_fails(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(repo)
        with (repo / "candidates" / "holdout.jsonl").open("a") as f:
            f.write(self.ROW + "\n")
        assert any("reserved twice" in e for e in self.errors(repo))

    def test_changing_a_preregistration_fails(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(repo)
        (repo / "candidates" / "preregistrations" / "1-1x2-v1.yaml").write_text("issue: 2\n")
        assert any("changed" in e for e in self.errors(repo))

    def test_changing_a_selection_fails(self, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(repo)
        (repo / "candidates" / "selections" / "1-1x2-v1.yaml").write_text("issue: 2\n")
        assert any("selections" in e for e in self.errors(repo))

    @pytest.mark.parametrize("folder", ["preregistrations", "selections"])
    def test_renaming_or_deleting_a_file_fails(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch, folder: str
    ) -> None:
        monkeypatch.chdir(repo)
        path = repo / "candidates" / folder / "1-1x2-v1.yaml"
        path.rename(path.with_name("1-1x2-v2.yaml"))
        assert any("deleted or renamed" in e for e in self.errors(repo))
        path.with_name("1-1x2-v2.yaml").unlink()
        assert any("deleted or renamed" in e for e in self.errors(repo))

    def test_another_market_may_reserve_the_same_issue(
        self, repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(repo)
        other = json.dumps({"kind": "reserved", "issue": 1, "market": "totals", "version": 1})
        with (repo / "candidates" / "holdout.jsonl").open("a") as f:
            f.write(other + "\n")
        assert self.errors(repo) == []
