"""Unit tests for scripts/check_candidate_versions.py."""

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_candidate_versions.py"

BASE_YAML = """\
name: back-longest
description: Back the longest price
version: 2
status: active
params:
  min_edge: 0.02
"""


def days_from_today(days: int) -> str:
    return (datetime.now(UTC).date() + timedelta(days=days)).isoformat()


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


class TestCheckCandidateVersions:
    @pytest.fixture
    def repo(self, tmp_path: Path) -> Path:
        git(tmp_path, "init", "-q", "-b", "main")
        git(tmp_path, "config", "user.email", "t@example.com")
        git(tmp_path, "config", "user.name", "t")
        (tmp_path / "candidates").mkdir()
        (tmp_path / "candidates" / "back-longest.yaml").write_text(BASE_YAML)
        git(tmp_path, "add", "candidates")
        git(tmp_path, "commit", "-q", "-m", "base")
        return tmp_path

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
        self.edit(repo, "version: 2", f"version: 3\nstart_date: {days_from_today(0)}")
        assert self.run(repo).returncode == 0

    def test_bump_with_a_back_dated_start_fails(self, repo: Path) -> None:
        self.edit(repo, "version: 2", f"version: 3\nstart_date: {days_from_today(-1)}")
        result = self.run(repo)
        assert result.returncode == 1 and "before today" in result.stderr

    def test_bump_without_a_start_date_fails(self, repo: Path) -> None:
        self.edit(repo, "version: 2", "version: 3")
        result = self.run(repo)
        assert result.returncode == 1 and "needs a start_date" in result.stderr

    def test_new_file_starting_today_or_later_passes(self, repo: Path) -> None:
        for offset, name in ((0, "today"), (5, "later")):
            (repo / "candidates" / f"{name}.yaml").write_text(
                f"name: {name}\nversion: 1\nstart_date: {days_from_today(offset)}\n"
            )
        assert self.run(repo).returncode == 0

    def test_new_file_with_a_back_dated_start_fails(self, repo: Path) -> None:
        (repo / "candidates" / "old.yaml").write_text(
            f"name: old\nversion: 1\nstart_date: {days_from_today(-1)}\n"
        )
        result = self.run(repo)
        assert result.returncode == 1
        assert "old.yaml" in result.stderr and "before today" in result.stderr

    def test_new_file_without_a_start_date_fails(self, repo: Path) -> None:
        (repo / "candidates" / "bare.yaml").write_text("name: bare\nversion: 1\n")
        assert self.run(repo).returncode == 1

    def test_version_decrease_fails(self, repo: Path) -> None:
        self.edit(repo, "version: 2", "version: 1")
        result = self.run(repo)
        assert result.returncode == 1 and "decreased" in result.stderr

    def test_deletion_fails(self, repo: Path) -> None:
        (repo / "candidates" / "back-longest.yaml").unlink()
        result = self.run(repo)
        assert result.returncode == 1 and "status: killed" in result.stderr

    def test_new_file_passes(self, repo: Path) -> None:
        (repo / "candidates" / "fresh.yaml").write_text(
            BASE_YAML.replace("back-longest", "fresh") + f"start_date: {days_from_today(1)}\n"
        )
        assert self.run(repo).returncode == 0

    def test_default_base_is_merge_base_with_main(self, repo: Path) -> None:
        git(repo, "checkout", "-q", "-b", "feature")
        self.edit(repo, "0.02", "0.05")
        self.edit(repo, "version: 2", f"version: 3\nstart_date: {days_from_today(1)}")
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
