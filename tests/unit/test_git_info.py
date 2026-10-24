"""Unit tests for the git SHA helper."""

import subprocess

import pytest
from odds_core import git_info
from odds_core.git_info import current_git_sha, resolve_git_sha


def fake_git(monkeypatch: pytest.MonkeyPatch, head: str, status: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def run(cmd, *args, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=head if cmd[1] == "rev-parse" else status)

    monkeypatch.setattr(git_info.subprocess, "run", run)
    return calls


class TestResolveGitSha:
    def test_returns_head_of_the_project_checkout(self) -> None:
        sha = resolve_git_sha()
        assert sha is not None and len(sha.removesuffix("-dirty")) == 40

    def test_clean_tree_is_the_bare_sha(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_git(monkeypatch, "abc\n", "")
        assert resolve_git_sha() == "abc"

    def test_uncommitted_changes_mark_the_sha_dirty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = fake_git(monkeypatch, "abc\n", " M file.py\n")
        assert resolve_git_sha() == "abc-dirty"
        assert ["git", "status", "--porcelain"] in calls

    def test_none_when_git_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(git_info.subprocess, "run", fail)
        assert resolve_git_sha() is None

    def test_none_when_git_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(git_info.subprocess, "run", missing)
        assert resolve_git_sha() is None


class TestCurrentGitSha:
    def test_is_the_value_resolved_at_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(git_info, "GIT_SHA", "fixed")
        calls = fake_git(monkeypatch, "other", "")
        assert current_git_sha() == "fixed"
        assert calls == []
