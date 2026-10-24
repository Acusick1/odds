"""Unit tests for the git SHA helper."""

import subprocess

import pytest
from odds_core import git_info
from odds_core.git_info import current_git_sha


class TestCurrentGitSha:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        current_git_sha.cache_clear()
        yield
        current_git_sha.cache_clear()

    def test_returns_head_of_the_project_checkout(self) -> None:
        sha = current_git_sha()
        assert sha is not None and len(sha) == 40

    def test_none_when_git_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*args, **kwargs):
            raise subprocess.CalledProcessError(128, "git")

        monkeypatch.setattr(git_info.subprocess, "run", fail)
        assert current_git_sha() is None

    def test_none_when_git_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(git_info.subprocess, "run", missing)
        assert current_git_sha() is None

    def test_resolved_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def run(*args, **kwargs):
            calls.append(1)
            return subprocess.CompletedProcess(args, 0, stdout="abc\n")

        monkeypatch.setattr(git_info.subprocess, "run", run)
        assert current_git_sha() == "abc"
        assert current_git_sha() == "abc"
        assert len(calls) == 1
