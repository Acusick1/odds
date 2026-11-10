"""Unit tests for recording a rehash that changes no bet."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
import yaml
from odds_analytics import candidate_rehash
from odds_core.candidates import load_candidates, load_rehashes
from odds_lambda.candidate_rules import RULES, rule_sha256

from tests.unit.conftest import git_commit

OLD = "a" * 64
REGISTERED = {
    "name": "gap",
    "description": "d",
    "kind": "mechanical",
    "market": "1x2",
    "sport": "soccer_epl",
    "version": 1,
    "issue": 1,
    "start_date": "2020-01-01",
    "status": "active",
    "rule": "exchange_gap",
    "rule_sha256": OLD,
    "params": {"threshold": 0.01, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
}
LINK = "https://github.com/Acusick1/odds/pull/1"


class TestRehash:
    @pytest.fixture
    def root(self, candidates_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        (candidates_repo / "candidates" / "gap.yaml").write_text(yaml.safe_dump(REGISTERED))
        git_commit(candidates_repo)
        monkeypatch.setattr(candidate_rehash, "_git", lambda root, *args: "base")
        return candidates_repo

    def run(
        self,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        replays: Callable[[Path], bytes],
        parity: tuple[bool, str] = (True, "gap v1: 3/3 matched"),
    ) -> candidate_rehash.RehashResult:
        monkeypatch.setattr(
            candidate_rehash, "replay_under", lambda checkout, *_: replays(checkout)
        )

        async def parity_summary(*_: object) -> tuple[bool, str]:
            return parity

        monkeypatch.setattr(candidate_rehash, "parity_summary", parity_summary)
        import asyncio

        return asyncio.run(
            candidate_rehash.rehash(
                root,
                "exchange_gap",
                "postgresql://x",
                date(2025, 1, 1),
                date(2025, 2, 1),
                LINK,
                "base",
            )
        )

    def test_identical_replays_record_the_rehash(
        self, root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result = self.run(root, monkeypatch, lambda _: b"same")
        assert result.refused is None
        (entry,) = load_rehashes(root / "candidates")
        assert (entry.rule, entry.old, entry.new) == (
            "exchange_gap",
            OLD,
            rule_sha256("exchange_gap"),
        )
        assert entry.evidence.link == LINK and "byte-identical" in entry.evidence.replay
        assert entry.evidence.parity == "gap v1: 3/3 matched"
        (candidate,) = load_candidates(root / "candidates")
        assert candidate.rule_sha256 == rule_sha256("exchange_gap")

    @pytest.mark.parametrize(
        ("replays", "parity"),
        [
            (lambda checkout: str(checkout).encode(), (True, "")),
            (lambda _: b"same", (False, "gap v1: 2/3 matched")),
        ],
        ids=["replays differ", "parity mismatch"],
    )
    def test_a_difference_writes_nothing(
        self,
        root: Path,
        monkeypatch: pytest.MonkeyPatch,
        replays: Callable[[Path], bytes],
        parity: tuple[bool, str],
    ) -> None:
        before = (root / "candidates" / "gap.yaml").read_text()
        result = self.run(root, monkeypatch, replays, parity)
        assert result.refused is not None
        assert load_rehashes(root / "candidates") == []
        assert (root / "candidates" / "gap.yaml").read_text() == before


class TestLedger:
    def test_the_committed_ledger_is_valid(self) -> None:
        from odds_core.candidates import candidates_dir

        for entry in load_rehashes(candidates_dir()):
            assert entry.rule in RULES
