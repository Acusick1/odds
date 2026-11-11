"""Unit tests for recording a rehash that changes no bet."""

from __future__ import annotations

import asyncio
import fcntl
import json
import subprocess
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from odds_analytics import candidate_rehash
from odds_analytics.candidate_rehash import RehashResult, Replay
from odds_core.candidates import candidates_dir, load_candidates
from odds_core.rehash_ledger import REHASH_LEDGER, Rehash, load_rehashes
from odds_lambda.candidate_rules import RULES

from scripts import check_candidate_versions as hook
from tests.unit.conftest import INVALID_LEDGER_ROWS, git_commit, ledger_row

OLD, NEW = "a" * 64, "b" * 64
NAMES = ("gap-one", "gap-two")
LINK = "https://github.com/Acusick1/odds/pull/1"
START, END = date(2025, 1, 1), date(2025, 2, 1)


def candidate(name: str, rule_sha256: str = OLD) -> dict[str, Any]:
    return {
        "name": name,
        "description": "d",
        "kind": "mechanical",
        "market": "1x2",
        "sports": ["soccer_epl"],
        "version": 1,
        "issue": 1,
        "start_date": "2020-01-01",
        "status": "active",
        "rule": "exchange_gap",
        "rule_sha256": rule_sha256,
        "params": {"threshold": 0.01, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
    }


def replay_output(decisions: int = 5, bets: int = 2) -> bytes:
    return json.dumps(
        {n: {"counts": {"with_input": decisions}, "bets": [{}] * bets} for n in NAMES}
    ).encode()


class Fakes:
    """What the rehash reads from git checkouts and the database, set per test."""

    def __init__(self) -> None:
        self.before = Replay(replay_output(), OLD)
        self.after = Replay(replay_output(), NEW)
        self.parity: dict[str, tuple[int, int]] = dict.fromkeys(NAMES, (3, 3))


class TestRehash:
    @pytest.fixture
    def root(self, candidates_repo: Path) -> Path:
        for name in NAMES:
            (candidates_repo / "candidates" / f"{name}.yaml").write_text(
                yaml.safe_dump(candidate(name))
            )
        git_commit(candidates_repo)
        return candidates_repo

    @pytest.fixture
    def fakes(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> Fakes:
        fakes = Fakes()
        checkout = root / "base-checkout"

        @asynccontextmanager
        async def frozen_snapshot(url: str) -> AsyncIterator[tuple[None, str]]:
            yield None, "snapshot-1"

        @contextmanager
        def base_checkout(root: Path, base: str) -> Iterator[Path]:
            yield checkout

        def replay_under(where: Path, *args: Any) -> Replay:
            assert args[-1] == "snapshot-1"
            return fakes.before if where == checkout else fakes.after

        async def parity_counts(connection: None, c: Any) -> tuple[int, int]:
            return fakes.parity[c.name]

        monkeypatch.setattr(candidate_rehash, "rule_sha256", lambda rule: NEW)
        monkeypatch.setattr(candidate_rehash, "frozen_snapshot", frozen_snapshot)
        monkeypatch.setattr(candidate_rehash, "base_checkout", base_checkout)
        monkeypatch.setattr(candidate_rehash, "replay_under", replay_under)
        monkeypatch.setattr(candidate_rehash, "parity_counts", parity_counts)
        return fakes

    def run(self, root: Path, start: date = START, end: date = END) -> RehashResult:
        return asyncio.run(
            candidate_rehash.rehash(
                root,
                "exchange_gap",
                "postgresql+asyncpg://odds_web:pw@db:5433/odds",
                start,
                end,
                LINK,
                "HEAD",
            )
        )

    def unchanged(self, root: Path) -> bool:
        directory = root / "candidates"
        return load_rehashes(directory) == [] and all(
            c.rule_sha256 == OLD for c in load_candidates(directory)
        )

    def test_identical_replays_record_every_candidate_of_the_rule(
        self, root: Path, fakes: Fakes
    ) -> None:
        result = self.run(root)
        assert result.refused is None
        (entry,) = load_rehashes(root / "candidates")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        ).stdout.strip()
        assert (entry.rule, entry.old, entry.new, entry.base, entry.head) == (
            "exchange_gap",
            OLD,
            NEW,
            head,
            head,
        )
        assert entry.database == "odds_web@db:5433/odds" and "pw" not in entry.model_dump_json()
        assert [(c.name, c.decisions, c.bets, c.parity_matched) for c in entry.candidates] == [
            (n, 5, 2, 3) for n in NAMES
        ]
        assert all(c.rule_sha256 == NEW for c in load_candidates(root / "candidates"))
        assert hook._row_problem(entry.model_dump_json()) is None

    def test_a_candidate_without_recorded_decisions_is_recorded_as_such(
        self, root: Path, fakes: Fakes
    ) -> None:
        fakes.parity["gap-two"] = (0, 0)
        assert self.run(root).refused is None
        (entry,) = load_rehashes(root / "candidates")
        assert (entry.candidates[1].parity_matched, entry.candidates[1].parity_total) == (0, 0)

    @pytest.mark.parametrize(
        ("setup", "reason"),
        [
            (lambda f: setattr(f, "before", Replay(replay_output(), "c" * 64)), "hash at base"),
            (lambda f: setattr(f, "after", Replay(replay_output(bets=3), NEW)), "differ"),
            (
                lambda f: (
                    setattr(f, "before", Replay(replay_output(decisions=0), OLD)),
                    setattr(f, "after", Replay(replay_output(decisions=0), NEW)),
                ),
                "no decision",
            ),
            (
                lambda f: (
                    setattr(f, "before", Replay(replay_output(bets=0), OLD)),
                    setattr(f, "after", Replay(replay_output(bets=0), NEW)),
                ),
                "no bet",
            ),
            (lambda f: f.parity.update({"gap-one": (2, 3)}), "parity matched 2 of 3"),
        ],
        ids=["base hash differs", "replays differ", "no decisions", "no bets", "parity mismatch"],
    )
    def test_missing_or_contrary_evidence_writes_nothing(
        self, root: Path, fakes: Fakes, setup: Callable[[Fakes], object], reason: str
    ) -> None:
        setup(fakes)
        result = self.run(root)
        assert result.refused is not None and reason in result.refused
        assert self.unchanged(root)

    def test_an_empty_range_is_refused(self, root: Path, fakes: Fakes) -> None:
        assert "after" in (self.run(root, start=END, end=START).refused or "")
        assert self.unchanged(root)

    def test_candidates_recording_different_hashes_are_refused(
        self, root: Path, fakes: Fakes
    ) -> None:
        path = root / "candidates" / "gap-two.yaml"
        path.write_text(yaml.safe_dump(candidate("gap-two", "c" * 64)))
        git_commit(root)
        assert "hash at base" in (self.run(root).refused or "")

    def test_uncommitted_code_is_refused(self, root: Path, fakes: Fakes) -> None:
        (root / "rule.py").write_text("x = 1\n")
        assert "commit" in (self.run(root).refused or "")
        assert self.unchanged(root)

    def test_a_concurrent_rehash_is_refused(self, root: Path, fakes: Fakes) -> None:
        with (root / ".git" / candidate_rehash.LOCK_FILE).open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert "lock" in (self.run(root).refused or "")
        assert self.unchanged(root)

    def test_a_comment_on_the_hash_line_is_kept_and_the_hash_updated(
        self, root: Path, fakes: Fakes
    ) -> None:
        path = root / "candidates" / "gap-one.yaml"
        path.write_text(
            path.read_text().replace(f"rule_sha256: {OLD}", f"rule_sha256: {OLD} # pinned")
        )
        git_commit(root)
        assert self.run(root).refused is None
        assert f"rule_sha256: {NEW} # pinned" in path.read_text()
        assert load_candidates(root / "candidates")[0].rule_sha256 == NEW

    @pytest.mark.parametrize(
        "edit",
        [
            lambda text: text.replace(f"rule_sha256: {OLD}", f"rule_sha256: '{OLD}'"),
            lambda text: text + f"rule_sha256: {OLD}\n",
        ],
        ids=["quoted value", "duplicate key"],
    )
    def test_a_hash_line_it_cannot_update_safely_is_refused(
        self, root: Path, fakes: Fakes, edit: Callable[[str], str]
    ) -> None:
        path = root / "candidates" / "gap-one.yaml"
        path.write_text(edit(path.read_text()))
        git_commit(root)
        before = path.read_text()
        result = self.run(root)
        assert result.refused is not None and "gap-one.yaml" in result.refused
        assert path.read_text() == before and load_rehashes(root / "candidates") == []

    def test_a_file_that_does_not_record_the_new_hash_after_writing_leaves_no_ledger_row(
        self, root: Path, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(candidate_rehash, "_write_atomically", lambda path, content: None)
        result = self.run(root)
        assert result.refused is not None and "after writing" in result.refused
        assert load_rehashes(root / "candidates") == []

    def test_a_failed_write_leaves_no_ledger_row(
        self, root: Path, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        written: list[Path] = []
        real = candidate_rehash._write_atomically

        def write(path: Path, content: str) -> None:
            if written:
                raise OSError("disk full")
            written.append(path)
            real(path, content)

        monkeypatch.setattr(candidate_rehash, "_write_atomically", write)
        with pytest.raises(OSError):
            self.run(root)
        assert load_rehashes(root / "candidates") == []


class TestLedger:
    def test_the_committed_ledger_is_valid(self) -> None:
        path = candidates_dir() / REHASH_LEDGER
        for line in path.read_text().splitlines() if path.exists() else []:
            assert hook._row_problem(line) is None
            assert Rehash.model_validate_json(line).rule in RULES

    def test_the_model_and_the_hook_accept_a_complete_row(self) -> None:
        Rehash.model_validate(ledger_row())
        assert hook._row_problem(json.dumps(ledger_row())) is None

    @pytest.mark.parametrize("case", list(INVALID_LEDGER_ROWS))
    def test_the_model_and_the_hook_reject_the_same_rows(self, case: str) -> None:
        row = INVALID_LEDGER_ROWS[case]
        line = row if isinstance(row, str) else json.dumps(row)
        with pytest.raises(ValueError):
            Rehash.model_validate_json(line)
        assert hook._row_problem(line) is not None
