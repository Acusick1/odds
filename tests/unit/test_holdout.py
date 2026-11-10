"""The sealed holdout: the seal in the replay library, the pre-registration, the durable
one-shot ledger, the pooled gate and the command."""

from __future__ import annotations

import asyncio
import json
import subprocess
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from odds_analytics import candidate_backtest as bt
from odds_analytics import holdout, holdout_run
from odds_analytics.candidate_report import Checkpoint, ScoredBet
from odds_cli.main import app
from odds_core.candidates import MechanicalCandidate
from typer.testing import CliRunner

from tests.unit.conftest import HOLDOUT_CELL, HOLDOUT_GRID, git_commit, preregistration
from tests.unit.test_candidates import MECHANICAL

RUNNER = CliRunner(env={"COLUMNS": "250"})
START, END = date(2025, 9, 1), date(2025, 9, 30)
NOW = datetime(2026, 11, 1, tzinfo=UTC)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


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


def scored_bet(event_id: int, p_close: float, year: int) -> ScoredBet:
    when = datetime(year, 9, 1, tzinfo=UTC)
    return ScoredBet("back-longest", 1, event_id, when, 1.0, 0.0, 2.0, p_close, when, timedelta(0))


def replayed(bets: list[ScoredBet]) -> bt.Replayed:
    counts = bt._Counts(decisions=len(bets), with_input=len(bets), with_bet=len(bets))
    return bt.Replayed(bets, counts, len(bets), len(bets), None, (), False)


class TestSealInTheLibrary:
    def test_direct_replay_of_a_sealed_range_is_refused_before_any_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        query = AsyncMock(return_value=[])
        monkeypatch.setattr(bt, "final_events", query)
        with pytest.raises(holdout.HoldoutRefused, match="sealed holdout"):
            asyncio.run(bt.replay(None, draft(), START, END, 1))  # type: ignore[arg-type]
        query.assert_not_awaited()

    def test_final_events_refuses_a_sealed_range_itself(self) -> None:
        session = AsyncMock()
        with pytest.raises(holdout.HoldoutRefused):
            asyncio.run(bt.final_events(session, "soccer_germany_bundesliga", START, END))
        session.execute.assert_not_awaited()

    def test_a_range_mixing_development_and_holdout_days_is_refused(self) -> None:
        with pytest.raises(holdout.HoldoutRefused):
            asyncio.run(
                bt.replay(None, draft(), date(2024, 8, 1), date(2026, 7, 31), 1)  # type: ignore[arg-type]
            )

    def test_development_days_of_a_sealed_league_still_replay(self) -> None:
        holdout.require_clear("soccer_spain_la_liga", date(2024, 8, 1), date(2025, 7, 31))

    def test_an_access_for_one_league_opens_no_other(self) -> None:
        access = reservation_access("soccer_spain_la_liga")
        holdout.require_clear("soccer_spain_la_liga", START, END, access)
        with pytest.raises(holdout.HoldoutRefused):
            holdout.require_clear("soccer_germany_bundesliga", START, END, access)

    def test_an_access_cannot_be_made_outside_a_reservation(self) -> None:
        window = holdout.windows_of("soccer_spain_la_liga")[0]
        with pytest.raises(holdout.HoldoutRefused):
            holdout.HoldoutAccess(window, object())

    def test_compare_cannot_open_a_sealed_league_for_a_development_primary(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(bt, "final_events", AsyncMock(return_value=[]))
        monkeypatch.setattr(
            "odds_cli.commands.candidates.candidates_dir", lambda: tmp_path / "candidates"
        )

        @asynccontextmanager
        async def session() -> Any:
            yield None

        async def starts(_session: object) -> dict[tuple[str, int], datetime]:
            return {}

        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
        monkeypatch.setattr("odds_cli.commands.candidates.load_decision_starts", starts)
        primary, other = tmp_path / "epl.yaml", tmp_path / "laliga.yaml"
        primary.write_text(yaml.safe_dump(draft(sport="soccer_epl").model_dump(mode="json")))
        other.write_text(yaml.safe_dump(draft(issue=999).model_dump(mode="json")))

        result = RUNNER.invoke(
            app,
            ["candidates", "backtest", str(primary), "--from", str(START), "--to", str(END)]
            + ["--compare", str(other)],
        )

        assert result.exit_code == 1 and "sealed holdout" in result.stdout

    def test_the_replay_script_never_replays_sealed_days(self) -> None:
        segments = holdout.development_segments("soccer_epl", date(2015, 8, 1), date(2026, 11, 9))
        assert segments == [
            (date(2015, 8, 1), date(2022, 7, 31)),
            (date(2024, 8, 1), date(2026, 11, 9)),
        ]
        assert holdout.development_segments(
            "soccer_spain_la_liga", date(2015, 8, 1), date(2026, 7, 31)
        ) == [(date(2015, 8, 1), date(2025, 7, 31))]
        assert holdout.development_segments("mlb", date(2015, 8, 1), date(2026, 7, 31)) == [
            (date(2015, 8, 1), date(2026, 7, 31))
        ]


def reservation_access(sport: str) -> holdout.HoldoutAccess:
    prereg = holdout.Preregistration.model_validate(preregistration())
    reservation = holdout.Reservation(prereg, Path("."), "a" * 40, "b" * 40)
    return reservation.access(sport)


class TestWindows:
    @pytest.mark.parametrize(
        ("sport", "day", "sealed"),
        [
            ("soccer_spain_la_liga", "2025-07-31", False),
            ("soccer_spain_la_liga", "2025-08-01", True),
            ("soccer_spain_la_liga", "2026-07-31", True),
            ("soccer_spain_la_liga", "2026-08-01", False),
            ("soccer_epl", "2025-09-01", False),
            ("soccer_epl", "2023-01-01", True),
            ("soccer_epl", "2022-07-31", False),
        ],
    )
    def test_inclusive_edges_by_sport(self, sport: str, day: str, sealed: bool) -> None:
        d = date.fromisoformat(day)
        assert bool(holdout.intersecting(sport, d, d)) is sealed

    def test_epl_2022_24_follows_the_switch_and_defaults_sealed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert holdout.EPL_2022_24_SEALED is True
        monkeypatch.setattr(holdout, "EPL_2022_24_SEALED", False)
        assert not holdout.intersecting("soccer_epl", date(2023, 1, 1), date(2023, 1, 2))

    def test_events_are_selected_by_utc_kickoff_day(self) -> None:
        session = MagicMock()
        rows = MagicMock()
        rows.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=rows)
        asyncio.run(
            bt.final_events(session, "soccer_spain_la_liga", date(2024, 9, 1), date(2024, 9, 30))
        )
        params = session.execute.await_args.args[0].compile().params
        assert datetime(2024, 9, 1, tzinfo=UTC) in params.values()
        assert datetime(2024, 10, 1, tzinfo=UTC) in params.values()


class TestPreregistration:
    def test_binds_the_strategy_grid_and_selection_rule(self) -> None:
        prereg = holdout.Preregistration.model_validate(preregistration())
        assert (prereg.name, prereg.version, prereg.rule, prereg.market) == (
            "back-longest",
            1,
            "exchange_gap",
            "1x2",
        )
        assert prereg.grid == HOLDOUT_GRID

    @pytest.mark.parametrize(
        "changes",
        [
            {"cell": {**HOLDOUT_CELL, "threshold": 0.03}},
            {"cell": {"threshold": 0.02}},
            {"grid": {**HOLDOUT_GRID, "extra": [1]}},
            {"selection_rule": "see the issue"},
            {"leagues": ["soccer_spain_la_liga", "soccer_spain_la_liga"]},
            {"version": "2"},
            {"unknown": 1},
        ],
    )
    def test_rejects_an_incomplete_or_inconsistent_file(self, changes: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            holdout.Preregistration.model_validate(preregistration(**changes))

    def test_a_league_without_a_window_is_refused(self, holdout_repo: Path) -> None:
        path = holdout_repo / "preregistrations" / "610.yaml"
        path.write_text(yaml.safe_dump(preregistration(leagues=["mlb"])))
        with pytest.raises(holdout.HoldoutRefused, match="no sealed window"):
            holdout.load_preregistration(path, holdout_repo)


class TestPreregistrationCommit:
    def commit_of(self, directory: Path, backfill: datetime | None = None) -> str:
        path = directory / "preregistrations" / "610.yaml"
        return holdout.preregistration_commit(path, directory.parent, backfill)

    def test_one_commit_before_the_backfill_passes(self, holdout_repo: Path) -> None:
        sha = self.commit_of(holdout_repo, datetime.now(UTC) + timedelta(days=1))
        assert sha == git(holdout_repo.parent, "rev-parse", "HEAD")

    def test_no_backfill_yet_passes(self, holdout_repo: Path) -> None:
        self.commit_of(holdout_repo, None)

    def test_committed_after_the_first_backfill_snapshot_is_refused(
        self, holdout_repo: Path
    ) -> None:
        with pytest.raises(holdout.HoldoutRefused, match="not before the first backfill"):
            self.commit_of(holdout_repo, datetime.now(UTC) - timedelta(days=1))

    def test_uncommitted_is_refused(self, holdout_repo: Path) -> None:
        other = holdout_repo / "preregistrations" / "611.yaml"
        other.write_text("issue: 611\n")
        with pytest.raises(holdout.HoldoutRefused, match="not committed"):
            holdout.preregistration_commit(other, holdout_repo.parent, None)

    def test_edited_since_commit_is_refused(self, holdout_repo: Path) -> None:
        path = holdout_repo / "preregistrations" / "610.yaml"
        path.write_text(path.read_text().replace("0.02", "0.05"))
        with pytest.raises(holdout.HoldoutRefused, match="differs from its committed"):
            self.commit_of(holdout_repo)

    def test_a_second_commit_touching_it_is_refused(self, holdout_repo: Path) -> None:
        path = holdout_repo / "preregistrations" / "610.yaml"
        path.write_text(yaml.safe_dump(preregistration(cell={**HOLDOUT_CELL, "threshold": 0.05})))
        git_commit(holdout_repo.parent)
        with pytest.raises(holdout.HoldoutRefused, match="exactly one commit"):
            self.commit_of(holdout_repo)


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


class TestLedger:
    def reserve(self, directory: Path, **changes: object) -> holdout.Reservation:
        prereg = holdout.Preregistration.model_validate(preregistration(**changes))
        path = directory / "preregistrations" / "610.yaml"
        with holdout.exclusive_lock(directory.parent):
            return holdout.reserve(prereg, path, directory, None, NOW)

    def test_reserving_writes_a_reserved_row(self, holdout_repo: Path) -> None:
        self.reserve(holdout_repo)
        (row,) = holdout.load_ledger(holdout_repo)
        assert isinstance(row, holdout.Reserved) and (row.issue, row.version) == (610, 1)
        assert row.git_sha == git(holdout_repo.parent, "rev-parse", "HEAD")

    def test_a_second_reservation_is_refused_until_the_first_is_committed(
        self, holdout_repo: Path
    ) -> None:
        self.reserve(holdout_repo)
        with pytest.raises(holdout.HoldoutRefused, match="uncommitted"):
            self.reserve(holdout_repo)

    def test_a_committed_reservation_consumes_the_version(self, holdout_repo: Path) -> None:
        self.reserve(holdout_repo)
        git_commit(holdout_repo.parent)
        with pytest.raises(holdout.HoldoutRefused, match="already ran"):
            self.reserve(holdout_repo)

    def test_erasing_the_committed_ledger_does_not_free_the_version(
        self, holdout_repo: Path
    ) -> None:
        self.reserve(holdout_repo)
        git_commit(holdout_repo.parent)
        (holdout_repo / "holdout.jsonl").write_text("")
        with pytest.raises(holdout.HoldoutRefused):
            self.reserve(holdout_repo)

    def test_a_held_lock_refuses_a_concurrent_run(self, holdout_repo: Path) -> None:
        with holdout.exclusive_lock(holdout_repo.parent):
            with pytest.raises(holdout.HoldoutRefused, match="holds the lock"):
                with holdout.exclusive_lock(holdout_repo.parent):
                    pass

    def test_a_ledger_missing_origin_mains_rows_is_refused(self, holdout_repo: Path) -> None:
        root = holdout_repo.parent
        self.reserve(holdout_repo)
        git_commit(root)
        git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
        (holdout_repo / "holdout.jsonl").write_text("")
        git_commit(root)
        with pytest.raises(holdout.HoldoutRefused, match="extend"):
            self.reserve(holdout_repo, version=2)

    def test_rows_validate_and_a_result_needs_its_reservation(self) -> None:
        with pytest.raises(ValueError, match="without one reservation"):
            holdout.parse_ledger(result_row().model_dump_json())

    def test_the_committed_ledger_validates(self) -> None:
        directory = Path(__file__).resolve().parents[2] / "candidates"
        assert (directory / holdout.HOLDOUT_LEDGER).exists()
        holdout.load_ledger(directory)
        for path in (directory / holdout.PREREGISTRATION_DIR).glob("*.yaml"):
            holdout.load_preregistration(path, directory)


def result_row() -> holdout.HoldoutResult:
    return holdout.HoldoutResult(
        issue=610,
        version=1,
        name="back-longest",
        rule="exchange_gap",
        rule_sha256="c" * 64,
        cell=HOLDOUT_CELL,
        windows=holdout.windows_of("soccer_spain_la_liga"),
        ran_on=date(2026, 11, 1),
        git_sha="a" * 40,
        preregistration_commit="b" * 40,
        result=holdout.gate(None),
    )


class TestRun:
    @pytest.fixture
    def collected(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, date, date, bool]]:
        """Stub the replay: la Liga's holdout bets are good, the Bundesliga's bad."""
        calls: list[tuple[str, date, date, bool]] = []

        async def collect(
            _session: object, candidate: MechanicalCandidate, start: date, end: date, **kw: Any
        ) -> bt.Replayed:
            calls.append((candidate.sport, start, end, kw["access"] is not None))
            p_close = 0.6 if candidate.sport == "soccer_spain_la_liga" else 0.3
            return replayed(
                [scored_bet(i + (0 if p_close > 0.5 else 100), p_close, 2025) for i in range(40)]
            )

        async def no_backfill(_session: object) -> None:
            return None

        monkeypatch.setattr(holdout_run, "collect", collect)
        monkeypatch.setattr(holdout_run, "earliest_backfill", no_backfill)
        return calls

    def run(self, directory: Path) -> holdout_run.HoldoutRun:
        path = directory / "preregistrations" / "610.yaml"
        return asyncio.run(holdout_run.run_holdout(None, path, directory, NOW))  # type: ignore[arg-type]

    def test_replays_each_leagues_window_with_access_and_pools_one_gate(
        self, holdout_repo: Path, collected: list[tuple[str, date, date, bool]]
    ) -> None:
        done = self.run(holdout_repo)

        assert collected == [
            ("soccer_spain_la_liga", date(2025, 8, 1), date(2026, 7, 31), True),
            ("soccer_germany_bundesliga", date(2025, 8, 1), date(2026, 7, 31), True),
        ]
        reserved, result = holdout.load_ledger(holdout_repo)
        assert isinstance(reserved, holdout.Reserved) and isinstance(result, holdout.HoldoutResult)
        assert result.result.bets == 80 and result.cell == HOLDOUT_CELL
        assert result.preregistration_commit == git(holdout_repo.parent, "rev-parse", "HEAD")
        assert done.row == result

    def test_a_second_run_is_refused_without_reading_any_holdout_data(
        self, holdout_repo: Path, collected: list[tuple[str, date, date, bool]]
    ) -> None:
        self.run(holdout_repo)
        git_commit(holdout_repo.parent)
        collected.clear()
        with pytest.raises(holdout.HoldoutRefused, match="already ran"):
            self.run(holdout_repo)
        assert collected == []

    def test_a_failure_after_reserving_spends_the_run(
        self,
        holdout_repo: Path,
        collected: list[tuple[str, date, date, bool]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def boom(*_a: object, **_k: object) -> bt.Replayed:
            raise RuntimeError("database went away")

        monkeypatch.setattr(holdout_run, "collect", boom)
        with pytest.raises(RuntimeError):
            self.run(holdout_repo)
        (row,) = holdout.load_ledger(holdout_repo)
        assert isinstance(row, holdout.Reserved)
        git_commit(holdout_repo.parent)
        with pytest.raises(holdout.HoldoutRefused, match="already ran"):
            self.run(holdout_repo)

    def test_a_bad_cell_params_cost_no_run(
        self, holdout_repo: Path, collected: list[tuple[str, date, date, bool]]
    ) -> None:
        path = holdout_repo / "preregistrations" / "610.yaml"
        bad = {
            "threshold": 0.02,
            "minutes_before_kickoff": [60],
            "max_input_age_minutes": 10,
            "x": 1,
        }
        grid = {**HOLDOUT_GRID, "x": [1]}
        path.write_text(yaml.safe_dump(preregistration(cell=bad, grid=grid)))
        git_commit(holdout_repo.parent)
        with pytest.raises(holdout.HoldoutRefused):
            asyncio.run(holdout_run.run_holdout(None, path, holdout_repo, NOW))  # type: ignore[arg-type]
        assert holdout.load_ledger(holdout_repo) == [] and collected == []

    def test_the_gate_is_the_holdouts_alone_not_a_mix_with_development(self) -> None:
        candidate = draft()
        bets = [scored_bet(i, 0.7 if i < 70 else 0.4, 2024 if i < 70 else 2025) for i in range(100)]
        coverage = bt.Coverage(100, 100, 100, 100, 0, 100, 100, 100, 100)
        mixed = bt.score(
            candidate, date(2024, 8, 1), date(2026, 7, 31), bets, coverage, 1, NOW, 1000, 0
        )
        pure = bt.score(candidate, START, END, bets[70:], coverage, 1, NOW, 1000, 0)
        assert holdout.gate(mixed.overall).verdict is holdout.GateVerdict.PASS
        assert holdout.gate(pure.overall).verdict is holdout.GateVerdict.FAIL

    def test_the_command_prints_the_gate_and_refuses_a_second_run(
        self,
        holdout_repo: Path,
        collected: list[tuple[str, date, date, bool]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        @asynccontextmanager
        async def session() -> Any:
            yield None

        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: holdout_repo)
        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
        path = str(holdout_repo / "preregistrations" / "610.yaml")

        first = RUNNER.invoke(app, ["candidates", "holdout", path])
        assert first.exit_code == 0, first.stdout
        assert "Holdout gate:" in first.stdout and "never promotes" in first.stdout

        git_commit(holdout_repo.parent)
        second = RUNNER.invoke(app, ["candidates", "holdout", path])
        assert second.exit_code == 1 and "already ran" in second.stdout
        rows = [
            json.loads(line) for line in (holdout_repo / "holdout.jsonl").read_text().splitlines()
        ]
        assert [r["kind"] for r in rows] == ["reserved", "result"]
