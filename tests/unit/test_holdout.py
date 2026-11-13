"""The sealed holdout: the seal in the replay library, the pre-registration, the durable
one-shot ledger, the pooled gate and the command."""

from __future__ import annotations

import asyncio
import json
import stat
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
from odds_core.vocabulary import Source
from typer.testing import CliRunner

from tests.unit.conftest import (
    HOLDOUT_CELL,
    HOLDOUT_GRID,
    STEM,
    git_commit,
    preregistration,
    selection,
)
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
        "sports": ["soccer_spain_la_liga"],
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


def scored_bet(
    event_id: int, p_close: float, year: int, sport: str = "soccer_spain_la_liga"
) -> ScoredBet:
    when = datetime(year, 9, 1, tzinfo=UTC)
    return ScoredBet(
        "back-longest",
        1,
        event_id,
        when,
        1.0,
        0.0,
        2.0,
        p_close,
        when,
        timedelta(0),
        sport,
        "home",
        None,
        "bet365",
        p_close,
        Source.BETFAIR,
    )


# The leagues of the draft collected, its day ranges per league and the leagues it had access to.
Collected = tuple[tuple[str, ...], dict[str, list[tuple[date, date]]], tuple[str, ...]]


def replayed(bets: list[ScoredBet]) -> bt.Replayed:
    sports = sorted({b.sport_key for b in bets})
    counts = {
        sport: bt._Counts(decisions=n, with_input=n, with_bet=n)
        for sport in sports
        if (n := sum(b.sport_key == sport for b in bets))
    }
    events = {sport: c.decisions for sport, c in counts.items()}
    return bt.Replayed(bets, counts, events, len(bets), None, (), False)


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
        primary.write_text(yaml.safe_dump(draft(sports=["soccer_epl"]).model_dump(mode="json")))
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
            (date(2015, 8, 1), date(2026, 11, 9)),
        ]
        assert holdout.development_segments(
            "soccer_spain_la_liga", date(2015, 8, 1), date(2026, 7, 31)
        ) == [(date(2015, 8, 1), date(2025, 7, 31))]
        assert holdout.development_segments("mlb", date(2015, 8, 1), date(2026, 7, 31)) == [
            (date(2015, 8, 1), date(2026, 7, 31))
        ]


def reservation_access(sport: str) -> holdout.HoldoutAccess:
    """An access from a reservation made without any repository: through the module's token."""
    prereg = holdout.Preregistration.model_validate(preregistration())
    grid = Path("grid.yaml")
    plan = holdout.Plan(
        prereg,
        holdout.Selection(
            issue=610, market="1x2", version=1, grid_sha256="d" * 64, cell=HOLDOUT_CELL
        ),
        grid,
        grid,
    )
    reservation = holdout.Reservation(
        plan, Path("."), "a" * 40, "b" * 40, "c" * 40, holdout._ISSUER
    )
    return reservation.access(sport)


class TestWindows:
    @pytest.mark.parametrize(
        ("sport", "day", "sealed"),
        [
            ("soccer_spain_la_liga", "2025-07-31", False),
            ("soccer_spain_la_liga", "2025-08-01", True),
            ("soccer_spain_la_liga", "2026-07-31", True),
            ("soccer_spain_la_liga", "2026-08-01", False),
            ("soccer_epl", "2023-01-01", False),
            ("soccer_epl", "2025-09-01", False),
        ],
    )
    def test_inclusive_edges_by_sport(self, sport: str, day: str, sealed: bool) -> None:
        d = date.fromisoformat(day)
        assert bool(holdout.intersecting(sport, d, d)) is sealed

    def test_only_the_five_leagues_are_sealed(self) -> None:
        assert [w.sport for w in holdout.holdout_windows()] == list(holdout.FIVE_LEAGUES)

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
    def test_the_grid_file_binds_strategy_grid_and_selection_rule_and_has_no_cell(self) -> None:
        prereg = holdout.Preregistration.model_validate(preregistration())
        assert (prereg.name, prereg.version, prereg.rule, prereg.market) == (
            "back-longest",
            1,
            "exchange_gap",
            "1x2",
        )
        assert prereg.grid == HOLDOUT_GRID and prereg.stem == STEM

    def test_a_grid_file_with_a_cell_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="cell"):
            holdout.Preregistration.model_validate(preregistration(cell=HOLDOUT_CELL))

    @pytest.mark.parametrize(
        "changes",
        [
            {"selection_rule": "see the issue"},
            {"leagues": ["soccer_spain_la_liga", "soccer_spain_la_liga"]},
            {"grid": {**HOLDOUT_GRID, "threshold": []}},
            {"version": "2"},
            {"market": "1X2 "},
            {"unknown": 1},
        ],
    )
    def test_rejects_an_incomplete_or_inconsistent_grid_file(
        self, changes: dict[str, object]
    ) -> None:
        with pytest.raises(ValueError):
            holdout.Preregistration.model_validate(preregistration(**changes))


class TestPreregisteredCells:
    CELLS: list[dict[str, object]] = [
        {"side": "under", "minutes_before_kickoff": [1440], "max_input_age_minutes": 60},
        {"side": "under", "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
    ]

    def listed(self) -> holdout.Preregistration:
        return holdout.Preregistration.model_validate(preregistration(grid=None, cells=self.CELLS))

    def test_a_listed_cells_file_pre_registers_exactly_those_cells(self) -> None:
        assert self.listed().candidate_cells() == self.CELLS

    def test_a_grid_pre_registers_its_cartesian_product(self) -> None:
        prereg = holdout.Preregistration.model_validate(preregistration())
        assert len(prereg.candidate_cells()) == 4
        assert HOLDOUT_CELL in prereg.candidate_cells()

    def test_a_cell_is_checked_against_the_list(self) -> None:
        holdout.check_cell(self.listed(), self.CELLS[1])
        mixed = {**self.CELLS[0], "max_input_age_minutes": 10}
        with pytest.raises(ValueError, match="not one of"):
            holdout.check_cell(self.listed(), mixed)

    @pytest.mark.parametrize(
        "changes",
        [
            {"cells": [{"side": "under"}]},
            {"grid": None, "cells": []},
            {"grid": None, "cells": [{}]},
            {"grid": None, "cells": [{"a": 1}, {"b": 1}]},
            {"grid": None, "cells": [{"a": 1}, {"a": 1}]},
        ],
    )
    def test_exactly_one_of_grid_and_cells_is_required(self, changes: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            holdout.Preregistration.model_validate(preregistration(**changes))


class TestPlan:
    def plan(self, directory: Path) -> holdout.Plan:
        return holdout.load_plan(directory / "selections" / f"{STEM}.yaml", directory)

    def rewrite_selection(self, directory: Path, **changes: object) -> Path:
        grid = directory / "preregistrations" / f"{STEM}.yaml"
        path = directory / "selections" / f"{STEM}.yaml"
        path.write_text(yaml.safe_dump(selection(grid, **changes)))
        return path

    def test_loads_a_selection_with_its_grid(self, holdout_repo: Path) -> None:
        plan = self.plan(holdout_repo)
        assert plan.cell == HOLDOUT_CELL and plan.prereg.grid == HOLDOUT_GRID

    @pytest.mark.parametrize(
        "cell",
        [
            {**HOLDOUT_CELL, "threshold": 0.03},
            {"threshold": 0.02},
            {**HOLDOUT_CELL, "extra": 1},
            {**HOLDOUT_CELL, "minutes_before_kickoff": [30]},
        ],
    )
    def test_a_cell_outside_the_grid_is_refused(
        self, holdout_repo: Path, cell: dict[str, object]
    ) -> None:
        self.rewrite_selection(holdout_repo, cell=cell)
        with pytest.raises(holdout.HoldoutRefused):
            self.plan(holdout_repo)

    def test_a_selection_of_another_grid_file_is_refused(self, holdout_repo: Path) -> None:
        self.rewrite_selection(holdout_repo, grid_sha256="0" * 64)
        with pytest.raises(holdout.HoldoutRefused, match="another grid file"):
            self.plan(holdout_repo)

    def test_files_are_named_by_issue_market_and_version(self, holdout_repo: Path) -> None:
        old = holdout_repo / "selections" / f"{STEM}.yaml"
        wrong = holdout_repo / "selections" / "610.yaml"
        wrong.write_text(old.read_text())
        with pytest.raises(holdout.HoldoutRefused, match="must be named 610-1x2-v1.yaml"):
            holdout.load_plan(wrong, holdout_repo)

    def test_a_second_market_and_a_second_version_have_their_own_files(
        self, holdout_repo: Path
    ) -> None:
        for market, version in (("totals", 1), ("1x2", 2)):
            stem = f"610-{market}-v{version}"
            grid = holdout_repo / "preregistrations" / f"{stem}.yaml"
            grid.write_text(yaml.safe_dump(preregistration(market=market, version=version)))
            chosen = holdout_repo / "selections" / f"{stem}.yaml"
            chosen.write_text(yaml.safe_dump(selection(grid, market=market, version=version)))
            assert holdout.load_plan(chosen, holdout_repo).prereg.stem == stem

    def test_a_league_without_a_window_is_set_aside_while_another_has_one(
        self, holdout_repo: Path
    ) -> None:
        grid = holdout_repo / "preregistrations" / f"{STEM}.yaml"
        grid.write_text(
            yaml.safe_dump(preregistration(leagues=["soccer_epl", "soccer_spain_la_liga"]))
        )
        self.rewrite_selection(holdout_repo)
        plan = self.plan(holdout_repo)
        assert (plan.holdout_leagues, plan.windowless_leagues) == (
            ["soccer_spain_la_liga"],
            ["soccer_epl"],
        )

    def test_epl_alone_is_refused(self, holdout_repo: Path) -> None:
        grid = holdout_repo / "preregistrations" / f"{STEM}.yaml"
        grid.write_text(yaml.safe_dump(preregistration(leagues=["soccer_epl"])))
        self.rewrite_selection(holdout_repo)
        with pytest.raises(holdout.HoldoutRefused, match="none of soccer_epl has a sealed"):
            self.plan(holdout_repo)

    def test_a_pre_registration_of_leagues_none_of_which_has_a_window_is_refused(
        self, holdout_repo: Path
    ) -> None:
        grid = holdout_repo / "preregistrations" / f"{STEM}.yaml"
        grid.write_text(yaml.safe_dump(preregistration(leagues=["mlb", "nba"])))
        self.rewrite_selection(holdout_repo)
        with pytest.raises(holdout.HoldoutRefused, match="none of mlb, nba has a sealed window"):
            self.plan(holdout_repo)


class TestCommittedOnce:
    def grid(self, directory: Path) -> Path:
        return directory / "preregistrations" / f"{STEM}.yaml"

    def commit_of(self, path: Path, backfill: datetime | None = None) -> str:
        return holdout.committed_once(path, path.parents[2], backfill)

    def test_one_commit_before_the_backfill_passes(self, holdout_repo: Path) -> None:
        past = datetime.now(UTC) + timedelta(days=1)
        assert self.commit_of(self.grid(holdout_repo), past) == git(
            holdout_repo.parent, "rev-parse", "HEAD~1"
        )

    def test_no_backfill_yet_passes(self, holdout_repo: Path) -> None:
        self.commit_of(self.grid(holdout_repo), None)

    def test_the_grid_committed_after_the_first_backfill_snapshot_is_refused(
        self, holdout_repo: Path
    ) -> None:
        with pytest.raises(holdout.HoldoutRefused, match="not before the first backfill"):
            self.commit_of(self.grid(holdout_repo), datetime.now(UTC) - timedelta(days=1))

    def test_uncommitted_is_refused(self, holdout_repo: Path) -> None:
        other = holdout_repo / "preregistrations" / "611-1x2-v1.yaml"
        other.write_text("issue: 611\n")
        with pytest.raises(holdout.HoldoutRefused, match="not committed"):
            self.commit_of(other)

    def test_edited_since_commit_is_refused(self, holdout_repo: Path) -> None:
        path = self.grid(holdout_repo)
        path.write_text(path.read_text().replace("0.02", "0.04"))
        with pytest.raises(holdout.HoldoutRefused, match="differs from its committed"):
            self.commit_of(path)

    def test_a_second_commit_touching_it_is_refused(self, holdout_repo: Path) -> None:
        path = self.grid(holdout_repo)
        path.write_text(path.read_text().replace("0.02", "0.04"))
        git_commit(holdout_repo.parent)
        with pytest.raises(holdout.HoldoutRefused, match="exactly one commit"):
            self.commit_of(path)

    def test_a_rename_does_not_hide_the_earlier_history(self, holdout_repo: Path) -> None:
        root = holdout_repo.parent
        path = self.grid(holdout_repo)
        moved = path.with_name("old.yaml")
        git(root, "mv", str(path), str(moved))
        moved.write_text(moved.read_text() + "\n# edited\n")
        git_commit(root)
        git(root, "mv", str(moved), str(path))
        git_commit(root)
        with pytest.raises(holdout.HoldoutRefused, match="exactly one commit"):
            self.commit_of(path)

    def test_a_plain_rename_to_a_fresh_name_keeps_its_history(self, holdout_repo: Path) -> None:
        root = holdout_repo.parent
        path = self.grid(holdout_repo)
        fresh = path.with_name("611-1x2-v1.yaml")
        git(root, "mv", str(path), str(fresh))
        git_commit(root)
        with pytest.raises(holdout.HoldoutRefused, match="exactly one commit"):
            self.commit_of(fresh)


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
    def reserve(self, directory: Path) -> holdout.Reservation:
        plan = holdout.load_plan(directory / "selections" / f"{STEM}.yaml", directory)
        with holdout.exclusive_lock(directory.parent):
            return holdout.reserve(plan, directory, None, NOW)

    def test_reserving_writes_a_reserved_row(self, holdout_repo: Path) -> None:
        self.reserve(holdout_repo)
        (row,) = holdout.load_ledger(holdout_repo)
        assert isinstance(row, holdout.Reserved) and (row.issue, row.market, row.version) == (
            610,
            "1x2",
            1,
        )
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
            self.reserve(holdout_repo)

    def test_rows_validate_and_a_result_needs_its_reservation(self) -> None:
        with pytest.raises(ValueError, match="without one reservation"):
            holdout.parse_ledger(result_row().model_dump_json())

    def test_the_committed_ledger_validates(self) -> None:
        directory = Path(__file__).resolve().parents[2] / "candidates"
        assert (directory / holdout.HOLDOUT_LEDGER).exists()
        holdout.load_ledger(directory)
        for path in (directory / holdout.SELECTION_DIR).glob("*.yaml"):
            holdout.load_plan(path, directory)

    def test_sibling_worktrees_cannot_reserve_the_same_version(
        self, holdout_repo: Path, tmp_path: Path
    ) -> None:
        root = holdout_repo.parent
        sibling = tmp_path / "sibling"
        git(root, "worktree", "add", "--detach", str(sibling), "HEAD")
        self.reserve(holdout_repo)
        with pytest.raises(holdout.HoldoutRefused, match="already reserved"):
            self.reserve(sibling / "candidates")
        assert holdout.load_ledger(sibling / "candidates") == []

    def test_a_spent_version_stays_spent_after_the_ledger_row_is_lost(
        self, holdout_repo: Path
    ) -> None:
        self.reserve(holdout_repo)
        subprocess.run(
            ["git", "checkout", "--", "candidates/holdout.jsonl"],
            cwd=holdout_repo.parent,
            check=True,
        )
        with pytest.raises(holdout.HoldoutRefused, match="already reserved"):
            self.reserve(holdout_repo)

    def test_the_spent_file_directory_is_synced_before_access(
        self, holdout_repo: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synced_directories: list[bool] = []
        real_fsync = holdout.os.fsync

        def track(fd: int) -> None:
            synced_directories.append(stat.S_ISDIR(holdout.os.fstat(fd).st_mode))
            real_fsync(fd)

        monkeypatch.setattr(holdout.os, "fsync", track)
        self.reserve(holdout_repo)
        assert any(synced_directories)

    def test_a_reservation_cannot_be_constructed_directly(self) -> None:
        with pytest.raises(holdout.HoldoutRefused, match="issued by reserve"):
            holdout.Reservation(None, Path("."), "a" * 40, "b" * 40, "c" * 40, object())  # type: ignore[arg-type]


def result_row() -> holdout.HoldoutResult:
    return holdout.HoldoutResult(
        issue=610,
        market="1x2",
        version=1,
        name="back-longest",
        rule="exchange_gap",
        rule_sha256="c" * 64,
        cell=HOLDOUT_CELL,
        windows=holdout.windows_of("soccer_spain_la_liga"),
        leagues_without_window=[],
        ran_on=date(2026, 11, 1),
        git_sha="a" * 40,
        preregistration_commit="b" * 40,
        selection_commit="c" * 40,
        result=holdout.gate(None),
    )


class TestRun:
    @pytest.fixture
    def collected(self, monkeypatch: pytest.MonkeyPatch) -> list[Collected]:
        """Stub the replay: la Liga's holdout bets are good, the Bundesliga's bad."""
        calls: list[Collected] = []

        async def collect(
            _session: object,
            candidate: MechanicalCandidate,
            ranges: dict[str, list[tuple[date, date]]],
            **kw: Any,
        ) -> bt.Replayed:
            granted = tuple(a.window.sport for a in kw["access"])
            calls.append((candidate.sports, dict(ranges), granted))
            return replayed(
                [
                    scored_bet(
                        n * 100 + i, 0.6 if sport == "soccer_spain_la_liga" else 0.3, 2025, sport
                    )
                    for n, sport in enumerate(candidate.sports)
                    for i in range(40)
                ]
            )

        async def no_backfill(_session: object) -> None:
            return None

        monkeypatch.setattr(holdout_run, "collect", collect)
        monkeypatch.setattr(holdout_run, "earliest_backfill", no_backfill)
        return calls

    def run(self, directory: Path) -> holdout_run.HoldoutRun:
        path = directory / "selections" / f"{STEM}.yaml"
        return asyncio.run(holdout_run.run_holdout(None, path, directory, NOW))  # type: ignore[arg-type]

    def test_replays_one_pooled_draft_on_each_leagues_window_and_judges_one_gate(
        self, holdout_repo: Path, collected: list[Collected]
    ) -> None:
        done = self.run(holdout_repo)

        leagues = ("soccer_spain_la_liga", "soccer_germany_bundesliga")
        window = [(date(2025, 8, 1), date(2026, 7, 31))]
        assert collected == [(leagues, dict.fromkeys(leagues, window), leagues)]
        assert [(r.window.sport, r.coverage.bets) for r in done.leagues] == [
            ("soccer_spain_la_liga", 40),
            ("soccer_germany_bundesliga", 40),
        ]
        reserved, result = holdout.load_ledger(holdout_repo)
        assert isinstance(reserved, holdout.Reserved) and isinstance(result, holdout.HoldoutResult)
        assert result.result.bets == 80 and result.cell == HOLDOUT_CELL
        assert result.leagues_without_window == []
        assert result.preregistration_commit == git(holdout_repo.parent, "rev-parse", "HEAD~1")
        assert result.selection_commit == git(holdout_repo.parent, "rev-parse", "HEAD")
        assert done.row == result

    def register_epl_and_la_liga(self, directory: Path) -> Path:
        """Commit a grid file of EPL and La Liga, unlike the fixture's so that git does not read
        it as a rename of it, and its selection."""
        cell = {"threshold": 0.03, "minutes_before_kickoff": [45], "max_input_age_minutes": 15}
        grid = directory / "preregistrations" / "611-1x2-v1.yaml"
        grid.write_text(
            yaml.safe_dump(
                preregistration(
                    issue=611,
                    name="back-shortest",
                    leagues=["soccer_epl", "soccer_spain_la_liga"],
                    grid={
                        "threshold": [0.03, 0.04, 0.06],
                        "minutes_before_kickoff": [[45], [20, 10]],
                        "max_input_age_minutes": [15, 20],
                    },
                    selection_rule="https://github.com/Acusick1/odds/issues/822#issuecomment-1",
                )
            )
        )
        path = directory / "selections" / "611-1x2-v1.yaml"
        path.write_text(yaml.safe_dump(selection(grid, issue=611, cell=cell)))
        git_commit(directory.parent)
        return path

    def test_epl_is_not_replayed_and_the_row_records_it_as_without_a_window(
        self, holdout_repo: Path, collected: list[Collected]
    ) -> None:
        path = self.register_epl_and_la_liga(holdout_repo)

        done = asyncio.run(holdout_run.run_holdout(None, path, holdout_repo, NOW))  # type: ignore[arg-type]

        replayed = ("soccer_spain_la_liga",)
        assert [(c[0], tuple(c[1]), c[2]) for c in collected] == [(replayed, replayed, replayed)]
        assert [r.window.sport for r in done.leagues] == list(replayed)
        assert done.row.leagues_without_window == ["soccer_epl"]
        assert done.row.result.bets == 40
        *_, stored = holdout.load_ledger(holdout_repo)
        assert stored == done.row
        assert {w.sport for w in done.row.windows} == set(replayed)

    def test_a_second_run_is_refused_without_reading_any_holdout_data(
        self, holdout_repo: Path, collected: list[Collected]
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
        collected: list[Collected],
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
        self, holdout_repo: Path, collected: list[Collected]
    ) -> None:
        path = holdout_repo / "selections" / f"{STEM}.yaml"
        grid_file = holdout_repo / "preregistrations" / f"{STEM}.yaml"
        bad = {
            "threshold": 0.02,
            "minutes_before_kickoff": [60],
            "max_input_age_minutes": 10,
            "x": 1,
        }
        grid = {**HOLDOUT_GRID, "x": [1]}
        grid_file.write_text(yaml.safe_dump(preregistration(grid=grid)))
        path.write_text(yaml.safe_dump(selection(grid_file, cell=bad)))
        git_commit(holdout_repo.parent)
        with pytest.raises(holdout.HoldoutRefused):
            asyncio.run(holdout_run.run_holdout(None, path, holdout_repo, NOW))  # type: ignore[arg-type]
        assert holdout.load_ledger(holdout_repo) == [] and collected == []

    def test_the_gate_is_the_holdouts_alone_not_a_mix_with_development(self) -> None:
        candidate = draft()
        bets = [scored_bet(i, 0.7 if i < 70 else 0.4, 2024 if i < 70 else 2025) for i in range(100)]
        mixed = bt.score(
            candidate, date(2024, 8, 1), date(2026, 7, 31), replayed(bets), 1, NOW, 1000, 0
        )
        pure = bt.score(candidate, START, END, replayed(bets[70:]), 1, NOW, 1000, 0)
        assert holdout.gate(mixed.overall).verdict is holdout.GateVerdict.PASS
        assert holdout.gate(pure.overall).verdict is holdout.GateVerdict.FAIL

    def test_the_command_names_the_leagues_that_had_no_window(
        self,
        holdout_repo: Path,
        collected: list[Collected],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        @asynccontextmanager
        async def session() -> Any:
            yield None

        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: holdout_repo)
        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
        path = self.register_epl_and_la_liga(holdout_repo)

        done = RUNNER.invoke(app, ["candidates", "holdout", str(path)])
        assert done.exit_code == 0, done.stdout
        assert "No sealed window, so no holdout bets: soccer_epl" in done.stdout

    def test_the_command_prints_the_gate_and_refuses_a_second_run(
        self,
        holdout_repo: Path,
        collected: list[Collected],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        @asynccontextmanager
        async def session() -> Any:
            yield None

        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: holdout_repo)
        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
        path = str(holdout_repo / "selections" / f"{STEM}.yaml")

        first = RUNNER.invoke(app, ["candidates", "holdout", path])
        assert first.exit_code == 0, first.stdout
        assert "Holdout gate:" in first.stdout and "never promotes" in first.stdout
        assert "No sealed window" not in first.stdout

        git_commit(holdout_repo.parent)
        second = RUNNER.invoke(app, ["candidates", "holdout", path])
        assert second.exit_code == 1 and "already ran" in second.stdout
        rows = [
            json.loads(line) for line in (holdout_repo / "holdout.jsonl").read_text().splitlines()
        ]
        assert [r["kind"] for r in rows] == ["reserved", "result"]


class TestSpentReplay:
    """A spent holdout replays again to be stored, and only if it reproduces the ledger."""

    @pytest.fixture
    def stub(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
        """La Liga's and the Bundesliga's holdout bets, ``per_league`` of them each."""
        state = {"per_league": 40, "access": 0}

        async def collect(
            _session: object, candidate: MechanicalCandidate, _ranges: object, **kw: Any
        ) -> bt.Replayed:
            state["access"] = len(kw["access"])
            return replayed(
                [
                    scored_bet(
                        n * 100 + i, 0.6 if sport == "soccer_spain_la_liga" else 0.3, 2025, sport
                    )
                    for n, sport in enumerate(candidate.sports)
                    for i in range(state["per_league"])
                ]
            )

        async def no_backfill(_session: object) -> None:
            return None

        monkeypatch.setattr(holdout_run, "collect", collect)
        monkeypatch.setattr(holdout_run, "earliest_backfill", no_backfill)
        return state

    @pytest.fixture
    def spent(self, holdout_repo: Path, stub: dict[str, int]) -> Path:
        path = holdout_repo / "selections" / f"{STEM}.yaml"
        asyncio.run(holdout_run.run_holdout(None, path, holdout_repo, NOW))  # type: ignore[arg-type]
        git_commit(holdout_repo.parent)
        stub["access"] = 0
        return path

    def replay(self, path: Path, directory: Path) -> holdout_run.SpentReplay:
        return asyncio.run(holdout_run.replay_spent(None, path, directory, NOW))  # type: ignore[arg-type]

    def test_a_replay_that_reproduces_the_ledger_returns_the_pooled_result(
        self, holdout_repo: Path, spent: Path, stub: dict[str, int]
    ) -> None:
        replay = self.replay(spent, holdout_repo)

        assert stub["access"] == 2
        assert replay.result.overall is not None and replay.result.overall.n == 80
        assert [w.sport for w in replay.windows] == [
            "soccer_spain_la_liga",
            "soccer_germany_bundesliga",
        ]
        assert replay.row == holdout.load_ledger(holdout_repo)[-1]

    def test_it_appends_nothing_to_the_ledger_and_spends_nothing(
        self, holdout_repo: Path, spent: Path
    ) -> None:
        before = (holdout_repo / "holdout.jsonl").read_text()

        self.replay(spent, holdout_repo)

        assert (holdout_repo / "holdout.jsonl").read_text() == before

    def test_a_replay_with_other_bets_than_the_ledger_is_a_mismatch(
        self, holdout_repo: Path, spent: Path, stub: dict[str, int]
    ) -> None:
        stub["per_league"] = 39

        with pytest.raises(holdout_run.HoldoutMismatch, match="bets: ledger 80, replay 78"):
            self.replay(spent, holdout_repo)

    def test_a_rule_that_hashes_differently_now_is_a_mismatch(
        self, holdout_repo: Path, spent: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(holdout_run, "rule_sha256", lambda _rule: "0" * 64)

        with pytest.raises(holdout_run.HoldoutMismatch, match="the code that ran is not"):
            self.replay(spent, holdout_repo)

    def test_an_idea_with_no_result_in_the_ledger_is_refused_without_reading_the_windows(
        self, holdout_repo: Path, stub: dict[str, int]
    ) -> None:
        path = holdout_repo / "selections" / f"{STEM}.yaml"

        with pytest.raises(holdout.HoldoutRefused, match="has no result"):
            self.replay(path, holdout_repo)
        assert stub["access"] == 0

    def test_a_spent_run_cannot_be_made_outside_spent_run(self, holdout_repo: Path) -> None:
        plan = holdout.load_plan(holdout_repo / "selections" / f"{STEM}.yaml", holdout_repo)

        with pytest.raises(holdout.HoldoutRefused, match="issued by spent_run"):
            holdout.SpentRun(plan, result_row(), object())

    def test_the_command_stores_the_replayed_run_and_leaves_the_ledger_alone(
        self,
        holdout_repo: Path,
        spent: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stored: list[Any] = []

        @asynccontextmanager
        async def session() -> Any:
            yield MagicMock(execute=AsyncMock())

        async def store(_session: object, records: list[Any]) -> None:
            stored.extend(records)

        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: holdout_repo)
        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
        monkeypatch.setattr("odds_analytics.backtest_store.store_runs", store)
        monkeypatch.setattr("odds_cli.commands.candidates.current_git_sha", lambda: "c" * 40)

        done = RUNNER.invoke(app, ["candidates", "store-holdout", str(spent)])

        assert done.exit_code == 0, done.stdout
        assert "reproduces the ledger" in done.stdout
        (record,) = stored
        assert (record.run.window_kind.value, record.run.bets, len(record.bets)) == (
            "holdout",
            80,
            80,
        )
        assert record.run.git_sha == "c" * 40

    def test_the_command_stops_on_a_mismatch_and_stores_nothing(
        self,
        holdout_repo: Path,
        spent: Path,
        stub: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stored: list[Any] = []

        @asynccontextmanager
        async def session() -> Any:
            yield MagicMock(execute=AsyncMock())

        async def store(_session: object, records: list[Any]) -> None:
            stored.extend(records)

        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: holdout_repo)
        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", session)
        monkeypatch.setattr("odds_analytics.backtest_store.store_runs", store)
        stub["per_league"] = 10

        done = RUNNER.invoke(app, ["candidates", "store-holdout", str(spent)])

        assert done.exit_code == 1 and "does not reproduce the ledger" in done.stdout
        assert stored == []
