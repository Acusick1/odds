"""Round 1's cell selection: blocks, eligibility, the interval, the pick and its tie-breaks,
what it refuses, and the command."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml
from odds_analytics import cell_selection as cs
from odds_analytics import holdout
from odds_analytics.candidate_report import ScoredBet
from odds_cli.main import app
from typer.testing import CliRunner

from tests.unit.conftest import HOLDOUT_GRID, STEM, preregistration

RUNNER = CliRunner(env={"COLUMNS": "250"})
SIX_LEAGUES = ["soccer_epl", *holdout.FIVE_LEAGUES]
EPL_2024_26, EPL_2020_22, FIVE_2024_25 = (
    cs.DEVELOPMENT_BLOCKS[1],
    cs.DEVELOPMENT_BLOCKS[0],
    cs.DEVELOPMENT_BLOCKS[2],
)


def bet(
    event_id: int, ev: float | None, price: float = 2.0, power: float | None = None
) -> ScoredBet:
    when = datetime(2025, 1, 1, tzinfo=UTC) + timedelta(hours=event_id)
    return ScoredBet(
        candidate_name="idea",
        candidate_version=1,
        event_id=event_id,
        commence_time=when,
        stake=1.0,
        pnl=0.0,
        price_decimal=price,
        p_close=None if ev is None else (1 + ev) / price,
        placed_at=when - timedelta(hours=1),
        close_grace=timedelta(days=7),
        sport_key="soccer_epl",
        selection="home",
        line=None,
        bookmaker="bet365",
        p_close_power=None if ev is None else (1 + (ev if power is None else power)) / price,
    )


def replay(
    index: int, *blocks: tuple[cs.Block, Sequence[ScoredBet]], cell: dict[str, object] | None = None
) -> cs.CellReplay:
    return cs.CellReplay(
        index, cell or {"threshold": index}, tuple((b, tuple(bets)) for b, bets in blocks)
    )


def cell_score(index: int, lower: float, mean: float, events: int) -> cs.CellScore:
    return cs.CellScore(
        index=index,
        cell={"threshold": index},
        bets=100,
        events=events,
        blocks=(),
        mean_ev=mean,
        lower_bound=lower,
        power_mean_ev=None,
        eligible=True,
    )


class TestBlocks:
    def test_six_leagues_have_the_three_blocks(self) -> None:
        assert cs.development_blocks(SIX_LEAGUES) == list(cs.DEVELOPMENT_BLOCKS)

    def test_each_block_is_limited_to_the_files_leagues(self) -> None:
        blocks = cs.development_blocks(["soccer_epl", "soccer_italy_serie_a"])

        assert [(b.name, b.leagues) for b in blocks] == [
            ("EPL 2020-22", ("soccer_epl",)),
            ("EPL 2024-26", ("soccer_epl",)),
            ("five leagues 2024-25", ("soccer_italy_serie_a",)),
        ]

    def test_a_block_with_none_of_the_files_leagues_is_left_out(self) -> None:
        blocks = cs.development_blocks(["soccer_spain_la_liga"])

        assert [(b.name, b.start, b.end) for b in blocks] == [
            ("five leagues 2024-25", date(2024, 8, 1), date(2025, 7, 31))
        ]

    def test_no_block_touches_the_sealed_holdout(self) -> None:
        for block in cs.DEVELOPMENT_BLOCKS:
            for league in block.leagues:
                assert holdout.intersecting(league, block.start, block.end) == []


class TestEligibility:
    def blocks(self, *pairs: tuple[int, float | None]) -> list[cs.BlockScore]:
        return [cs.BlockScore(f"b{i}", n, mean) for i, (n, mean) in enumerate(pairs)]

    def test_fifty_bets_in_thirty_events_positive_in_every_block_is_eligible(self) -> None:
        assert cs.is_eligible(50, 30, self.blocks((30, 0.01), (20, 0.02)))

    def test_under_fifty_bets_is_not(self) -> None:
        assert not cs.is_eligible(49, 30, self.blocks((49, 0.05)))

    def test_under_thirty_events_is_not(self) -> None:
        assert not cs.is_eligible(50, 29, self.blocks((50, 0.05)))

    @pytest.mark.parametrize("mean", [0.0, -0.01])
    def test_a_block_of_twenty_bets_without_positive_mean_ev_is_not(self, mean: float) -> None:
        assert not cs.is_eligible(60, 40, self.blocks((40, 0.05), (20, mean)))

    def test_a_block_under_twenty_bets_may_lose(self) -> None:
        assert cs.is_eligible(60, 40, self.blocks((41, 0.05), (19, -0.5), (0, None)))

    def test_bets_without_a_close_do_not_count(self) -> None:
        bets = [bet(i, 0.05) for i in range(49)] + [bet(100 + i, None) for i in range(10)]

        score = cs.score_cell(replay(0, (EPL_2024_26, bets)), longshots=False)

        assert (score.bets, score.events, score.eligible) == (49, 49, False)

    def test_events_are_counted_across_blocks(self) -> None:
        epl = [bet(i // 2, 0.05) for i in range(40)]
        five = [bet(1000 + i, 0.05) for i in range(10)]

        score = cs.score_cell(replay(0, (EPL_2024_26, epl), (FIVE_2024_25, five)), False)

        assert (score.bets, score.events, score.eligible) == (50, 30, True)
        assert [(b.name, b.bets) for b in score.blocks] == [
            ("EPL 2024-26", 40),
            ("five leagues 2024-25", 10),
        ]

    def test_a_losing_block_makes_a_winning_pool_ineligible(self) -> None:
        good = [bet(i, 0.2) for i in range(60)]
        bad = [bet(1000 + i, -0.05) for i in range(20)]

        score = cs.score_cell(replay(0, (EPL_2024_26, good), (FIVE_2024_25, bad)), False)

        assert score.mean_ev is not None and score.mean_ev > 0
        assert not score.eligible


class TestInterval:
    def test_a_constant_sample_has_its_ev_as_the_bound(self) -> None:
        assert cs.lower_bound([bet(i, 0.03) for i in range(40)]) == pytest.approx(0.03)

    def test_two_events_of_opposite_ev_have_the_worst_resample_as_the_bound(self) -> None:
        # Resamples of two events give means -1, 0 and +1 with probabilities 1/4, 1/2, 1/4.
        assert cs.lower_bound([bet(1, 1.0), bet(2, -1.0)]) == pytest.approx(-1.0)

    def test_an_events_bets_are_resampled_together_and_weigh_by_count(self) -> None:
        # Event 1 holds three bets of +1, event 2 one of -1: a resample of both events twice
        # over has mean (3 - 1) / 4; the worst, event 2 twice, has -1.
        sample = [bet(1, 1.0), bet(1, 1.0), bet(1, 1.0), bet(2, -1.0)]

        assert cs.lower_bound(sample) == pytest.approx(-1.0)
        assert cs.score_cell(replay(0, (EPL_2024_26, sample)), False).mean_ev == 0.5

    def test_a_fixed_sample_gives_its_known_bound(self) -> None:
        sample = [bet(i, ((i * 37) % 21 - 8) / 100) for i in range(60)]

        assert cs.lower_bound(sample) == pytest.approx(0.007, abs=1e-12)

    def test_the_bound_does_not_depend_on_the_order_of_the_bets(self) -> None:
        sample = [bet(i, ((i * 37) % 21 - 8) / 100) for i in range(60)]

        assert cs.lower_bound(sample[::-1]) == cs.lower_bound(sample)


class TestPick:
    def test_the_highest_lower_bound_wins(self) -> None:
        cells = [cell_score(0, 0.01, 0.09, 90), cell_score(1, 0.02, 0.03, 30)]

        assert cs.pick(cells) == cells[1]

    def test_a_tied_bound_goes_to_the_higher_mean_ev(self) -> None:
        cells = [cell_score(0, 0.02, 0.03, 90), cell_score(1, 0.02, 0.04, 30)]

        assert cs.pick(cells) == cells[1]

    def test_a_tied_bound_and_mean_go_to_more_events(self) -> None:
        cells = [cell_score(0, 0.02, 0.04, 30), cell_score(1, 0.02, 0.04, 31)]

        assert cs.pick(cells) == cells[1]

    def test_a_full_tie_goes_to_grid_order(self) -> None:
        cells = [cell_score(i, 0.02, 0.04, 30) for i in range(3)]

        assert cs.pick(cells[::-1]) == cells[0]

    def test_an_ineligible_cell_is_never_picked(self) -> None:
        cells = [
            dataclasses.replace(cell_score(0, 0.5, 0.6, 99), eligible=False),
            cell_score(1, 0.01, 0.02, 30),
        ]

        assert cs.pick(cells) == cells[1]

    def test_no_eligible_cell_picks_none(self) -> None:
        assert cs.pick([dataclasses.replace(cell_score(0, 0.5, 0.6, 99), eligible=False)]) is None


class TestPowerDevigScore:
    def test_an_idea_betting_above_four_reports_the_power_mean_on_every_cell(self) -> None:
        replays = [
            replay(0, (EPL_2024_26, [bet(1, 0.1, price=4.5, power=0.02)])),
            replay(1, (EPL_2024_26, [bet(2, 0.1, power=0.04)])),
        ]

        selected = cs.select_from(
            holdout.Preregistration.model_validate(preregistration()), "0" * 64, replays
        )

        assert selected.longshots
        assert [s.power_mean_ev for s in selected.scores] == [
            pytest.approx(0.02),
            pytest.approx(0.04),
        ]

    def test_an_idea_betting_at_four_or_below_does_not(self) -> None:
        replays = [replay(0, (EPL_2024_26, [bet(1, 0.1, price=4.0, power=0.02)]))]

        selected = cs.select_from(
            holdout.Preregistration.model_validate(preregistration()), "0" * 64, replays
        )

        assert not selected.longshots and selected.scores[0].power_mean_ev is None

    def test_it_never_changes_the_pick(self) -> None:
        strong = [bet(i, 0.1, price=5.0, power=-0.2) for i in range(50)]
        weak = [bet(100 + i, 0.05, price=5.0, power=0.3) for i in range(50)]

        selected = cs.select_from(
            holdout.Preregistration.model_validate(preregistration()),
            "0" * 64,
            [replay(0, (EPL_2024_26, weak)), replay(1, (EPL_2024_26, strong))],
        )

        assert selected.picked is not None and selected.picked.index == 1


@pytest.fixture
def directory(tmp_path: Path) -> Path:
    """``candidates/`` with the grid file of issue 610's `1x2` version 1."""
    directory = tmp_path / "candidates"
    (directory / "preregistrations").mkdir(parents=True)
    write_grid(directory)
    return directory


def write_grid(directory: Path, **changes: object) -> Path:
    path = directory / "preregistrations" / f"{STEM}.yaml"
    path.write_text(yaml.safe_dump(preregistration(**changes)))
    return path


TODAY = date(2026, 11, 11)


class TestRefusals:
    def test_a_grid_file_follows_round_ones_rule(self, directory: Path) -> None:
        path = write_grid(
            directory, selection_rule="https://github.com/Acusick1/odds/issues/999#rule"
        )
        with pytest.raises(cs.SelectionRefused, match="not round 1's rule"):
            cs.load_selectable(path, directory, TODAY)

    def test_an_unregistered_rule_is_refused_by_name(self, directory: Path) -> None:
        path = write_grid(directory, rule="no_such_rule")
        with pytest.raises(cs.SelectionRefused, match="unknown rule 'no_such_rule'"):
            cs.load_selectable(path, directory, TODAY)

    def test_a_cell_the_rule_rejects_is_refused_before_any_replay(self, directory: Path) -> None:
        path = write_grid(directory, grid={**HOLDOUT_GRID, "nonsense": [1]})
        with pytest.raises(cs.SelectionRefused, match="nonsense"):
            cs.load_selectable(path, directory, TODAY)

    def test_an_existing_selection_is_never_replaced(self, directory: Path) -> None:
        (directory / "selections").mkdir()
        (directory / "selections" / f"{STEM}.yaml").write_text("{}\n")
        with pytest.raises(cs.SelectionRefused, match="never changed"):
            cs.load_selectable(directory / "preregistrations" / f"{STEM}.yaml", directory, TODAY)

    def test_a_file_outside_preregistrations_is_refused(
        self, directory: Path, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / f"{STEM}.yaml"
        elsewhere.write_text(yaml.safe_dump(preregistration()))
        with pytest.raises(cs.SelectionRefused, match="preregistrations"):
            cs.load_selectable(elsewhere, directory, TODAY)

    def test_a_file_named_for_another_identity_is_refused(self, directory: Path) -> None:
        path = directory / "preregistrations" / "611-1x2-v1.yaml"
        path.write_text(yaml.safe_dump(preregistration()))
        with pytest.raises(cs.SelectionRefused, match="must be named"):
            cs.load_selectable(path, directory, TODAY)


class TestCommand:
    @pytest.fixture
    def replays(self, monkeypatch: pytest.MonkeyPatch, directory: Path) -> list[cs.CellReplay]:
        """Stub the replay with the cells' bets, set by each test, and the session."""
        stubbed: list[cs.CellReplay] = []

        async def replay_cells(_session: object, prereg: holdout.Preregistration, _today: date):
            return stubbed

        session = AsyncMock()

        class Maker:
            async def __aenter__(self) -> Any:
                return session

            async def __aexit__(self, *_exc: object) -> None:
                return None

        monkeypatch.setattr(cs, "replay_cells", replay_cells)
        monkeypatch.setattr("odds_cli.commands.candidates.candidates_dir", lambda: directory)
        monkeypatch.setattr("odds_cli.commands.candidates.async_session_maker", Maker)
        return stubbed

    def cells(self) -> list[dict[str, object]]:
        return holdout.Preregistration.model_validate(preregistration()).candidate_cells()

    def invoke(self, directory: Path, *extra: str) -> Any:
        path = str(directory / "preregistrations" / f"{STEM}.yaml")
        return RUNNER.invoke(app, ["candidates", "select", path, *extra])

    def test_writes_the_picked_cell_as_a_selection_the_holdout_loads(
        self, directory: Path, replays: list[cs.CellReplay], tmp_path: Path
    ) -> None:
        cells = self.cells()
        replays += [
            replay(0, (FIVE_2024_25, [bet(i, 0.02) for i in range(60)]), cell=cells[0]),
            replay(1, (FIVE_2024_25, [bet(i, 0.05) for i in range(60)]), cell=cells[1]),
            replay(2, (FIVE_2024_25, [bet(i, 0.09) for i in range(10)]), cell=cells[2]),
            replay(3, (FIVE_2024_25, []), cell=cells[3]),
        ]
        out = tmp_path / "selection.json"

        result = self.invoke(directory, "--json", str(out))

        assert result.exit_code == 0, result.stdout
        assert "Picked cell 2" in result.stdout
        written = directory / "selections" / f"{STEM}.yaml"
        plan = holdout.load_plan(written, directory)
        assert plan.cell == cells[1]
        payload = json.loads(out.read_text())
        assert payload["picked"] == 1
        assert [c["eligible"] for c in payload["cells"]] == [True, True, False, False]
        placed = payload["cells"][1]["placed"][0]
        assert placed["block"] == "five leagues 2024-25"
        assert placed["ev"] == pytest.approx(0.05)
        assert set(placed) == {
            "block",
            "event_id",
            "league",
            "kickoff",
            "decision_time",
            "outcome",
            "line",
            "bookmaker",
            "price",
            "p_close",
            "ev",
        }

    def test_no_eligible_cell_kills_the_idea_and_writes_nothing(
        self, directory: Path, replays: list[cs.CellReplay]
    ) -> None:
        replays += [
            replay(i, (FIVE_2024_25, [bet(j, -0.01) for j in range(60)]), cell=cell)
            for i, cell in enumerate(self.cells())
        ]

        result = self.invoke(directory)

        assert result.exit_code == 0, result.stdout
        assert "killed on development data" in result.stdout
        assert not (directory / "selections").exists()

    def test_a_refusal_exits_one_without_replaying(
        self, directory: Path, replays: list[cs.CellReplay]
    ) -> None:
        write_grid(directory, rule="no_such_rule")

        result = self.invoke(directory)

        assert result.exit_code == 1 and "unknown rule 'no_such_rule'" in result.stdout
