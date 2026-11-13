"""Round 1's cell selection end to end: a small grid replayed on stored development events,
the picked cell written and loaded back as the holdout's plan."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from odds_analytics import cell_selection as cs
from odds_analytics import holdout

from tests.integration.test_candidate_backtest import add_event

pytestmark = pytest.mark.integration

STEM = "900-1x2-v1"
GRID = {
    "threshold": [0.02, 0.05],
    "minutes_before_kickoff": [[60]],
    "max_input_age_minutes": [10],
}
EPL_START = datetime(2025, 1, 4, 15, 0, tzinfo=UTC)
LA_LIGA_START = datetime(2025, 3, 1, 15, 0, tzinfo=UTC)
# Against the exchange's fair home probability of 0.5: EV +10% and +4%.
STRONG = (2.2, 3.0, 3.0)
WEAK = (2.08, 3.0, 3.0)


def prereg() -> dict[str, object]:
    return {
        "issue": 900,
        "name": "small-grid",
        "version": 1,
        "rule": "exchange_gap",
        "market": "1x2",
        "leagues": ["soccer_epl", "soccer_spain_la_liga"],
        "grid": GRID,
        "selection_rule": cs.ROUND_ONE_RULE,
    }


async def add_events(session: Any) -> None:
    """30 EPL and 30 La Liga events, a day apart: 25 of each league at the strong price, 5 at
    the weak one, and one La Liga event in the sealed 2025-26 season at the strong price."""
    for i in range(30):
        retail = STRONG if i < 25 else WEAK
        await add_event(session, 1 + i, EPL_START + timedelta(days=i), retail=retail)
        await add_event(
            session,
            101 + i,
            LA_LIGA_START + timedelta(days=i),
            retail=retail,
            sport="soccer_spain_la_liga",
        )
    await add_event(
        session, 999, datetime(2025, 9, 1, 15, 0, tzinfo=UTC), sport="soccer_spain_la_liga"
    )


class TestSelectCell:
    async def test_picks_from_a_small_grid_and_the_holdout_loads_the_selection(
        self, pglite_async_session: Any, tmp_path: Path
    ) -> None:
        await add_events(pglite_async_session)
        directory = tmp_path / "candidates"
        (directory / "preregistrations").mkdir(parents=True)
        path = directory / "preregistrations" / f"{STEM}.yaml"
        path.write_text(yaml.safe_dump(prereg()))

        result = await cs.select_cell(pglite_async_session, path, directory, date(2026, 11, 11))

        loose, strict = result.scores
        assert [b.name for b in result.blocks] == [
            "EPL 2022-24",
            "EPL 2024-26",
            "five leagues 2024-25",
        ]
        assert (loose.bets, loose.events) == (60, 60)
        assert [(b.bets, b.mean_ev) for b in loose.blocks] == [
            (0, None),
            (30, pytest.approx(0.09)),
            (30, pytest.approx(0.09)),
        ]
        assert loose.mean_ev == pytest.approx(0.09) and loose.eligible
        assert loose.lower_bound is not None and loose.lower_bound < 0.1
        assert (strict.bets, strict.events, strict.eligible) == (50, 50, True)
        assert strict.lower_bound == pytest.approx(0.1)
        assert not result.longshots and strict.power_mean_ev is None
        assert result.picked == strict
        sealed = [b for r in result.replays for _, bets in r.blocks for b in bets]
        assert 999 not in {b.event_id for b in sealed}

        selection = result.selection
        assert selection is not None
        written = cs.write_selection(selection, directory)
        plan = holdout.load_plan(written, directory)
        assert plan.cell == {
            "threshold": 0.05,
            "minutes_before_kickoff": [60],
            "max_input_age_minutes": 10,
        }
        assert plan.prereg.stem == STEM
