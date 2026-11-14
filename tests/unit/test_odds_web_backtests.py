"""Stored backtests on the pages: the figures, the killed list and what is never drawn."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from odds_analytics.backtest_store import kill_reason, stored_running_score
from odds_analytics.candidate_report import CandidateReport
from odds_core.candidate_backtest_models import (
    CandidateBacktest,
    CandidateBacktestBet,
    WindowKind,
)
from odds_core.vocabulary import Source
from odds_web import backtests as bt
from odds_web.app import create_app
from odds_web.db import build_web_database
from odds_web.ledger import LedgerPage

from tests.odds_web_helpers import bet, mechanical, report
from tests.odds_web_helpers import stub_stored_backtests as stub

STORED = datetime(2026, 11, 13, 12, tzinfo=UTC)


def run(
    *,
    name: str = "idea",
    kind: WindowKind = WindowKind.DEVELOPMENT,
    cell_index: int | None = None,
    selected: bool = False,
    bets: int = 60,
    mean_ev: float | None = 0.02,
    sha: str = "a" * 40,
    stored: datetime = STORED,
    events: int | None = None,
    params: dict[str, object] | None = None,
) -> CandidateBacktest:
    return CandidateBacktest(
        id=1,
        candidate_name=name,
        candidate_version=1,
        issue=900,
        market="1x2",
        rule="exchange_gap",
        params=params or {"threshold": 0.02},
        window_kind=kind,
        start_date=date(2025, 8, 1),
        end_date=date(2026, 7, 31),
        leagues=["soccer_epl"],
        git_sha=sha,
        cell_index=cell_index,
        eligible=None if cell_index is None else selected,
        selected=selected,
        events=bets if events is None else events,
        bets=bets,
        bets_with_close=bets,
        stake=float(bets),
        pnl=3.0,
        mean_ev=mean_ev,
        ci_lower=None if mean_ev is None else mean_ev - 0.03,
        ci_upper=None if mean_ev is None else mean_ev + 0.03,
        ci_level=None if mean_ev is None else 0.9,
        created_at=stored,
    )


def stored_bet(position: int, ev: float, pnl: float) -> CandidateBacktestBet:
    when = datetime(2025, 9, 1, tzinfo=UTC) + timedelta(hours=position)
    return CandidateBacktestBet(
        backtest_id=1,
        position=position,
        event_id=position,
        league="soccer_epl",
        kickoff=when + timedelta(hours=1),
        decision_time=when,
        outcome="home",
        bookmaker="bet365",
        price=2.0,
        stake=1.0,
        pnl=pnl,
        p_close=(1 + ev) / 2,
        ev=ev,
        close_source=Source.BETFAIR,
    )


class TestKillReason:
    def test_too_few_bets_names_the_most_any_cell_reached(self) -> None:
        cells = [run(cell_index=i, bets=n) for i, n in enumerate((23, 46, 37))]

        assert kill_reason(cells) == "no cell reached 50 bets with a close (the most is 46)"

    def test_every_cell_negative_is_named(self) -> None:
        cells = [run(cell_index=i, mean_ev=-0.01 * (i + 1)) for i in range(3)]

        assert kill_reason(cells) == "no cell had a positive mean EV"

    def test_otherwise_it_is_the_eligibility_rule(self) -> None:
        cells = [run(cell_index=0, mean_ev=0.02), run(cell_index=1, mean_ev=-0.01)]

        assert kill_reason(cells).startswith("no cell was eligible")


class TestKilledIdeas:
    def test_lists_a_grid_with_no_selected_cell_with_its_best_cell(self) -> None:
        cells = [
            run(cell_index=0, mean_ev=0.01, params={"threshold": 0.01}),
            run(cell_index=1, mean_ev=0.04, params={"threshold": 0.02}),
            run(cell_index=2, mean_ev=-0.02, params={"threshold": 0.03}),
        ]

        (idea,) = bt.killed_ideas(cells)

        assert (idea.name, idea.issue, idea.cells) == ("idea", 900, 3)
        assert idea.best is not None and idea.best.mean_ev == 0.04
        assert idea.best.ci == pytest.approx((0.01, 0.07)) and idea.best.level == 0.9
        assert idea.best_cell == {"threshold": 0.02}

    def test_a_grid_that_selected_a_cell_is_not_killed(self) -> None:
        cells = [run(cell_index=0), run(cell_index=1, selected=True)]

        assert bt.killed_ideas(cells) == []

    def test_the_latest_grid_of_a_version_decides(self) -> None:
        old = [run(cell_index=0, sha="a" * 40, stored=STORED)]
        new = [run(cell_index=0, selected=True, sha="b" * 40, stored=STORED + timedelta(days=1))]

        assert bt.killed_ideas(old + new) == []

    def test_a_cell_without_bets_has_no_best_estimate(self) -> None:
        (idea,) = bt.killed_ideas([run(cell_index=0, bets=0, mean_ev=None)])

        assert idea.best is None and idea.best_cell is None


class TestFigures:
    def test_a_run_with_its_bets_has_the_series_and_its_own_interval(self) -> None:
        stored = run(selected=True, cell_index=0, bets=3, mean_ev=0.04)
        bets = [stored_bet(1, 0.1, 1.0), stored_bet(2, -0.1, -1.0), stored_bet(3, 0.1, 1.0)]

        figures = bt.figures_of(stored, bets)

        assert figures.series is not None
        assert figures.series.running_ev_percent == pytest.approx([10.0, 0.0, 10 / 3])
        assert figures.series.pnl_units == [1.0, 0.0, 1.0]
        (interval,) = figures.series.intervals
        assert (interval.n, interval.level, interval.promote_percent) == (3, 0.9, None)
        assert interval.ci_percent == pytest.approx((1.0, 7.0))
        assert figures.series.checkpoints == []
        assert figures.roi == pytest.approx(1.0)

    def test_a_run_without_bets_draws_no_series(self) -> None:
        assert bt.figures_of(run(cell_index=3), []).series is None

    def test_the_running_score_is_the_reports_over_settled_bets_with_a_close(self) -> None:
        pending = stored_bet(2, 0.1, 1.0)
        pending.pnl = None
        closeless = stored_bet(3, 0.1, 1.0)
        closeless.p_close = closeless.ev = None

        score = stored_running_score(run(), [stored_bet(1, 0.2, 1.0), pending, closeless])

        assert score is not None and score.mean_ev == (pytest.approx(0.2),)

    def test_blocks_come_back_as_figures(self) -> None:
        stored = run(cell_index=0, selected=True)
        stored.blocks = [
            {
                "name": "EPL 2024-26",
                "start": "2024-08-01",
                "end": "2026-07-31",
                "leagues": ["soccer_epl"],
                "bets": 5,
                "mean_ev": 0.01,
            }
        ]

        (block,) = bt.figures_of(stored, []).blocks

        assert (block.name, block.bets, block.start) == ("EPL 2024-26", 5, date(2024, 8, 1))


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    registry = [mechanical()]

    async def read(*_: object, **__: object) -> CandidateReport:
        return report([bet(i, 0.3) for i in range(100)], registry)

    async def no_decisions(*_: object, **__: object) -> None:
        return None

    async def no_trades(*_: object, **__: object) -> LedgerPage:
        return LedgerPage([], 1, 1, 0)

    monkeypatch.setattr("odds_web.app.candidate_report", read)
    monkeypatch.setattr("odds_web.app.load_decision_history", no_decisions)
    monkeypatch.setattr("odds_web.app.load_ledger_page", no_trades)
    monkeypatch.setattr("odds_web.app.load_candidates", lambda _directory: registry)
    database = build_web_database("postgresql+asyncpg://odds_web:pw@localhost:1/odds")
    overview = bt.BacktestOverview(
        {("gap", 1): bt.backtest_forest(WindowKind.HOLDOUT, 30, 0.037, (0.019, 0.056), 0.95)},
        bt.killed_ideas(
            [
                run(name="dead-idea", cell_index=i, bets=n, params={"threshold": i})
                for i, n in enumerate((23, 46))
            ]
        ),
    )
    figures = [
        bt.figures_of(
            run(name="gap", cell_index=0, selected=True, bets=2, mean_ev=0.03),
            [stored_bet(1, 0.1, 1.0), stored_bet(2, 0.1, -1.0)],
        ),
        bt.figures_of(run(name="gap", kind=WindowKind.HOLDOUT, bets=30, mean_ev=0.037), []),
    ]
    stub(monkeypatch, overview, figures)
    with TestClient(create_app(database, Path("."))) as test_client:
        yield test_client


class TestPages:
    def test_the_candidate_page_shows_development_and_holdout_above_the_forward_test(
        self, client: TestClient
    ) -> None:
        body = client.get("/c/gap/1").text

        assert body.index("Development") < body.index("Holdout") < body.index("Forward test")
        assert body.index("Forward test") < body.index("Bet ledger")
        assert "+3.7%" in body and "90% CI" in body and "95% CI" not in body.split("Holdout")[0]

    def test_a_run_that_kept_its_bets_draws_its_chart_and_one_without_does_not(
        self, client: TestClient
    ) -> None:
        body = client.get("/c/gap/1").text

        charts = re.findall(r'data-chart="(ev|cumulative)" data-series=\'(.*?)\'', body)
        backtest = body[: body.index("Forward test")]
        assert backtest.count('data-chart="ev"') == 1
        ev = json.loads(re.findall(r'data-chart="ev" data-series=\'(.*?)\'', backtest)[0])
        assert ev["checkpoints"] == [] and ev["intervals"][0]["promote"] is None
        assert ev["intervals"][0]["level"] == 0.9
        assert len(charts) >= 2

    def test_the_portfolio_shows_the_backtest_beside_the_live_estimate(
        self, client: TestClient
    ) -> None:
        body = client.get("/").text

        assert "Backtest mean EV" in body
        assert "holdout backtest" in body and "+3.7%" in body

    def test_the_killed_list_shows_each_grid_without_charting_it(self, client: TestClient) -> None:
        body = client.get("/").text

        killed = body[body.index('id="killed"') :]
        assert "dead-idea" in killed
        assert "no cell reached 50 bets with a close (the most is 46)" in killed
        assert "data-chart" not in killed
        assert re.search(r"<td class=\"r num\">2</td>", killed)
