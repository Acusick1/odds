"""The candidate page: its figures are the report data's, and what it must not show."""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from odds_analytics.candidate_report import CandidateSummary, ScoredBet
from odds_core.paper_trade_models import TradeResult
from odds_web.app import create_app
from odds_web.candidate_page import AXIS_FROM_BETS, build_series
from odds_web.db import build_web_database
from odds_web.decisions import Decision, DecisionDay, DecisionHistory, DecisionKind
from odds_web.ledger import LedgerPage, LedgerRow
from sqlalchemy.exc import DBAPIError

from tests.odds_web_helpers import NOW, agent, bet, mechanical, summaries


def ledger_row(trade_id: int = 1, **fields: object) -> LedgerRow:
    defaults: dict[str, object] = {
        "trade_id": trade_id,
        "event": "Arsenal v Spurs",
        "commence_time": NOW,
        "selection": "home",
        "line": None,
        "bookmaker": "bet365",
        "price": 3.0,
        "stake": 1.0,
        "result": TradeResult.WIN,
        "pnl": 2.0,
        "placed_at": NOW,
        "p_close": 0.4,
    }
    return LedgerRow(**{**defaults, **fields})  # type: ignore[arg-type]


def history(*kinds: DecisionKind, total: int | None = None, missed: int = 0) -> DecisionHistory:
    found = [
        Decision(kind, "Arsenal v Spurs", NOW, int(kind is DecisionKind.BET)) for kind in kinds
    ]
    return DecisionHistory(
        total=len(found) if total is None else total,
        missed=missed,
        days=[DecisionDay(NOW.date(), found)],
    )


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """What the app's readers return; ``error`` makes the report raise."""
    state: dict[str, object] = {
        "summaries": [],
        "trades": [],
        "decisions": None,
        "ledger": LedgerPage([ledger_row()], 1, 1, 1),
        "error": None,
        "registry": [],
        "pages": [],
    }

    async def report(*_: object, **__: object) -> list[CandidateSummary]:
        if isinstance(state["error"], Exception):
            raise state["error"]
        return state["summaries"]  # type: ignore[return-value]

    async def trades(_session: object, candidate: tuple[str, int]) -> list[ScoredBet]:
        return [t for t in state["trades"] if (t.candidate_name, t.candidate_version) == candidate]  # type: ignore[attr-defined,union-attr]

    async def decisions(*_: object, **__: object) -> DecisionHistory:
        return state["decisions"]  # type: ignore[return-value]

    async def ledger(_session: object, _name: str, _version: int, page: int) -> LedgerPage:
        state["pages"].append(page)  # type: ignore[attr-defined]
        return state["ledger"]  # type: ignore[return-value]

    monkeypatch.setattr("odds_web.app.candidate_report", report)
    monkeypatch.setattr("odds_web.app.load_scored_trades", trades)
    monkeypatch.setattr("odds_web.app.load_decision_history", decisions)
    monkeypatch.setattr("odds_web.app.load_ledger_page", ledger)
    monkeypatch.setattr("odds_web.app.load_candidates", lambda _directory: state["registry"])
    return state


@pytest.fixture
def client(data: dict[str, object]) -> Iterator[TestClient]:
    database = build_web_database("postgresql+asyncpg://odds_web:pw@localhost:1/odds")
    with TestClient(create_app(database, Path("."))) as test_client:
        yield test_client


def seed(data: dict[str, object], bets: list[ScoredBet], *registry: object) -> CandidateSummary:
    data["registry"] = list(registry)
    data["trades"] = bets
    data["summaries"] = summaries(bets, list(registry))
    return data["summaries"][0]  # type: ignore[index]


def chart_series(body: str, kind: str) -> str:
    [found] = re.findall(rf'data-chart="{kind}" data-series=\'(.*?)\'', body)
    return found


class TestCandidatePage:
    def test_shows_the_numbers_of_the_report(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        summary = seed(data, [bet(i, 0.5) for i in range(100)], mechanical())
        assert summary.current is not None

        body = client.get("/c/gap/1").text

        assert f"{summary.current.mean_ev * 100:+.1f}%" in body
        assert f"{summary.current.ci95[0] * 100:+.1f}%" in body
        assert f"{summary.pnl:+.1f}u" in body
        assert f"ROI {summary.roi * 100:+.1f}%" in body
        assert re.findall(r'<span class="verdict">(.*?)</span>', body) == ["promote"]
        assert "at 100 bets" in body and "soccer_epl · 1x2" in body

    def test_no_promote_or_kill_below_100_bets_with_a_close(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(i, 0.9) for i in range(99)], mechanical())

        body = client.get("/c/gap/1").text

        assert "99<span" in body and "/ 100" in body
        assert re.findall(r'<span class="verdict">(.*?)</span>', body) == ["not yet"]
        assert "promote" not in body.replace("promote CI", "").replace('"promote"', "")

    def test_the_ledger_is_listed_with_void_trades_labelled(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(i, 0.5) for i in range(3)], mechanical())
        data["ledger"] = LedgerPage(
            [
                ledger_row(1),
                ledger_row(2, result=TradeResult.VOID, pnl=0.0, selection="draw"),
                ledger_row(3, result=None, pnl=None, p_close=None),
            ],
            1,
            1,
            3,
        )

        body = client.get("/c/gap/1").text

        assert 'class="void"' in body
        assert re.findall(r"<td class=\"text\">(win|void|pending)</td>", body) == [
            "win",
            "void",
            "pending",
        ]
        assert "3 trades" in body

    def test_a_mechanical_candidate_shows_its_decision_timeline_and_missed_share(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(i, 0.5) for i in range(3)], mechanical())
        data["decisions"] = history(
            DecisionKind.BET, DecisionKind.NO_BET, DecisionKind.MISSED, total=148, missed=9
        )

        body = client.get("/c/gap/1").text

        assert body.count('class="mark mark-') == 3
        assert "mark-bet" in body and "mark-no-bet" in body and "mark-missed" in body
        assert "9 of 148" in body and ">6%<" in body

    def test_an_agent_has_no_decision_timeline(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(i, 0.5, name="mlb-agent") for i in range(3)], agent())
        data["decisions"] = history(DecisionKind.BET)  # must not be read for an agent

        body = client.get("/c/mlb-agent/1").text

        assert "Decisions" not in body and "mark-" not in body
        assert "mlb-agent" in body and "baseball_mlb · agent" in body

    def test_a_candidate_with_no_close_shows_pnl_and_the_ledger_but_no_ev_charts(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        summary = seed(
            data, [bet(i, 0.0, name="mlb-agent", p_close=False) for i in range(5)], agent()
        )

        body = client.get("/c/mlb-agent/1").text

        assert "EV not scoreable" in body
        assert f"{summary.pnl:+.1f}u" in body
        assert 'id="ledger"' in body and "Arsenal v Spurs" in body
        assert "data-chart" not in body

    def test_a_mechanical_candidate_whose_market_has_no_close_is_not_scoreable_before_any_bet(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        mlb = mechanical("mlb-gap", sport="baseball_mlb", market="h2h")
        data["registry"] = [mlb]
        data["summaries"] = summaries([], [mlb])

        body = client.get("/c/mlb-gap/1").text

        assert "EV not scoreable" in body

    def test_the_ledger_shows_the_stake_of_each_trade(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(0, 0.5)], mechanical())
        data["ledger"] = LedgerPage([ledger_row(stake=2.5)], 1, 1, 1)

        body = client.get("/c/gap/1").text

        assert '<th class="r">Stake</th>' in body and '<td class="r">2.5</td>' in body

    def test_an_unknown_candidate_is_not_found(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(0, 0.5)], mechanical())

        assert client.get("/c/gap/9").status_code == 404
        assert client.get("/c/nope/1").status_code == 404

    def test_a_report_that_failed_before_any_load_renders_unavailable(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        data["error"] = DBAPIError("select", {}, TimeoutError("canceling statement"))

        response = client.get("/c/gap/1")

        assert response.status_code == 200
        assert "Unavailable · last good data never" in response.text

    def test_a_ledger_that_times_out_renders_unavailable(
        self, client: TestClient, data: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(data, [bet(i, 0.5) for i in range(3)], mechanical())

        async def timed_out(*_: object, **__: object) -> LedgerPage:
            raise DBAPIError("select", {}, TimeoutError("canceling statement"))

        monkeypatch.setattr("odds_web.app.load_ledger_page", timed_out)

        response = client.get("/c/gap/1")

        assert response.status_code == 200
        assert "Unavailable · last good data never" in response.text

    def test_the_ledger_page_is_requested_and_linked_by_path(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(i, 0.5) for i in range(3)], mechanical())
        data["ledger"] = LedgerPage([ledger_row()], 2, 3, 60)

        body = client.get("/c/gap/1?page=2").text
        fragment = client.get("/fragments/c/gap/1?page=2").text

        assert data["pages"] == [2]  # the fragment shares the cached page
        assert 'href="/c/gap/1?page=1"' in body and 'href="/c/gap/1?page=3"' in body
        assert "<table" in fragment and "<html" not in fragment
        assert "http://" not in body and "https://" not in body

    def test_the_portfolio_links_to_the_page(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(0, 0.5)], mechanical())

        assert 'href="/c/gap/1"' in client.get("/").text

    def test_the_report_is_read_once_for_the_portfolio_and_the_page(
        self, client: TestClient, data: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(data, [bet(0, 0.5)], mechanical())
        calls: list[int] = []
        original = data["summaries"]

        async def counted(*_: object, **__: object) -> object:
            calls.append(1)
            return original

        monkeypatch.setattr("odds_web.app.candidate_report", counted)

        client.get("/")
        client.get("/c/gap/1")

        assert len(calls) == 1


class TestSeries:
    def test_intervals_are_only_at_reached_checkpoints_and_the_current_n(self) -> None:
        bets = [bet(i, 1.0 if i % 2 else -1.0) for i in range(230)]
        [summary] = summaries(bets, [mechanical()])

        series = build_series(summary, bets, NOW)

        assert [i.n for i in series.intervals] == [100, 200, 230]
        assert [i.at_checkpoint for i in series.intervals] == [True, True, False]
        assert len(series.running_ev_percent) == 230
        assert series.checkpoints == [100, 200, 300]

    def test_a_version_exactly_at_a_checkpoint_has_one_interval_there(self) -> None:
        bets = [bet(i, 1.0 if i % 2 else -1.0) for i in range(200)]
        [summary] = summaries(bets, [mechanical()])

        series = build_series(summary, bets, NOW)

        assert [i.n for i in series.intervals] == [100, 200]
        assert series.checkpoints == [100, 200, 300]

    def test_below_a_checkpoint_only_the_current_estimate_has_an_interval(self) -> None:
        bets = [bet(i, 0.5) for i in range(40)]
        [summary] = summaries(bets, [mechanical()])

        series = build_series(summary, bets, NOW)

        assert [i.n for i in series.intervals] == [40]
        assert series.checkpoints == [100]

    def test_the_running_mean_is_the_stake_weighted_ev_of_the_bets_so_far(self) -> None:
        bets = [bet(0, 0.3), bet(1, -0.1), bet(2, 0.5)]
        [summary] = summaries(bets, [mechanical()])

        series = build_series(summary, bets, NOW)

        assert series.running_ev_percent == pytest.approx([30.0, 10.0, 23.3333], rel=1e-3)
        assert series.expected_units == pytest.approx([0.3, 0.2, 0.7])
        assert series.pnl_units == pytest.approx([2.0, 1.0, 0.0])

    def test_the_cohort_is_the_scoreable_prefix_with_a_close(self) -> None:
        bets = [bet(0, 0.5), bet(1, 0.0, p_close=False), bet(2, 0.5)]
        unsettled = ScoredBet(
            **{
                **bets[2].__dict__,
                "pnl": None,
                "placed_at": bets[2].placed_at + timedelta(minutes=5),
            }
        )
        [summary] = summaries([*bets, unsettled, bet(9, 0.5)], [mechanical()])

        series = build_series(summary, [*bets, unsettled, bet(9, 0.5)], NOW)

        assert len(series.running_ev_percent) == summary.bets_with_close == 2

    def test_early_swings_do_not_set_the_axis(self) -> None:
        bets = [bet(0, 0.9)] + [bet(i, 0.01) for i in range(1, AXIS_FROM_BETS + 20)]
        [summary] = summaries(bets, [mechanical()])

        series = build_series(summary, bets, datetime(2026, 6, 1, tzinfo=UTC))

        assert series.running_ev_percent[0] == pytest.approx(90.0)
        assert series.axis_percent < 40
