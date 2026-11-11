"""The candidate page: its figures are the report data's, and what it must not show."""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import odds_web
import pytest
from fastapi.testclient import TestClient
from odds_analytics.candidate_report import (
    CandidateReport,
    CandidateSummary,
    RunningScore,
    ScoredBet,
)
from odds_core.candidates import CandidateStatus
from odds_core.paper_trade_models import TradeResult
from odds_web.app import create_app
from odds_web.cache import SingleFlightCache
from odds_web.candidate_page import AXIS_FROM_BETS, build_series
from odds_web.db import build_web_database
from odds_web.decisions import Decision, DecisionDay, DecisionHistory, DecisionKind
from odds_web.ledger import LedgerPage, LedgerRow
from sqlalchemy.exc import DBAPIError

from tests.odds_web_helpers import LONG_AGO, NOW, agent, bet, mechanical, report


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
    """What the app's readers return; the report is read from ``trades`` and ``registry`` when
    the app asks for it, and ``error`` makes it raise."""
    state: dict[str, object] = {
        "trades": [],
        "decisions": None,
        "ledger": LedgerPage([ledger_row()], 1, 1, 1),
        "error": None,
        "registry": [],
        "pages": [],
    }

    async def read(*_: object, **__: object) -> CandidateReport:
        if isinstance(state["error"], Exception):
            raise state["error"]
        return report(state["trades"], state["registry"])  # type: ignore[arg-type]

    async def decisions(*_: object, **__: object) -> DecisionHistory:
        return state["decisions"]  # type: ignore[return-value]

    async def ledger(_session: object, _name: str, _version: int, page: int) -> LedgerPage:
        state["pages"].append(page)  # type: ignore[attr-defined]
        return state["ledger"]  # type: ignore[return-value]

    monkeypatch.setattr("odds_web.app.candidate_report", read)
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
    return report(bets, list(registry)).summaries[0]


def scored(
    bets: list[ScoredBet],
    registry: list[object] | None = None,
    starts: dict[tuple[str, int], datetime] | None = None,
) -> tuple[CandidateSummary, RunningScore]:
    """The report's summary and running score of the version ``gap`` v1."""
    found = report(bets, registry or [mechanical()], starts)
    [summary] = [s for s in found.summaries if (s.name, s.version) == ("gap", 1)]
    return summary, found.running[("gap", 1)]


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

    def test_a_killed_candidate_shows_its_status_and_no_next_checkpoint(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        killed = mechanical().model_copy(update={"status": CandidateStatus.KILLED})
        seed(data, [], killed)

        body = client.get("/c/gap/1").text

        assert re.findall(r'<span class="status">(.*?)</span>', body) == ["killed"]
        assert re.findall(r'<span class="verdict">(.*?)</span>', body) == ["not yet"]
        assert "/ 100" not in body

    def test_a_promoted_candidate_shows_no_next_checkpoint(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        promoted = mechanical().model_copy(update={"status": CandidateStatus.PROMOTED})
        seed(data, [], promoted)

        body = client.get("/c/gap/1").text

        assert re.findall(r'<span class="status">(.*?)</span>', body) == ["promoted"]
        assert "/ 100" not in body

    def test_a_historical_version_takes_not_the_current_versions_scope_or_progress(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        v2 = mechanical(sport="basketball_nba", market="h2h").model_copy(update={"version": 2})
        seed(data, [bet(i, 0.5) for i in range(3)], v2)

        body = client.get("/c/gap/1").text

        assert re.findall(r'<span class="status">(.*?)</span>', body) == ["not current"]
        assert "basketball_nba" not in body and "/ 100" not in body

    def test_a_historical_version_shows_its_timeline_only_if_it_recorded_decisions(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        v2 = mechanical().model_copy(update={"version": 2})
        seed(data, [bet(i, 0.5) for i in range(3)], v2)
        data["decisions"] = history(DecisionKind.BET, DecisionKind.MISSED, missed=1)

        recorded = client.get("/c/gap/1").text
        data["decisions"] = history(total=0)
        none_recorded = client.get("/c/gap/1?page=2").text  # the cache is per page

        assert "mark-bet" in recorded and "1 of 2" in recorded
        assert "mark-" not in none_recorded and "Decisions" not in none_recorded

    def test_a_historical_version_with_decisions_only_is_shown_with_its_timeline(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        v2 = mechanical().model_copy(update={"version": 2})
        seed(data, [bet(i, 0.5) for i in range(3)], v2)
        data["trades"] = []
        data["decisions"] = history(DecisionKind.NO_BET, DecisionKind.MISSED, missed=1)

        response = client.get("/c/gap/1")

        assert response.status_code == 200
        assert re.findall(r'<span class="status">(.*?)</span>', response.text) == ["not current"]
        assert "mark-no-bet" in response.text and "1 of 2" in response.text
        assert "/ 100" not in response.text

    def test_a_version_with_no_trades_and_no_decisions_is_not_found(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        v2 = mechanical().model_copy(update={"version": 2})
        seed(data, [], v2)
        data["decisions"] = history(total=0)

        assert client.get("/c/gap/1").status_code == 404

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

    def test_loads_the_status_strip_and_dates_its_data(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(0, 0.5)], mechanical())

        page = client.get("/c/gap/1").text
        fragment = client.get("/fragments/c/gap/1").text

        assert 'hx-get="/fragments/status" hx-trigger="load, every 60s"' in page
        assert re.search(r'<p class="as-of sub">as of \d\d:\d\d UTC</p>', page)
        assert "fragments/status" not in fragment

    def test_a_report_that_failed_before_any_load_renders_unavailable(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        data["error"] = DBAPIError("select", {}, TimeoutError("canceling statement"))

        response = client.get("/c/gap/1")

        assert response.status_code == 200
        assert "Database unavailable · no data loaded yet" in response.text

    def test_a_ledger_that_times_out_renders_unavailable(
        self, client: TestClient, data: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(data, [bet(i, 0.5) for i in range(3)], mechanical())

        async def timed_out(*_: object, **__: object) -> LedgerPage:
            raise DBAPIError("select", {}, TimeoutError("canceling statement"))

        monkeypatch.setattr("odds_web.app.load_ledger_page", timed_out)

        response = client.get("/c/gap/1")

        assert response.status_code == 200
        assert "Database unavailable · no data loaded yet" in response.text

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
        original = report(data["trades"], data["registry"])  # type: ignore[arg-type]

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
        summary, running = scored(bets)

        series = build_series(summary, running)

        assert [i.n for i in series.intervals] == [100, 200, 230]
        assert [i.at_checkpoint for i in series.intervals] == [True, True, False]
        assert len(series.running_ev_percent) == 230
        assert series.checkpoints == [100, 200, 300]

    def test_a_version_exactly_at_a_checkpoint_has_one_interval_there(self) -> None:
        bets = [bet(i, 1.0 if i % 2 else -1.0) for i in range(200)]
        summary, running = scored(bets)

        series = build_series(summary, running)

        assert [i.n for i in series.intervals] == [100, 200]
        assert series.checkpoints == [100, 200, 300]

    def test_below_a_checkpoint_only_the_current_estimate_has_an_interval(self) -> None:
        bets = [bet(i, 0.5) for i in range(40)]
        summary, running = scored(bets)

        series = build_series(summary, running)

        assert [i.n for i in series.intervals] == [40]
        assert series.checkpoints == [100]

    def test_the_running_mean_is_the_stake_weighted_ev_of_the_bets_so_far(self) -> None:
        bets = [bet(0, 0.3), bet(1, -0.1), bet(2, 0.5)]
        summary, running = scored(bets)

        series = build_series(summary, running)

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
        summary, running = scored([*bets, unsettled, bet(9, 0.5)])

        series = build_series(summary, running)

        assert len(series.running_ev_percent) == summary.bets_with_close == 2

    def test_early_swings_do_not_set_the_axis(self) -> None:
        bets = [bet(0, 0.9)] + [bet(i, 0.01) for i in range(1, AXIS_FROM_BETS + 20)]
        summary, running = scored(bets)

        series = build_series(summary, running)

        assert series.running_ev_percent[0] == pytest.approx(90.0)
        assert series.axis_percent < 40


class TestOneSnapshot:
    def test_the_charts_describe_the_cohort_of_the_cached_report(
        self, client: TestClient, data: dict[str, object]
    ) -> None:
        seed(data, [bet(i, 0.5) for i in range(99)], mechanical())
        client.get("/")  # the report is cached at 99 bets with a close
        data["trades"] = [bet(i, 0.5) for i in range(100)]

        body = client.get("/c/gap/1").text

        ev = json.loads(chart_series(body, "ev"))
        cumulative = json.loads(chart_series(body, "cumulative"))
        assert "99<span" in body
        assert len(ev["running"]) == len(cumulative["pnl"]) == 99
        assert [i["n"] for i in ev["intervals"]] == [99]


class TestCurrentBesideACheckpoint:
    def test_a_current_estimate_at_a_checkpoints_n_with_a_larger_family_is_kept(self) -> None:
        registry = [mechanical("gap"), mechanical("sibling")]
        # The 100th bet is placed 99 minutes after LONG_AGO; the sibling starts a day later.
        starts = {("gap", 1): LONG_AGO, ("sibling", 1): LONG_AGO + timedelta(days=1)}
        bets = [bet(i, 1.0 if i % 2 else -1.0) for i in range(100)]
        summary, running = scored(bets, registry, starts)
        assert summary.checkpoint is not None and summary.current is not None
        assert (summary.checkpoint.family_versions, summary.current.family_versions) == (1, 2)

        series = build_series(summary, running)

        assert [(i.n, i.at_checkpoint) for i in series.intervals] == [(100, True), (100, False)]
        assert series.intervals[1].promote_percent == (
            summary.current.promote_ci[0] * 100,
            summary.current.promote_ci[1] * 100,
        )


class TestNoScoringInOddsWeb:
    def test_the_series_is_the_reports_running_score(self) -> None:
        summary, running = scored([bet(0, 0.3), bet(1, -0.1), bet(2, 0.5)])

        series = build_series(summary, running)

        assert series.running_ev_percent == [ev * 100 for ev in running.mean_ev]
        assert series.expected_units == list(running.expected)
        assert series.pnl_units == list(running.pnl)

    def test_odds_web_takes_only_report_data_and_ledger_closes_from_odds_analytics(self) -> None:
        allowed = {
            "CandidateReport",
            "CandidateSummary",
            "Checkpoint",
            "CheckpointProgress",
            "LeagueBreakdown",
            "RunningScore",
            "Verdict",
            "candidate_report",
            "checkpoint_progress",
            "checkpoints_through_next",
            "close_probabilities",
            "expected_value",
            "unscored_version",
        }
        imported: set[str] = set()
        for path in Path(odds_web.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    imported |= {a.name for a in node.names if a.name.startswith("odds_analytics")}
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "odds_analytics"
                ):
                    imported |= {a.name for a in node.names}
        assert imported <= allowed


class TestLastGoodData:
    def test_a_stale_report_dates_the_page_by_the_report_not_the_listings(
        self, data: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seed(data, [bet(i, 0.5) for i in range(3)], mechanical())
        loads = iter(datetime(2026, 6, 1, 12, minute, tzinfo=UTC) for minute in range(0, 60, 5))

        class Clock:
            @staticmethod
            def now(_tz: object) -> datetime:
                return next(loads)

        monkeypatch.setattr("odds_web.cache.datetime", Clock)
        database = build_web_database("postgresql+asyncpg://odds_web:pw@localhost:1/odds")
        with TestClient(create_app(database, Path("."), SingleFlightCache(ttl=0.0))) as client:
            client.get("/c/gap/1")  # the report loads at 12:00 and the listings at 12:05
            data["error"] = RuntimeError("database is down")
            body = client.get("/c/gap/1").text  # the listings reload at 12:10

        assert (
            "odds-web error (RuntimeError), see the server log · showing data as of 12:00 UTC"
            in body
        )
