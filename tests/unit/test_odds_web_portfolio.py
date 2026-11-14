from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from pathlib import Path

import odds_web
import pytest
from fastapi.testclient import TestClient
from odds_analytics.candidate_report import CandidateReport
from odds_core.candidates import CandidateStatus
from odds_web.app import create_app
from odds_web.cache import SingleFlightCache
from odds_web.db import build_web_database
from odds_web.portfolio import AXIS_PERCENT, forest_for, portfolio_rows
from sqlalchemy.exc import DBAPIError

from tests.odds_web_helpers import agent, bet, mechanical, stub_stored_backtests, summaries


@pytest.fixture
def report(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """What the app's ``candidate_report`` returns; ``error`` makes it raise."""
    state: dict[str, object] = {"summaries": [], "error": None, "calls": 0}

    async def fake(*_: object, **__: object) -> CandidateReport:
        state["calls"] = int(state["calls"]) + 1  # type: ignore[call-overload]
        if isinstance(state["error"], Exception):
            raise state["error"]
        return CandidateReport(state["summaries"], {})  # type: ignore[arg-type]

    monkeypatch.setattr("odds_web.app.candidate_report", fake)
    stub_stored_backtests(monkeypatch)
    return state


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> list:
    entries: list = []
    monkeypatch.setattr("odds_web.app.load_candidates", lambda _directory: entries)
    return entries


@pytest.fixture
def client(report: dict[str, object], registry: list) -> Iterator[TestClient]:
    database = build_web_database("postgresql+asyncpg://odds_web:pw@localhost:1/odds")
    with TestClient(create_app(database, Path("."))) as test_client:
        yield test_client


class TestPortfolioPage:
    def test_shows_the_numbers_of_the_report(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        registry.append(mechanical())
        bets = [bet(i, 0.5) for i in range(100)]
        report["summaries"] = summaries(bets, registry)
        summary = report["summaries"][0]  # type: ignore[index]
        assert summary.current is not None

        body = client.get("/").text

        assert "gap" in body and "soccer_epl · 1x2" in body
        assert f"{summary.current.mean_ev * 100:+.1f}%" in body
        assert f"{summary.pnl:+.1f}u" in body
        assert re.findall(r'<span class="verdict">(.*?)</span>', body) == ["promote"]
        assert "at 100 bets" in body

    def test_never_shows_promote_or_kill_below_100_bets_with_a_close(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        registry.append(mechanical())
        # Strongly positive and strongly negative, each short of a checkpoint.
        report["summaries"] = summaries(
            [bet(i, 0.9) for i in range(99)] + [bet(i, -0.9, name="loser") for i in range(99)],
            registry,
        )

        body = client.get("/").text

        assert "99 / 100" in body
        assert re.findall(r'<span class="verdict">(.*?)</span>', body) == ["not yet", "not yet"]

    def test_an_agent_without_a_close_is_listed_with_trades_only(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        registry.append(agent())
        report["summaries"] = summaries(
            [bet(i, 0.0, name="mlb-agent", p_close=False) for i in range(5)], registry
        )

        body = client.get("/").text

        assert "mlb-agent" in body and "baseball_mlb · agent" in body
        assert "EV not scoreable" in body
        assert "no close" in body

    def test_p_and_l_never_takes_the_ev_colours(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        registry.append(mechanical())
        report["summaries"] = summaries([bet(i, 0.5) for i in range(10)], registry)

        body = client.get("/").text

        assert 'class="r num">' in body

    def test_a_report_raising_a_dbapi_error_renders_unavailable_but_this_does_not_prove_a_timeout_cancels(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        report["error"] = DBAPIError("select", {}, TimeoutError("canceling statement"))

        response = client.get("/")

        assert response.status_code == 200
        assert "Database unavailable · no data loaded yet" in response.text
        assert "as of" not in response.text

    def test_after_a_good_load_a_failure_shows_the_last_cached_time(
        self, report: dict[str, object], registry: list
    ) -> None:
        registry.append(mechanical())
        report["summaries"] = summaries([bet(i, 0.5) for i in range(10)], registry)
        database = build_web_database("postgresql+asyncpg://odds_web:pw@localhost:1/odds")
        with TestClient(create_app(database, Path("."), SingleFlightCache(ttl=0.0))) as client:
            client.get("/")
            report["error"] = DBAPIError("select", {}, ConnectionRefusedError("refused"))
            body = client.get("/fragments/portfolio").text

        assert re.search(r"Database unavailable · showing data as of \d\d:\d\d UTC", body)
        assert "gap" in body

    def test_a_fault_is_named_apart_from_an_outage(
        self, client: TestClient, report: dict[str, object]
    ) -> None:
        report["error"] = KeyError("home")

        body = client.get("/").text

        assert "odds-web error (KeyError), see the server log · no data loaded yet" in body
        assert "Database unavailable" not in body

    def test_a_healthy_section_says_when_its_data_was_loaded(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        registry.append(mechanical())
        report["summaries"] = summaries([bet(i, 0.5) for i in range(10)], registry)

        body = client.get("/").text

        assert re.search(r'<p class="as-of sub">as of \d\d:\d\d UTC</p>', body)
        assert "unavailable" not in body

    def test_the_fragment_is_the_portfolio_without_the_page(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        registry.append(mechanical())
        report["summaries"] = summaries([bet(i, 0.5) for i in range(10)], registry)

        fragment = client.get("/fragments/portfolio").text

        assert "<table" in fragment and "<html" not in fragment

    def test_the_page_links_by_path_and_serves_its_static_files(self, client: TestClient) -> None:
        body = client.get("/").text

        assert "http://" not in body and "https://" not in body
        assert 'href="/static/app.css"' in body
        assert client.get("/static/app.css").status_code == 200
        assert client.get("/static/vendor/htmx.min.js").status_code == 200

    def test_the_report_is_cached_between_requests(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        client.get("/")
        client.get("/")
        client.get("/fragments/portfolio")

        assert report["calls"] == 1


class TestPortfolioRows:
    def test_a_version_that_is_not_in_the_registry_is_not_current(self) -> None:
        old = summaries([bet(i, 0.5, name="ghost") for i in range(3)], [])

        [row] = portfolio_rows(old)

        assert row.status == "not current" and row.scope == "—"

    def test_a_killed_candidate_shows_its_status_and_no_next_checkpoint(self) -> None:
        killed = mechanical().model_copy(update={"status": CandidateStatus.KILLED})

        [row] = portfolio_rows(summaries([], [killed]))

        assert row.status == "killed" and row.progress is None
        assert row.verdict == "not yet"

    def test_a_promoted_candidate_shows_no_next_checkpoint(self) -> None:
        promoted = mechanical().model_copy(update={"status": CandidateStatus.PROMOTED})

        [row] = portfolio_rows(summaries([], [promoted]))

        assert row.status == "promoted" and row.progress is None

    def test_a_historical_version_does_not_take_the_current_versions_scope(self) -> None:
        v2 = mechanical(sport="basketball_nba", market="h2h").model_copy(update={"version": 2})
        old, current = portfolio_rows(summaries([bet(i, 0.5) for i in range(3)], [v2]))

        assert (old.version, old.scope, old.status) == (1, "—", "not current")
        assert (current.version, current.scope) == (2, "basketball_nba · h2h")

    def test_progress_is_toward_the_next_checkpoint(self) -> None:
        noise = [bet(i, 1.0 if i % 2 else -1.0) for i in range(130)]
        [row] = portfolio_rows(summaries(noise, [mechanical()]))

        assert row.progress is not None
        assert (row.progress.bets_with_close, row.progress.next_checkpoint) == (130, 200)
        assert row.verdict == "continue" and row.verdict_note == "at 100 bets"

    def test_forest_clips_an_interval_past_the_axis_and_marks_it(self) -> None:
        [summary] = summaries([bet(i, 1.0 if i % 2 else -1.0) for i in range(4)], [mechanical()])
        assert summary.current is not None

        forest = forest_for(summary.current)

        assert forest.clipped_low and forest.clipped_high
        assert forest.ci95_x[0] == 0.0
        assert forest.zero_x == pytest.approx(70.0)
        assert AXIS_PERCENT == 10.0


class TestWebAppBoundaries:
    def test_nothing_in_odds_web_imports_the_shared_session_maker(self) -> None:
        offenders: list[str] = []
        for path in Path(odds_web.__file__).parent.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.ImportFrom | ast.Import)
                    else []
                )
                module = node.module if isinstance(node, ast.ImportFrom) else ""
                if "async_session_maker" in names or module == "odds_core.database":
                    offenders.append(str(path))
        assert offenders == []
