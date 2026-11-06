from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import odds_web
import pytest
from fastapi.testclient import TestClient
from odds_analytics.candidate_report import (
    CandidateSummary,
    ScoredBet,
    summarize_candidates,
)
from odds_core.candidates import AgentCandidate, CandidateStatus, MechanicalCandidate
from odds_web.app import create_app
from odds_web.cache import SingleFlightCache
from odds_web.db import build_web_database
from odds_web.portfolio import AXIS_PERCENT, forest_for, portfolio_rows
from sqlalchemy.exc import DBAPIError

NOW = datetime(2026, 6, 1, tzinfo=UTC)
LONG_AGO = NOW - timedelta(days=30)
N_BOOT = 400


def bet(index: int, ev: float, *, name: str = "gap", p_close: bool = True) -> ScoredBet:
    """A settled trade at price 3.0 whose EV is ``ev``; without a close when ``p_close`` is False."""
    return ScoredBet(
        candidate_name=name,
        candidate_version=1,
        event_id=index + 1,
        commence_time=LONG_AGO,
        stake=1.0,
        pnl=2.0 if index % 3 == 0 else -1.0,
        price_decimal=3.0,
        p_close=(1 + ev) / 3 if p_close else None,
        placed_at=LONG_AGO + timedelta(minutes=index),
        close_grace=timedelta(0),
    )


def mechanical(name: str = "gap") -> MechanicalCandidate:
    return MechanicalCandidate(
        name=name,
        description="d",
        sport="soccer_epl",
        version=1,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="mechanical",
        market="1x2",
        rule="exchange_gap",
        rule_sha256="0" * 64,
    )


def agent(name: str = "mlb-agent") -> AgentCandidate:
    return AgentCandidate(
        name=name,
        description="d",
        sport="baseball_mlb",
        version=1,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="agent",
    )


def summaries(bets: list[ScoredBet], registry: list) -> list[CandidateSummary]:
    return summarize_candidates(bets, registry, NOW, n_boot=N_BOOT)


@pytest.fixture
def report(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """What the app's ``candidate_report`` returns; ``error`` makes it raise."""
    state: dict[str, object] = {"summaries": [], "error": None, "calls": 0}

    async def fake(*_: object, **__: object) -> list[CandidateSummary]:
        state["calls"] = int(state["calls"]) + 1  # type: ignore[call-overload]
        if isinstance(state["error"], Exception):
            raise state["error"]
        return state["summaries"]  # type: ignore[return-value]

    monkeypatch.setattr("odds_web.app.candidate_report", fake)
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

    def test_a_failing_report_renders_unavailable_not_a_failed_page(
        self, client: TestClient, report: dict[str, object], registry: list
    ) -> None:
        report["error"] = DBAPIError("select", {}, TimeoutError("canceling statement"))

        response = client.get("/")

        assert response.status_code == 200
        assert "Unavailable" in response.text
        assert "last good data never" in response.text

    def test_after_a_good_load_a_failure_shows_the_last_cached_time(
        self, report: dict[str, object], registry: list
    ) -> None:
        registry.append(mechanical())
        report["summaries"] = summaries([bet(i, 0.5) for i in range(10)], registry)
        database = build_web_database("postgresql+asyncpg://odds_web:pw@localhost:1/odds")
        with TestClient(create_app(database, Path("."), SingleFlightCache(ttl=0.0))) as client:
            client.get("/")
            report["error"] = RuntimeError("database is down")
            body = client.get("/fragments/portfolio").text

        assert "Unavailable · last good data" in body and "never" not in body
        assert "gap" in body

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

        [row] = portfolio_rows(old, [])

        assert row.not_current and row.scope == "—"

    def test_progress_is_toward_the_next_checkpoint(self) -> None:
        noise = [bet(i, 1.0 if i % 2 else -1.0) for i in range(130)]
        [row] = portfolio_rows(summaries(noise, [mechanical()]), [mechanical()])

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
