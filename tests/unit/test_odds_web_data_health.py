from __future__ import annotations

import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from odds_core.close import CloseKind
from odds_core.vocabulary import Source
from odds_lambda.storage.readers import BurnProjection, KeyQuota
from odds_web.app import create_app
from odds_web.closes import CloseClass, CloseWeeks
from odds_web.coverage import CandidateCoverage, CoverageCell, CoverageRow
from odds_web.db import build_web_database
from odds_web.status import ChipState, JobChip, QuotaChip, SourceChip, StatusStrip
from sqlalchemy.exc import DBAPIError

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
TIMEOUT = DBAPIError("select", {}, TimeoutError("canceling statement due to statement timeout"))


def strip(**overrides: object) -> StatusStrip:
    fields: dict[str, object] = {
        "now": NOW,
        "jobs": [
            JobChip("check-health", ChipState.OK, NOW - timedelta(minutes=4), 2, NOW, False),
            JobChip("run-candidates-epl", ChipState.UNKNOWN, None, None, NOW, True),
        ],
        "sources": [
            SourceChip("oddsportal epl 1x2", ChipState.LATE, NOW - timedelta(hours=5), 3.0),
            SourceChip("oddsportal epl totals", ChipState.UNKNOWN, None, None),
        ],
        "quotas": [
            QuotaChip(0, KeyQuota(("soccer_epl",), 18240, 1760), BurnProjection(80, 228, NOW)),
            QuotaChip(1, KeyQuota(("basketball_nba",), 400, 100), None),
        ],
    }
    fields.update(overrides)
    return StatusStrip(**fields)  # type: ignore[arg-type]


def coverage() -> list[CandidateCoverage]:
    kickoff = NOW - timedelta(days=1)
    cells = [
        CoverageCell(
            m, kickoff - timedelta(minutes=m), kickoff, True, m == 30, ("retail",), m == 30
        )
        for m in (60, 30)
    ]
    row = CoverageRow(1, "Arsenal v Chelsea", kickoff, cells)
    return [
        CandidateCoverage(
            "exchange-gap-epl",
            3,
            "soccer_epl",
            "1x2",
            ("retail", "exchange"),
            [60, 30],
            (60, 30),
            [row],
        )
    ]


def closes() -> list[CloseWeeks]:
    weeks = [date(2026, 11, 2), date(2026, 11, 9)]
    return [
        CloseWeeks(
            "soccer_epl",
            "1x2",
            weeks,
            {
                weeks[0]: Counter(
                    {
                        CloseClass(CloseKind.LIVE): 6,
                        CloseClass(CloseKind.FALLBACK, Source.FOOTBALL_DATA_UK): 1,
                    }
                ),
                weeks[1]: Counter({CloseClass(CloseKind.PENDING): 3}),
            },
        )
    ]


@pytest.fixture
def readers(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """What the app's readers return; an exception makes that reader raise it."""
    state: dict[str, object] = {"status": strip(), "coverage": coverage(), "closes": closes()}

    def fake(key: str) -> Callable[..., Awaitable[object]]:
        async def read(*_: object, **__: object) -> object:
            value = state[key]
            if isinstance(value, Exception):
                raise value
            return value

        return read

    async def one_candidate(*args: object, **kwargs: object) -> object:
        value = await fake("coverage")(*args, **kwargs)
        assert isinstance(value, list)
        return value[0]

    monkeypatch.setattr("odds_web.app.load_status", fake("status"))
    monkeypatch.setattr("odds_web.app.candidate_coverage", one_candidate)
    monkeypatch.setattr("odds_web.app.coverage_candidates", lambda _registry: [object()])
    monkeypatch.setattr("odds_web.app.load_candidates", lambda _directory: [])
    monkeypatch.setattr("odds_web.app.close_coverage", fake("closes"))
    return state


@pytest.fixture
def client(readers: dict[str, object]) -> Iterator[TestClient]:
    database = build_web_database("postgresql+asyncpg://odds_web:pw@localhost:1/odds")
    with TestClient(create_app(database, Path("."))) as test_client:
        yield test_client


class TestStatusStrip:
    def test_every_page_loads_the_strip(self, client: TestClient) -> None:
        for path in ("/", "/data"):
            body = client.get(path).text
            assert 'hx-get="/fragments/status" hx-trigger="load, every 60s"' in body

    def test_shows_each_job_source_and_key(self, client: TestClient) -> None:
        body = client.get("/fragments/status").text

        assert '<span class="dot ok"></span>check-health <span class="num">4m</span>' in body
        assert (
            '<span class="dot late"></span>oddsportal epl 1x2 <span class="num">5h</span>' in body
        )
        assert "18,240 / 20,000" in body and "~228d</span> left" in body
        assert re.search(r"as of \d\d:\d\d UTC", body)

    def test_anything_without_enough_history_is_unknown(self, client: TestClient) -> None:
        body = client.get("/fragments/status").text

        assert (
            '<span class="dot unknown"></span>run-candidates-epl <span class="num">unknown</span>'
            in body
        )
        assert (
            '<span class="dot unknown"></span>oddsportal epl totals <span class="num">unknown</span>'
            in body
        )
        assert "400 / 500</span> · unknown" in body

    def test_no_quota_reading_is_unknown(
        self, client: TestClient, readers: dict[str, object]
    ) -> None:
        readers["status"] = strip(quotas=[])

        assert "quota unknown" in client.get("/fragments/status").text

    def test_a_database_timeout_renders_unavailable(
        self, client: TestClient, readers: dict[str, object]
    ) -> None:
        readers["status"] = TIMEOUT

        response = client.get("/fragments/status")

        assert response.status_code == 200
        assert "Database unavailable · no data loaded yet" in response.text

    def test_the_strip_never_uses_the_ev_colours(self, client: TestClient) -> None:
        body = client.get("/fragments/status").text

        assert "pos" not in body and "neg" not in body


class TestDataHealthPage:
    def test_renders_the_heatmap_and_the_weekly_closes(self, client: TestClient) -> None:
        body = client.get("/data").text

        assert "exchange-gap-epl" in body and "Arsenal v Chelsea" in body
        assert body.count('class="cell all recorded"') == 1
        assert body.count('class="cell some"') == 1
        assert re.findall(r'<rect class="close (\w+)"', body) == ["live", "fallback", "pending"]
        assert "week of 02 Nov: 1 FDUK fallback" in body
        assert len(re.findall(r'<p class="as-of sub">as of \d\d:\d\d UTC</p>', body)) == 2

    def test_a_database_timeout_renders_each_section_unavailable(
        self, client: TestClient, readers: dict[str, object]
    ) -> None:
        readers["coverage"] = TIMEOUT
        readers["closes"] = TIMEOUT

        response = client.get("/data")

        assert response.status_code == 200
        assert response.text.count("Database unavailable · no data loaded yet") == 2

    def test_one_failing_section_leaves_the_other_standing(
        self, client: TestClient, readers: dict[str, object]
    ) -> None:
        readers["coverage"] = KeyError("home")

        body = client.get("/data").text

        assert "odds-web error (KeyError), see the server log · no data loaded yet" in body
        assert "week of 02 Nov" in body

    def test_the_fragments_are_the_sections_without_the_page(self, client: TestClient) -> None:
        for name in ("coverage", "closes"):
            fragment = client.get(f"/fragments/{name}").text
            assert f'id="{name}"' in fragment and "<html" not in fragment
