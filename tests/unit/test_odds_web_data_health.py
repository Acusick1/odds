from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from odds_core.close import CloseKind
from odds_core.vocabulary import Source
from odds_lambda.health_readers import Freshness, SourceFreshness
from odds_lambda.storage.readers import BurnProjection, KeyQuota
from odds_web.app import create_app
from odds_web.closes import CloseClass, CloseWeeks
from odds_web.coverage import CandidateCoverage, CoverageCell, CoverageRow
from odds_web.db import build_web_database
from odds_web.status import (
    STALE_FACTOR,
    ChipState,
    JobChip,
    QuotaChip,
    SourceChip,
    StatusStrip,
    _job_state,
    _source_state,
)
from sqlalchemy.exc import DBAPIError

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
TIMEOUT = DBAPIError("select", {}, TimeoutError("canceling statement due to statement timeout"))


def strip(**overrides: object) -> StatusStrip:
    """EPL and MLB collection, an idle NBA source, and one job and source in each problem state."""
    fields: dict[str, object] = {
        "now": NOW,
        "jobs": [
            JobChip("check-health", ChipState.OK, NOW - timedelta(minutes=4), 2, NOW, False),
            JobChip(
                "run-candidates-epl",
                ChipState.UNKNOWN,
                None,
                None,
                NOW + timedelta(hours=2, minutes=30),
                True,
            ),
            JobChip(
                "run-candidates-mlb",
                ChipState.OK,
                NOW - timedelta(hours=1),
                6,
                NOW + timedelta(minutes=30),
                False,
            ),
            JobChip("fetch-odds-epl", ChipState.STALE, NOW - timedelta(hours=9), 2, NOW, False),
        ],
        "sources": [
            SourceChip("oddsportal epl 1x2", "epl", ChipState.LATE, NOW - timedelta(hours=5), 3.0),
            SourceChip("oddsportal epl totals", "epl", ChipState.UNKNOWN, None, None),
            SourceChip("betfair epl 1x2", "epl", ChipState.OK, NOW - timedelta(minutes=2), 1.0),
            SourceChip("oddsportal mlb h2h", "mlb", ChipState.OK, NOW - timedelta(minutes=9), 1.0),
            SourceChip("polymarket mlb h2h", "mlb", ChipState.STALE, NOW - timedelta(hours=8), 1.0),
            SourceChip("odds_api nba h2h", "nba", ChipState.IDLE, NOW - timedelta(days=60), None),
        ],
        "quotas": [
            QuotaChip(0, KeyQuota(("soccer_epl",), 18240, 1760), BurnProjection(80, 228, NOW)),
            QuotaChip(1, KeyQuota(("basketball_nba",), 400, 100), None),
        ],
    }
    fields.update(overrides)
    return StatusStrip(**fields)  # type: ignore[arg-type]


def healthy_strip() -> StatusStrip:
    return strip(
        jobs=[
            JobChip("check-health", ChipState.OK, NOW - timedelta(minutes=4), 2, NOW, False),
            JobChip(
                "run-candidates-epl",
                ChipState.OK,
                NOW - timedelta(hours=1),
                6,
                NOW + timedelta(minutes=30),
                False,
            ),
        ],
        sources=[
            SourceChip("betfair epl 1x2", "epl", ChipState.OK, NOW - timedelta(minutes=2), 1.0),
            SourceChip("odds_api nba h2h", "nba", ChipState.IDLE, NOW - timedelta(days=60), None),
        ],
    )


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


def chips(body: str) -> list[str]:
    """The strip's chips, as their visible text."""
    found = re.findall(
        r'<(?:span|a) class="chip[^"]*"[^>]*>(.*?)</(?:span|a)>\s*(?=<|$)', body, re.S
    )
    return [html.unescape(re.sub(r"<[^>]+>", "", chip)).strip() for chip in found]


class TestStatusSplit:
    def test_a_sport_shows_its_worst_source(self) -> None:
        epl, mlb = strip().collections()

        assert (epl.sport, epl.state, epl.last_snapshot, epl.sources) == (
            "epl",
            ChipState.LATE,
            NOW - timedelta(hours=5),
            3,
        )
        assert (mlb.sport, mlb.state, mlb.last_snapshot) == (
            "mlb",
            ChipState.STALE,
            NOW - timedelta(hours=8),
        )

    def test_a_sport_with_only_idle_sources_is_left_off(self) -> None:
        assert [c.sport for c in strip().collections()] == ["epl", "mlb"]

    def test_the_next_candidate_run_is_the_soonest_across_sports(self) -> None:
        run = strip().next_candidate_run()

        assert run is not None
        assert (run.job, run.state, run.next_run) == (
            "run-candidates-mlb",
            ChipState.OK,
            NOW + timedelta(minutes=30),
        )

    def test_no_candidate_job_means_no_next_run(self) -> None:
        assert strip(jobs=[]).next_candidate_run() is None

    def test_quota_sums_the_keys_and_takes_the_worst_days_left(self) -> None:
        quota = strip().quota()

        assert quota is not None
        assert (quota.remaining, quota.limit, quota.days_left) == (18640, 20500, 228)

    def test_problems_are_late_stale_and_unknown_worst_first(self) -> None:
        status = strip()

        assert [j.name for j in status.problem_jobs()] == ["fetch-odds-epl", "run-candidates-epl"]
        assert [s.label for s in status.problem_sources()] == [
            "polymarket mlb h2h",
            "oddsportal epl 1x2",
            "oddsportal epl totals",
        ]

    def test_healthy_and_idle_are_not_problems(self) -> None:
        status = healthy_strip()

        assert status.problem_jobs() == [] and status.problem_sources() == []


class TestChipStates:
    def test_a_job_is_late_then_stale_by_its_expectation(self) -> None:
        beat = lambda hours: NOW - timedelta(hours=hours)  # noqa: E731

        assert _job_state(beat(1.9), 2, NOW) is ChipState.OK
        assert _job_state(beat(2.1), 2, NOW) is ChipState.LATE
        assert _job_state(beat(2 * STALE_FACTOR + 0.1), 2, NOW) is ChipState.STALE
        assert _job_state(None, 2, NOW) is ChipState.UNKNOWN
        assert _job_state(beat(500), None, NOW) is ChipState.OK

    def test_a_source_is_late_then_stale_by_its_threshold(self) -> None:
        def freshness(active_hours: float | None, status: Freshness) -> SourceFreshness:
            return SourceFreshness(
                "oddsportal",
                "j",
                "soccer_epl",
                "1x2",
                status,
                NOW,
                active_hours,
                3.0,  # type: ignore[arg-type]
            )

        assert _source_state(freshness(4.0, Freshness.STALE)) is ChipState.LATE
        assert (
            _source_state(freshness(3.0 * STALE_FACTOR + 0.1, Freshness.STALE)) is ChipState.STALE
        )
        assert _source_state(freshness(None, Freshness.IDLE)) is ChipState.IDLE


class TestStatusStrip:
    def test_every_page_loads_the_strip(self, client: TestClient) -> None:
        for path in ("/", "/data"):
            body = client.get(path).text
            assert 'hx-get="/fragments/status" hx-trigger="load, every 60s"' in body

    def test_the_summary_row_is_a_chip_per_sport_the_next_run_and_the_quota(
        self, client: TestClient, readers: dict[str, object]
    ) -> None:
        readers["status"] = healthy_strip()

        assert chips(client.get("/fragments/status").text) == [
            "epl 2m",
            "run-candidates next 12:30",
            "Odds API quota 18,640 / 20,500 · ~228d left",
            "all →",
        ]

    def test_a_healthy_strip_lists_no_job_or_source(
        self, client: TestClient, readers: dict[str, object]
    ) -> None:
        readers["status"] = healthy_strip()

        body = client.get("/fragments/status").text

        assert "check-health" not in body and "betfair epl 1x2" not in body
        assert "nba" not in body and 'class="strip"' in body

    def test_problems_follow_the_summary_with_amber_and_red_dots(self, client: TestClient) -> None:
        body = client.get("/fragments/status").text

        assert '<span class="dot late"></span>epl <span class="num">5h</span>' in body
        assert '<span class="dot stale"></span>mlb <span class="num">8h</span>' in body
        assert '<span class="dot stale"></span>fetch-odds-epl <span class="num">9h</span>' in body
        assert (
            '<span class="dot late"></span>oddsportal epl 1x2 <span class="num">5h</span>' in body
        )
        assert (
            '<span class="dot unknown"></span>run-candidates-epl <span class="num">unknown</span>'
            in body
        )
        assert (
            '<span class="dot unknown"></span>oddsportal epl totals <span class="num">unknown</span>'
            in body
        )
        assert "check-health" not in body and "betfair epl 1x2" not in body
        assert "odds_api nba h2h" not in body

    def test_the_next_run_chip_carries_its_state(self, client: TestClient) -> None:
        body = client.get("/fragments/status").text

        assert (
            '<span class="dot ok"></span>run-candidates next <span class="num">12:30</span>' in body
        )

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
        for path in ("/fragments/status", "/fragments/status-detail"):
            body = client.get(path).text

            assert "pos" not in body and "neg" not in body


class TestStatusDetail:
    def test_data_health_lists_every_job_source_and_key(self, client: TestClient) -> None:
        body = client.get("/data").text

        for job in strip().jobs:
            assert f"<td>{job.name}" in body
        for source in strip().sources:
            assert f"<td>{source.label}</td>" in body
        assert "<td>key 0</td><td>soccer_epl</td>" in body and "18,240 / 20,000" in body
        assert "<td>key 1</td><td>basketball_nba</td>" in body and "400 / 500" in body
        assert "~228d left" in body

    def test_idle_sources_show_only_on_data_health(self, client: TestClient) -> None:
        assert "odds_api nba h2h" not in client.get("/fragments/status").text
        assert '<span class="dot idle"></span> idle' in client.get("/data").text

    def test_the_fragment_is_the_section_without_the_page(self, client: TestClient) -> None:
        fragment = client.get("/fragments/status-detail").text

        assert 'id="status"' in fragment and "<html" not in fragment

    def test_a_database_timeout_renders_the_section_unavailable(
        self, client: TestClient, readers: dict[str, object]
    ) -> None:
        readers["status"] = TIMEOUT

        response = client.get("/data")

        assert response.status_code == 200
        assert response.text.count("Database unavailable · no data loaded yet") == 1


class TestDataHealthPage:
    def test_renders_the_heatmap_and_the_weekly_closes(self, client: TestClient) -> None:
        body = client.get("/data").text

        assert "exchange-gap-epl" in body and "Arsenal v Chelsea" in body
        assert body.count('class="cell all recorded"') == 1
        assert body.count('class="cell some"') == 1
        assert re.findall(r'<rect class="close (\w+)"', body) == ["live", "fallback", "pending"]
        assert "week of 02 Nov: 1 FDUK fallback" in body
        assert len(re.findall(r'<p class="as-of sub">as of \d\d:\d\d UTC</p>', body)) == 3

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
