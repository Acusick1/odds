"""Every reader behind every odds-web page, populated, under the ``odds_web`` role and its grants.

A reader that starts using a table the grants file does not list fails these tests, because each
sport's branch of each reader runs against rows and the role may read only what the file grants.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from odds_core.bookmakers import BookSet
from odds_core.candidate_backtest_models import (
    CandidateBacktest,
    CandidateBacktestBet,
    WindowKind,
)
from odds_core.candidates import AgentCandidate, CandidateStatus, MechanicalCandidate
from odds_core.config import SchedulerConfig, Settings, get_settings
from odds_core.mlb_data_models import MlbProbablePitchers
from odds_core.models import AlertHistory, EventStatus, FetchLog
from odds_core.paper_trade_models import CandidateDecision, PaperTrade, TradeResult
from odds_lambda.candidate_rules.pinning import rule_sha256
from odds_web.app import create_app
from odds_web.db import WebDatabase
from odds_web.role_check import READ_TABLES
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests.test_helpers import (
    ODDS_WEB_GRANTS,
    as_odds_web_login,
    betfair_exchange_bookmaker,
    book,
    client_of,
    make_epl_event,
    make_nba_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)

NOW = datetime.now(UTC).replace(second=0, microsecond=0)
PLAYED = NOW - timedelta(days=1)
UPCOMING = NOW + timedelta(hours=3)
PARAMS = {"threshold": 0.01, "minutes_before_kickoff": [60, 30], "max_input_age_minutes": 10}
SPORTS = ["soccer_epl", "basketball_nba", "baseball_mlb"]
JOBS = ["check-health", "fetch-oddsportal", "fetch-betfair-exchange", "run-candidates"]


def mechanical(name: str, sport: str, market: str, rule: str) -> MechanicalCandidate:
    return MechanicalCandidate(
        name=name,
        description="d",
        sports=(sport,),
        version=1,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="mechanical",
        books=BookSet.UK_RETAIL,
        market=market,
        rule=rule,
        rule_sha256=rule_sha256(rule),
        params=PARAMS,
    )


CANDIDATES: list[MechanicalCandidate | AgentCandidate] = [
    mechanical("epl-gap", "soccer_epl", "1x2", "exchange_gap"),
    mechanical("nba-gap", "basketball_nba", "h2h", "polymarket_gap"),
    mechanical("mlb-gap", "baseball_mlb", "h2h", "exchange_gap"),
    AgentCandidate(
        name="mlb-agent",
        description="d",
        sports=("baseball_mlb",),
        version=1,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="agent",
    ),
]
MARKETS = {"epl-gap": "1x2", "nba-gap": "h2h", "mlb-gap": "h2h", "mlb-agent": "h2h"}
PLAYED_EVENT = {"epl-gap": 1, "nba-gap": 2, "mlb-gap": 3, "mlb-agent": 3}


def settings() -> Settings:
    base = get_settings()
    return base.model_copy(
        update={
            "scheduler": SchedulerConfig(bootstrap_jobs=JOBS),
            "data_collection": base.data_collection.model_copy(update={"sports": SPORTS}),
        }
    )


@pytest.fixture
async def populated(odds_web_session: AsyncSession) -> AsyncSession:
    """A played and an upcoming event of each sport, a bet and a decision of each candidate, and
    the rows the status strip reads: schedules, heartbeats, quota readings and the MLB schedule."""
    session = odds_web_session
    session.add_all(
        [
            make_epl_event(1, PLAYED, status=EventStatus.FINAL),
            make_nba_event(2, PLAYED, status=EventStatus.FINAL),
            make_nba_event(3, PLAYED, status=EventStatus.FINAL, sport_key="baseball_mlb"),
            make_epl_event(4, UPCOMING),
            make_nba_event(5, UPCOMING),
            make_nba_event(6, UPCOMING, sport_key="baseball_mlb"),
        ]
    )
    await session.flush()
    for event_id, minutes, books, source in [
        (1, 65, [three_way_bookmaker("bet365", (2.0, 3.0, 4.0))], "oddsportal_live"),
        (1, 10, [betfair_exchange_bookmaker((2.0, 4.0, 4.0))], "betfair_api"),
    ]:
        await store_tagged(
            session,
            make_tagged_snapshot(event_id, PLAYED - timedelta(minutes=minutes), books, source),
        )
    stored = CandidateBacktest(
        candidate_name="epl-gap",
        candidate_version=1,
        issue=1,
        market="1x2",
        rule="exchange_gap",
        params=PARAMS,
        window_kind=WindowKind.DEVELOPMENT,
        start_date=PLAYED.date(),
        end_date=PLAYED.date(),
        leagues=["soccer_epl"],
        git_sha="a" * 40,
        cell_index=0,
        eligible=True,
        selected=True,
        events=1,
        bets=1,
        bets_with_close=1,
        stake=1.0,
        pnl=1.0,
        mean_ev=0.02,
        ci_lower=0.0,
        ci_upper=0.04,
        ci_level=0.9,
    )
    session.add(stored)
    await session.flush()
    session.add(
        CandidateBacktestBet(
            backtest_id=stored.id,
            position=1,
            event_id=1,
            league="soccer_epl",
            kickoff=PLAYED,
            decision_time=PLAYED - timedelta(minutes=60),
            outcome="home",
            bookmaker="bet365",
            price=3.0,
            stake=1.0,
            pnl=2.0,
            p_close=0.4,
            ev=0.2,
        )
    )
    for candidate in CANDIDATES:
        event_id = PLAYED_EVENT[candidate.name]
        session.add(
            PaperTrade(
                event_id=event_id,
                market=MARKETS[candidate.name],
                selection="home",
                bookmaker="bet365",
                price_decimal=2.2,
                stake=1.0,
                candidate_name=candidate.name,
                candidate_version=candidate.version,
                placed_at=PLAYED - timedelta(hours=2),
                settled_at=PLAYED + timedelta(hours=3),
                result=TradeResult.WIN,
                pnl=1.2,
            )
        )
        if candidate.kind == "mechanical":
            session.add(
                CandidateDecision(
                    candidate_name=candidate.name,
                    candidate_version=candidate.version,
                    sports=["soccer_epl"],
                    event_id=event_id,
                    minutes_before_kickoff=60,
                    decision_time=PLAYED - timedelta(minutes=60),
                    kickoff_at_decision=PLAYED,
                    known_at=PLAYED - timedelta(minutes=59),
                )
            )
    session.add(
        MlbProbablePitchers(
            game_pk=1,
            commence_time=UPCOMING,
            fetched_at=NOW - timedelta(hours=1),
            home_team="Yankees",
            away_team="Red Sox",
        )
    )
    session.add_all(
        FetchLog(
            fetch_time=NOW - timedelta(hours=hours),
            sport_key="soccer_epl",
            events_count=1,
            bookmakers_count=1,
            success=True,
            api_quota_remaining=remaining,
            api_quota_used=500 - remaining,
        )
        for hours, remaining in ((30, 450), (2, 420), (1, 416))
    )
    for job in ("check-health", "fetch-oddsportal-soccer_epl", "run-candidates-soccer_epl"):
        session.add(
            AlertHistory(
                alert_type=f"heartbeat:{job}",
                severity="info",
                message="ok",
                sent_at=NOW - timedelta(minutes=10),
            )
        )
        await book(session, job, NOW + timedelta(minutes=30))
    await session.commit()
    return session


async def fetch_every_page(session: AsyncSession, registry_dir: Path) -> dict[str, str]:
    """Every page and fragment of the app over ``session``'s engine, by path, as the role."""
    engine = session.bind
    assert isinstance(engine, AsyncEngine)
    makers = async_sessionmaker(engine, expire_on_commit=False)
    for candidate in CANDIDATES:
        (registry_dir / f"{candidate.name}.yaml").write_text(
            candidate.model_dump_json()  # YAML is a superset of JSON
        )
    paths = [
        "/",
        "/data",
        "/fragments/portfolio",
        "/fragments/status",
        "/fragments/status-detail",
        "/fragments/coverage",
        "/fragments/closes",
        *(f"/c/{c.name}/{c.version}" for c in CANDIDATES),
    ]
    app = create_app(WebDatabase(engine, makers), registry_dir, settings=settings())
    with as_odds_web_login(engine):
        async with client_of(app) as client:
            return {path: (await client.get(path)).text for path in paths}


class TestEveryReaderUnderTheRole:
    async def test_every_page_renders_from_populated_tables(
        self, populated: AsyncSession, tmp_path: Path
    ) -> None:
        pages = await fetch_every_page(populated, tmp_path)

        for path, body in pages.items():
            assert 'class="unavailable"' not in body, path
        assert all(name in pages["/"] for name in MARKETS)
        # The sport branches that read different tables: the strip's calendars and sources, the
        # closes of EPL's FDUK, NBA's Polymarket and MLB's none, and each candidate's coverage.
        status = pages["/fragments/status-detail"]
        assert "run-candidates-soccer_epl" in status and "416" in status
        for sport in ("epl", "nba", "mlb"):
            assert f" {sport} " in status, sport
        closes = pages["/fragments/closes"]
        assert all(market in closes for market in ("soccer_epl", "basketball_nba", "baseball_mlb"))
        coverage = pages["/fragments/coverage"]
        assert all(c.name in coverage for c in CANDIDATES if c.kind == "mechanical")
        for candidate in CANDIDATES:
            assert "bet365" in pages[f"/c/{candidate.name}/{candidate.version}"]

    def test_the_grants_file_lists_exactly_the_tables_the_readers_use(self) -> None:
        granted = {
            table.strip()
            for listed in re.findall(
                r"GRANT SELECT ON(.*?)TO odds_web", ODDS_WEB_GRANTS.read_text(), re.S
            )
            for table in listed.split(",")
        }

        assert granted == set(READ_TABLES)

    @pytest.mark.parametrize("table", READ_TABLES)
    async def test_a_table_the_role_cannot_read_fails_a_page(
        self, populated: AsyncSession, tmp_path: Path, table: str
    ) -> None:
        await populated.execute(text(f"REVOKE SELECT ON {table} FROM odds_web"))
        await populated.commit()

        pages = await fetch_every_page(populated, tmp_path)

        assert any("odds-web error" in body for body in pages.values()), table
