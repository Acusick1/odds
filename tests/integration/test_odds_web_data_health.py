"""The data-health page's readers against a database: decision-window coverage, close coverage,
and the status strip under the ``odds_web`` role."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from odds_core.candidates import (
    CandidateStatus,
    MechanicalCandidate,
    candidates_dir,
    load_candidates,
)
from odds_core.close import CloseKind
from odds_core.config import SchedulerConfig, Settings, get_settings
from odds_core.models import AlertHistory, Event, FetchLog
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_lambda.candidate_rules import load_market_state, max_input_age, resolve_rule
from odds_lambda.candidate_rules.pinning import rule_sha256
from odds_web.app import create_app
from odds_web.closes import CloseClass, close_coverage, week_of
from odds_web.coverage import RETAIL, candidate_coverage
from odds_web.db import WebDatabase
from odds_web.status import ChipState, load_status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from tests.test_helpers import (
    as_odds_web_login,
    betfair_exchange_bookmaker,
    book,
    client_of,
    create_scheduler_tables,
    make_epl_event,
    make_tagged_snapshot,
    polymarket_bookmaker,
    store_tagged,
    three_way_bookmaker,
)

NOW = datetime.now(UTC).replace(second=0, microsecond=0)
KICKOFF = NOW - timedelta(days=1)
BOOKS = [
    three_way_bookmaker("bet365", (2.0, 3.0, 4.0)),
    betfair_exchange_bookmaker((2.0, 4.0, 4.0)),
]


def candidate(rule: str = "exchange_gap", name: str = "exchange-gap-epl") -> MechanicalCandidate:
    return MechanicalCandidate(
        name=name,
        description="d",
        sports=("soccer_epl",),
        version=3,
        issue=1,
        start_date="2026-01-01",
        status=CandidateStatus.ACTIVE,
        kind="mechanical",
        market="1x2",
        rule=rule,
        rule_sha256=rule_sha256(rule),
        params={"threshold": 0.01, "minutes_before_kickoff": [60, 30], "max_input_age_minutes": 10},
    )


async def store(
    session: AsyncSession,
    minutes_before: int,
    books: list,
    source: str = "oddsportal_live",
    stored_after: timedelta = timedelta(0),
) -> None:
    snapshot = make_tagged_snapshot(1, KICKOFF - timedelta(minutes=minutes_before), books, source)
    snapshot.created_at = snapshot.snapshot_time + stored_after
    await store_tagged(session, snapshot)


@pytest.fixture
async def seeded(pglite_async_session: AsyncSession) -> AsyncIterator[AsyncSession]:
    """One played EPL event: retail with the exchange 115, 65 and 35 minutes before kick-off,
    retail alone at 85, Polymarket at 62, and a snapshot at 25 stored 20 minutes late."""
    session = pglite_async_session
    session.add(make_epl_event(1, KICKOFF))
    await session.flush()
    for minutes in (115, 65, 35):
        await store(session, minutes, BOOKS)
    await store(session, 85, BOOKS[:1])
    await store(session, 62, [polymarket_bookmaker((0.45, 0.28, 0.27))], "polymarket")
    await store(session, 25, BOOKS, stored_after=timedelta(minutes=20))
    await session.commit()
    yield session


class TestDecisionWindowCoverage:
    @pytest.mark.parametrize("rule", ["exchange_gap", "polymarket_gap"])
    async def test_each_bucket_equals_load_market_state_as_known_then(
        self, seeded: AsyncSession, rule: str
    ) -> None:
        entry = candidate(rule)
        spec, params = resolve_rule(entry)
        event = await seeded.get(Event, 1)
        assert event is not None

        [row] = (await candidate_coverage(seeded, entry, NOW)).rows

        for cell in row.cells:
            state = await load_market_state(
                seeded,
                event,
                entry.market,
                cell.time,
                max_input_age(params),
                spec.sharp_inputs,
                known_at=cell.known_at,
            )
            assert cell.complete == state.has_input, cell
            for name, needs in [
                (RETAIL, frozenset()),
                *((i.value, frozenset({i})) for i in spec.sharp_inputs),
            ]:
                alone = await load_market_state(
                    seeded,
                    event,
                    entry.market,
                    cell.time,
                    max_input_age(params),
                    needs,
                    known_at=cell.known_at,
                )
                assert (name in cell.fresh) == alone.has_input, (name, cell)
        assert {c.complete for c in row.cells} == {True, False}

    async def test_buckets_name_which_inputs_were_fresh(self, seeded: AsyncSession) -> None:
        coverage = await candidate_coverage(seeded, candidate(), NOW)
        cells = {c.minutes_before: c for c in coverage.rows[0].cells}

        assert coverage.inputs == (RETAIL, "exchange")
        assert coverage.minutes == [120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
        assert (cells[110].fresh, cells[110].complete) == ((RETAIL, "exchange"), True)
        assert (cells[80].fresh, cells[80].complete) == ((RETAIL,), False)
        assert (cells[100].fresh, cells[100].complete) == ((), False)

    async def test_a_snapshot_stored_late_is_unseen_at_its_bucket(
        self, seeded: AsyncSession
    ) -> None:
        cells = {
            c.minutes_before: c
            for c in (await candidate_coverage(seeded, candidate(), NOW)).rows[0].cells
        }

        # Stamped at 25 but stored at 5: at 20 the 35-minute snapshot is too old and the
        # 25-minute one not yet stored.
        assert not cells[20].complete

    async def test_a_recorded_decision_is_read_as_of_its_known_at(
        self, seeded: AsyncSession
    ) -> None:
        decided = KICKOFF - timedelta(minutes=20)
        seeded.add(
            CandidateDecision(
                candidate_name="exchange-gap-epl",
                candidate_version=3,
                sports=["soccer_epl"],
                event_id=1,
                minutes_before_kickoff=30,
                decision_time=decided,
                kickoff_at_decision=KICKOFF,
                known_at=KICKOFF - timedelta(minutes=4),
            )
        )
        await seeded.commit()

        cells = {
            c.minutes_before: c
            for c in (await candidate_coverage(seeded, candidate(), NOW)).rows[0].cells
        }

        recorded = cells[30]
        assert recorded.recorded and recorded.time == decided
        assert recorded.known_at == KICKOFF - timedelta(minutes=4)
        # As known at 4 minutes before kick-off the late 25-minute snapshot is visible.
        assert recorded.complete

    async def test_buckets_still_ahead_are_not_evaluated(
        self, pglite_async_session: AsyncSession
    ) -> None:
        pglite_async_session.add(make_epl_event(1, NOW + timedelta(minutes=45)))
        await pglite_async_session.commit()

        [row] = (await candidate_coverage(pglite_async_session, candidate(), NOW)).rows

        assert [c.minutes_before for c in row.cells if c.evaluated] == [
            120,
            110,
            100,
            90,
            80,
            70,
            60,
            50,
        ]

    def test_every_registered_rule_hash_still_matches_its_registry_entry(self) -> None:
        for entry in load_candidates(candidates_dir()):
            if isinstance(entry, MechanicalCandidate):
                assert rule_sha256(entry.rule) == entry.rule_sha256, entry.name


class TestCloseCoverage:
    async def test_bet_lines_are_counted_by_close_and_kickoff_week(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(1, NOW - timedelta(days=10)),
                make_epl_event(2, NOW - timedelta(days=10)),
                make_epl_event(3, NOW + timedelta(hours=1)),
            ]
        )
        await session.flush()
        live = make_tagged_snapshot(
            1, NOW - timedelta(days=10, minutes=5), BOOKS[1:], "betfair_api"
        )
        await store_tagged(session, live)
        for event_id in (1, 1, 2, 3):
            session.add(
                PaperTrade(
                    event_id=event_id,
                    market="1x2",
                    selection="home",
                    bookmaker="bet365",
                    price_decimal=2.2,
                    stake=1.0,
                    candidate_name="exchange-gap-epl",
                    candidate_version=3,
                    placed_at=NOW - timedelta(days=11),
                )
            )
        await session.commit()

        [weeks] = await close_coverage(session, NOW)

        assert (weeks.sport, weeks.market) == ("soccer_epl", "1x2")
        past = week_of(NOW - timedelta(days=10))
        # Event 2 is past FDUK's grace with no close; event 3 has not kicked off.
        live, pending, none = (
            CloseClass(k) for k in (CloseKind.LIVE, CloseKind.PENDING, CloseKind.NONE)
        )
        assert weeks.counts[past] == {live: 1, none: 1}
        assert weeks.counts[week_of(NOW + timedelta(hours=1))] == {pending: 1}
        assert weeks.classes == [live, pending, none]
        assert [c.label for c in weeks.classes] == ["live Betfair", "pending", "no close"]


def bootstrapping(*jobs: str) -> Settings:
    settings = get_settings()
    return settings.model_copy(
        update={
            "scheduler": SchedulerConfig(bootstrap_jobs=list(jobs)),
            "data_collection": settings.data_collection.model_copy(
                update={"sports": ["soccer_epl"]}
            ),
        }
    )


class TestDeadChains:
    @pytest.fixture
    async def scheduler(self, pglite_async_session: AsyncSession) -> AsyncSession:
        await create_scheduler_tables(pglite_async_session)
        return pglite_async_session

    async def test_a_candidate_job_that_ran_and_booked_nothing_is_stale(
        self, scheduler: AsyncSession
    ) -> None:
        scheduler.add(
            AlertHistory(
                alert_type="heartbeat:run-candidates-epl",
                severity="info",
                message="ok",
                sent_at=NOW - timedelta(minutes=10),
            )
        )
        await book(scheduler, "run-candidates-epl", None)
        await scheduler.commit()

        strip = await load_status(scheduler, bootstrapping("run-candidates"), NOW)

        [chip] = strip.jobs
        assert (chip.name, chip.state, chip.next_run) == (
            "run-candidates-epl",
            ChipState.STALE,
            None,
        )
        assert chip.last_success == NOW - timedelta(minutes=10)
        assert strip.problem_jobs() == [chip]

    async def test_an_expected_job_with_no_schedule_row_is_listed_as_stale(
        self, scheduler: AsyncSession
    ) -> None:
        await book(scheduler, "check-health", NOW + timedelta(minutes=30))
        await scheduler.commit()

        strip = await load_status(scheduler, bootstrapping("check-health", "run-candidates"), NOW)

        assert [(j.name, j.state) for j in strip.jobs] == [
            ("check-health", ChipState.UNKNOWN),
            ("run-candidates-epl", ChipState.STALE),
        ]

    async def test_a_run_in_flight_is_not_a_dead_chain(self, scheduler: AsyncSession) -> None:
        await book(scheduler, "run-candidates-epl", None)
        await scheduler.execute(
            text(
                "INSERT INTO jobs (id, task_id, args, kwargs, schedule_id, executor, metadata, "
                "created_at) VALUES (:id, 'run-candidates-epl', '', '', 'run-candidates-epl', "
                "'async', '{}', :now)"
            ),
            {"id": uuid.uuid4(), "now": NOW},
        )
        await scheduler.commit()

        strip = await load_status(scheduler, bootstrapping("run-candidates"), NOW)

        assert [(j.name, j.state) for j in strip.jobs] == [
            ("run-candidates-epl", ChipState.UNKNOWN)
        ]


class TestUnderTheRole:
    async def test_the_readers_run_on_the_grants_in_read_only_transactions(
        self, odds_web_session: AsyncSession, tmp_path: Path
    ) -> None:
        session = odds_web_session
        await create_scheduler_tables(session)
        for job in ("check-health", "fetch-oddsportal-epl"):
            await book(session, job, NOW + timedelta(minutes=30))
        session.add(make_epl_event(1, NOW + timedelta(hours=3)))
        # Two quota readings an hour apart: a balance, but too little history for a projection.
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
            for hours, remaining in ((2, 420), (1, 416))
        )
        await session.commit()
        engine = session.bind
        assert isinstance(engine, AsyncEngine)
        makers = async_sessionmaker(engine, expire_on_commit=False)
        (tmp_path / "exchange-gap-epl.yaml").write_text(
            candidate().model_dump_json()  # YAML is a superset of JSON
        )

        with as_odds_web_login(engine):
            async with makers() as reader:
                strip = await load_status(
                    reader, bootstrapping("check-health", "fetch-oddsportal"), NOW
                )
                assert (
                    await reader.scalar(text("SELECT current_setting('transaction_read_only')"))
                    == "on"
                )
            database = WebDatabase(engine, makers)
            async with client_of(create_app(database, tmp_path)) as client:
                data = (await client.get("/data")).text
                fragment = (await client.get("/fragments/status")).text

        assert [(j.name, j.state) for j in strip.jobs] == [
            ("check-health", ChipState.UNKNOWN),
            ("fetch-oddsportal-epl", ChipState.UNKNOWN),
        ]
        [quota] = strip.quotas
        assert (quota.quota.remaining, quota.projection) == (416, None)
        assert any(s.state is ChipState.UNKNOWN for s in strip.sources)
        assert "unavailable" not in data and "odds-web error" not in data
        assert "exchange-gap-epl" in data
        assert "check-health" in fragment and "unknown" in fragment
        assert "416 / 500</span> · unknown" in fragment
