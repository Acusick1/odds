"""Integration tests for the production data copier, using a second schema as production."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from odds_cli.commands import copy_from_prod as copy_cli
from odds_core.models import (
    Event,
    EventExternalId,
    EventStatus,
    HealthCheckAcknowledgement,
    OddsSnapshot,
    Price,
)
from odds_core.vocabulary import HealthCheck, Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.copy_from_production import CopyStats, ProductionDataCopier
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlmodel import SQLModel
from typer.testing import CliRunner

from tests.test_helpers import make_epl_event

PROD_SCHEMA = "prod_source"
KICKOFF = datetime(2026, 8, 22, 15, 0, tzinfo=UTC)
WINDOW = (KICKOFF - timedelta(days=1), KICKOFF + timedelta(days=1))
SPORT = "soccer_epl"


@pytest.fixture
async def prod_session_maker(
    pglite_async_engine: AsyncEngine, pglite_async_session: AsyncSession
) -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    prod_engine = pglite_async_engine.execution_options(schema_translate_map={None: PROD_SCHEMA})
    async with pglite_async_engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA IF EXISTS "{PROD_SCHEMA}" CASCADE'))
        await conn.execute(text(f'CREATE SCHEMA "{PROD_SCHEMA}"'))
    async with prod_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield async_sessionmaker(prod_engine, expire_on_commit=False)
    async with pglite_async_engine.begin() as conn:
        await conn.execute(text(f'DROP SCHEMA "{PROD_SCHEMA}" CASCADE'))


@pytest.fixture
def local_session_maker(
    pglite_async_engine: AsyncEngine, pglite_async_session: AsyncSession
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(pglite_async_engine, expire_on_commit=False)


async def _seed_prod_event(
    prod_session_maker: async_sessionmaker[AsyncSession],
    event_id: str,
    first_snapshot_id: int,
    kickoff: datetime = KICKOFF,
) -> None:
    async with prod_session_maker() as session:
        session.add(
            Event(
                id=event_id,
                sport_key=SPORT,
                sport_title="EPL",
                commence_time=kickoff,
                home_team=f"Home {event_id}",
                away_team=f"Away {event_id}",
                status=EventStatus.FINAL,
                home_score=2,
                away_score=1,
                completed_at=kickoff + timedelta(hours=2),
            )
        )
        await session.flush()
        for offset, (source, time_basis, hours_before) in enumerate(
            [
                (Source.FOOTBALL_DATA_UK, TimeBasis.UPPER_BOUND, 25),
                (Source.ODDS_API, TimeBasis.OBSERVED, 1),
                (Source.BETFAIR, TimeBasis.OBSERVED, 1),
            ]
        ):
            session.add(
                OddsSnapshot(
                    id=first_snapshot_id + offset,
                    event_id=event_id,
                    snapshot_time=kickoff - timedelta(hours=hours_before),
                    source=source,
                    market=Market.ONE_X_TWO,
                    time_basis=time_basis,
                )
            )
        await session.commit()


def _copier(
    prod: async_sessionmaker[AsyncSession],
    local: async_sessionmaker[AsyncSession],
    skip_existing: bool = True,
) -> ProductionDataCopier:
    return ProductionDataCopier(prod, local, skip_existing=skip_existing)


class TestCopyEvents:
    @pytest.mark.asyncio
    async def test_copies_the_events_snapshots(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)

        stats = await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)

        assert (stats.events_copied, stats.events_failed, stats.snapshots_copied) == (1, 0, 3)

    @pytest.mark.asyncio
    async def test_recopy_with_overwrite_replaces_without_duplicating(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)

        stats = await _copier(
            prod_session_maker, local_session_maker, skip_existing=False
        ).copy_events(*WINDOW, SPORT)

        assert stats.events_failed == 0
        assert stats.snapshots_copied == 3
        async with local_session_maker() as session:
            assert len((await session.execute(select(OddsSnapshot))).scalars().all()) == 3


class TestCopyPreservesColumns:
    @pytest.mark.asyncio
    async def test_snapshot_keeps_its_shape_and_creation_time(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        async with prod_session_maker() as session:
            prod_snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()

        await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)

        async with local_session_maker() as session:
            local_snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()

        def keyed(rows: Sequence[OddsSnapshot]) -> dict[tuple[datetime, Any], Any]:
            return {
                (r.snapshot_time, r.source): (r.market, r.time_basis, r.created_at) for r in rows
            }

        assert keyed(local_snapshots) == keyed(prod_snapshots)
        assert Source.FOOTBALL_DATA_UK in {s.source for s in local_snapshots}

    @pytest.mark.asyncio
    async def test_event_keeps_result_and_overwrite_refreshes_it(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        async with local_session_maker() as session:
            session.add(
                Event(
                    id="evt_a",
                    sport_key=SPORT,
                    sport_title="EPL",
                    commence_time=KICKOFF,
                    home_team="Home evt_a",
                    away_team="Away evt_a",
                )
            )
            await session.commit()

        await _copier(prod_session_maker, local_session_maker, skip_existing=False).copy_events(
            *WINDOW, SPORT
        )

        async with local_session_maker() as session:
            event = await session.get(Event, "evt_a")
        assert event is not None
        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 2, 1)
        assert event.completed_at == KICKOFF + timedelta(hours=2)


class TestCopyFailures:
    @pytest.mark.asyncio
    async def test_failed_event_is_rolled_back_and_reported_not_counted_as_copied(
        self,
        monkeypatch: pytest.MonkeyPatch,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _seed_prod_event(
            prod_session_maker, "evt_b", first_snapshot_id=600, kickoff=KICKOFF + timedelta(hours=3)
        )
        real_copy_identity = ProductionDataCopier._copy_identity

        async def failing_copy_identity(
            self: ProductionDataCopier, session: AsyncSession, event: Event, *args: Any
        ) -> None:
            if event.id == "evt_b":
                raise RuntimeError("identity copy failed")
            await real_copy_identity(self, session, event, *args)

        monkeypatch.setattr(ProductionDataCopier, "_copy_identity", failing_copy_identity)
        copier = _copier(prod_session_maker, local_session_maker)

        stats = await copier.copy_events(*WINDOW, SPORT)

        assert stats.events_copied == 1
        assert stats.events_failed == 1
        assert copier.stats.failed_event_ids == ["evt_b"]
        async with local_session_maker() as session:
            assert {e.id for e in (await session.execute(select(Event))).scalars()} == {"evt_a"}
            snapshot_events = (await session.execute(select(OddsSnapshot.event_id))).scalars().all()
        assert set(snapshot_events) == {"evt_a"}

    def test_cli_exits_non_zero_when_any_event_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_copy(**kwargs: Any) -> CopyStats:
            return CopyStats(events_copied=1, failed_event_ids=["evt_a"])

        monkeypatch.setattr(copy_cli, "copy_from_prod", fake_copy)

        result = CliRunner().invoke(copy_cli.app, ["2026-08-01", "2026-08-31"])

        assert result.exit_code == 1


class TestOverwriteReplacesLocalRows:
    @pytest.mark.asyncio
    async def test_snapshots_sharing_a_time_stay_distinct(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        for _ in range(2):
            await _copier(prod_session_maker, local_session_maker, skip_existing=False).copy_events(
                *WINDOW, SPORT
            )

        async with local_session_maker() as session:
            snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        last_hour = [s for s in snapshots if s.snapshot_time == KICKOFF - timedelta(hours=1)]
        assert len(snapshots) == 3
        assert {s.source for s in last_hour} == {Source.ODDS_API, Source.BETFAIR}

    @pytest.mark.asyncio
    async def test_price_rows_equal_production_and_link_to_the_local_snapshot_copy(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _seed_new_shape(prod_session_maker, "evt_a", snapshot_id=502)
        for _ in range(2):
            await _copier(prod_session_maker, local_session_maker, skip_existing=False).copy_events(
                *WINDOW, SPORT
            )

        async with local_session_maker() as session:
            snapshot_ids = (
                (await session.execute(select(OddsSnapshot.id).order_by(OddsSnapshot.id)))
                .scalars()
                .all()
            )
            [price] = (await session.execute(select(Price))).scalars().all()
        assert price.snapshot_id == snapshot_ids[2]

    @pytest.mark.asyncio
    async def test_a_local_snapshot_production_lacks_is_replaced(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)
        async with local_session_maker() as session:
            session.add(
                OddsSnapshot(
                    event_id="evt_a",
                    snapshot_time=KICKOFF - timedelta(hours=30),
                    source=Source.ODDS_API,
                    market=Market.ONE_X_TWO,
                    time_basis=TimeBasis.OBSERVED,
                )
            )
            await session.commit()

        await _copier(prod_session_maker, local_session_maker, skip_existing=False).copy_events(
            *WINDOW, SPORT
        )

        async with local_session_maker() as session:
            times = {
                s.snapshot_time for s in (await session.execute(select(OddsSnapshot))).scalars()
            }
        assert KICKOFF - timedelta(hours=30) not in times

    @pytest.mark.asyncio
    async def test_skip_existing_leaves_existing_event_untouched(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)

        stats = await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)

        assert (stats.events_skipped, stats.events_copied) == (1, 0)


class TestCopyLeavesLocalTwins:
    @pytest.mark.asyncio
    async def test_local_twin_and_its_rows_survive_and_are_reported(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        twin_id = "fduk_2026-2027_A_B_2026-08-22"
        async with local_session_maker() as session:
            session.add(
                Event(
                    id=twin_id,
                    sport_key=SPORT,
                    sport_title="EPL",
                    commence_time=KICKOFF + timedelta(minutes=30),
                    home_team="Home evt_a",
                    away_team="Away evt_a",
                    status=EventStatus.SCHEDULED,
                )
            )
            await session.flush()
            session.add(
                OddsSnapshot(
                    event_id=twin_id,
                    snapshot_time=KICKOFF,
                    source=Source.FOOTBALL_DATA_UK,
                    market=Market.ONE_X_TWO,
                    time_basis=TimeBasis.CLOSE,
                )
            )
            await session.commit()

        stats = await _copier(
            prod_session_maker, local_session_maker, skip_existing=False
        ).copy_events(*WINDOW, SPORT)

        assert stats.events_failed == 0
        assert stats.local_twin_matches == 1
        async with local_session_maker() as session:
            assert await session.get(Event, twin_id) is not None
            twin_snapshots = (
                (
                    await session.execute(
                        select(OddsSnapshot).where(OddsSnapshot.event_id == twin_id)
                    )
                )
                .scalars()
                .all()
            )
        assert len(twin_snapshots) == 1


async def _seed_new_shape(
    prod_session_maker: async_sessionmaker[AsyncSession], event_id: str, snapshot_id: int
) -> None:
    async with prod_session_maker() as session:
        event = await session.get(Event, event_id)
        assert event is not None and event.surrogate_id is not None
        session.add_all(
            [
                Price(
                    snapshot_id=snapshot_id,
                    bookmaker="bet365",
                    outcome=PriceOutcome.HOME,
                    side=Side.BACK,
                    price=Decimal("2.5"),
                ),
                EventExternalId(
                    source=Source.ODDSPORTAL, external_id="Kl1AOZN9", event_id=event.surrogate_id
                ),
                HealthCheckAcknowledgement(
                    event_id=event.surrogate_id,
                    check_name=HealthCheck.MISSING_MLB_GAME_PK,
                    reason="exhibition",
                ),
            ]
        )
        await session.commit()


class TestCopyNewShape:
    @pytest.mark.asyncio
    async def test_prices_external_ids_and_acknowledgements_follow_the_local_keys(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _seed_new_shape(prod_session_maker, "evt_a", snapshot_id=502)
        async with local_session_maker() as session:
            session.add(make_epl_event("local_first", KICKOFF + timedelta(days=30)))
            await session.commit()

        for skip_existing in (True, False):
            stats = await _copier(
                prod_session_maker, local_session_maker, skip_existing=skip_existing
            ).copy_events(*WINDOW, SPORT)
            assert stats.events_failed == 0

        async with local_session_maker() as session:
            event = await session.get(Event, "evt_a")
            assert event is not None
            [price] = (await session.execute(select(Price))).scalars().all()
            snapshot = await session.get(OddsSnapshot, price.snapshot_id)
            assert snapshot is not None and snapshot.event_id == "evt_a"
            assert snapshot.snapshot_time == KICKOFF - timedelta(hours=1)
            [external] = (await session.execute(select(EventExternalId))).scalars().all()
            assert (external.external_id, external.event_id) == ("Kl1AOZN9", event.surrogate_id)
            [ack] = (await session.execute(select(HealthCheckAcknowledgement))).scalars().all()
            assert (ack.event_id, ack.reason) == (event.surrogate_id, "exhibition")

    @pytest.mark.asyncio
    async def test_the_local_event_holding_an_id_is_the_production_event(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _seed_new_shape(prod_session_maker, "evt_a", snapshot_id=502)
        twin_key = await _hold_locally(local_session_maker, "local_twin", "Kl1AOZN9")

        skipped = await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)
        copied = await _copier(
            prod_session_maker, local_session_maker, skip_existing=False
        ).copy_events(*WINDOW, SPORT)

        assert (skipped.events_skipped, copied.events_copied, copied.events_failed) == (1, 1, 0)
        async with local_session_maker() as session:
            assert [e.id for e in (await session.execute(select(Event))).scalars()] == [
                "local_twin"
            ]
            event = await session.get(Event, "local_twin")
            assert event is not None
            assert (event.surrogate_id, event.home_team) == (twin_key, "Home evt_a")
            snapshot_events = set((await session.execute(select(OddsSnapshot.event_id))).scalars())
            assert snapshot_events == {"local_twin"}
            [external] = (await session.execute(select(EventExternalId))).scalars().all()
            assert external.event_id == twin_key

    @pytest.mark.asyncio
    async def test_ids_held_by_several_local_events_fail_the_event(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _seed_new_shape(prod_session_maker, "evt_a", snapshot_id=502)
        async with prod_session_maker() as session:
            event = await session.get(Event, "evt_a")
            assert event is not None
            session.add(
                EventExternalId(
                    source=Source.BETFAIR, external_id="35000001", event_id=event.surrogate_id
                )
            )
            await session.commit()
        await _hold_locally(local_session_maker, "local_one", "Kl1AOZN9")
        await _hold_locally(local_session_maker, "local_two", "35000001", Source.BETFAIR)

        stats = await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)

        assert stats.failed_event_ids == ["evt_a"]

    @pytest.mark.asyncio
    async def test_an_event_without_local_ids_is_matched_by_its_fixture(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        async with local_session_maker() as session:
            session.add(
                Event(
                    id="fduk_local",
                    sport_key=SPORT,
                    sport_title="EPL",
                    commence_time=KICKOFF,
                    home_team="Home evt_a",
                    away_team="Away evt_a",
                )
            )
            await session.commit()

        stats = await _copier(
            prod_session_maker, local_session_maker, skip_existing=False
        ).copy_events(*WINDOW, SPORT)

        assert (stats.events_copied, stats.events_failed) == (1, 0)
        async with local_session_maker() as session:
            assert [e.id for e in (await session.execute(select(Event))).scalars()] == [
                "fduk_local"
            ]
            event = await session.get(Event, "fduk_local")
            assert event is not None and event.status is EventStatus.FINAL


async def _hold_locally(
    local_session_maker: async_sessionmaker[AsyncSession],
    event_id: str,
    external_id: str,
    source: Source = Source.ODDSPORTAL,
) -> int:
    async with local_session_maker() as session:
        event = make_epl_event(event_id, KICKOFF + timedelta(days=3))
        session.add(event)
        await session.flush()
        assert event.surrogate_id is not None
        session.add(
            EventExternalId(source=source, external_id=external_id, event_id=event.surrogate_id)
        )
        await session.commit()
        return event.surrogate_id


class TestCopyMatchesAnEarlierCopy:
    @pytest.mark.asyncio
    async def test_an_event_whose_kickoff_moved_is_matched_by_its_id(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)
        moved = KICKOFF + timedelta(hours=3)
        async with prod_session_maker() as session:
            event = await session.get(Event, "evt_a")
            assert event is not None
            event.commence_time = moved
            await session.commit()

        skipped = await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)
        copied = await _copier(
            prod_session_maker, local_session_maker, skip_existing=False
        ).copy_events(*WINDOW, SPORT)

        assert (skipped.events_skipped, copied.events_copied, copied.events_failed) == (1, 1, 0)
        async with local_session_maker() as session:
            event = await session.get(Event, "evt_a")
        assert event is not None and event.commence_time == moved

    @pytest.mark.asyncio
    async def test_a_fixture_holding_another_id_of_the_same_source_is_not_matched(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _seed_new_shape(prod_session_maker, "evt_a", snapshot_id=502)
        async with local_session_maker() as session:
            twin = Event(
                id="local_twin",
                sport_key=SPORT,
                sport_title="EPL",
                commence_time=KICKOFF,
                home_team="Home evt_a",
                away_team="Away evt_a",
            )
            session.add(twin)
            await session.flush()
            assert twin.surrogate_id is not None
            twin_key = twin.surrogate_id
            session.add(
                EventExternalId(source=Source.ODDSPORTAL, external_id="Other123", event_id=twin_key)
            )
            await session.commit()

        for _ in range(2):
            stats = await _copier(
                prod_session_maker, local_session_maker, skip_existing=False
            ).copy_events(*WINDOW, SPORT)
            assert (stats.events_copied, stats.events_failed) == (1, 0)

        async with local_session_maker() as session:
            held = {
                (x.external_id, x.event_id == twin_key)
                for x in (await session.execute(select(EventExternalId))).scalars()
            }
            ids = {e.id for e in (await session.execute(select(Event))).scalars()}
        assert ids == {"local_twin", "evt_a"}
        assert held == {("Other123", True), ("Kl1AOZN9", False)}
