"""Integration tests for the constraints of the prices, payload and event-identity tables."""

from __future__ import annotations

from decimal import Decimal

import pytest
from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    HealthCheckAcknowledgement,
    OddsSnapshot,
    Price,
    SnapshotPayload,
)
from odds_core.vocabulary import HealthCheck, Market, PriceOutcome, Side, Source
from odds_lambda.storage.event_merge import EventMerger
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import TWIN_KICKOFF, make_epl_event


async def rejected(session: AsyncSession, *rows: object) -> bool:
    """Whether flushing ``rows`` violates a constraint; the session stays usable either way."""
    try:
        async with session.begin_nested():
            session.add_all(rows)
            await session.flush()
    except IntegrityError:
        return True
    return False


def price(snapshot_id: int | None, **fields: object) -> Price:
    values: dict[str, object] = {
        "snapshot_id": snapshot_id,
        "bookmaker": "betfair_exchange",
        "outcome": PriceOutcome.HOME,
        "side": Side.BACK,
        "price": Decimal("2.0"),
    }
    return Price(**(values | fields))


class TestEventId:
    @pytest.mark.asyncio
    async def test_every_event_gets_a_distinct_id(self, pglite_async_session) -> None:
        events = [
            Event(
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=TWIN_KICKOFF,
                home_team="Arsenal",
                away_team="Chelsea",
            )
            for _ in range(3)
        ]
        pglite_async_session.add_all(events)
        await pglite_async_session.flush()

        keys = [e.id for e in events]
        assert None not in keys
        assert len(set(keys)) == 3

    @pytest.mark.asyncio
    async def test_unmatched_is_a_stored_status(self, pglite_async_session) -> None:
        pglite_async_session.add(make_epl_event(1, status=EventStatus.UNMATCHED))
        await pglite_async_session.flush()
        pglite_async_session.expunge_all()

        stored = await pglite_async_session.get(Event, 1)
        assert stored is not None and stored.status is EventStatus.UNMATCHED


class TestEventExternalIds:
    @pytest.mark.asyncio
    async def test_one_event_per_source_id(self, pglite_async_session) -> None:
        a, b = make_epl_event(1), make_epl_event(2)
        pglite_async_session.add_all([a, b])
        await pglite_async_session.flush()
        pglite_async_session.add(
            EventExternalId(source=Source.BETFAIR, external_id="1", event_id=a.id)
        )
        await pglite_async_session.flush()

        assert await rejected(
            pglite_async_session,
            EventExternalId(source=Source.BETFAIR, external_id="1", event_id=b.id),
        )

    @pytest.mark.asyncio
    async def test_one_id_per_source_per_event(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        key = stored_epl_event.id
        pglite_async_session.add(
            EventExternalId(source=Source.BETFAIR, external_id="1", event_id=key)
        )
        await pglite_async_session.flush()

        assert await rejected(
            pglite_async_session,
            EventExternalId(source=Source.BETFAIR, external_id="2", event_id=key),
        )
        assert not await rejected(
            pglite_async_session,
            EventExternalId(source=Source.ODDS_API, external_id="2", event_id=key),
        )

    @pytest.mark.asyncio
    async def test_a_source_that_issues_no_ids_is_refused(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        assert await rejected(
            pglite_async_session,
            EventExternalId(
                source=Source.FOOTBALL_DATA_UK,
                external_id="1",
                event_id=stored_epl_event.id,
            ),
        )

    @pytest.mark.asyncio
    async def test_a_merge_moves_the_duplicates_ids_to_the_kept_event(
        self, pglite_async_session
    ) -> None:
        keep, duplicate = make_epl_event(1), make_epl_event(2)
        pglite_async_session.add_all([keep, duplicate])
        await pglite_async_session.flush()
        pglite_async_session.add(
            EventExternalId(source=Source.ODDSPORTAL, external_id="op-1", event_id=duplicate.id)
        )
        await pglite_async_session.flush()

        await EventMerger(pglite_async_session).merge(keep, [duplicate])
        await pglite_async_session.flush()

        moved = await pglite_async_session.get(EventExternalId, (Source.ODDSPORTAL, "op-1"))
        assert moved is not None and moved.event_id == keep.id


class TestMergeClashes:
    @pytest.fixture
    async def twins(self, pglite_async_session) -> tuple[Event, Event]:
        keep, duplicate = make_epl_event(1), make_epl_event(2)
        pglite_async_session.add_all([keep, duplicate])
        await pglite_async_session.flush()
        return keep, duplicate

    @pytest.mark.asyncio
    async def test_a_duplicates_id_from_a_source_the_kept_event_holds_is_dropped(
        self, pglite_async_session, twins
    ) -> None:
        keep, duplicate = twins
        pglite_async_session.add_all(
            [
                EventExternalId(source=Source.ODDSPORTAL, external_id="op-keep", event_id=keep.id),
                EventExternalId(
                    source=Source.ODDSPORTAL, external_id="op-dup", event_id=duplicate.id
                ),
                EventExternalId(source=Source.BETFAIR, external_id="bf-dup", event_id=duplicate.id),
            ]
        )
        await pglite_async_session.flush()

        report = await EventMerger(pglite_async_session).merge(keep, [duplicate])
        await pglite_async_session.flush()

        held = await pglite_async_session.execute(
            select(EventExternalId.source, EventExternalId.external_id).where(
                EventExternalId.event_id == keep.id
            )
        )
        assert set(held.all()) == {(Source.ODDSPORTAL, "op-keep"), (Source.BETFAIR, "bf-dup")}
        assert report.rows_deleted == {"event_external_ids": 1}
        [log] = (await pglite_async_session.execute(select(DataQualityLog))).scalars().all()
        assert log.issue_type == "merge_dropped_external_id"
        assert log.raw_data is not None and log.raw_data["external_id"] == "op-dup"

    @pytest.mark.asyncio
    async def test_a_duplicates_acknowledgement_of_a_check_the_kept_event_has_is_dropped(
        self, pglite_async_session, twins
    ) -> None:
        keep, duplicate = twins
        for event, reason in ((keep, "kept reason"), (duplicate, "duplicate reason")):
            pglite_async_session.add(
                HealthCheckAcknowledgement(
                    event_id=event.id,
                    check_name=HealthCheck.MISSING_MLB_GAME_PK,
                    reason=reason,
                )
            )
        await pglite_async_session.flush()

        report = await EventMerger(pglite_async_session).merge(keep, [duplicate])
        await pglite_async_session.flush()

        reasons = await pglite_async_session.execute(select(HealthCheckAcknowledgement.reason))
        assert reasons.scalars().all() == ["kept reason"]
        assert report.rows_deleted == {"health_check_acknowledgements": 1}
        [log] = (await pglite_async_session.execute(select(DataQualityLog))).scalars().all()
        assert log.issue_type == "merge_dropped_acknowledgement"


class TestHealthCheckAcknowledgements:
    @pytest.mark.asyncio
    async def test_one_acknowledgement_per_event_and_check(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        def ack() -> HealthCheckAcknowledgement:
            return HealthCheckAcknowledgement(
                event_id=stored_epl_event.id,
                check_name=HealthCheck.MISSING_MLB_GAME_PK,
                reason="exhibition game",
            )

        pglite_async_session.add(ack())
        await pglite_async_session.flush()

        assert await rejected(pglite_async_session, ack())


class TestSnapshotKey:
    @pytest.mark.asyncio
    async def test_one_snapshot_per_event_source_market_and_time(
        self, pglite_async_session, epl_snapshot
    ) -> None:
        twin = OddsSnapshot(
            event_id=epl_snapshot.event_id,
            snapshot_time=epl_snapshot.snapshot_time,
            source=epl_snapshot.source,
            market=epl_snapshot.market,
        )

        assert await rejected(pglite_async_session, twin)

    @pytest.mark.asyncio
    async def test_snapshots_not_yet_backfilled_are_not_keyed(
        self, pglite_async_session, stored_epl_event
    ) -> None:
        legacy = [
            OddsSnapshot(
                event_id=stored_epl_event.id,
                snapshot_time=TWIN_KICKOFF,
            )
            for _ in range(2)
        ]

        assert not await rejected(pglite_async_session, *legacy)

    @pytest.mark.asyncio
    async def test_market_and_source_are_closed_sets(
        self, pglite_async_session, epl_snapshot
    ) -> None:
        for column, value in (
            ("source", "mlb_stats_api"),
            ("market", "btts"),
            ("time_basis", "x"),
            ("market_status", "OPEN"),
        ):
            with pytest.raises(IntegrityError):
                async with pglite_async_session.begin_nested():
                    await pglite_async_session.execute(
                        text(f"UPDATE odds_snapshots SET {column} = :value WHERE id = :id"),
                        {"value": value, "id": epl_snapshot.id},
                    )

    @pytest.mark.asyncio
    async def test_a_snapshot_references_its_payload(
        self, pglite_async_session, epl_snapshot
    ) -> None:
        payload = SnapshotPayload(payload={"markets": []})
        pglite_async_session.add(payload)
        await pglite_async_session.flush()
        epl_snapshot.payload_id = payload.id
        await pglite_async_session.flush()

        assert payload.id is not None and payload.received_at is not None


class TestPrices:
    @pytest.mark.asyncio
    async def test_a_price_must_exceed_one(self, pglite_async_session, epl_snapshot) -> None:
        assert await rejected(pglite_async_session, price(epl_snapshot.id, price=Decimal("1.0")))

    @pytest.mark.asyncio
    async def test_a_missing_line_is_one_key(self, pglite_async_session, epl_snapshot) -> None:
        pglite_async_session.add(price(epl_snapshot.id))
        await pglite_async_session.flush()

        assert await rejected(pglite_async_session, price(epl_snapshot.id, price=Decimal("2.1")))
        assert not await rejected(pglite_async_session, price(epl_snapshot.id, side=Side.LAY))

    @pytest.mark.asyncio
    async def test_lines_are_part_of_the_key(self, pglite_async_session, epl_snapshot) -> None:
        over = {"outcome": PriceOutcome.OVER}
        assert not await rejected(
            pglite_async_session,
            price(epl_snapshot.id, line=Decimal("2.5"), **over),
            price(epl_snapshot.id, line=Decimal("3.5"), **over),
        )

    @pytest.mark.asyncio
    async def test_outcome_and_side_are_closed_sets(
        self, pglite_async_session, epl_snapshot
    ) -> None:
        pglite_async_session.add(price(epl_snapshot.id))
        await pglite_async_session.flush()
        for column, value in (("outcome", "Arsenal"), ("side", "ask")):
            with pytest.raises(IntegrityError):
                async with pglite_async_session.begin_nested():
                    await pglite_async_session.execute(
                        text(f"UPDATE prices SET {column} = :v"), {"v": value}
                    )

    @pytest.mark.asyncio
    async def test_prices_go_with_their_snapshot(self, pglite_async_session, epl_snapshot) -> None:
        pglite_async_session.add(price(epl_snapshot.id))
        await pglite_async_session.flush()

        await pglite_async_session.delete(epl_snapshot)
        await pglite_async_session.flush()

        assert (await pglite_async_session.execute(select(Price))).first() is None

    @pytest.mark.asyncio
    async def test_values_read_back_as_typed_members(
        self, pglite_async_session, epl_snapshot
    ) -> None:
        pglite_async_session.add(price(epl_snapshot.id, size=Decimal("150.25")))
        await pglite_async_session.flush()
        pglite_async_session.expunge_all()

        stored = (await pglite_async_session.execute(select(Price))).scalar_one()
        snapshot = await pglite_async_session.get(OddsSnapshot, epl_snapshot.id)
        assert stored.outcome is PriceOutcome.HOME
        assert stored.side is Side.BACK
        assert stored.size == Decimal("150.25")
        assert snapshot is not None and snapshot.market is Market.ONE_X_TWO
