"""The backfill of stored snapshots into the new shape, and the verify step that checks it."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    HealthCheckAcknowledgement,
    Odds,
    OddsSnapshot,
    Price,
    SnapshotPayload,
)
from odds_core.paper_trade_models import CandidateDecision
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketMarketType,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.shape_backfill import snapshots as snapshots_module
from odds_lambda.shape_backfill.events import (
    IdOutcome,
    MergeDropsRefused,
    attach_oddsportal_history,
    merge_twins,
    migrate_unmatched,
    oddsportal_history_games,
    plan_twin_merges,
    record_stored_ids,
    stored_external_ids,
)
from odds_lambda.shape_backfill.legacy_polymarket import migrate_legacy_polymarket
from odds_lambda.shape_backfill.run import run_backfill
from odds_lambda.shape_backfill.snapshots import (
    SnapshotScan,
    SplitSnapshotReferenced,
    backfill_snapshots,
    capture_cutoff,
    scan_snapshots,
)
from odds_lambda.shape_backfill.verify import Check, verify_shape
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import add_event, make_epl_event, three_way_bookmaker

NOW = datetime.now(UTC).replace(microsecond=0)
KICKOFF = NOW - timedelta(days=3)


def _odds_api_envelope(event_id: str = "abc123") -> dict[str, Any]:
    return {
        "id": event_id,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "williamhill",
                "title": "William Hill",
                "last_update": "2026-10-20T11:00:00Z",
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.2},
                            {"name": "Draw", "price": 3.4},
                            {"name": "Chelsea", "price": 3.1},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.9, "point": 2.5},
                            {"name": "Under", "price": 1.95, "point": 2.5},
                        ],
                    },
                ],
            },
            {
                "key": "betfair_ex_uk",
                "title": "Betfair",
                "last_update": "2026-10-20T11:00:00Z",
                "markets": [
                    {
                        "key": "h2h_lay",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.3},
                            {"name": "Draw", "price": 3.6},
                            {"name": "Chelsea", "price": 3.3},
                        ],
                    }
                ],
            },
        ],
    }


def _betfair_envelope() -> dict[str, Any]:
    return {
        "source": "betfair_api",
        "betfair_event_id": "35000001",
        "bookmakers": [
            {
                "key": "betfair_exchange",
                "title": "Betfair Exchange",
                "last_update": "2026-10-22T10:00:00+00:00",
                "markets": [{"key": "1x2", "outcomes": []}],
                "betfair_meta": {
                    "inplay": False,
                    "market_status": "OPEN",
                    "total_matched": 1234.5,
                    "market_start_time": "2026-10-22T15:00:00+00:00",
                    "runners": [
                        {
                            "runner_name": "Arsenal",
                            "best_back": 2.0,
                            "back_size": 10.0,
                            "best_lay": 2.04,
                            "lay_size": 12.0,
                            "last_price_traded": 2.02,
                        },
                        {"runner_name": "The Draw", "best_back": 3.5, "best_lay": 3.6},
                        {"runner_name": "Chelsea", "best_back": 4.0, "best_lay": 4.2},
                    ],
                },
            }
        ],
    }


def _oddsportal_envelope(tag: str, price: float = 2.1, **extra: Any) -> dict[str, Any]:
    return {
        "source": tag,
        **extra,
        "bookmakers": [three_way_bookmaker("bet365", (price, 3.4, 3.5))],
    }


def _fduk_envelope(basis: str | None, price: float) -> dict[str, Any]:
    envelope: dict[str, Any] = {
        "source": "football_data_uk",
        "bookmakers": [three_way_bookmaker("pinnacle", (price, 3.4, 3.5))],
    }
    if basis is not None:
        envelope["timing"] = {"basis": basis, "kickoff_time_assumed": False}
    return envelope


async def _legacy(
    session: AsyncSession,
    event_id: str,
    raw_data: dict[str, Any],
    snapshot_time: datetime,
    api_request_id: str | None = None,
) -> OddsSnapshot:
    snapshot, _ = await OddsWriter(session).store_odds_snapshot(
        event_id, raw_data, snapshot_time, validate=False, api_request_id=api_request_id
    )
    return snapshot


async def _count(session: AsyncSession, model: Any) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def _table_counts(session: AsyncSession) -> dict[str, int]:
    models = (
        Event,
        OddsSnapshot,
        Price,
        SnapshotPayload,
        Odds,
        DataQualityLog,
        EventExternalId,
        HealthCheckAcknowledgement,
    )
    return {m.__tablename__: await _count(session, m) for m in models}  # type: ignore[attr-defined]


async def _scan(session: AsyncSession, batch_size: int = 2) -> SnapshotScan:
    cutoff = await capture_cutoff(session)
    return await scan_snapshots(session, cutoff=cutoff, batch_size=batch_size)


async def _backfill(session: AsyncSession, batch_size: int = 2, **kwargs: Any) -> Any:
    scan = await _scan(session, batch_size)
    return await backfill_snapshots(session, scan, batch_size=batch_size, **kwargs)


@pytest.fixture
async def stored(pglite_async_session: AsyncSession) -> dict[str, int]:
    """One EPL event with a legacy snapshot from every source."""
    session = pglite_async_session
    await add_event(session, make_epl_event("abc123", commence_time=KICKOFF))
    ids = {
        "odds_api_history": await _legacy(
            session, "abc123", _odds_api_envelope(), KICKOFF - timedelta(days=2)
        ),
        "odds_api_live": await _legacy(
            session,
            "abc123",
            {**_odds_api_envelope(), "bookmakers": _odds_api_envelope()["bookmakers"][:1]},
            NOW,
        ),
        "betfair": await _legacy(
            session, "abc123", _betfair_envelope(), KICKOFF - timedelta(hours=5)
        ),
        "oddsportal_live": await _legacy(
            session,
            "abc123",
            _oddsportal_envelope("oddsportal_live", oddsportal_event_id="AbCd1234"),
            KICKOFF - timedelta(hours=4),
        ),
        "oddsportal": await _legacy(
            session, "abc123", _oddsportal_envelope("oddsportal"), KICKOFF - timedelta(hours=6)
        ),
        "fduk_opening": await _legacy(
            session, "abc123", _fduk_envelope(None, 2.0), KICKOFF - timedelta(hours=24)
        ),
        "fduk_closing": await _legacy(
            session, "abc123", _fduk_envelope(None, 2.1), KICKOFF - timedelta(hours=1)
        ),
        "fduk_tagged": await _legacy(
            session,
            "abc123",
            _fduk_envelope("fduk_collection_deadline", 2.05),
            KICKOFF - timedelta(hours=30),
        ),
    }
    await session.commit()
    return {name: s.id for name, s in ids.items() if s.id is not None}


class TestSnapshotBackfill:
    async def test_fills_source_market_and_time_basis_per_source(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        await _backfill(pglite_async_session)

        rows = {
            s.id: s for s in (await pglite_async_session.execute(select(OddsSnapshot))).scalars()
        }
        expected = {
            "odds_api_history": (Source.ODDS_API, Market.ONE_X_TWO, TimeBasis.UPPER_BOUND),
            "odds_api_live": (Source.ODDS_API, Market.ONE_X_TWO, TimeBasis.OBSERVED),
            "betfair": (Source.BETFAIR, Market.ONE_X_TWO, TimeBasis.OBSERVED),
            "oddsportal_live": (Source.ODDSPORTAL, Market.ONE_X_TWO, TimeBasis.OBSERVED),
            "oddsportal": (Source.ODDSPORTAL, Market.ONE_X_TWO, TimeBasis.SOURCE_STAMPED),
            "fduk_opening": (Source.FOOTBALL_DATA_UK, Market.ONE_X_TWO, TimeBasis.UPPER_BOUND),
            "fduk_closing": (Source.FOOTBALL_DATA_UK, Market.ONE_X_TWO, TimeBasis.CLOSE),
            "fduk_tagged": (Source.FOOTBALL_DATA_UK, Market.ONE_X_TWO, TimeBasis.UPPER_BOUND),
        }
        for name, (source, market, basis) in expected.items():
            row = rows[stored[name]]
            assert (row.source, row.market, row.time_basis) == (source, market, basis), name
            assert row.payload_id is not None

    async def test_splits_a_multi_market_snapshot_and_moves_odds_rows(
        self, pglite_async_session: AsyncSession, stored: dict[str, int], tmp_path: Path
    ) -> None:
        session = pglite_async_session
        legacy_id = stored["odds_api_history"]
        original = (await session.get(OddsSnapshot, legacy_id)).raw_data  # type: ignore[union-attr]
        map_out = tmp_path / "map.jsonl"

        report = await _backfill(session, map_out=map_out)

        parts = list(
            (
                await session.execute(
                    select(OddsSnapshot).where(
                        OddsSnapshot.payload_id
                        == (await session.get(OddsSnapshot, legacy_id)).payload_id
                    )  # type: ignore[union-attr]
                )
            ).scalars()
        )
        by_market = {p.market: p for p in parts}
        assert set(by_market) == {Market.ONE_X_TWO, Market.TOTALS}
        assert by_market[Market.ONE_X_TWO].id == legacy_id
        assert by_market[Market.TOTALS].created_at == by_market[Market.ONE_X_TWO].created_at
        for market, part in by_market.items():
            keys = {m["key"] for b in part.raw_data["bookmakers"] for m in b["markets"]}
            assert keys == ({"1x2", "h2h_lay"} if market is Market.ONE_X_TWO else {"totals"})
            odds_keys = set(
                (
                    await session.execute(
                        select(Odds.market_key).where(Odds.snapshot_id == part.id)
                    )
                ).scalars()
            )
            assert odds_keys == keys
        payload = await session.get(SnapshotPayload, parts[0].payload_id)
        assert payload is not None and payload.payload == original
        lay = (
            (
                await session.execute(
                    select(Price).where(Price.snapshot_id == legacy_id, Price.side == Side.LAY)
                )
            )
            .scalars()
            .all()
        )
        assert {p.outcome for p in lay} == {PriceOutcome.HOME, PriceOutcome.DRAW, PriceOutcome.AWAY}
        # The live Odds API snapshot splits too.
        assert report.split == 2 and report.new_snapshots == 2
        assert f'"old_snapshot_id": {legacy_id}' in map_out.read_text()

    async def test_writes_betfair_market_metadata(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        await _backfill(pglite_async_session)

        snapshot = await pglite_async_session.get(OddsSnapshot, stored["betfair"])
        assert snapshot is not None
        assert snapshot.in_play is False
        assert float(snapshot.total_matched or 0) == 1234.5
        assert snapshot.market_start_time == datetime(2026, 10, 22, 15, tzinfo=UTC)
        sides = (
            (
                await pglite_async_session.execute(
                    select(Price.side).where(Price.snapshot_id == snapshot.id)
                )
            )
            .scalars()
            .all()
        )
        assert sorted(s.value for s in sides).count("traded") == 1

    async def test_a_second_run_writes_nothing(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        await _backfill(pglite_async_session)
        before = await _table_counts(pglite_async_session)

        report = await _backfill(pglite_async_session)

        assert await _table_counts(pglite_async_session) == before
        assert report.shaped == 0 and report.payloads == 0

    async def test_a_stopped_run_resumes_where_it_stopped(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        first = await _backfill(pglite_async_session, limit=3)
        pending = await _count_pending(pglite_async_session)
        rest = await _backfill(pglite_async_session)

        assert first.shaped == 3
        assert pending == len(stored) - 3
        assert rest.shaped == len(stored) - 3
        assert await _count_pending(pglite_async_session) == 0

    async def test_preflight_counts_and_writes_nothing(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        before = await _table_counts(pglite_async_session)

        report = (await _scan(pglite_async_session)).report

        assert await _table_counts(pglite_async_session) == before
        assert report.shaped == len(stored)
        assert report.split == 2
        applied = await _backfill(pglite_async_session)
        for name in ("shaped", "prices", "new_snapshots", "odds_moved", "odds_deleted", "deleted"):
            assert getattr(applied, name) == getattr(report, name), name

    @pytest.mark.parametrize("batch_size", [1, 2])
    async def test_a_same_time_historical_oddsportal_pair_keeps_the_closing(
        self, pglite_async_session: AsyncSession, batch_size: int
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("op_x", commence_time=KICKOFF))
        at = KICKOFF - timedelta(hours=2)
        opening = await _legacy(
            session, "op_x", _oddsportal_envelope("oddsportal", 2.0), at, "oddsportal"
        )
        closing = await _legacy(
            session, "op_x", _oddsportal_envelope("oddsportal", 2.5), at, "oddsportal"
        )
        opening_id, closing_id, opening_raw = opening.id, closing.id, opening.raw_data
        await session.commit()

        report = await _backfill(session, batch_size=batch_size)

        assert await session.get(OddsSnapshot, opening_id) is None
        kept_source = await session.scalar(
            select(OddsSnapshot.source).where(OddsSnapshot.id == closing_id)
        )
        assert kept_source is Source.ODDSPORTAL
        assert (
            await session.execute(select(func.count()).where(Odds.snapshot_id == opening_id))
        ).scalar_one() == 0
        log = (
            await session.execute(
                select(DataQualityLog).where(DataQualityLog.issue_type == "snapshot_key_conflict")
            )
        ).scalar_one()
        assert log.raw_data["kept_snapshot_id"] == closing_id  # type: ignore[index]
        archived = await session.get(SnapshotPayload, log.raw_data["payload_id"])  # type: ignore[index]
        assert archived is not None and archived.payload == opening_raw
        assert report.deleted == 1 and len(report.conflicts) == 1

    async def test_a_same_time_live_pair_keeps_the_first(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("op_live_x", commence_time=KICKOFF))
        at = KICKOFF - timedelta(hours=2)
        first = await _legacy(
            session,
            "op_live_x",
            _oddsportal_envelope("oddsportal_live", 2.0),
            at,
            "oddsportal_live",
        )
        second = await _legacy(
            session,
            "op_live_x",
            _oddsportal_envelope("oddsportal_live", 2.5),
            at,
            "oddsportal_live",
        )
        first_id, second_id = first.id, second.id
        await session.commit()

        report = await _backfill(session)

        assert await session.get(OddsSnapshot, first_id) is not None
        assert await session.get(OddsSnapshot, second_id) is None
        assert report.conflicts[0].held_by == first_id

    async def test_a_group_of_three_keeps_the_first(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("op_x", commence_time=KICKOFF))
        at = KICKOFF - timedelta(hours=2)
        ids = [
            (
                await _legacy(
                    session, "op_x", _oddsportal_envelope("oddsportal", price), at, "oddsportal"
                )
            ).id
            for price in (2.0, 2.5, 2.7)
        ]
        await session.commit()

        report = await _backfill(session)

        assert sorted(c.snapshot_id for c in report.conflicts) == ids[1:]
        assert {c.held_by for c in report.conflicts} == {ids[0]}

    async def test_a_mixed_market_group_keeps_the_first(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("op_x", commence_time=KICKOFF))
        at = KICKOFF - timedelta(hours=2)
        first = await _legacy(
            session, "op_x", _oddsportal_envelope("oddsportal", 2.0), at, "oddsportal"
        )
        both = _oddsportal_envelope("oddsportal", 2.5)
        both["bookmakers"][0]["markets"].append(
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "price_decimal": 1.9, "point": 2.5},
                    {"name": "Under", "price_decimal": 1.95, "point": 2.5},
                ],
            }
        )
        second = await _legacy(session, "op_x", both, at, "oddsportal")
        first_id, second_id = first.id, second.id
        await session.commit()

        report = await _backfill(session)

        assert [(c.snapshot_id, c.market, c.held_by) for c in report.conflicts] == [
            (second_id, Market.ONE_X_TWO, first_id)
        ]
        kept_market = await session.scalar(
            select(OddsSnapshot.market).where(OddsSnapshot.id == second_id)
        )
        assert kept_market is Market.TOTALS

    async def test_stops_when_a_decision_recorded_a_snapshot_it_would_split(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        session.add(
            CandidateDecision(
                candidate_name="c",
                candidate_version=1,
                event_id="abc123",
                minutes_before_kickoff=60,
                decision_time=KICKOFF - timedelta(hours=1),
                kickoff_at_decision=KICKOFF,
                known_at=KICKOFF - timedelta(hours=1),
                input_snapshot_ids=[stored["odds_api_history"]],
            )
        )
        await session.commit()
        before = await _table_counts(session)

        with pytest.raises(SplitSnapshotReferenced):
            await _backfill(session, limit=1)

        assert await _table_counts(session) == before

    async def test_leaves_snapshots_stored_after_the_cutoff(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        cutoff = await capture_cutoff(session)
        late = await _legacy(session, "abc123", _oddsportal_envelope("oddsportal", 1.9), NOW)
        late_id = late.id
        await session.execute(
            update(OddsSnapshot)
            .where(OddsSnapshot.id == late_id)
            .values(created_at=cutoff + timedelta(seconds=1))
        )
        await session.commit()

        scan = await scan_snapshots(session, cutoff=cutoff, batch_size=50)
        await backfill_snapshots(session, scan, batch_size=50)

        late_source = await session.scalar(
            select(OddsSnapshot.source).where(OddsSnapshot.id == late_id)
        )
        assert late_source is None


class _UniqueViolation(Exception):
    sqlstate = "23505"


class TestBatchRetry:
    async def test_a_batch_that_loses_a_key_race_is_planned_again(
        self,
        pglite_async_session: AsyncSession,
        stored: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        real = snapshots_module._backfill_batch
        calls: list[int] = []

        async def flaky(session: AsyncSession, scan: SnapshotScan, batch: Any) -> Any:
            calls.append(len(batch))
            if len(calls) == 1:
                raise IntegrityError("INSERT", {}, _UniqueViolation())
            return await real(session, scan, batch)

        monkeypatch.setattr(snapshots_module, "_backfill_batch", flaky)

        report = await _backfill(pglite_async_session)

        assert report.shaped == len(stored)
        assert await _count_pending(pglite_async_session) == 0

    async def test_another_database_error_is_raised(
        self,
        pglite_async_session: AsyncSession,
        stored: dict[str, int],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def broken(session: AsyncSession, scan: SnapshotScan, batch: Any) -> Any:
            raise IntegrityError("INSERT", {}, Exception("not null"))

        monkeypatch.setattr(snapshots_module, "_backfill_batch", broken)

        with pytest.raises(IntegrityError):
            await _backfill(pglite_async_session)


async def _count_pending(session: AsyncSession) -> int:
    return (
        await session.execute(
            select(func.count()).where(OddsSnapshot.source.is_(None))  # type: ignore[union-attr]
        )
    ).scalar_one()


class TestVerify:
    async def test_a_backfilled_database_verifies(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        await _backfill(pglite_async_session)
        cutoff = await capture_cutoff(pglite_async_session)

        report = await verify_shape(pglite_async_session, cutoff=cutoff, batch_size=3)

        assert report.ok, report.examples
        # Both Odds API snapshots split in two.
        assert report.snapshots == len(stored) + 2

    async def test_reports_a_snapshot_not_yet_backfilled(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        cutoff = await capture_cutoff(pglite_async_session)

        report = await verify_shape(pglite_async_session, cutoff=cutoff, batch_size=3)

        assert report.mismatches[Check.NO_SOURCE] == len(stored)

    async def test_catches_a_planted_price_mismatch(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await _backfill(session)
        price = (
            (await session.execute(select(Price).where(Price.snapshot_id == stored["betfair"])))
            .scalars()
            .first()
        )
        assert price is not None
        price.price = price.price + 1
        await session.commit()
        cutoff = await capture_cutoff(session)

        report = await verify_shape(session, cutoff=cutoff, batch_size=3)

        # The snapshot no longer matches its raw_data, nor its payload.
        assert report.mismatches == {Check.PRICES_DIFFER: 1, Check.PAYLOAD_PRICES: 1}
        assert {m.snapshot_id for m in report.examples} == {stored["betfair"]}

    async def test_catches_changed_market_metadata(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await _backfill(session)
        await session.execute(
            update(OddsSnapshot).where(OddsSnapshot.id == stored["betfair"]).values(in_play=True)
        )
        await session.commit()
        cutoff = await capture_cutoff(session)

        report = await verify_shape(session, cutoff=cutoff, batch_size=3)

        assert report.mismatches == {Check.MARKET_META: 1}

    async def test_catches_an_odds_row_on_another_markets_snapshot(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await _backfill(session)
        totals = (
            await session.execute(
                select(OddsSnapshot.id).where(
                    OddsSnapshot.market == Market.TOTALS,
                    OddsSnapshot.payload_id
                    == select(OddsSnapshot.payload_id)
                    .where(OddsSnapshot.id == stored["odds_api_history"])
                    .scalar_subquery(),
                )
            )
        ).scalar_one()
        await session.execute(
            update(Odds)
            .where(Odds.snapshot_id == totals)
            .values(snapshot_id=stored["odds_api_history"])
        )
        await session.commit()
        cutoff = await capture_cutoff(session)

        report = await verify_shape(session, cutoff=cutoff, batch_size=3)

        assert report.mismatches[Check.ODDS_MARKET] == 1


def _decision(event_id: str) -> CandidateDecision:
    return CandidateDecision(
        candidate_name="c",
        candidate_version=1,
        event_id=event_id,
        minutes_before_kickoff=60,
        decision_time=KICKOFF - timedelta(hours=1),
        kickoff_at_decision=KICKOFF,
        known_at=KICKOFF - timedelta(hours=1),
    )


class TestVerifyPayloads:
    async def _split_part(
        self, session: AsyncSession, legacy_id: int, market: Market
    ) -> OddsSnapshot:
        payload_id = select(OddsSnapshot.payload_id).where(OddsSnapshot.id == legacy_id)
        return (
            await session.execute(
                select(OddsSnapshot).where(
                    OddsSnapshot.payload_id == payload_id.scalar_subquery(),
                    OddsSnapshot.market == market,
                )
            )
        ).scalar_one()

    async def _verify(self, session: AsyncSession) -> Any:
        cutoff = await capture_cutoff(session)
        return await verify_shape(session, cutoff=cutoff, batch_size=3)

    async def test_catches_a_bookmaker_dropped_from_a_split_cut(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await _backfill(session)
        part = await self._split_part(session, stored["odds_api_history"], Market.ONE_X_TWO)
        cut = dict(part.raw_data)
        cut["bookmakers"] = [b for b in cut["bookmakers"] if b["key"] != "betfair_ex_uk"]
        await session.execute(
            update(OddsSnapshot).where(OddsSnapshot.id == part.id).values(raw_data=cut)
        )
        await session.execute(
            delete(Price).where(Price.snapshot_id == part.id, Price.side == Side.LAY)
        )
        await session.commit()

        report = await self._verify(session)

        assert report.mismatches == {Check.PAYLOAD_PRICES: 1}

    async def test_catches_a_market_dropped_from_a_split(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await _backfill(session)
        part = await self._split_part(session, stored["odds_api_history"], Market.TOTALS)
        await session.execute(delete(Odds).where(Odds.snapshot_id == part.id))
        await session.execute(delete(OddsSnapshot).where(OddsSnapshot.id == part.id))
        await session.commit()

        report = await self._verify(session)

        assert report.mismatches == {Check.PAYLOAD_LOST: 1}

    async def test_catches_a_deletion_whose_kept_snapshot_does_not_hold_the_key(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("op_live_x", commence_time=KICKOFF))
        at = KICKOFF - timedelta(hours=2)
        first = await _legacy(
            session, "op_live_x", _oddsportal_envelope("oddsportal_live", 2.0), at
        )
        await _legacy(session, "op_live_x", _oddsportal_envelope("oddsportal_live", 2.5), at)
        first_id = first.id
        await session.commit()
        await _backfill(session)
        await session.execute(
            update(OddsSnapshot)
            .where(OddsSnapshot.id == first_id)
            .values(snapshot_time=at + timedelta(minutes=1))
        )
        await session.commit()

        report = await self._verify(session)

        assert report.mismatches == {Check.PAYLOAD_LOST: 1}

    async def test_counts_orphan_odds_and_fails_on_new_ones(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await _backfill(session)
        await session.execute(
            update(Odds).where(Odds.snapshot_id == stored["betfair"]).values(snapshot_id=None)
        )
        orphans = await session.scalar(
            select(func.count()).where(Odds.snapshot_id.is_(None))  # type: ignore[union-attr]
        )
        await session.commit()

        old = await self._verify(session)
        session.add(SnapshotPayload(payload=[], received_at=NOW - timedelta(days=30)))
        await session.commit()
        new = await self._verify(session)

        assert old.orphan_odds == orphans and Check.NEW_ORPHAN_ODDS not in old.mismatches
        assert new.mismatches[Check.NEW_ORPHAN_ODDS] == orphans


class TestTwinMerge:
    async def test_merges_twins_with_the_kickoff_of_the_highest_observing_source(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("abcdef", commence_time=KICKOFF))
        # Its id says football-data.co.uk, but OddsPortal holds an id on it and stored its prices.
        twin = await add_event(
            session,
            make_epl_event("fduk_ARS_CHE", commence_time=KICKOFF + timedelta(minutes=30)),
        )
        assert twin.surrogate_id is not None
        session.add(
            EventExternalId(
                source=Source.ODDSPORTAL, external_id="AbCd1234", event_id=twin.surrogate_id
            )
        )
        await _legacy(
            session,
            "fduk_ARS_CHE",
            _oddsportal_envelope("oddsportal"),
            KICKOFF - timedelta(hours=3),
        )
        await _legacy(session, "abcdef", _odds_api_envelope("abcdef"), KICKOFF - timedelta(days=1))
        await session.commit()

        plans = await plan_twin_merges(session)
        merges = await merge_twins(session, plans, allow_drops=False)

        assert len(merges) == 1 and merges[0].error is None
        assert (merges[0].kept_id, merges[0].kickoff_source) == ("abcdef", Source.ODDSPORTAL)
        assert plans[0].rows_moved == merges[0].rows_moved
        kept = await session.get(Event, "abcdef")
        assert kept is not None and kept.commence_time == KICKOFF + timedelta(minutes=30)
        assert await session.get(Event, "fduk_ARS_CHE") is None
        assert set((await session.execute(select(OddsSnapshot.event_id))).scalars()) == {"abcdef"}
        assert await plan_twin_merges(session) == []

    async def test_preflight_lists_the_group_and_merges_nothing(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("abcdef", commence_time=KICKOFF))
        await add_event(session, make_epl_event("fduk_ARS_CHE", commence_time=KICKOFF))
        await session.commit()

        plans = await plan_twin_merges(session)

        assert [m.merged_ids for m in plans] == [("fduk_ARS_CHE",)]
        assert await _count(session, Event) == 2

    async def test_refuses_a_merge_that_drops_decisions_unless_allowed(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("abcdef", commence_time=KICKOFF))
        await add_event(session, make_epl_event("fduk_ARS_CHE", commence_time=KICKOFF))
        session.add_all([_decision("abcdef"), _decision("fduk_ARS_CHE")])
        await session.commit()

        plans = await plan_twin_merges(session)
        assert plans[0].rows_deleted == {"candidate_decisions": 1}
        before = await _table_counts(session)
        with pytest.raises(MergeDropsRefused):
            await merge_twins(session, plans, allow_drops=False)
        assert await _table_counts(session) == before

        merges = await merge_twins(session, plans, allow_drops=True)
        assert merges[0].rows_deleted == {"candidate_decisions": 1}
        assert await _count(session, CandidateDecision) == 1


def _mlb_event(event_id: str, status: EventStatus, reason: str | None, **fields: Any) -> Event:
    return Event(
        id=event_id,
        sport_key="baseball_mlb",
        sport_title="MLB",
        commence_time=KICKOFF,
        home_team="Boston Red Sox",
        away_team="Tampa Bay Rays" if "2" not in event_id else "New York Yankees",
        status=status,
        no_game_pk_reason=reason,
        **fields,
    )


class TestUnmatched:
    async def test_marks_events_with_a_reason_and_keeps_final_status(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await add_event(session, _mlb_event("op_live_1", EventStatus.SCHEDULED, "not in MLB"))
        await add_event(session, _mlb_event("op_live_2", EventStatus.FINAL, "All-Star Game"))
        await session.commit()

        report = await migrate_unmatched(session, apply=True)
        again = await migrate_unmatched(session, apply=True)

        assert sorted(report.marked) == ["op_live_1", "op_live_2"]
        assert (again.marked, again.already) == ([], 2)
        scheduled = await session.get(Event, "op_live_1")
        final = await session.get(Event, "op_live_2")
        assert scheduled is not None and scheduled.status is EventStatus.UNMATCHED
        assert final is not None and final.status is EventStatus.FINAL
        assert await _count(session, HealthCheckAcknowledgement) == 2


class TestExternalIds:
    async def test_records_each_sources_ids_from_stored_data(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await add_event(
            session, _mlb_event("op_live_9", EventStatus.FINAL, None, mlb_game_pk=745001)
        )
        await session.commit()

        report = await record_stored_ids(session, await stored_external_ids(session), apply=True)

        held = {
            (row.source, row.external_id)
            for row in (await session.execute(select(EventExternalId))).scalars()
        }
        assert held == {
            (Source.ODDS_API, "abc123"),
            (Source.BETFAIR, "35000001"),
            (Source.ODDSPORTAL, "AbCd1234"),
            (Source.MLB_STATS_API, "745001"),
        }
        assert report.outcomes[(Source.BETFAIR, IdOutcome.NEW)] == 1
        again = await record_stored_ids(session, await stored_external_ids(session), apply=True)
        assert set(again.outcomes) == {(s, IdOutcome.HELD) for s, _ in held}

    async def test_refuses_a_second_id_and_keeps_a_newer_odds_api_id(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        event = await session.get(Event, "abc123")
        assert event is not None and event.surrogate_id is not None
        session.add(
            EventExternalId(
                source=Source.ODDS_API, external_id="newer", event_id=event.surrogate_id
            )
        )
        session.add(
            EventExternalId(source=Source.BETFAIR, external_id="999", event_id=event.surrogate_id)
        )
        await session.commit()

        report = await record_stored_ids(session, await stored_external_ids(session), apply=True)

        assert report.outcomes[(Source.ODDS_API, IdOutcome.KEPT_CURRENT)] == 1
        assert report.outcomes[(Source.BETFAIR, IdOutcome.REFUSED)] == 1
        logs = (
            await session.execute(
                select(func.count()).where(DataQualityLog.issue_type == "external_id_refused")
            )
        ).scalar_one()
        assert logs == 1
        await record_stored_ids(session, await stored_external_ids(session), apply=True)
        assert (
            await session.execute(
                select(func.count()).where(DataQualityLog.issue_type == "external_id_refused")
            )
        ).scalar_one() == 1

    async def test_preflight_counts_without_writing(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        report = await record_stored_ids(
            pglite_async_session, await stored_external_ids(pglite_async_session), apply=False
        )

        assert sum(report.outcomes.values()) == 3
        assert await _count(pglite_async_session, EventExternalId) == 0

    async def test_attaches_historical_oddsportal_ids_from_the_scrapes(
        self, pglite_async_session: AsyncSession, tmp_path: Path
    ) -> None:
        session = pglite_async_session
        await add_event(
            session, make_epl_event("op_2024", commence_time=datetime(2024, 10, 4, 16, tzinfo=UTC))
        )
        await session.commit()
        (tmp_path / "epl_2024-2025.json").write_text(
            '[{"home_team": "Arsenal", "away_team": "Chelsea", "match_date": '
            '"2024-10-04 19:00:00 UTC", "match_link": '
            '"https://www.oddsportal.com/football/england/premier-league/arsenal-chelsea-AbCd1234/"}]'
        )
        (tmp_path / "epl_2024-2025_links.json").write_text("[]")

        games = oddsportal_history_games(tmp_path)
        report = await attach_oddsportal_history(session, games, apply=True)

        assert (report.games, report.attached) == (1, 1)
        row = (await session.execute(select(EventExternalId))).scalar_one()
        assert (row.source, row.external_id) == (Source.ODDSPORTAL, "AbCd1234")


async def _legacy_polymarket(session: AsyncSession) -> None:
    tip = KICKOFF
    await add_event(
        session,
        Event(
            id="nba1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=tip,
            home_team="Milwaukee Bucks",
            away_team="Dallas Mavericks",
            status=EventStatus.FINAL,
        ),
    )
    matched = PolymarketEvent(
        pm_event_id="pm1",
        ticker="nba-dal-mil-x",
        slug="s",
        title="t",
        event_id="nba1",
        start_date=tip,
        end_date=tip,
    )
    unmatched = PolymarketEvent(
        pm_event_id="pm2", ticker="nba-bos-nyk-x", slug="s", title="t", start_date=tip, end_date=tip
    )
    session.add_all([matched, unmatched])
    await session.flush()
    moneyline = PolymarketMarket(
        polymarket_event_id=matched.id,
        pm_market_id="m1",
        condition_id="c",
        question="q",
        clob_token_ids=["a", "b"],
        outcomes=["Mavericks", "Bucks"],
        market_type=PolymarketMarketType.MONEYLINE,
    )
    total = PolymarketMarket(
        polymarket_event_id=matched.id,
        pm_market_id="m2",
        condition_id="c",
        question="q",
        clob_token_ids=["a", "b"],
        outcomes=["Over", "Under"],
        market_type=PolymarketMarketType.TOTAL,
        point=220.5,
    )
    session.add_all([moneyline, total])
    await session.flush()
    session.add_all(
        [
            PolymarketPriceSnapshot(
                polymarket_market_id=moneyline.id,
                snapshot_time=tip - timedelta(hours=2),
                outcome_0_price=0.4,
                outcome_1_price=0.6,
            ),
            PolymarketPriceSnapshot(
                polymarket_market_id=moneyline.id,
                snapshot_time=tip - timedelta(hours=1),
                outcome_0_price=0.0,
                outcome_1_price=1.0,
            ),
            PolymarketPriceSnapshot(
                polymarket_market_id=total.id,
                snapshot_time=tip - timedelta(hours=1),
                outcome_0_price=0.5,
                outcome_1_price=0.5,
            ),
            PolymarketOrderBookSnapshot(
                polymarket_market_id=moneyline.id,
                snapshot_time=tip - timedelta(hours=2),
                token_id="a",
                raw_book={"bids": [], "asks": []},
            ),
        ]
    )
    await session.commit()


class TestLegacyPolymarket:
    async def test_migrates_moneyline_prices_as_mid_rows(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _legacy_polymarket(session)

        report = await migrate_legacy_polymarket(session, apply=True)

        snapshot = (await session.execute(select(OddsSnapshot))).scalar_one()
        assert (snapshot.source, snapshot.market, snapshot.time_basis) == (
            Source.POLYMARKET,
            Market.H2H,
            TimeBasis.OBSERVED,
        )
        prices = {p.outcome: p for p in (await session.execute(select(Price))).scalars()}
        assert {p.side for p in prices.values()} == {Side.MID}
        assert float(prices[PriceOutcome.AWAY].price) == pytest.approx(1 / 0.4)
        assert float(prices[PriceOutcome.HOME].price) == pytest.approx(1 / 0.6)
        assert report.unmatched_games == 1
        assert report.invalid_rows == 1
        assert report.rows_skipped["total"] == 1
        assert report.order_books == 1
        assert await _count(session, SnapshotPayload) == 2

    async def test_a_second_run_writes_nothing_and_verify_passes(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _legacy_polymarket(session)
        await migrate_legacy_polymarket(session, apply=True)
        before = await _table_counts(session)

        again = await migrate_legacy_polymarket(session, apply=True)

        assert await _table_counts(session) == before
        assert (again.snapshots, again.existing, again.order_books_existing) == (0, 1, 1)
        cutoff = await capture_cutoff(session)
        assert (await verify_shape(session, cutoff=cutoff, batch_size=10)).ok


class TestRun:
    async def test_preflight_writes_nothing_and_apply_then_verifies(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await _legacy_polymarket(session)
        before = await _table_counts(session)

        preflight = await run_backfill(session, apply=False, batch_size=3)
        assert await _table_counts(session) == before

        applied = await run_backfill(session, apply=True, batch_size=3)
        assert applied.snapshots.shaped == preflight.snapshots.shaped
        assert applied.legacy_polymarket.snapshots == preflight.legacy_polymarket.snapshots
        after = await _table_counts(session)

        await run_backfill(session, apply=True, batch_size=3)
        assert await _table_counts(session) == after
        cutoff = await capture_cutoff(session)
        assert (await verify_shape(session, cutoff=cutoff, batch_size=3)).ok


class TestRunGates:
    async def test_a_referenced_split_stops_the_run_before_any_write(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await add_event(session, _mlb_event("op_live_1", EventStatus.SCHEDULED, "not in MLB"))
        decision = _decision("abc123")
        decision.input_snapshot_ids = [stored["odds_api_history"]]
        session.add(decision)
        await session.commit()
        before = await _table_counts(session)

        with pytest.raises(SplitSnapshotReferenced):
            await run_backfill(session, apply=True, batch_size=3)

        assert await _table_counts(session) == before

    async def test_a_merge_dropping_decisions_stops_the_run_before_any_write(
        self, pglite_async_session: AsyncSession, stored: dict[str, int]
    ) -> None:
        session = pglite_async_session
        await add_event(session, make_epl_event("fduk_ARS_CHE", commence_time=KICKOFF))
        session.add_all([_decision("abc123"), _decision("fduk_ARS_CHE")])
        await session.commit()
        before = await _table_counts(session)

        with pytest.raises(MergeDropsRefused):
            await run_backfill(session, apply=True, batch_size=3)

        assert await _table_counts(session) == before
