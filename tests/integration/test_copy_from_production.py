"""Integration tests for the production data copier, using a second schema as production."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_cli.commands import copy_from_prod as copy_cli
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.odds_math import american_to_decimal
from odds_core.prediction_models import Prediction
from odds_lambda.copy_from_production import CopyStats, ProductionDataCopier
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlmodel import SQLModel
from typer.testing import CliRunner

PROD_SCHEMA = "prod_source"
KICKOFF = datetime(2026, 8, 22, 15, 0, tzinfo=UTC)
WINDOW = (KICKOFF - timedelta(days=1), KICKOFF + timedelta(days=1))
SPORT = "soccer_epl"


def _raw_data() -> dict:
    return {
        "bookmakers": [
            {
                "key": "bet365",
                "title": "Bet365",
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Home", "price": 150},
                            {"name": "Away", "price": 200},
                            {"name": "Draw", "price": 230},
                        ],
                    }
                ],
            }
        ]
    }


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
        for offset, (tag, tier) in enumerate([("football_data_uk", "early"), (None, "closing")]):
            session.add(
                OddsSnapshot(
                    id=first_snapshot_id + offset,
                    event_id=event_id,
                    snapshot_time=kickoff - timedelta(hours=24 * (1 - offset) + 1),
                    raw_data=_raw_data(),
                    bookmaker_count=1,
                    api_request_id=tag,
                    fetch_tier=tier,
                    hours_until_commence=24.0 if offset == 0 else 1.0,
                )
            )
        session.add(
            OddsSnapshot(
                id=first_snapshot_id + 2,
                event_id=event_id,
                snapshot_time=kickoff - timedelta(hours=1),
                raw_data=_raw_data(),
                bookmaker_count=1,
                api_request_id=None,
                fetch_tier="closing",
                hours_until_commence=1.0,
            )
        )
        for outcome, price in (("Home", 150), ("Away", 200), ("Draw", 230)):
            session.add(
                Odds(
                    event_id=event_id,
                    bookmaker_key="bet365",
                    bookmaker_title="Bet365",
                    market_key="1x2",
                    outcome_name=outcome,
                    price_decimal=american_to_decimal(price),
                    odds_timestamp=kickoff - timedelta(hours=1),
                    last_update=kickoff - timedelta(hours=1),
                )
            )
        await session.flush()
        session.add(
            Prediction(
                event_id=event_id,
                snapshot_id=first_snapshot_id + 1,
                model_name="clv",
                model_version="v1",
                predicted_clv=0.02,
            )
        )
        await session.commit()


def _copier(
    prod: async_sessionmaker[AsyncSession],
    local: async_sessionmaker[AsyncSession],
    skip_existing: bool = True,
) -> ProductionDataCopier:
    return ProductionDataCopier(prod, local, skip_existing=skip_existing)


class TestCopyPredictions:
    @pytest.mark.asyncio
    async def test_prediction_follows_its_snapshot_when_ids_differ(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)

        stats = await _copier(prod_session_maker, local_session_maker).copy_events(*WINDOW, SPORT)

        assert stats.events_copied == 1
        assert stats.events_failed == 0
        assert stats.predictions_copied == 1
        async with local_session_maker() as session:
            prediction = (await session.execute(select(Prediction))).scalar_one()
            snapshot = await session.get(OddsSnapshot, prediction.snapshot_id)
        assert snapshot is not None
        assert snapshot.fetch_tier == "closing"

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
        assert stats.predictions_copied == 1
        async with local_session_maker() as session:
            assert len((await session.execute(select(OddsSnapshot))).scalars().all()) == 3
            assert len((await session.execute(select(Prediction))).scalars().all()) == 1


class TestCopyPreservesColumns:
    @pytest.mark.asyncio
    async def test_snapshot_keeps_source_tag_tier_and_creation_time(
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

        def keyed(rows: Sequence[OddsSnapshot]) -> dict[tuple[datetime, str], Any]:
            return {
                (r.snapshot_time, r.fetch_tier or ""): (
                    r.api_request_id,
                    r.hours_until_commence,
                    r.created_at,
                )
                for r in rows
            }

        assert keyed(local_snapshots) == keyed(prod_snapshots)
        assert "football_data_uk" in {s.api_request_id for s in local_snapshots}

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
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        await _seed_prod_event(
            prod_session_maker, "evt_b", first_snapshot_id=600, kickoff=KICKOFF + timedelta(hours=3)
        )
        async with prod_session_maker() as session:
            session.add(
                Prediction(
                    event_id="evt_b",
                    snapshot_id=500,
                    model_name="other",
                    model_version="v1",
                    predicted_clv=0.0,
                )
            )
            await session.commit()
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
    async def test_snapshots_sharing_time_and_tag_stay_distinct_and_keep_their_predictions(
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
            prediction = (await session.execute(select(Prediction))).scalar_one()
        closing = [s for s in snapshots if s.fetch_tier == "closing"]
        assert len(snapshots) == 3
        assert len({(s.snapshot_time, s.api_request_id) for s in closing}) == 1
        assert prediction.snapshot_id in {s.id for s in closing}
        local_closing_with_prediction = [s for s in closing if s.id == prediction.snapshot_id]
        assert len(local_closing_with_prediction) == 1

    @pytest.mark.asyncio
    async def test_odds_rows_equal_production_and_are_stable_across_reruns(
        self,
        prod_session_maker: async_sessionmaker[AsyncSession],
        local_session_maker: async_sessionmaker[AsyncSession],
    ) -> None:
        await _seed_prod_event(prod_session_maker, "evt_a", first_snapshot_id=500)
        for _ in range(2):
            await _copier(prod_session_maker, local_session_maker, skip_existing=False).copy_events(
                *WINDOW, SPORT
            )

        async with prod_session_maker() as session:
            prod_count = len((await session.execute(select(Odds))).scalars().all())
        async with local_session_maker() as session:
            local_count = len((await session.execute(select(Odds))).scalars().all())
        assert prod_count == local_count == 3

    @pytest.mark.asyncio
    async def test_old_untagged_local_snapshot_is_replaced(
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
                    raw_data=_raw_data(),
                    bookmaker_count=1,
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
