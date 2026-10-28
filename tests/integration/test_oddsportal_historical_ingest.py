"""Integration tests for the historical OddsPortal loader's resolver and snapshot writes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from odds_core.models import (
    Event,
    EventExternalId,
    EventStatus,
    OddsSnapshot,
    Price,
    SnapshotPayload,
)
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.oddsportal_common import IngestionStats, event_id_from_match_link
from sqlalchemy import func, select

from scripts.ingest_oddsportal import (
    MARKET_CONFIGS,
    SPORT_CONFIGS,
    _ingest_one_game,
    _normalize_team_passthrough,
    get_market_config,
)
from tests.test_helpers import make_epl_event

MATCH_LINK = "https://www.oddsportal.com/football/england/premier-league/arsenal-chelsea-AbCd1234/"
MATCH_DT = datetime(2024, 8, 17, 12, 30, tzinfo=UTC)


def _history(opening: float, closing: float) -> dict[str, Any]:
    return {
        "opening_odds": {"timestamp": "2024-08-14T10:00:00", "odds": opening},
        "odds_history": [{"timestamp": "2024-08-17T11:00:00", "odds": closing}],
    }


def _record(**fields: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "match_date": "2024-08-17 12:30:00 UTC",
        "match_link": MATCH_LINK,
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "home_score": "2",
        "away_score": "1",
        "1x2_market": [
            {
                "bookmaker_name": "bet365",
                "odds_history_data": [_history(2.0, 1.9), _history(3.5, 3.4), _history(4.0, 4.2)],
            },
            {
                "bookmaker_name": "Betway",
                "odds_history_data": [_history(2.1, 2.0), _history(3.6, 3.5), _history(3.9, 4.1)],
            },
        ],
    }
    record.update(fields)
    return record


async def _ingest(session, record: dict[str, Any]) -> IngestionStats:
    stats = IngestionStats()
    await _ingest_one_game(
        session,
        record,
        "2024-2025",
        stats,
        sport_config=SPORT_CONFIGS["soccer"],
        market_config=MARKET_CONFIGS["1x2"],
        json_market_key="1x2_market",
        normalize_fn=_normalize_team_passthrough,
        canonical_to_abbrev=None,
    )
    await session.flush()
    return stats


async def _snaps(session, event_id: str) -> list[OddsSnapshot]:
    result = await session.execute(
        select(OddsSnapshot)
        .where(OddsSnapshot.event_id == event_id)
        .order_by(OddsSnapshot.snapshot_time)
    )
    return list(result.scalars().all())


class TestMatchLinkId:
    def test_reads_the_trailing_event_id(self) -> None:
        assert event_id_from_match_link(MATCH_LINK) == "AbCd1234"

    @pytest.mark.parametrize("link", ["", None, "https://www.oddsportal.com/football/"])
    def test_a_link_without_one_has_none(self, link: str | None) -> None:
        assert event_id_from_match_link(link) is None

    def test_a_head_to_head_link_takes_the_id_from_its_fragment(self) -> None:
        link = "https://www.oddsportal.com/football/h2h/arsenal-OOOOOOOO/chelsea-TTTTTTTT/#AbCd1234"
        assert event_id_from_match_link(link) == "AbCd1234"

    def test_a_head_to_head_link_without_a_fragment_names_no_event(self) -> None:
        link = "https://www.oddsportal.com/football/h2h/arsenal-OOOOOOOO/chelsea-TTTTTTTT/"
        assert event_id_from_match_link(link) is None

    def test_a_query_string_is_ignored(self) -> None:
        assert event_id_from_match_link(MATCH_LINK + "?x=1") == "AbCd1234"


class TestNewEvent:
    @pytest.mark.asyncio
    async def test_creates_the_event_with_its_result_and_the_oddsportal_id(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session

        stats = await _ingest(session, _record())

        event = await session.get(Event, "op_2024-2025_ARS_CHE_2024-08-17")
        assert event is not None
        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 2, 1)
        assert event.commence_time == MATCH_DT and event.completed_at == MATCH_DT
        ids = await session.execute(
            select(EventExternalId).where(EventExternalId.event_id == event.surrogate_id)
        )
        assert [(r.source, r.external_id) for r in ids.scalars()] == [
            (Source.ODDSPORTAL, "AbCd1234")
        ]
        assert (stats.events_created, stats.snapshots_inserted) == (1, 2)

    @pytest.mark.asyncio
    async def test_writes_source_stamped_snapshots_with_prices_and_one_payload(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        await _ingest(session, _record())

        snaps = await _snaps(session, "op_2024-2025_ARS_CHE_2024-08-17")

        assert [(s.source, s.market, s.time_basis) for s in snaps] == [
            (Source.ODDSPORTAL, Market.ONE_X_TWO, TimeBasis.SOURCE_STAMPED)
        ] * 2
        assert snaps[0].payload_id == snaps[1].payload_id is not None
        payload = await session.get(SnapshotPayload, snaps[0].payload_id)
        assert payload is not None
        assert [e["source"] for e in payload.payload] == ["oddsportal"] * 2
        closing = (
            (await session.execute(select(Price).where(Price.snapshot_id == snaps[1].id)))
            .scalars()
            .all()
        )
        assert {(p.bookmaker, p.outcome, p.side): p.price for p in closing} == {
            ("bet365", PriceOutcome.HOME, Side.BACK): Decimal("1.9"),
            ("bet365", PriceOutcome.DRAW, Side.BACK): Decimal("3.4"),
            ("bet365", PriceOutcome.AWAY, Side.BACK): Decimal("4.2"),
            ("betway", PriceOutcome.HOME, Side.BACK): Decimal("2.0"),
            ("betway", PriceOutcome.DRAW, Side.BACK): Decimal("3.5"),
            ("betway", PriceOutcome.AWAY, Side.BACK): Decimal("4.1"),
        }

    @pytest.mark.asyncio
    async def test_totals_snapshots_carry_their_line(self, pglite_async_session) -> None:
        session = pglite_async_session
        record = {
            "over_under_2_5_market": [
                {
                    "bookmaker_name": "bet365",
                    "odds_history_data": [_history(1.9, 1.8), _history(1.95, 2.0)],
                }
            ],
            **_record(),
        }
        stats = IngestionStats()
        await _ingest_one_game(
            session,
            record,
            "2024-2025",
            stats,
            sport_config=SPORT_CONFIGS["soccer"],
            market_config=get_market_config("over_under_2_5"),
            json_market_key="over_under_2_5_market",
            normalize_fn=_normalize_team_passthrough,
            canonical_to_abbrev=None,
        )
        await session.flush()

        [snap, *_] = await _snaps(session, "op_2024-2025_ARS_CHE_2024-08-17")
        assert snap.market is Market.TOTALS
        prices = (
            (await session.execute(select(Price).where(Price.snapshot_id == snap.id)))
            .scalars()
            .all()
        )
        assert {(p.outcome, p.line) for p in prices} == {
            (PriceOutcome.OVER, Decimal("2.5")),
            (PriceOutcome.UNDER, Decimal("2.5")),
        }

    @pytest.mark.asyncio
    async def test_rerun_adds_nothing(self, pglite_async_session) -> None:
        session = pglite_async_session
        await _ingest(session, _record())

        stats = await _ingest(session, _record())

        assert (stats.events_matched, stats.snapshots_inserted) == (1, 0)
        assert len(await _snaps(session, "op_2024-2025_ARS_CHE_2024-08-17")) == 2
        count = await session.execute(select(func.count()).select_from(SnapshotPayload))
        assert count.scalar_one() == 1


class TestExistingEvent:
    @pytest.mark.asyncio
    async def test_attaches_to_an_event_a_day_off_and_keeps_its_result_and_time(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        stored = make_epl_event("odds_api_hex", MATCH_DT + timedelta(hours=20))
        session.add(stored)
        await session.flush()

        stats = await _ingest(session, _record())

        assert (stats.events_matched, stats.events_created) == (1, 0)
        assert stored.status is EventStatus.SCHEDULED and stored.home_score is None
        assert stored.commence_time == MATCH_DT + timedelta(hours=20)
        assert len(await _snaps(session, "odds_api_hex")) == 2

    @pytest.mark.asyncio
    async def test_finds_an_event_by_its_recorded_oddsportal_id(self, pglite_async_session) -> None:
        session = pglite_async_session
        stored = make_epl_event("op_live_x", MATCH_DT + timedelta(days=3))
        session.add(stored)
        await session.flush()
        session.add(
            EventExternalId(
                source=Source.ODDSPORTAL, external_id="AbCd1234", event_id=stored.surrogate_id
            )
        )
        await session.flush()

        await _ingest(session, _record())

        assert len(await _snaps(session, "op_live_x")) == 2
