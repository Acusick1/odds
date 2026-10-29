"""The quote readers: odds_core.quotes.quotes and the candidate rules' own query."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.quotes import quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis
from odds_lambda.candidate_rules.inputs import SharpInput, rule_quotes

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_tagged_snapshot,
    polymarket_bookmaker,
    store_tagged,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

KICKOFF = datetime(2026, 11, 7, 15, 0, tzinfo=UTC)
T = KICKOFF - timedelta(hours=1)
EXCHANGE = frozenset({SharpInput.EXCHANGE})
POLYMARKET = frozenset({SharpInput.POLYMARKET})


@pytest.fixture
async def stored(pglite_async_session) -> dict[str, int]:
    """Two events: an OddsPortal snapshot of retail, Pinnacle and exchange prices, a live
    Betfair snapshot, and another event's OddsPortal snapshot."""
    session = pglite_async_session
    session.add(make_epl_event(1, KICKOFF))
    session.add(make_epl_event(2, KICKOFF, away_team="Spurs"))
    oddsportal = await store_tagged(
        session,
        make_tagged_snapshot(
            1,
            T - timedelta(minutes=5),
            [
                three_way_bookmaker("bet365", (2.0, 3.0, 4.0)),
                three_way_bookmaker("pinnacle", (2.1, 3.1, 4.1)),
                betfair_exchange_bookmaker((2.0, 4.0, 4.0)),
            ],
            "oddsportal_live",
        ),
    )
    oddsportal.created_at = T + timedelta(minutes=5)
    betfair = await store_tagged(
        session,
        make_tagged_snapshot(
            1,
            T - timedelta(minutes=1),
            [
                betfair_exchange_bookmaker(
                    (2.0, 4.0, 4.0),
                    runners=[
                        {"runner_name": name, "best_back": back, "best_lay": back + 0.02}
                        for name, back in (("Arsenal", 2.0), ("The Draw", 4.0), ("Chelsea", 4.0))
                    ],
                )
            ],
            "betfair_api",
        ),
    )
    other = await store_tagged(
        session,
        make_tagged_snapshot(
            2,
            T - timedelta(minutes=5),
            [
                three_way_bookmaker("bet365", (2.0, 3.0, 4.0), away_team="Spurs"),
                betfair_exchange_bookmaker((2.0, 4.0, 4.0), away_team="Spurs"),
            ],
            "oddsportal_live",
        ),
    )
    await session.commit()
    assert oddsportal.id and betfair.id and other.id
    return {"oddsportal": oddsportal.id, "betfair": betfair.id, "other": other.id}


class TestQuotes:
    async def test_carries_the_snapshot_fields(self, pglite_async_session, stored) -> None:
        rows = await quotes(pglite_async_session, [1], Market.ONE_X_TWO, sources=[Source.BETFAIR])
        assert {q.snapshot_id for q in rows} == {stored["betfair"]}
        back = next(q for q in rows if q.side is Side.BACK and q.outcome == "home")
        assert (back.source, back.time_basis, back.in_play) == (
            Source.BETFAIR,
            TimeBasis.OBSERVED,
            False,
        )
        assert back.price == 2.0 and isinstance(back.price, float)
        assert {q.side for q in rows} == {Side.BACK, Side.LAY}

    async def test_filters_narrow_in_sql(self, pglite_async_session, stored) -> None:
        session = pglite_async_session
        backs = await quotes(
            session, [1, 2], Market.ONE_X_TWO, bookmakers=["bet365"], sides=[Side.BACK]
        )
        assert {(q.event_id, q.bookmaker) for q in backs} == {(1, "bet365"), (2, "bet365")}
        window = await quotes(session, [1], Market.ONE_X_TWO, since=T - timedelta(minutes=2))
        assert {q.snapshot_id for q in window} == {stored["betfair"]}
        known = await quotes(session, [1], Market.ONE_X_TWO, known_at=T)
        assert {q.snapshot_id for q in known} == {stored["betfair"]}
        by_id = await quotes(session, [1], Market.ONE_X_TWO, snapshot_ids=[stored["betfair"]])
        assert {q.snapshot_id for q in by_id} == {stored["betfair"]}
        assert await quotes(session, [1], Market.TOTALS) == []


class TestRuleQuotes:
    async def test_reads_only_retail_sources_backs_and_rule_books(
        self, pglite_async_session, stored
    ) -> None:
        rows = await rule_quotes(pglite_async_session, 1, "1x2", EXCHANGE)
        assert {q.snapshot_id for q in rows} == {stored["oddsportal"]}
        assert {q.bookmaker for q in rows} == {"bet365", "betfair_exchange"}
        assert {q.side for q in rows} == {Side.BACK}

    async def test_polymarket_mids_are_read_only_when_requested(
        self, pglite_async_session, stored
    ) -> None:
        polymarket = await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                1,
                T - timedelta(minutes=3),
                [polymarket_bookmaker((0.5, 0.3, 0.2))],
                "polymarket",
            ),
        )
        await pglite_async_session.commit()

        without = await rule_quotes(pglite_async_session, 1, "1x2", EXCHANGE)
        with_polymarket = await rule_quotes(pglite_async_session, 1, "1x2", POLYMARKET)

        assert polymarket.id not in {q.snapshot_id for q in without}
        mids = [q for q in with_polymarket if q.snapshot_id == polymarket.id]
        assert mids and {q.side for q in mids} == {Side.MID}
        assert {q.snapshot_id for q in with_polymarket} >= {stored["oddsportal"], polymarket.id}

    async def test_recorded_ids_are_read_whatever_their_event(
        self, pglite_async_session, stored
    ) -> None:
        rows = await rule_quotes(
            pglite_async_session, 1, "1x2", EXCHANGE, snapshot_ids=[stored["other"]]
        )
        assert {q.event_id for q in rows} == {2}

    async def test_known_at_drops_snapshots_stored_later(
        self, pglite_async_session, stored
    ) -> None:
        assert await rule_quotes(pglite_async_session, 1, "1x2", EXCHANGE, known_at=T) == []
