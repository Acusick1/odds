"""The parity check: replaying recorded decisions against the bets placed live."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.paper_trade_models import CandidateDecision, PaperTrade

from tests.integration.test_candidate_backtest import (
    EXCHANGE,
    KICKOFF,
    POLYMARKET,
    RETAIL,
    T,
    betfair,
    candidate,
    model_candidate,
    polymarket_candidate,
)
from tests.test_helpers import (
    make_epl_event,
    make_tagged_snapshot,
    polymarket_bookmaker,
    store_tagged,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

KNOWN_AT = T + timedelta(minutes=1)


async def add_event(
    session: Any,
    retail: tuple[float, float, float] = RETAIL,
    late_retail: tuple[float, float, float] | None = None,
    late_stored: timedelta = timedelta(minutes=-1),
) -> list[int]:
    """An event whose OddsPortal snapshot was stored before the run, and that snapshot's id.

    ``late_retail`` adds a snapshot stamped 2 minutes before the decision time and stored
    ``late_stored`` after ``KNOWN_AT`` (negative: before it); the run did not see it, so its
    id is not returned.
    """
    session.add(make_epl_event("e1", KICKOFF))
    snapshot = make_tagged_snapshot(
        "e1",
        T - timedelta(minutes=5),
        [three_way_bookmaker("bet365", retail), betfair(EXCHANGE)],
        "oddsportal_live",
    )
    snapshot.created_at = KNOWN_AT - timedelta(hours=1)
    await store_tagged(session, snapshot)
    if late_retail is not None:
        late = make_tagged_snapshot(
            "e1",
            T - timedelta(minutes=2),
            [three_way_bookmaker("bet365", late_retail), betfair(EXCHANGE)],
            "oddsportal_live",
        )
        late.created_at = KNOWN_AT + late_stored
        await store_tagged(session, late)
    await session.commit()
    assert snapshot.id is not None
    return [snapshot.id]


async def add_polymarket_snapshot(session: Any) -> int:
    """A Polymarket snapshot stamped 5 minutes before the decision and stored before the run,
    and its id."""
    snapshot = make_tagged_snapshot(
        "e1",
        T - timedelta(minutes=5),
        [polymarket_bookmaker(POLYMARKET)],
        "polymarket",
    )
    snapshot.created_at = KNOWN_AT - timedelta(hours=1)
    await store_tagged(session, snapshot)
    await session.commit()
    assert snapshot.id is not None
    return snapshot.id


async def add_decision(
    session: Any,
    input_ids: list[int],
    trades: list[tuple[str, str, float]] | None = None,
    missed: bool = False,
    version: int = 1,
    decision_time: datetime = T,
    kickoff: datetime = KICKOFF,
    known_at: datetime | None = None,
) -> None:
    """A decision of ``gap`` on ``e1`` with the given (selection, book, price) unit-stake trades."""
    trades = trades or []
    decision = CandidateDecision(
        candidate_name="gap",
        candidate_version=version,
        event_id="e1",
        minutes_before_kickoff=int((kickoff - decision_time).total_seconds() // 60),
        decision_time=decision_time,
        kickoff_at_decision=kickoff,
        known_at=known_at or decision_time + timedelta(minutes=1),
        input_snapshot_ids=input_ids,
        bets_placed=len(trades),
        missed=missed,
    )
    session.add(decision)
    await session.flush()
    for selection, book, price in trades:
        session.add(
            PaperTrade(
                event_id="e1",
                market="1x2",
                selection=selection,
                bookmaker=book,
                price_decimal=price,
                stake=1.0,
                candidate_name="gap",
                candidate_version=version,
                decision_id=decision.id,
                placed_at=KNOWN_AT,
            )
        )
    await session.commit()


class TestCheckParity:
    async def test_matching_bets(self, pglite_async_session) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.2)])

        report = await check_parity(pglite_async_session, candidate())

        assert (report.checked, len(report.of(ParityStatus.MATCH)), report.ok) == (1, 1, True)
        assert report.data_differences == []

    async def test_zero_bet_decision_matches_a_zero_bet_replay(self, pglite_async_session) -> None:
        ids = await add_event(pglite_async_session, retail=(1.9, 3.0, 3.0))
        await add_decision(pglite_async_session, ids)

        report = await check_parity(pglite_async_session, candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]

    async def test_a_different_bet_set_is_a_mismatch(self, pglite_async_session) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.3)])

        report = await check_parity(pglite_async_session, candidate())

        [mismatch] = report.of(ParityStatus.MISMATCH)
        assert mismatch.live == (("home", None, "bet365", 2.3, 1.0),)
        assert mismatch.replayed == (("home", None, "bet365", 2.2, 1.0),)
        assert not report.ok

    async def test_a_recorded_snapshot_missing_from_the_database_is_a_mismatch(
        self, pglite_async_session
    ) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, [*ids, 99999], [("home", "bet365", 2.2)])

        [d] = (await check_parity(pglite_async_session, candidate())).decisions

        assert d.status is ParityStatus.MISMATCH and d.missing_snapshot_ids == (99999,)

    async def test_a_polymarket_decision_replays_to_the_same_bets(
        self, pglite_async_session
    ) -> None:
        [retail_id] = await add_event(pglite_async_session)
        polymarket_id = await add_polymarket_snapshot(pglite_async_session)
        await add_decision(
            pglite_async_session, [retail_id, polymarket_id], [("home", "bet365", 2.2)]
        )

        report = await check_parity(pglite_async_session, polymarket_candidate())

        assert (report.checked, len(report.of(ParityStatus.MATCH)), report.ok) == (1, 1, True)
        assert report.data_differences == []

    async def test_a_recorded_polymarket_snapshot_missing_from_the_database_is_a_mismatch(
        self, pglite_async_session
    ) -> None:
        [retail_id] = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, [retail_id, 99999], [("home", "bet365", 2.2)])

        [d] = (await check_parity(pglite_async_session, polymarket_candidate())).decisions

        assert d.status is ParityStatus.MISMATCH and d.missing_snapshot_ids == (99999,)

    async def test_a_snapshot_stored_before_known_at_but_unseen_is_a_data_difference(
        self, pglite_async_session
    ) -> None:
        # The live run bet nothing; a snapshot stamped before the decision and stored a
        # minute before the run is in the database now, so the database replay bets.
        ids = await add_event(pglite_async_session, retail=(1.9, 3.0, 3.0), late_retail=RETAIL)
        await add_decision(pglite_async_session, ids)

        report = await check_parity(pglite_async_session, candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]
        [difference] = report.data_differences
        assert difference.replayed == () and difference.db_replayed == (
            ("home", None, "bet365", 2.2, 1.0),
        )
        assert report.ok

    async def test_a_snapshot_stored_after_known_at_is_no_difference(
        self, pglite_async_session
    ) -> None:
        ids = await add_event(
            pglite_async_session,
            retail=(1.9, 3.0, 3.0),
            late_retail=RETAIL,
            late_stored=timedelta(minutes=10),
        )
        await add_decision(pglite_async_session, ids)

        report = await check_parity(pglite_async_session, candidate())

        assert report.data_differences == [] and report.ok

    async def test_missed_decisions_and_other_versions_are_skipped(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session)
        await add_decision(pglite_async_session, [], missed=True)

        assert (await check_parity(pglite_async_session, candidate())).checked == 0

    async def test_version_other_than_the_registered_one_is_an_error(
        self, pglite_async_session
    ) -> None:
        with pytest.raises(ValueError, match="version 1"):
            await check_parity(pglite_async_session, candidate(), version=2)

    async def test_a_model_rule_is_refused(self, pglite_async_session) -> None:
        with pytest.raises(ValueError, match="needs a fitted model"):
            await check_parity(pglite_async_session, model_candidate())

    async def test_an_outcome_bet_at_an_earlier_decision_is_not_replayed_again(
        self, pglite_async_session
    ) -> None:
        later_ids = await add_event(pglite_async_session)
        early_time = KICKOFF - timedelta(minutes=90)
        early = make_tagged_snapshot(
            "e1",
            early_time - timedelta(minutes=5),
            [three_way_bookmaker("bet365", RETAIL), betfair(EXCHANGE)],
            "oddsportal_live",
        )
        early.created_at = early_time - timedelta(minutes=4)
        await store_tagged(pglite_async_session, early)
        await pglite_async_session.commit()
        assert early.id is not None
        await add_decision(
            pglite_async_session, [early.id], [("home", "bet365", 2.2)], decision_time=early_time
        )
        # The later decision saw the same home edge and bet nothing, as home was already bet.
        await add_decision(pglite_async_session, later_ids)
        two_times = candidate(
            params={
                "threshold": 0.02,
                "minutes_before_kickoff": [60, 90],
                "max_input_age_minutes": 10,
            }
        )

        report = await check_parity(pglite_async_session, two_times)

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH, ParityStatus.MATCH]
        assert report.data_differences == []

    async def test_an_earlier_mismatch_does_not_spread_to_a_later_decision(
        self, pglite_async_session
    ) -> None:
        later_ids = await add_event(pglite_async_session)
        early_time = KICKOFF - timedelta(minutes=90)
        # The early decision records a snapshot missing from the database, so its replay bets
        # nothing; live bet home there, so the later decision correctly bet nothing.
        await add_decision(
            pglite_async_session, [99999], [("home", "bet365", 2.2)], decision_time=early_time
        )
        await add_decision(pglite_async_session, later_ids)
        two_times = candidate(
            params={
                "threshold": 0.02,
                "minutes_before_kickoff": [60, 90],
                "max_input_age_minutes": 10,
            }
        )

        report = await check_parity(pglite_async_session, two_times)

        assert [d.status for d in report.decisions] == [
            ParityStatus.MISMATCH,
            ParityStatus.MATCH,
        ]

    async def test_decisions_are_replayed_in_live_order_after_a_kickoff_moved_earlier(
        self, pglite_async_session
    ) -> None:
        # KO-90 bet home; the kick-off then moved 31 minutes earlier, so the KO-60 decision,
        # taken next, has an earlier decision time and bet nothing because home was bet.
        await add_event(pglite_async_session)
        snapshot = make_tagged_snapshot(
            "e1",
            KICKOFF - timedelta(minutes=95),
            [three_way_bookmaker("bet365", RETAIL), betfair(EXCHANGE)],
            "oddsportal_live",
        )
        snapshot.created_at = KICKOFF - timedelta(minutes=94)
        await store_tagged(pglite_async_session, snapshot)
        await pglite_async_session.commit()
        assert snapshot.id is not None
        await add_decision(
            pglite_async_session,
            [snapshot.id],
            [("home", "bet365", 2.2)],
            decision_time=KICKOFF - timedelta(minutes=90),
        )
        moved = KICKOFF - timedelta(minutes=31)
        await add_decision(
            pglite_async_session,
            [snapshot.id],
            decision_time=moved - timedelta(minutes=60),
            kickoff=moved,
            known_at=KICKOFF - timedelta(minutes=88),
        )
        two_times = candidate(
            params={
                "threshold": 0.02,
                "minutes_before_kickoff": [60, 90],
                "max_input_age_minutes": 10,
            }
        )

        report = await check_parity(pglite_async_session, two_times)

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH, ParityStatus.MATCH]

    async def test_decision_time_is_not_the_events_current_kickoff(
        self, pglite_async_session
    ) -> None:
        # The event moved after the decision; the replay uses the kick-off the decision saw.
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.2)])
        event = await pglite_async_session.get(type(make_epl_event("x")), "e1")
        event.commence_time = datetime(2026, 4, 11, 15, 0, tzinfo=UTC)
        await pglite_async_session.commit()

        report = await check_parity(pglite_async_session, candidate())

        assert report.ok and report.checked == 1
