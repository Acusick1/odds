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
    RETAIL,
    T,
    betfair,
    candidate,
)
from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

pytestmark = pytest.mark.integration

KNOWN_AT = T + timedelta(minutes=1)


async def add_event(
    session: Any,
    retail: tuple[float, float, float] | None = RETAIL,
    late_retail: tuple[float, float, float] | None = None,
    late_stored: timedelta = timedelta(minutes=1),
) -> None:
    """An event whose retail snapshot, and optionally a later one, was stored before the run.

    The late snapshot is stamped 5 minutes before the decision time and stored ``late_stored``
    before ``KNOWN_AT``.
    """
    session.add(make_epl_event("e1", KICKOFF))
    stored = KNOWN_AT - timedelta(hours=1)
    snapshots = [
        make_tagged_snapshot("e1", T - timedelta(minutes=5), [betfair(EXCHANGE)], "betfair_api"),
    ]
    if retail is not None:
        snapshots.append(
            make_tagged_snapshot(
                "e1",
                T - timedelta(minutes=20),
                [three_way_bookmaker("bet365", retail)],
                "oddsportal_live",
            )
        )
    for snapshot in snapshots:
        snapshot.created_at = stored
    if late_retail is not None:
        late = make_tagged_snapshot(
            "e1",
            T - timedelta(minutes=5),
            [three_way_bookmaker("bet365", late_retail)],
            "oddsportal_live",
        )
        late.created_at = KNOWN_AT - late_stored
        snapshots.append(late)
    session.add_all(snapshots)
    await session.commit()


async def add_decision(
    session: Any,
    trades: list[tuple[str, str, float]] | None = None,
    missed: bool = False,
    version: int = 1,
) -> None:
    """A decision of ``gap`` on ``e1`` with the given (selection, book, price) unit-stake trades."""
    trades = trades or []
    decision = CandidateDecision(
        candidate_name="gap",
        candidate_version=version,
        event_id="e1",
        decision_time=T,
        kickoff_at_decision=KICKOFF,
        known_at=KNOWN_AT,
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
        await add_event(pglite_async_session)
        await add_decision(pglite_async_session, [("home", "bet365", 2.2)])

        report = await check_parity(pglite_async_session, candidate())

        assert (report.checked, len(report.of(ParityStatus.MATCH)), report.ok) == (1, 1, True)

    async def test_zero_bet_decision_matches_a_zero_bet_replay(self, pglite_async_session) -> None:
        await add_event(pglite_async_session, retail=(1.9, 3.0, 3.0))
        await add_decision(pglite_async_session)

        report = await check_parity(pglite_async_session, candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]

    async def test_a_different_bet_set_is_a_mismatch(self, pglite_async_session) -> None:
        await add_event(pglite_async_session)
        await add_decision(pglite_async_session, [("home", "bet365", 2.3)])

        report = await check_parity(pglite_async_session, candidate())

        [mismatch] = report.of(ParityStatus.MISMATCH)
        assert mismatch.live == (("home", "bet365", 2.3, 1.0),)
        assert mismatch.replayed == (("home", "bet365", 2.2, 1.0),)
        assert not report.ok

    async def test_a_snapshot_stored_just_before_known_at_is_storage_lag(
        self, pglite_async_session
    ) -> None:
        # The live run did not see the 2.2 price, stored a minute before it, so it bet nothing.
        await add_event(pglite_async_session, retail=(1.9, 3.0, 3.0), late_retail=RETAIL)
        await add_decision(pglite_async_session)

        report = await check_parity(pglite_async_session, candidate())

        [lag] = report.of(ParityStatus.STORAGE_LAG)
        assert lag.lag_known_at is not None and lag.lag_known_at < KNOWN_AT
        assert report.ok

    async def test_a_snapshot_stored_long_before_known_at_is_a_mismatch(
        self, pglite_async_session
    ) -> None:
        await add_event(
            pglite_async_session,
            retail=(1.9, 3.0, 3.0),
            late_retail=RETAIL,
            late_stored=timedelta(minutes=10),
        )
        await add_decision(pglite_async_session)

        report = await check_parity(pglite_async_session, candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MISMATCH]

    async def test_missed_decisions_and_other_versions_are_skipped(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session)
        await add_decision(pglite_async_session, missed=True)

        assert (await check_parity(pglite_async_session, candidate())).checked == 0

    async def test_version_other_than_the_registered_one_is_an_error(
        self, pglite_async_session
    ) -> None:
        with pytest.raises(ValueError, match="version 1"):
            await check_parity(pglite_async_session, candidate(), version=2)

    async def test_decision_time_is_not_the_events_current_kickoff(
        self, pglite_async_session
    ) -> None:
        # The event moved after the decision; the replay uses the kick-off the decision saw.
        await add_event(pglite_async_session)
        await add_decision(pglite_async_session, [("home", "bet365", 2.2)])
        event = await pglite_async_session.get(type(make_epl_event("x")), "e1")
        event.commence_time = datetime(2026, 4, 11, 15, 0, tzinfo=UTC)
        await pglite_async_session.commit()

        report = await check_parity(pglite_async_session, candidate())

        assert report.ok and report.checked == 1
