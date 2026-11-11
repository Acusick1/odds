"""The retail-consensus rules of round 1 through the stored-data paths: the replay of
retail-quorum lag over its lookback snapshots, and the parity check of each rule."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.candidates import MechanicalCandidate
from odds_lambda.candidate_rules.pinning import rule_sha256
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.test_candidate_backtest import (
    KICKOFF,
    T,
    betfair,
    candidate,
    run,
)
from tests.integration.test_candidate_parity import add_decision
from tests.test_helpers import (
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

GROUPED = ("paddypower", "ladbrokes", "williamhill", "betvictor")
LAGGING = (2.2, 3.4, 4.0)
LAG_PARAMS = {
    "threshold": 0.02,
    "lookbacks": [{"minutes_before_decision": 30, "max_input_age_minutes": 10}],
    "minutes_before_kickoff": [60],
    "max_input_age_minutes": 10,
}


def _priced(home: float, draw: float, away: float) -> tuple[float, float, float]:
    """Prices with a 5% margin whose proportional devig gives these probabilities."""
    return (1 / (home * 1.05), 1 / (draw * 1.05), 1 / (away * 1.05))


def groups_at(*p: float) -> list[dict[str, Any]]:
    return [three_way_bookmaker(book, _priced(*p)) for book in GROUPED]


def rule_candidate(rule: str, params: dict[str, Any]) -> MechanicalCandidate:
    return candidate(rule=rule, rule_sha256=rule_sha256(rule), params=params)


async def add_match(session: AsyncSession) -> None:
    """Event 1, which home won 2-1, with a Betfair close."""
    session.add(make_epl_event(1, KICKOFF, status="final", home_score=2, away_score=1))
    await session.flush()
    await store_tagged(
        session,
        make_tagged_snapshot(
            1, KICKOFF - timedelta(minutes=2), [betfair((2.0, 4.0, 4.0))], "betfair_api"
        ),
    )
    await session.commit()


async def add_snapshot(
    session: AsyncSession, minutes_before_t: float, books: list[dict[str, Any]]
) -> int:
    """An OddsPortal snapshot of event 1, stored a minute after its stamp, and its id."""
    stamp: datetime = T - timedelta(minutes=minutes_before_t)
    snapshot = make_tagged_snapshot(1, stamp, books, "oddsportal_live")
    snapshot.created_at = stamp + timedelta(minutes=1)
    stored = await store_tagged(session, snapshot)
    await session.commit()
    assert stored.id is not None
    return stored.id


async def add_lag(
    session: AsyncSession, lagging_then: tuple[float, float, float] = LAGGING
) -> list[int]:
    """Four groups that raised the home probability from 0.48 to 0.50 between the lookback and
    the decision, with bet365 at ``lagging_then`` and then 2.2 on the home win; the snapshots'
    ids, the decision's first."""
    then = await add_snapshot(
        session, 35, [*groups_at(0.48, 0.28, 0.24), three_way_bookmaker("bet365", lagging_then)]
    )
    now = await add_snapshot(
        session, 5, [*groups_at(0.50, 0.27, 0.23), three_way_bookmaker("bet365", LAGGING)]
    )
    return [now, then]


async def replay_lag(session: AsyncSession) -> Any:
    return await run(session, rule_candidate("retail_quorum_lag", LAG_PARAMS))


class TestRetailQuorumLagReplay:
    async def test_backs_the_lagging_price_from_the_lookback_snapshots(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(pglite_async_session)
        await add_lag(pglite_async_session)

        result = await replay_lag(pglite_async_session)

        assert (result.coverage.with_input, result.coverage.bets) == (1, 1)
        # Home won at 2.2 for a unit stake.
        assert result.summary.pnl == pytest.approx(1.2)

    async def test_a_price_that_moved_since_the_lookback_is_not_backed(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(pglite_async_session)
        await add_lag(pglite_async_session, lagging_then=(2.1, 3.4, 4.0))

        result = await replay_lag(pglite_async_session)

        assert (result.coverage.with_input, result.coverage.bets) == (1, 0)

    async def test_no_lookback_snapshot_is_no_input(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(pglite_async_session)
        await add_snapshot(
            pglite_async_session,
            5,
            [*groups_at(0.50, 0.27, 0.23), three_way_bookmaker("bet365", LAGGING)],
        )

        result = await replay_lag(pglite_async_session)

        assert (result.coverage.decisions, result.coverage.with_input) == (1, 0)


async def add_outlier(session: AsyncSession) -> list[int]:
    """bet365's home 2.5 against William Hill's 2.3, and an exchange fair price of 0.5."""
    books = [
        three_way_bookmaker("bet365", (2.5, 3.0, 3.0)),
        three_way_bookmaker("williamhill", (2.3, 3.0, 3.0)),
        three_way_bookmaker("coral", (2.2, 3.1, 3.1)),
        betfair((2.0, 4.0, 4.0)),
    ]
    return [await add_snapshot(session, 5, books)]


async def add_subsidy(session: AsyncSession) -> list[int]:
    """bet365 at 2.1 on the home win, 0.05 above a consensus of 0.5, with its margin on the
    draw and the away win."""
    books = [*groups_at(0.50, 0.27, 0.23), three_way_bookmaker("bet365", (2.1, 3.2, 3.7))]
    return [await add_snapshot(session, 5, books)]


PARITY_CASES = [
    ("retail_quorum_lag", LAG_PARAMS, add_lag, ("home", "bet365", 2.2)),
    (
        "single_book_outlier",
        {"min_ratio": 1.05, "gap": 0, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
        add_outlier,
        ("home", "bet365", 2.5),
    ),
    (
        "subsidised_outcome",
        {
            "threshold": 0.02,
            "min_overround": 0.04,
            "minutes_before_kickoff": [60],
            "max_input_age_minutes": 10,
        },
        add_subsidy,
        ("home", "bet365", 2.1),
    ),
]


class TestParity:
    @pytest.mark.parametrize(("rule", "params", "add_inputs", "bet"), PARITY_CASES)
    async def test_the_recorded_decision_replays_to_its_bet(
        self,
        pglite_async_session: AsyncSession,
        rule: str,
        params: dict[str, Any],
        add_inputs: Any,
        bet: tuple[str, str, float],
    ) -> None:
        await add_match(pglite_async_session)
        ids = await add_inputs(pglite_async_session)
        await add_decision(pglite_async_session, ids, [bet])

        report = await check_parity(pglite_async_session, rule_candidate(rule, params))

        [decision] = report.decisions
        assert decision.status is ParityStatus.MATCH
        assert report.data_differences == []

    @pytest.mark.parametrize(("rule", "params", "add_inputs", "bet"), PARITY_CASES)
    async def test_a_different_recorded_bet_is_a_mismatch(
        self,
        pglite_async_session: AsyncSession,
        rule: str,
        params: dict[str, Any],
        add_inputs: Any,
        bet: tuple[str, str, float],
    ) -> None:
        await add_match(pglite_async_session)
        ids = await add_inputs(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("draw", bet[1], 3.0)])

        report = await check_parity(pglite_async_session, rule_candidate(rule, params))

        assert [d.status for d in report.decisions] == [ParityStatus.MISMATCH]
