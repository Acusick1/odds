"""A rule that declares lookback points: the loader, and the replay and parity over them."""

from __future__ import annotations

from collections.abc import Collection
from datetime import timedelta
from typing import Any

import pytest
from odds_analytics.candidate_backtest import replay
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.vocabulary import Market
from odds_lambda.candidate_rules import Lookback, MarketState, SharpInput, load_market_state
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import CrossMarketRun
from tests.integration.test_candidate_backtest import KICKOFF, N_BOOT, NOW, T, candidate
from tests.integration.test_candidate_parity import add_decision
from tests.test_helpers import make_epl_event

pytestmark = pytest.mark.integration

MAX_AGE = timedelta(minutes=10)
EXCHANGE = frozenset({SharpInput.EXCHANGE})
SIX_HOURS = Lookback(360, timedelta(minutes=60))
# Stamped 5 minutes before the 6 h lookback point.
AT_SIX_HOURS = 365
LOOKBACK_PARAMS = {
    "threshold": 0.02,
    "minutes_before_kickoff": [60],
    "max_input_age_minutes": 10,
    "lookbacks": [{"minutes_before_decision": 360, "max_input_age_minutes": 60}],
}


def lookback_candidate() -> Any:
    return candidate(params=LOOKBACK_PARAMS)


async def load(
    session: AsyncSession,
    *,
    snapshot_ids: Collection[int] | None = None,
    known_at: Any = None,
    lookbacks: tuple[Lookback, ...] = (SIX_HOURS,),
) -> MarketState:
    event = make_epl_event(1, KICKOFF)
    event.id = 1
    return await load_market_state(
        session,
        event,
        Market.ONE_X_TWO,
        T,
        MAX_AGE,
        EXCHANGE,
        UK_RETAIL_BOOKMAKERS,
        known_at=known_at,
        snapshot_ids=snapshot_ids,
        lookbacks=lookbacks,
    )


class TestLoader:
    async def test_the_state_at_the_lookback_point_is_loaded(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        now, _ = await add_cross_market_run(5, totals=False)
        earlier, _ = await add_cross_market_run(AT_SIX_HOURS, totals=False)
        state = await load(pglite_async_session)
        assert state.input_snapshot_ids == [now, earlier]
        assert state.lookbacks[360].decision_time == T - timedelta(hours=6)

    async def test_a_missing_lookback_gives_no_input(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        await add_cross_market_run(5, totals=False)
        assert not (await load(pglite_async_session)).has_input
        assert (await load(pglite_async_session, lookbacks=())).has_input

    async def test_a_stale_lookback_gives_no_input(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        await add_cross_market_run(5, totals=False)
        await add_cross_market_run(360 + 61, totals=False)
        assert not (await load(pglite_async_session)).has_input

    async def test_a_snapshot_after_the_lookback_point_is_not_its_state(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        await add_cross_market_run(5, totals=False)
        await add_cross_market_run(280, totals=False)
        assert not (await load(pglite_async_session)).has_input

    async def test_known_at_bounds_the_lookback_snapshot(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        stored = T - timedelta(minutes=4)
        await add_cross_market_run(5, totals=False, created_at=stored)
        earlier, _ = await add_cross_market_run(
            AT_SIX_HOURS, totals=False, created_at=stored + timedelta(minutes=1)
        )
        before = await load(pglite_async_session, known_at=stored + timedelta(seconds=30))
        after = await load(pglite_async_session, known_at=stored + timedelta(minutes=2))
        assert not before.has_input
        assert after.has_input and after.lookbacks[360].snapshot_id == earlier

    async def test_recorded_ids_replay_the_lookback(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        now, _ = await add_cross_market_run(5, totals=False)
        earlier, _ = await add_cross_market_run(AT_SIX_HOURS, totals=False)
        both = await load(pglite_async_session, snapshot_ids=[now, earlier])
        assert both.input_snapshot_ids == [now, earlier]
        assert not (await load(pglite_async_session, snapshot_ids=[now])).has_input


class TestBacktestAndParity:
    async def _add_event(self, session: AsyncSession) -> None:
        session.add(make_epl_event(1, KICKOFF, status="final", home_score=2, away_score=1))
        await session.flush()

    async def test_the_replay_decides_through_the_same_loader(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        await self._add_event(pglite_async_session)
        await add_cross_market_run(5, totals=False)
        await add_cross_market_run(AT_SIX_HOURS, totals=False)
        result = await replay(
            pglite_async_session,
            lookback_candidate(),
            KICKOFF.date() - timedelta(days=9),
            KICKOFF.date() + timedelta(days=20),
            1,
            now=NOW,
            n_boot=N_BOOT,
        )
        assert (result.coverage.decisions, result.coverage.with_input) == (1, 1)

    async def test_the_replay_misses_a_decision_without_the_lookback(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        await self._add_event(pglite_async_session)
        await add_cross_market_run(5, totals=False)
        result = await replay(
            pglite_async_session,
            lookback_candidate(),
            KICKOFF.date() - timedelta(days=9),
            KICKOFF.date() + timedelta(days=20),
            1,
            now=NOW,
            n_boot=N_BOOT,
        )
        assert (result.coverage.decisions, result.coverage.with_input) == (1, 0)

    async def test_parity_matches_when_the_lookback_is_recorded(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        now, _ = await add_cross_market_run(5, totals=False)
        earlier, _ = await add_cross_market_run(AT_SIX_HOURS, totals=False)
        await add_decision(pglite_async_session, [now, earlier], [("home", "bet365", 2.2)])

        report = await check_parity(pglite_async_session, lookback_candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]

    async def test_parity_checks_the_lookbacks_recorded_input(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        now, _ = await add_cross_market_run(5, totals=False)
        earlier, _ = await add_cross_market_run(AT_SIX_HOURS, totals=False)
        await add_decision(pglite_async_session, [now, earlier + 999], [("home", "bet365", 2.2)])

        [d] = (await check_parity(pglite_async_session, lookback_candidate())).decisions

        assert d.status is ParityStatus.MISMATCH and d.missing_snapshot_ids == (earlier + 999,)

    async def test_parity_without_the_recorded_lookback_replays_no_bets(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        now, _ = await add_cross_market_run(5, totals=False)
        await add_cross_market_run(AT_SIX_HOURS, totals=False)
        await add_decision(pglite_async_session, [now], [("home", "bet365", 2.2)])

        [d] = (await check_parity(pglite_async_session, lookback_candidate())).decisions

        assert d.status is ParityStatus.MISMATCH and d.replayed == ()
