"""The exchange-steam rule through the replay backtest and the parity check, over lookback
snapshots stored as the writers store them."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from odds_analytics.candidate_backtest import BacktestResult, replay
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_lambda.candidate_rules.pinning import rule_sha256
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.test_candidate_backtest import KICKOFF, N_BOOT, NOW, T, betfair, candidate
from tests.integration.test_candidate_parity import add_decision
from tests.test_helpers import (
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

# Home's exchange fair probability: 0.5 at the decision, 0.4118 three hours earlier.
EXCHANGE_NOW = (2.0, 4.0, 4.0)
EXCHANGE_BEFORE = (2.5, 3.5, 3.5)
RETAIL = (2.6, 3.0, 3.0)
PARAMS = {
    "min_rise": 0.05,
    "gap": 0,
    "lookbacks": [{"minutes_before_decision": 180, "max_input_age_minutes": 30}],
    "minutes_before_kickoff": [60],
    "max_input_age_minutes": 10,
}


def steam_candidate(rule: str = "exchange_steam", **params: Any) -> Any:
    return candidate(rule=rule, rule_sha256=rule_sha256(rule), params={**PARAMS, **params})


async def add_snapshot(
    session: AsyncSession, minutes_before_t: int, exchange: tuple[float, float, float]
) -> int:
    snapshot = await store_tagged(
        session,
        make_tagged_snapshot(
            1,
            T - timedelta(minutes=minutes_before_t),
            [three_way_bookmaker("bet365", RETAIL), betfair(exchange)],
            "oddsportal_live",
        ),
    )
    await session.commit()
    return snapshot.id


async def add_event(session: AsyncSession) -> None:
    session.add(make_epl_event(1, KICKOFF, status="final", home_score=2, away_score=1))
    await session.flush()


async def run(session: AsyncSession, rule: str = "exchange_steam", **params: Any) -> BacktestResult:
    return await replay(
        session,
        steam_candidate(rule, **params),
        KICKOFF.date() - timedelta(days=9),
        KICKOFF.date() + timedelta(days=20),
        1,
        now=NOW,
        n_boot=N_BOOT,
    )


class TestReplay:
    @pytest.mark.parametrize("rule", ["exchange_steam", "team_news_lag"])
    async def test_a_rise_over_the_lookback_is_bet_at_the_decision(
        self, pglite_async_session: AsyncSession, rule: str
    ) -> None:
        await add_event(pglite_async_session)
        await add_snapshot(pglite_async_session, 5, EXCHANGE_NOW)
        await add_snapshot(pglite_async_session, 185, EXCHANGE_BEFORE)
        coverage = (await run(pglite_async_session, rule)).coverage
        assert (coverage.with_input, coverage.with_bet, coverage.bets) == (1, 1, 1)

    async def test_a_flat_exchange_is_decided_without_a_bet(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_event(pglite_async_session)
        await add_snapshot(pglite_async_session, 5, EXCHANGE_NOW)
        await add_snapshot(pglite_async_session, 185, EXCHANGE_NOW)
        coverage = (await run(pglite_async_session)).coverage
        assert (coverage.with_input, coverage.bets) == (1, 0)

    async def test_a_rise_below_min_rise_is_not_bet(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_event(pglite_async_session)
        await add_snapshot(pglite_async_session, 5, EXCHANGE_NOW)
        await add_snapshot(pglite_async_session, 185, EXCHANGE_BEFORE)
        assert (await run(pglite_async_session, min_rise=0.1)).coverage.bets == 0

    async def test_a_decision_without_the_lookback_has_no_input(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_event(pglite_async_session)
        await add_snapshot(pglite_async_session, 5, EXCHANGE_NOW)
        coverage = (await run(pglite_async_session)).coverage
        assert (coverage.decisions, coverage.with_input, coverage.bets) == (1, 0, 0)

    async def test_a_lookback_without_an_exchange_price_has_no_input(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await add_event(pglite_async_session)
        await add_snapshot(pglite_async_session, 5, EXCHANGE_NOW)
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                1,
                T - timedelta(minutes=185),
                [three_way_bookmaker("bet365", RETAIL)],
                "oddsportal_live",
            ),
        )
        await pglite_async_session.commit()
        assert (await run(pglite_async_session)).coverage.with_input == 0


class TestParity:
    async def test_a_decision_recording_both_snapshots_replays_to_its_bet(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event(1, KICKOFF))
        now = await add_snapshot(session, 5, EXCHANGE_NOW)
        before = await add_snapshot(session, 185, EXCHANGE_BEFORE)
        await add_decision(session, [now, before], [("home", "bet365", 2.6)])

        report = await check_parity(session, steam_candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]

    async def test_a_decision_without_the_lookback_recorded_replays_no_bets(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event(1, KICKOFF))
        now = await add_snapshot(session, 5, EXCHANGE_NOW)
        await add_snapshot(session, 185, EXCHANGE_BEFORE)
        await add_decision(session, [now], [("home", "bet365", 2.6)])

        [d] = (await check_parity(session, steam_candidate())).decisions

        assert d.status is ParityStatus.MISMATCH and d.replayed == ()
