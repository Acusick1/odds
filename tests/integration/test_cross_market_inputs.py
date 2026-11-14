"""A rule that declares an extra market: the loader, and the replay and parity over both."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import replace
from datetime import timedelta

import pytest
from odds_analytics.candidate_backtest import replay
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS
from odds_core.vocabulary import Market
from odds_lambda.candidate_rules import MarketState, SharpInput, load_market_state
from odds_lambda.candidate_rules.registry import RULES
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import CrossMarketRun
from tests.integration.test_candidate_backtest import KICKOFF, N_BOOT, NOW, T, candidate
from tests.integration.test_candidate_parity import add_decision
from tests.test_helpers import make_epl_event

pytestmark = pytest.mark.integration

MAX_AGE = timedelta(minutes=10)
EXCHANGE = frozenset({SharpInput.EXCHANGE})
TOTALS = frozenset({Market.TOTALS})


@pytest.fixture
def declares_totals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(RULES, "exchange_gap", replace(RULES["exchange_gap"], also_markets=TOTALS))


async def load(
    session: AsyncSession,
    also: frozenset[Market] = TOTALS,
    *,
    snapshot_ids: Collection[int] | None = None,
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
        also_markets=also,
        snapshot_ids=snapshot_ids,
    )


class TestLoader:
    async def test_both_markets_come_from_the_same_run(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        one_x_two, totals = await add_cross_market_run(5)
        state = await load(pglite_async_session)
        assert state.input_snapshot_ids == [one_x_two, totals]
        assert state.also[Market.TOTALS].exchange[2.5].fair["over"] == pytest.approx(0.5)

    async def test_the_extra_market_must_be_in_the_primarys_run(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        older, older_totals = await add_cross_market_run(8)
        newer, newer_totals = await add_cross_market_run(2, totals=False)
        state = await load(pglite_async_session)
        assert newer_totals is None
        assert state.input_snapshot_ids == [older, older_totals]
        assert newer not in state.input_snapshot_ids

    async def test_a_missing_extra_market_gives_no_input(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        await add_cross_market_run(5, totals=False)
        assert not (await load(pglite_async_session)).has_input
        assert (await load(pglite_async_session, also=frozenset())).has_input

    async def test_recorded_ids_replay_both_markets(
        self, pglite_async_session: AsyncSession, add_cross_market_run: CrossMarketRun
    ) -> None:
        one_x_two, totals = await add_cross_market_run(5)
        both = await load(pglite_async_session, snapshot_ids=[one_x_two, totals])
        assert both.input_snapshot_ids == [one_x_two, totals]
        assert not (await load(pglite_async_session, snapshot_ids=[one_x_two])).has_input


class TestBacktestAndParity:
    async def test_the_replay_decides_through_the_same_loader(
        self,
        pglite_async_session: AsyncSession,
        add_cross_market_run: CrossMarketRun,
        declares_totals: None,
    ) -> None:
        pglite_async_session.add(
            make_epl_event(1, KICKOFF, status="final", home_score=2, away_score=1)
        )
        await pglite_async_session.flush()
        await add_cross_market_run(5)
        result = await replay(
            pglite_async_session,
            candidate(),
            KICKOFF.date() - timedelta(days=9),
            KICKOFF.date() + timedelta(days=20),
            1,
            now=NOW,
            n_boot=N_BOOT,
        )
        assert (result.coverage.decisions, result.coverage.with_input) == (1, 1)

    async def test_parity_matches_when_both_inputs_are_recorded(
        self,
        pglite_async_session: AsyncSession,
        add_cross_market_run: CrossMarketRun,
        declares_totals: None,
    ) -> None:
        one_x_two, totals = await add_cross_market_run(5)
        await add_decision(pglite_async_session, [one_x_two, totals], [("home", "bet365", 2.2)])

        report = await check_parity(pglite_async_session, candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]

    async def test_parity_checks_the_extra_markets_recorded_input(
        self,
        pglite_async_session: AsyncSession,
        add_cross_market_run: CrossMarketRun,
        declares_totals: None,
    ) -> None:
        one_x_two, totals = await add_cross_market_run(5)
        await add_decision(
            pglite_async_session, [one_x_two, totals + 999], [("home", "bet365", 2.2)]
        )

        [d] = (await check_parity(pglite_async_session, candidate())).decisions

        assert d.status is ParityStatus.MISMATCH and d.missing_snapshot_ids == (totals + 999,)

    async def test_parity_without_the_recorded_extra_input_replays_no_bets(
        self,
        pglite_async_session: AsyncSession,
        add_cross_market_run: CrossMarketRun,
        declares_totals: None,
    ) -> None:
        one_x_two, _ = await add_cross_market_run(5)
        await add_decision(pglite_async_session, [one_x_two], [("home", "bet365", 2.2)])

        [d] = (await check_parity(pglite_async_session, candidate())).decisions

        assert d.status is ParityStatus.MISMATCH and d.replayed == ()
