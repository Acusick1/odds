"""The backfill of live candidate trades' estimated EV from replays of their decisions."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from odds_analytics.estimate_backfill import apply_backfill, plan_backfill
from odds_core.paper_trade_models import PaperTrade
from sqlalchemy import select
from sqlmodel import col

from tests.integration.test_candidate_backtest import T, candidate
from tests.integration.test_candidate_parity import add_decision, add_event

pytestmark = pytest.mark.integration

# The exchange's fair home probability 0.5 against the retail home price 2.2.
HOME_ESTIMATE = 0.5 * 2.2 - 1


async def estimates(session: Any) -> list[float | None]:
    found = await session.execute(select(col(PaperTrade.estimated_ev)).order_by(col(PaperTrade.id)))
    return list(found.scalars())


class TestPlan:
    async def test_a_trade_takes_the_estimate_of_its_replayed_bet(
        self, pglite_async_session: Any
    ) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.2)])

        plan = await plan_backfill(pglite_async_session, [candidate()])

        [version] = plan.versions
        [trade] = version.trades
        assert trade.matched and trade.estimated_ev == pytest.approx(HOME_ESTIMATE)
        assert version.to_write == [trade] and version.unmatched == []

    async def test_a_trade_with_no_replayed_bet_is_unmatched(
        self, pglite_async_session: Any
    ) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.3)])

        [version] = (await plan_backfill(pglite_async_session, [candidate()])).versions

        [trade] = version.unmatched
        assert (trade.key, trade.estimated_ev) == (("home", None, "bet365", 2.3), None)
        assert version.to_write == []

    async def test_a_decision_replays_dropping_only_the_selections_traded_before(
        self, pglite_async_session: Any
    ) -> None:
        # The earlier decision placed nothing live, so the later one may bet home again.
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, decision_time=T - timedelta(minutes=30))
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.2)])

        [version] = (await plan_backfill(pglite_async_session, [candidate()])).versions

        [trade] = version.trades
        assert trade.matched and trade.estimated_ev == pytest.approx(HOME_ESTIMATE)

    async def test_an_earlier_version_is_replayed_with_the_registrys_rule(
        self, pglite_async_session: Any
    ) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.2)], version=1)

        [version] = (await plan_backfill(pglite_async_session, [candidate(version=2)])).versions

        assert (version.name, version.version) == ("gap", 1)
        assert [t.matched for t in version.trades] == [True]

    async def test_a_name_the_registry_does_not_hold_is_skipped(
        self, pglite_async_session: Any
    ) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.2)])

        plan = await plan_backfill(pglite_async_session, [candidate(name="other")])

        assert (plan.versions, plan.skipped) == ((), {("gap", 1): 1})


class TestApply:
    async def test_writes_the_matched_estimates_and_a_rerun_writes_nothing(
        self, pglite_async_session: Any
    ) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.2)])

        written = await apply_backfill(
            pglite_async_session, await plan_backfill(pglite_async_session, [candidate()])
        )

        assert written == 1
        assert await estimates(pglite_async_session) == [pytest.approx(HOME_ESTIMATE)]
        again = await plan_backfill(pglite_async_session, [candidate()])
        assert again.versions[0].to_write == []
        assert await apply_backfill(pglite_async_session, again) == 0

    async def test_an_unmatched_trade_stays_without_an_estimate(
        self, pglite_async_session: Any
    ) -> None:
        ids = await add_event(pglite_async_session)
        await add_decision(pglite_async_session, ids, [("home", "bet365", 2.3)])

        await apply_backfill(
            pglite_async_session, await plan_backfill(pglite_async_session, [candidate()])
        )

        assert await estimates(pglite_async_session) == [None]
