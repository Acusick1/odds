"""Counterfactual book replays of a version's recorded decisions."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
from odds_analytics.book_replays import book_replays, replay_variants
from odds_analytics.candidate_parity import check_parity
from odds_analytics.candidate_report import candidate_report, load_scored_trades
from odds_core.bookmakers import BOOK_SETS, BookSet
from odds_core.paper_trade_models import PaperTrade, TradeResult
from sqlalchemy import select

from tests.integration.test_candidate_backtest import (
    EXCHANGE,
    N_BOOT,
    NOW,
    T,
    add_event,
    betfair,
    candidate,
)
from tests.integration.test_candidate_parity import add_decision
from tests.test_helpers import make_tagged_snapshot, store_tagged, three_way_bookmaker

pytestmark = pytest.mark.integration

OWN = BookSet.UK_RETAIL
ODDS_API = BookSet.ODDS_API_UK


async def add_snapshot(
    session: Any, home_prices: dict[str, float], minutes_before: int = 5, exchange: Any = EXCHANGE
) -> int:
    """A snapshot of each book's 1x2 prices (home price given, draw and away 3.0) and the
    exchange, stamped before the decision."""
    snapshot = make_tagged_snapshot(
        1,
        T - timedelta(minutes=minutes_before),
        [
            *(three_way_bookmaker(book, (home, 3.0, 3.0)) for book, home in home_prices.items()),
            betfair(exchange),
        ],
        "oddsportal_live",
    )
    await store_tagged(session, snapshot)
    await session.commit()
    assert snapshot.id is not None
    return snapshot.id


@pytest.fixture
async def midnite_event(pglite_async_session: Any) -> list[int]:
    """A won home bet's event whose snapshot quotes midnite at 2.6 (not in the Odds API's
    region), williamhill at 2.4 and bet365 at 2.2, all against an exchange fair of 0.5."""
    await add_event(pglite_async_session, 1, retail=None, exchange=None)
    return [
        await add_snapshot(
            pglite_async_session, {"midnite": 2.6, "williamhill": 2.4, "bet365": 2.2}
        )
    ]


class TestReplayVariants:
    async def test_the_registered_set_reproduces_the_live_bets(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])
        assert (await check_parity(pglite_async_session, candidate())).ok

        replayed = await replay_variants(pglite_async_session, candidate())

        assert replayed is not None
        assert replayed.decisions == 1
        [bet] = replayed.by_set[OWN]
        assert (bet.bookmaker, bet.selection, bet.price_decimal) == ("midnite", "home", 2.6)
        assert bet.p_close == pytest.approx(0.5) and bet.pnl == pytest.approx(1.6)

    async def test_another_set_bets_only_at_its_own_books(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])

        replayed = await replay_variants(pglite_async_session, candidate())

        assert replayed is not None
        [bet] = replayed.by_set[ODDS_API]
        assert (bet.bookmaker, bet.price_decimal, bet.pnl) == (
            "williamhill",
            2.4,
            pytest.approx(1.4),
        )

    async def test_leaving_a_book_out_takes_the_next_best_price(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])

        replayed = await replay_variants(pglite_async_session, candidate())

        assert replayed is not None
        left_out = replayed.without
        assert [b.bookmaker for b in left_out["midnite"]] == ["williamhill"]
        assert [b.bookmaker for b in left_out["williamhill"]] == ["midnite"]
        assert left_out.keys() == BOOK_SETS[BookSet.UK_RETAIL]
        assert replayed.by_set.keys() == set(BookSet)

    async def test_a_selection_is_bet_once_per_match_in_each_variant(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        later = await add_snapshot(pglite_async_session, {"williamhill": 2.4}, minutes_before=0)
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])
        await add_decision(
            pglite_async_session,
            [later],
            [],
            decision_time=T + timedelta(minutes=1),
            known_at=T + timedelta(minutes=2),
        )

        replayed = await replay_variants(pglite_async_session, candidate())

        assert replayed is not None
        assert len(replayed.by_set[OWN]) == len(replayed.by_set[ODDS_API]) == 1

    async def test_a_missed_decision_is_not_replayed(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, [], missed=True)

        replayed = await replay_variants(pglite_async_session, candidate())

        assert replayed is not None and replayed.decisions == 0
        assert not any(replayed.by_set.values()) and not any(replayed.without.values())


class TestBookReplays:
    async def trades(self, session: Any) -> list[Any]:
        for trade in (await session.scalars(select(PaperTrade))).all():
            trade.result, trade.pnl = TradeResult.WIN, 1.6
        await session.commit()
        return await load_scored_trades(session)

    async def test_scores_each_set_and_the_books_that_matter(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])
        trades = await self.trades(pglite_async_session)

        replays = await book_replays(pglite_async_session, [candidate()], trades, NOW)

        replay = replays[("gap", 1)]
        own, api = replay.sets
        assert (own.books, own.registered) == (BookSet.UK_RETAIL, True)
        assert (own.score.bets, own.score.settled_bets, own.score.bets_with_close) == (1, 1, 1)
        assert own.score.mean_ev == pytest.approx(0.3) and own.score.pnl == pytest.approx(1.6)
        assert own.running_mean_ev == pytest.approx((0.3,))
        assert (api.books, api.registered) == (BookSet.ODDS_API_UK, False)
        assert api.score.mean_ev == pytest.approx(0.2) and api.score.pnl == pytest.approx(1.4)
        [row] = replay.books
        assert row.book == "midnite"
        assert (row.placed.bets, row.placed.pnl) == (1, pytest.approx(1.6))
        assert row.placed.mean_ev == pytest.approx(0.3)
        assert row.without.mean_ev == pytest.approx(0.2) and row.without.bets == 1

    async def test_the_registered_set_is_the_live_running_score(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])
        await self.trades(pglite_async_session)

        report = await candidate_report(
            pglite_async_session, [candidate()], N_BOOT, now=NOW, replays=True
        )

        live = report.running[("gap", 1)].mean_ev
        [own] = [s for s in report.book_replays[("gap", 1)].sets if s.registered]
        assert own.running_mean_ev == pytest.approx(live)

    async def test_a_report_without_replays_holds_none(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])

        report = await candidate_report(pglite_async_session, [candidate()], N_BOOT, now=NOW)

        assert report.book_replays == {}

    async def test_a_version_the_registry_does_not_hold_has_no_replay(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])
        await self.trades(pglite_async_session)
        other = candidate(version=2)

        report = await candidate_report(
            pglite_async_session, [other], N_BOOT, now=NOW, replays=True
        )

        assert list(report.book_replays) == [("gap", 2)]

    async def test_the_replay_changes_no_summary(
        self, pglite_async_session: Any, midnite_event: list[int]
    ) -> None:
        await add_decision(pglite_async_session, midnite_event, [("home", "midnite", 2.6)])
        await self.trades(pglite_async_session)

        plain = await candidate_report(pglite_async_session, [candidate()], N_BOOT, now=NOW)
        replayed = await candidate_report(
            pglite_async_session, [candidate()], N_BOOT, now=NOW, replays=True
        )

        assert plain.summaries == replayed.summaries and plain.running == replayed.running
