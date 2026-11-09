"""The candidate page's readers: the bet ledger and the decision history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import CandidateDecision, PaperTrade, TradeResult
from odds_web.decisions import DecisionKind, load_decision_history
from odds_web.ledger import load_ledger_page
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    totals_book,
)

KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
NOW = KICKOFF + timedelta(days=1)


def paper_trade(event_id: int, minutes: int, **fields: object) -> PaperTrade:
    defaults: dict[str, object] = {
        "event_id": event_id,
        "market": "1x2",
        "selection": "home",
        "bookmaker": "bet365",
        "price_decimal": 2.2,
        "stake": 1.0,
        "candidate_name": "gap",
        "candidate_version": 1,
        "placed_at": KICKOFF - timedelta(days=1) + timedelta(minutes=minutes),
        "settled_at": KICKOFF + timedelta(hours=3),
        "result": TradeResult.WIN,
        "pnl": 1.2,
    }
    return PaperTrade(**{**defaults, **fields})  # type: ignore[arg-type]


async def add_event_with_close(session: AsyncSession, event_id: int = 1) -> Event:
    event = make_epl_event(event_id, KICKOFF, status=EventStatus.FINAL)
    session.add(event)
    await session.flush()
    await store_tagged(
        session,
        make_tagged_snapshot(
            event_id,
            KICKOFF - timedelta(minutes=10),
            [betfair_exchange_bookmaker((2.0, 4.0, 4.0))],
            "betfair_api",
        ),
    )
    return event


class TestLedger:
    async def test_void_trades_are_listed_and_labelled(self, test_session: AsyncSession) -> None:
        await add_event_with_close(test_session)
        test_session.add_all(
            [
                paper_trade(1, 1),
                paper_trade(1, 2, selection="draw", result=TradeResult.VOID, pnl=0.0),
                paper_trade(1, 3, selection="away", result=None, pnl=None, settled_at=None),
            ]
        )
        await test_session.flush()

        ledger = await load_ledger_page(test_session, "gap", 1)

        assert ledger.total == 3
        by_selection = {row.selection: row for row in ledger.rows}
        assert by_selection["draw"].is_void and by_selection["draw"].result is TradeResult.VOID
        assert not by_selection["home"].is_void
        assert by_selection["away"].result is None

    async def test_each_row_carries_its_identity_and_close(
        self, test_session: AsyncSession
    ) -> None:
        await add_event_with_close(test_session)
        test_session.add(paper_trade(1, 1))
        await test_session.flush()

        [row] = (await load_ledger_page(test_session, "gap", 1)).rows

        assert row.trade_id is not None
        assert row.event == "Arsenal v Chelsea"
        assert (row.selection, row.line, row.bookmaker) == ("home", None, "bet365")
        assert (row.price, row.stake, row.pnl) == (2.2, 1.0, 1.2)
        assert row.p_close == pytest.approx(0.5)
        assert row.ev == pytest.approx(0.5 * 2.2 - 1)

    async def test_totals_trades_take_the_close_of_their_own_line(
        self, test_session: AsyncSession
    ) -> None:
        event = await add_event_with_close(test_session)
        await store_tagged(
            test_session,
            make_tagged_snapshot(
                event.id,
                KICKOFF - timedelta(minutes=10),
                [totals_book("betfair_exchange", {2.5: (2.1, 1.9)}, betfair_runners=True)],
                "betfair_api",
            ),
        )
        test_session.add_all(
            [
                paper_trade(1, 1, market="totals", selection="over", line=2.5, price_decimal=2.3),
                paper_trade(1, 2, market="totals", selection="under", line=3.5, price_decimal=1.6),
            ]
        )
        await test_session.flush()

        ledger = await load_ledger_page(test_session, "gap", 1)

        by_line = {row.line: row for row in ledger.rows}
        assert by_line[2.5].p_close == pytest.approx((1 / 2.1) / (1 / 2.1 + 1 / 1.9))
        assert by_line[3.5].p_close is None and by_line[3.5].ev is None

    async def test_other_versions_and_non_candidate_trades_are_not_listed(
        self, test_session: AsyncSession
    ) -> None:
        await add_event_with_close(test_session)
        test_session.add_all(
            [
                paper_trade(1, 1),
                paper_trade(1, 2, candidate_version=2),
                paper_trade(1, 3, candidate_name="other"),
                paper_trade(
                    1, 4, candidate_name=None, candidate_version=None, bankroll_before=100.0
                ),
            ]
        )
        await test_session.flush()

        assert (await load_ledger_page(test_session, "gap", 1)).total == 1

    async def test_pages_run_newest_first_and_clamp(self, test_session: AsyncSession) -> None:
        await add_event_with_close(test_session)
        test_session.add_all(
            [paper_trade(1, minutes, bookmaker=f"b{minutes}") for minutes in range(5)]
        )
        await test_session.flush()

        first = await load_ledger_page(test_session, "gap", 1, page=1, page_size=2)
        last = await load_ledger_page(test_session, "gap", 1, page=3, page_size=2)
        beyond = await load_ledger_page(test_session, "gap", 1, page=9, page_size=2)

        assert [r.bookmaker for r in first.rows] == ["b4", "b3"]
        assert [r.bookmaker for r in last.rows] == ["b0"]
        assert (first.total, first.pages, last.page, beyond.page) == (5, 3, 3, 3)

    async def test_a_version_without_trades_has_one_empty_page(
        self, test_session: AsyncSession
    ) -> None:
        ledger = await load_ledger_page(test_session, "gap", 1)

        assert (ledger.rows, ledger.total, ledger.pages, ledger.page) == ([], 0, 1, 1)


def decision(event_id: int, days_ago: float, **fields: object) -> CandidateDecision:
    when = NOW - timedelta(days=days_ago)
    defaults: dict[str, object] = {
        "candidate_name": "gap",
        "candidate_version": 1,
        "event_id": event_id,
        "minutes_before_kickoff": 60,
        "decision_time": when,
        "kickoff_at_decision": when + timedelta(hours=1),
        "known_at": when,
    }
    return CandidateDecision(**{**defaults, **fields})  # type: ignore[arg-type]


class TestDecisionHistory:
    async def test_decisions_are_bet_no_bet_or_missed(self, test_session: AsyncSession) -> None:
        for event_id in (1, 2, 3):
            test_session.add(make_epl_event(event_id, KICKOFF))
        await test_session.flush()
        test_session.add_all(
            [
                decision(1, 2, bets_placed=1),
                decision(2, 2),
                decision(3, 2, missed=True),
            ]
        )
        await test_session.flush()

        history = await load_decision_history(test_session, "gap", 1, NOW)

        kinds = [d.kind for day in history.days for d in day.decisions]
        assert kinds == [DecisionKind.BET, DecisionKind.NO_BET, DecisionKind.MISSED]
        assert (history.total, history.missed) == (3, 1)
        assert history.missed_share == pytest.approx(1 / 3)

    async def test_the_counts_cover_every_decision_and_the_days_only_the_window(
        self, test_session: AsyncSession
    ) -> None:
        for event_id in (1, 2):
            test_session.add(make_epl_event(event_id, KICKOFF))
        await test_session.flush()
        test_session.add_all([decision(1, 2), decision(2, 40, missed=True)])
        await test_session.flush()

        history = await load_decision_history(test_session, "gap", 1, NOW)

        assert (history.total, history.missed) == (2, 1)
        assert sum(len(day.decisions) for day in history.days) == 1
        assert len(history.days) == 14 and history.days[-1].day == NOW.date()

    async def test_other_versions_are_not_counted(self, test_session: AsyncSession) -> None:
        test_session.add(make_epl_event(1, KICKOFF))
        await test_session.flush()
        test_session.add_all(
            [decision(1, 2), decision(1, 2, candidate_version=2, minutes_before_kickoff=30)]
        )
        await test_session.flush()

        assert (await load_decision_history(test_session, "gap", 1, NOW)).total == 1

    async def test_no_decisions_has_no_missed_share(self, test_session: AsyncSession) -> None:
        history = await load_decision_history(test_session, "gap", 1, NOW)

        assert (history.total, history.missed, history.missed_share) == (0, 0, None)
