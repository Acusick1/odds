"""The replay backtest: as-of-t state, final kick-off, twins, coverage and shared scoring."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from odds_analytics.candidate_backtest import HashStatus, replay
from odds_analytics.candidate_report import candidate_report, load_decision_starts
from odds_core.candidates import MechanicalCandidate
from odds_core.event_source import canonical_event_id
from odds_core.models import EventStatus
from odds_core.paper_trade_models import CandidateDecision, PaperTrade, TradeResult
from odds_lambda.candidate_rules import rule_sha256

from tests.test_helpers import (
    API_EVENT_ID,
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    make_epl_event,
    make_tagged_snapshot,
    three_way_bookmaker,
)

pytestmark = pytest.mark.integration

N_BOOT = 200
NOW = datetime(2026, 9, 1, tzinfo=UTC)
KICKOFF = datetime(2026, 4, 10, 15, 0, tzinfo=UTC)
# 60 minutes before kick-off.
T = KICKOFF - timedelta(minutes=60)
# Exchange fair (0.5, 0.25, 0.25) against a retail home price of 2.2: EV +10%.
EXCHANGE = (2.0, 4.0, 4.0)
RETAIL = (2.2, 3.0, 3.0)


def candidate(**overrides: Any) -> MechanicalCandidate:
    fields: dict[str, Any] = {
        "name": "gap",
        "description": "d",
        "kind": "mechanical",
        "sport": "soccer_epl",
        "version": 1,
        "issue": 626,
        "start_date": "2026-01-01",
        "status": "active",
        "rule": "exchange_gap",
        "rule_sha256": rule_sha256("exchange_gap"),
        "params": {"threshold": 0.02, "minutes_before_kickoff": 60},
        **overrides,
    }
    return MechanicalCandidate.model_validate(fields)


def betfair(prices: tuple[float, float, float]) -> dict[str, Any]:
    return three_way_bookmaker(
        "betfair_exchange", prices, betfair_meta={"inplay": False, "market_status": "OPEN"}
    )


async def add_event(
    session: Any,
    event_id: str,
    kickoff: datetime = KICKOFF,
    *,
    retail: tuple[float, float, float] | None = RETAIL,
    exchange: tuple[float, float, float] | None = EXCHANGE,
    close: tuple[float, float, float] | None = EXCHANGE,
    snapshot_offset: timedelta = timedelta(0),
    scores: tuple[int, int] | None = (2, 1),
    **fields: Any,
) -> None:
    """An event with retail and exchange snapshots 10 and 5 minutes before the decision."""
    t = kickoff - timedelta(minutes=60)
    event = make_epl_event(
        event_id,
        kickoff,
        status=EventStatus.FINAL if scores else EventStatus.SCHEDULED,
        home_score=scores[0] if scores else None,
        away_score=scores[1] if scores else None,
        **fields,
    )
    session.add(event)
    if retail is not None:
        session.add(
            make_tagged_snapshot(
                event_id,
                t - timedelta(minutes=10) + snapshot_offset,
                [three_way_bookmaker("bet365", retail)],
                "oddsportal_live",
            )
        )
    if exchange is not None:
        session.add(
            make_tagged_snapshot(
                event_id,
                t - timedelta(minutes=5) + snapshot_offset,
                [betfair(exchange)],
                "betfair_api",
            )
        )
    if close is not None:
        session.add(
            make_tagged_snapshot(
                event_id, kickoff - timedelta(minutes=2), [betfair(close)], "betfair_api"
            )
        )
    await session.commit()


async def run(session: Any, cand: MechanicalCandidate | None = None, **kwargs: Any):
    return await replay(
        session,
        cand or candidate(),
        date(2026, 4, 1),
        date(2026, 4, 30),
        1,
        now=NOW,
        n_boot=N_BOOT,
        **kwargs,
    )


class TestReplay:
    async def test_bets_are_scored_at_the_price_taken_against_the_close(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session, "e1", close=(2.5, 3.6, 3.6))

        result = await run(pglite_async_session)

        [stats] = [result.summary]
        assert stats.settled_bets == 1 and stats.bets_with_close == 1
        # Home won 2-1 at 2.2 for a unit stake.
        assert stats.pnl == pytest.approx(1.2)

    async def test_only_snapshots_stamped_at_or_before_the_decision_time_count(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session, "late", snapshot_offset=timedelta(minutes=20))

        result = await run(pglite_async_session)

        assert (result.coverage.with_retail, result.coverage.bets) == (0, 0)

    async def test_decision_time_uses_the_events_final_kickoff(self, pglite_async_session) -> None:
        # A snapshot 30 minutes before the 15:00 kick-off is invisible at 14:00 but would be
        # visible had the event been decided 60 minutes before 15:30.
        await add_event(pglite_async_session, "moved", snapshot_offset=timedelta(minutes=30))
        later = candidate(params={"threshold": 0.02, "minutes_before_kickoff": 30})

        assert (await run(pglite_async_session, later)).coverage.bets == 1
        assert (await run(pglite_async_session)).coverage.bets == 0

    async def test_twins_of_one_match_count_once(self, pglite_async_session) -> None:
        kickoff = TWIN_KICKOFF
        await add_event(pglite_async_session, TWIN_EVENT_ID, kickoff)
        await add_event(pglite_async_session, API_EVENT_ID, kickoff + timedelta(minutes=30))
        canonical = canonical_event_id([TWIN_EVENT_ID, API_EVENT_ID])
        other = API_EVENT_ID if canonical == TWIN_EVENT_ID else TWIN_EVENT_ID

        result = await run(pglite_async_session)

        assert result.coverage.events == 1
        assert canonical != other

    async def test_events_outside_the_range_or_not_final_are_excluded(
        self, pglite_async_session
    ) -> None:
        await add_event(pglite_async_session, "before", datetime(2026, 3, 31, 23, 0, tzinfo=UTC))
        await add_event(pglite_async_session, "last-day", datetime(2026, 4, 30, 23, 0, tzinfo=UTC))
        await add_event(pglite_async_session, "scheduled", scores=None)

        result = await run(pglite_async_session)

        assert result.coverage.events == 1

    async def test_coverage_counts_each_stage(self, pglite_async_session) -> None:
        day = timedelta(days=1)
        await add_event(pglite_async_session, "bet", KICKOFF)
        await add_event(pglite_async_session, "no-exchange", KICKOFF + day, exchange=None)
        await add_event(pglite_async_session, "no-edge", KICKOFF + 2 * day, retail=(1.9, 3.0, 3.0))
        await add_event(
            pglite_async_session, "nothing", KICKOFF + 3 * day, retail=None, exchange=None
        )

        coverage = (await run(pglite_async_session)).coverage

        assert (
            coverage.events,
            coverage.with_exchange,
            coverage.with_retail,
            coverage.with_bet,
            coverage.bets,
            coverage.bets_with_close,
        ) == (4, 2, 3, 1, 1, 1)

    async def test_empty_range_has_no_verdict(self, pglite_async_session) -> None:
        result = await run(pglite_async_session)

        assert result.coverage.events == 0
        assert result.overall is None
        assert result.summary.verdict == "not_yet"


class TestHashStatus:
    async def test_reports_whether_the_recorded_hash_matches(self, pglite_async_session) -> None:
        assert (await run(pglite_async_session)).hash_status is HashStatus.MATCH
        stale = candidate(rule_sha256="0" * 64)
        assert (await run(pglite_async_session, stale)).hash_status is HashStatus.MISMATCH
        draft = candidate(rule_sha256=None)
        assert (await run(pglite_async_session, draft)).hash_status is HashStatus.ABSENT


class TestSharedScoring:
    async def test_same_bets_give_the_same_verdict_as_the_forward_report(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        trades = []
        for i in range(100):
            kickoff = KICKOFF + timedelta(hours=3 * i)  # beyond the twin window
            close = (2.0, 4.0, 4.0) if i % 2 else (2.4, 3.6, 3.6)
            await add_event(session, f"e{i}", kickoff, close=close)
            trades.append(
                PaperTrade(
                    event_id=f"e{i}",
                    market="1x2",
                    selection="home",
                    bookmaker="bet365",
                    price_decimal=2.2,
                    stake=1.0,
                    candidate_name="gap",
                    candidate_version=1,
                    placed_at=kickoff - timedelta(minutes=60),
                    settled_at=kickoff,
                    result=TradeResult.WIN,
                    pnl=1.2,
                )
            )
        session.add_all(trades)
        await session.commit()

        result = await run(session)
        [forward] = await candidate_report(session, [], N_BOOT, now=NOW)

        assert result.summary.bets_with_close == 100
        assert result.summary.checkpoint == forward.checkpoint
        assert result.summary.verdict == forward.verdict
        assert result.summary.pnl == pytest.approx(forward.pnl)


class TestFamilySize:
    async def test_report_starts_earlier_versions_at_their_first_decision(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        await add_event(session, "e1", KICKOFF)
        await add_event(session, "e2", KICKOFF + timedelta(days=1))
        for version, event_id, when in (
            (1, "e1", KICKOFF),
            (1, "e2", KICKOFF + timedelta(days=1)),
            (2, "e2", KICKOFF + timedelta(days=1)),
        ):
            session.add(
                CandidateDecision(
                    candidate_name="gap",
                    candidate_version=version,
                    event_id=event_id,
                    decision_time=when,
                    kickoff_at_decision=when,
                    known_at=when,
                )
            )
        await session.commit()

        starts = await load_decision_starts(session)
        assert starts == {("gap", 1): KICKOFF, ("gap", 2): KICKOFF + timedelta(days=1)}

        registry = [candidate(version=3), candidate(name="sibling")]
        summaries = await candidate_report(session, registry, N_BOOT, now=NOW)
        assert {(s.name, s.version): s.family_versions for s in summaries} == {
            ("gap", 3): 4,
            ("sibling", 1): 4,
        }
