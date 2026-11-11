"""The crowded-kick-off and last-result rules decided from stored events: the replay, the live
runner and the parity check read the same fixture context, across a pooled candidate's leagues."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from odds_analytics.candidate_backtest import BacktestResult, replay
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.candidates import MechanicalCandidate
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_lambda.candidate_rules.pinning import rule_sha256
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import AddMatch
from tests.integration.test_candidate_backtest import (
    KICKOFF,
    N_BOOT,
    add_event,
    candidate,
)
from tests.integration.test_candidate_backtest import NOW as REPLAY_NOW
from tests.integration.test_candidate_end_to_end import NOW, decide, seed

pytestmark = pytest.mark.integration

BASE: dict[str, Any] = {"minutes_before_kickoff": [60], "max_input_age_minutes": 10}
# Exchange fair probabilities of about (0.57, 0.24, 0.19) against the retail home price 2.2.
HEAVY = (1.7, 4.0, 5.0)
LA_LIGA = "soccer_spain_la_liga"
POOLED = ["soccer_epl", LA_LIGA]
SIX_DAYS_BEFORE = KICKOFF - timedelta(days=6)
# A pooled replay needs a kick-off before the five leagues' sealed holdout windows.
DEV_KICKOFF = datetime(2025, 4, 12, 15, 0, tzinfo=UTC)


def crowded(sports: list[str], at_least: int) -> MechanicalCandidate:
    params = {"min_concurrent_kickoffs": at_least, "threshold": 0.0, **BASE}
    return candidate(
        rule="crowded_gap",
        rule_sha256=rule_sha256("crowded_gap"),
        sports=sports,
        params=params,
    )


def last_result(min_margin: int = 2) -> MechanicalCandidate:
    return candidate(
        rule="last_result_overreaction",
        rule_sha256=rule_sha256("last_result_overreaction"),
        params={"min_margin": min_margin, **BASE},
    )


def favourite(min_probability: float = 0.55) -> MechanicalCandidate:
    return candidate(
        rule="favourite_price",
        rule_sha256=rule_sha256("favourite_price"),
        params={"min_probability": min_probability, "threshold": 0.0, **BASE},
    )


async def replay_event_one(
    session: AsyncSession,
    cand: MechanicalCandidate,
    kickoff: datetime = KICKOFF,
    **event: Any,
) -> BacktestResult:
    await add_event(session, 1, kickoff, **event)
    return await replay(
        session,
        cand,
        kickoff.date() - timedelta(days=9),
        kickoff.date() + timedelta(days=20),
        1,
        now=REPLAY_NOW,
        n_boot=N_BOOT,
        only_events=[1],
    )


async def add_concurrent_matches(add_match: AddMatch, kickoff: datetime = KICKOFF) -> None:
    """A La Liga kick-off at the match's own time and an EPL one 5 minutes later."""
    await add_match(2, kickoff, "Real Madrid", "Girona", sport_key=LA_LIGA)
    await add_match(3, kickoff + timedelta(minutes=5), "Villa", "Everton")


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("odds_lambda.jobs.run_candidates.candidates_dir", lambda: tmp_path)
    monkeypatch.setattr("odds_lambda.paper_trading.candidates_dir", lambda: tmp_path)
    return tmp_path


def register(registry: Path, cand: MechanicalCandidate) -> None:
    (registry / f"{cand.name}.yaml").write_text(yaml.safe_dump(cand.model_dump(mode="json")))


class TestFavouritePriceReplay:
    async def test_the_favourite_is_bet_when_its_gap_is_positive(
        self, pglite_async_session: AsyncSession
    ) -> None:
        result = await replay_event_one(pglite_async_session, favourite(), exchange=HEAVY)

        assert (result.coverage.with_input, result.coverage.bets) == (1, 1)

    async def test_no_outcome_reaches_the_minimum_probability(
        self, pglite_async_session: AsyncSession
    ) -> None:
        result = await replay_event_one(pglite_async_session, favourite(0.6), exchange=HEAVY)

        assert (result.coverage.with_input, result.coverage.bets) == (1, 0)


class TestLastResultReplay:
    async def test_a_result_completed_long_after_the_decision_counts(
        self, pglite_async_session: AsyncSession, add_match: AddMatch
    ) -> None:
        # A backfill stores results long after their matches.
        await add_match(
            2,
            SIX_DAYS_BEFORE,
            "Arsenal",
            "Everton",
            score=(0, 3),
            completed_at=REPLAY_NOW + timedelta(days=30),
        )

        result = await replay_event_one(pglite_async_session, last_result())

        assert (result.coverage.with_input, result.coverage.bets) == (1, 1)
        # Home won 2-1 at the retail price 2.2 for a unit stake.
        assert result.summary.pnl == pytest.approx(1.2)

    async def test_a_smaller_loss_than_the_margin_is_not_bet(
        self, pglite_async_session: AsyncSession, add_match: AddMatch
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(0, 1))

        result = await replay_event_one(pglite_async_session, last_result())

        assert (result.coverage.with_input, result.coverage.bets) == (1, 0)

    async def test_a_first_match_of_the_season_is_decided_without_a_bet(
        self, pglite_async_session: AsyncSession
    ) -> None:
        result = await replay_event_one(pglite_async_session, last_result())

        assert (result.coverage.with_input, result.coverage.bets) == (1, 0)

    async def test_a_previous_match_without_a_result_misses_the_decision(
        self, pglite_async_session: AsyncSession, add_match: AddMatch
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton")

        result = await replay_event_one(pglite_async_session, last_result())

        assert (result.coverage.with_input, result.coverage.bets) == (0, 0)


class TestCrowdedGapReplay:
    async def test_a_pooled_candidate_counts_kickoffs_across_its_leagues(
        self, pglite_async_session: AsyncSession, add_match: AddMatch
    ) -> None:
        await add_concurrent_matches(add_match, DEV_KICKOFF)

        result = await replay_event_one(pglite_async_session, crowded(POOLED, 3), DEV_KICKOFF)

        assert result.coverage.bets == 1

    async def test_a_league_the_candidate_does_not_bet_in_is_not_counted(
        self, pglite_async_session: AsyncSession, add_match: AddMatch
    ) -> None:
        await add_concurrent_matches(add_match, DEV_KICKOFF)

        result = await replay_event_one(
            pglite_async_session, crowded(["soccer_epl"], 3), DEV_KICKOFF
        )

        assert result.coverage.bets == 0

    async def test_the_count_includes_the_match_itself(
        self, pglite_async_session: AsyncSession, add_match: AddMatch
    ) -> None:
        await add_concurrent_matches(add_match, DEV_KICKOFF)

        result = await replay_event_one(pglite_async_session, crowded(POOLED, 4), DEV_KICKOFF)

        assert result.coverage.bets == 0


class TestLiveRunner:
    async def test_a_pooled_candidate_counts_kickoffs_across_its_leagues(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        register(registry, crowded(POOLED, 3))
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await add_concurrent_matches(add_match)
        await pglite_async_session.commit()

        await decide(pglite_async_session)

        [decision] = (
            (
                await pglite_async_session.execute(
                    select(CandidateDecision).where(CandidateDecision.event_id == 1)
                )
            )
            .scalars()
            .all()
        )
        assert not decision.missed and decision.bets_placed == 1
        assert decision.input_fixture == {"concurrent_kickoffs": 3}
        assert decision.input_event_ids == [2, 3]

    async def test_an_epl_only_candidate_does_not_count_the_other_league(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        register(registry, crowded(["soccer_epl"], 3))
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await add_concurrent_matches(add_match)
        await pglite_async_session.commit()

        await decide(pglite_async_session)

        trades = (await pglite_async_session.execute(select(PaperTrade))).scalars().all()
        assert trades == []

    async def test_a_last_result_bet_is_placed_and_matches_in_parity(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        register(registry, last_result())
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(0, 3))
        await pglite_async_session.commit()

        await decide(pglite_async_session)

        [trade] = (await pglite_async_session.execute(select(PaperTrade))).scalars().all()
        assert (trade.selection, trade.bookmaker) == ("home", "bet365")
        report = await check_parity(pglite_async_session, last_result())
        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]

    async def test_a_result_completed_after_the_run_misses_the_decision(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        register(registry, last_result())
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await add_match(
            2,
            SIX_DAYS_BEFORE,
            "Arsenal",
            "Everton",
            score=(0, 3),
            completed_at=NOW + timedelta(days=1),
        )
        await pglite_async_session.commit()

        await decide(pglite_async_session)

        [decision] = (
            (
                await pglite_async_session.execute(
                    select(CandidateDecision).where(CandidateDecision.event_id == 1)
                )
            )
            .scalars()
            .all()
        )
        assert decision.missed and decision.bets_placed == 0

    async def test_a_pooled_crowded_decision_matches_in_parity(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        cand = crowded(POOLED, 3)
        register(registry, cand)
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await add_concurrent_matches(add_match)
        await pglite_async_session.commit()
        await decide(pglite_async_session)

        report = await check_parity(pglite_async_session, cand)

        assert [d.status for d in report.decisions if d.event_id == 1] == [ParityStatus.MATCH]
