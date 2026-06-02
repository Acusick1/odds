"""Unit tests for the unified scheduling decision engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.scheduling.decision import (
    CadenceConfig,
    decide_backward,
    decide_forward,
)
from odds_lambda.storage.writers import OddsWriter

NOW = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)

# Distinct interval per tier so the test can assert which tier was selected.
CADENCE = CadenceConfig(
    closing=0.5,
    pregame=3.0,
    sharp=12.0,
    early=24.0,
    opening=48.0,
    no_game=99.0,
    db_fallback=1.0,
)


def _hours_after(base: datetime, dt: datetime) -> float:
    return (dt - base).total_seconds() / 3600


class TestCadenceConfig:
    """Tier -> interval mapping."""

    def test_interval_per_tier(self) -> None:
        assert CADENCE.interval_for(FetchTier.CLOSING) == 0.5
        assert CADENCE.interval_for(FetchTier.PREGAME) == 3.0
        assert CADENCE.interval_for(FetchTier.SHARP) == 12.0
        assert CADENCE.interval_for(FetchTier.EARLY) == 24.0
        assert CADENCE.interval_for(FetchTier.OPENING) == 48.0

    def test_in_play_shares_closing(self) -> None:
        assert CADENCE.interval_for(FetchTier.IN_PLAY) == CADENCE.closing


class TestDecideForward:
    """Proximity gate + tier classification keyed off the next game."""

    @pytest.mark.asyncio
    async def test_no_game_gates_off(self) -> None:
        decision = await decide_forward("soccer_epl", CADENCE, now=NOW, next_kickoff=None)
        assert decision.should_execute is False
        assert decision.tier is None
        assert _hours_after(NOW, decision.next_execution) == CADENCE.no_game

    @pytest.mark.asyncio
    async def test_game_beyond_lookahead_gates_off(self) -> None:
        decision = await decide_forward(
            "soccer_epl",
            CADENCE,
            now=NOW,
            lookahead_days=7,
            next_kickoff=NOW + timedelta(days=10),
        )
        assert decision.should_execute is False
        assert _hours_after(NOW, decision.next_execution) == CADENCE.no_game

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("hours_until", "tier", "interval"),
        [
            (1.0, FetchTier.CLOSING, 0.5),
            (8.0, FetchTier.PREGAME, 3.0),
            (18.0, FetchTier.SHARP, 12.0),
            (48.0, FetchTier.EARLY, 24.0),
            (120.0, FetchTier.OPENING, 48.0),
        ],
    )
    async def test_tier_selection(
        self, hours_until: float, tier: FetchTier, interval: float
    ) -> None:
        decision = await decide_forward(
            "soccer_epl",
            CADENCE,
            now=NOW,
            lookahead_days=14,
            next_kickoff=NOW + timedelta(hours=hours_until),
        )
        assert decision.should_execute is True
        assert decision.tier == tier
        assert _hours_after(NOW, decision.next_execution) == interval

    @pytest.mark.asyncio
    async def test_overnight_skip_applied(self) -> None:
        # 23:00 UTC + 3h = 02:00, inside EPL window 22-06 -> pushed to 06:00.
        late = datetime(2026, 6, 2, 23, 0, tzinfo=UTC)
        decision = await decide_forward(
            "soccer_epl",
            CADENCE,
            now=late,
            overnight=(22, 6),
            next_kickoff=late + timedelta(hours=8),  # PREGAME -> 3h
        )
        assert decision.next_execution == datetime(2026, 6, 3, 6, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_sport_scoping_queries_own_sport(
        self, test_session, mock_session_factory
    ) -> None:
        """EPL gate must ignore an imminent MLB fixture (the all-sports bug)."""
        writer = OddsWriter(test_session)
        # Imminent MLB game, distant EPL game.
        await writer.upsert_event(
            Event(
                id="mlb_soon",
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=NOW + timedelta(hours=1),
                home_team="Yankees",
                away_team="Red Sox",
                status=EventStatus.SCHEDULED,
            )
        )
        await writer.upsert_event(
            Event(
                id="epl_far",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=NOW + timedelta(hours=120),
                home_team="Arsenal",
                away_team="Chelsea",
                status=EventStatus.SCHEDULED,
            )
        )
        await test_session.commit()

        import unittest.mock

        # get_next_kickoff imports async_session_maker from odds_core.database
        # at call time, so patch it there.
        with unittest.mock.patch("odds_core.database.async_session_maker", mock_session_factory):
            epl = await decide_forward("soccer_epl", CADENCE, now=NOW, lookahead_days=14)
            mlb = await decide_forward("baseball_mlb", CADENCE, now=NOW, lookahead_days=14)

        # EPL keys off its own 120h game (OPENING), not the 1h MLB game.
        assert epl.tier == FetchTier.OPENING
        # MLB keys off its own 1h game (CLOSING).
        assert mlb.tier == FetchTier.CLOSING


class TestDecideBackward:
    """Recent-games gate for scores / status jobs."""

    @pytest.mark.asyncio
    async def test_runs_when_non_final_game_in_window(
        self, test_session, mock_session_factory
    ) -> None:
        writer = OddsWriter(test_session)
        await writer.upsert_event(
            Event(
                id="recent_live",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=NOW - timedelta(hours=1),
                home_team="A",
                away_team="B",
                status=EventStatus.LIVE,
            )
        )
        await test_session.commit()

        decision = await decide_backward(
            "soccer_epl",
            window=timedelta(days=3),
            active_interval=6.0,
            idle_interval=12.0,
            now=NOW,
            session_factory=mock_session_factory,
        )
        assert decision.should_execute is True
        assert _hours_after(NOW, decision.next_execution) == 6.0

    @pytest.mark.asyncio
    async def test_idles_when_all_final(self, test_session, mock_session_factory) -> None:
        writer = OddsWriter(test_session)
        await writer.upsert_event(
            Event(
                id="recent_final",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=NOW - timedelta(hours=1),
                home_team="A",
                away_team="B",
                status=EventStatus.FINAL,
                home_score=1,
                away_score=0,
            )
        )
        await test_session.commit()

        decision = await decide_backward(
            "soccer_epl",
            window=timedelta(days=3),
            active_interval=6.0,
            idle_interval=12.0,
            now=NOW,
            session_factory=mock_session_factory,
        )
        assert decision.should_execute is False
        assert _hours_after(NOW, decision.next_execution) == 12.0

    @pytest.mark.asyncio
    async def test_status_filter_only_counts_scheduled(
        self, test_session, mock_session_factory
    ) -> None:
        """With statuses_needing_update={SCHEDULED}, a LIVE game does not trigger."""
        writer = OddsWriter(test_session)
        await writer.upsert_event(
            Event(
                id="already_live",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=NOW - timedelta(hours=1),
                home_team="A",
                away_team="B",
                status=EventStatus.LIVE,
            )
        )
        await test_session.commit()

        decision = await decide_backward(
            "soccer_epl",
            window=timedelta(hours=4),
            active_interval=1.0,
            idle_interval=6.0,
            statuses_needing_update={EventStatus.SCHEDULED},
            now=NOW,
            session_factory=mock_session_factory,
        )
        assert decision.should_execute is False

    @pytest.mark.asyncio
    async def test_sport_scoped(self, test_session, mock_session_factory) -> None:
        """An MLB game needing updates must not trigger the EPL backward gate."""
        writer = OddsWriter(test_session)
        await writer.upsert_event(
            Event(
                id="mlb_live",
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=NOW - timedelta(hours=1),
                home_team="A",
                away_team="B",
                status=EventStatus.LIVE,
            )
        )
        await test_session.commit()

        decision = await decide_backward(
            "soccer_epl",
            window=timedelta(days=3),
            active_interval=6.0,
            idle_interval=12.0,
            now=NOW,
            session_factory=mock_session_factory,
        )
        assert decision.should_execute is False
