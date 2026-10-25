"""Unit tests for the unified scheduling decision engine."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.scheduling.decision import (
    CadenceConfig,
    Lead,
    ScheduleDecision,
    decide_backward,
    decide_backward_resilient,
    decide_forward,
    decide_forward_resilient,
)

from tests.test_helpers import add_event

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
        assert decision.next_execution == NOW + timedelta(days=3)

    @pytest.mark.asyncio
    async def test_check_in_beats_a_distant_wake(self) -> None:
        decision = await decide_forward(
            "soccer_epl",
            CADENCE,
            now=NOW,
            lookahead_days=7,
            next_kickoff=NOW + timedelta(days=90),
        )
        assert decision.should_execute is False
        assert _hours_after(NOW, decision.next_execution) == CADENCE.no_game

    @pytest.mark.asyncio
    async def test_a_check_in_ignores_the_overnight_window(self) -> None:
        night = datetime(2026, 6, 2, 20, 0, tzinfo=UTC)
        decision = await decide_forward(
            "soccer_epl",
            CADENCE,
            now=night,
            overnight=(22, 6),
            lookahead_days=7,
            next_kickoff=night + timedelta(days=90),
        )
        assert decision.next_execution == night + timedelta(hours=CADENCE.no_game)

    @pytest.mark.asyncio
    async def test_a_game_exactly_at_the_lookahead_runs(self) -> None:
        decision = await decide_forward(
            "soccer_epl",
            CADENCE,
            now=NOW,
            lookahead_days=7,
            next_kickoff=NOW + timedelta(days=7),
        )
        assert decision.should_execute is True

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
    @pytest.mark.parametrize(
        ("hours_until", "tier", "interval"),
        [
            # Exactly on a boundary: strict ``<`` means the game falls into the
            # NEXT tier up, not the tier whose ``max_hours`` it equals. This is
            # where classify_tier intentionally differs from calculate_tier (``<=``).
            (0.0, FetchTier.CLOSING, 0.5),  # IN_PLAY is only < 0
            (3.0, FetchTier.PREGAME, 3.0),  # not CLOSING (max 3.0)
            (12.0, FetchTier.SHARP, 12.0),  # not PREGAME (max 12.0)
            (24.0, FetchTier.EARLY, 24.0),  # not SHARP (max 24.0)
            (72.0, FetchTier.OPENING, 48.0),  # not EARLY (max 72.0)
        ],
    )
    async def test_tier_boundary_uses_strict_less_than(
        self, hours_until: float, tier: FetchTier, interval: float
    ) -> None:
        decision = await decide_forward(
            "soccer_epl",
            CADENCE,
            now=NOW,
            lookahead_days=14,
            next_kickoff=NOW + timedelta(hours=hours_until),
        )
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
        # Imminent MLB game, distant EPL game.
        await add_event(
            test_session,
            Event(
                id="mlb_soon",
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=NOW + timedelta(hours=1),
                home_team="Yankees",
                away_team="Red Sox",
                status=EventStatus.SCHEDULED,
            ),
        )
        await add_event(
            test_session,
            Event(
                id="epl_far",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=NOW + timedelta(hours=120),
                home_team="Arsenal",
                away_team="Chelsea",
                status=EventStatus.SCHEDULED,
            ),
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


class TestDecideForwardResilient:
    """Soonest-of-many-sports gate with a DB-failure fallback cadence."""

    @pytest.mark.asyncio
    async def test_keys_off_soonest_sport(self, monkeypatch) -> None:
        """The decision names the sport that owns the soonest kickoff."""
        kickoffs = {
            "soccer_epl": NOW + timedelta(hours=48),  # EARLY
            "baseball_mlb": NOW + timedelta(hours=2),  # CLOSING — soonest
        }

        async def fake_kickoff(sport_key: str, *, now: datetime) -> datetime:
            return kickoffs[sport_key]

        monkeypatch.setattr("odds_lambda.scheduling.decision.get_next_kickoff", fake_kickoff)

        decision = await decide_forward_resilient(["soccer_epl", "baseball_mlb"], CADENCE, now=NOW)
        assert decision.tier == FetchTier.CLOSING
        assert "baseball_mlb" in decision.reason
        assert _hours_after(NOW, decision.next_execution) == CADENCE.closing

    @pytest.mark.asyncio
    async def test_db_failure_falls_back(self, monkeypatch) -> None:
        """A kickoff-query failure yields should_execute=True at db_fallback cadence."""

        async def boom(sport_key: str, *, now: datetime) -> datetime:
            raise RuntimeError("DB connection refused")

        monkeypatch.setattr("odds_lambda.scheduling.decision.get_next_kickoff", boom)

        decision = await decide_forward_resilient(["soccer_epl"], CADENCE, now=NOW)
        assert decision.should_execute is True
        assert decision.tier is None
        assert _hours_after(NOW, decision.next_execution) == CADENCE.db_fallback

    @pytest.mark.asyncio
    async def test_an_injected_fixture_source_replaces_the_stored_events(self, monkeypatch) -> None:
        async def stored(sport_key: str, *, now: datetime) -> None:
            return None

        async def published(sport_key: str, *, now: datetime) -> datetime:
            return NOW + timedelta(hours=2)

        monkeypatch.setattr("odds_lambda.scheduling.decision.get_next_kickoff", stored)

        decision = await decide_forward_resilient(
            ["soccer_epl"], CADENCE, now=NOW, next_kickoff_lookup=published
        )

        assert decision.should_execute is True
        assert decision.tier == FetchTier.CLOSING


class TestLeadRuns:
    """``leads`` add a run that long before each kickoff the cadence would step over."""

    LEAD_CADENCE = CadenceConfig(
        closing=0.25,
        pregame=0.5,
        sharp=2.0,
        early=2.0,
        opening=2.0,
        no_game=2.0,
    )
    LEADS = [Lead(timedelta(minutes=5))]

    @staticmethod
    def _kickoffs(monkeypatch: pytest.MonkeyPatch, kickoffs: list[datetime]) -> None:
        async def fake_kickoff(sport_key: str, *, now: datetime) -> datetime | None:
            return min((ko for ko in kickoffs if ko > now), default=None)

        monkeypatch.setattr("odds_lambda.scheduling.decision.get_next_kickoff", fake_kickoff)

    async def _decide(
        self, now: datetime = NOW, leads: list[Lead] | None = None, **kwargs: Any
    ) -> ScheduleDecision:
        return await decide_forward_resilient(
            ["soccer_epl"],
            self.LEAD_CADENCE,
            now=now,
            leads=self.LEADS if leads is None else leads,
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_pulls_next_run_to_lead_before_kickoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kickoff = NOW + timedelta(minutes=12)
        self._kickoffs(monkeypatch, [kickoff])

        decision = await self._decide()

        assert decision.tier == FetchTier.CLOSING
        assert decision.next_execution == kickoff - timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_keeps_cadence_when_it_lands_before_lead_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._kickoffs(monkeypatch, [NOW + timedelta(minutes=50)])

        decision = await self._decide()

        assert decision.next_execution == NOW + timedelta(minutes=15)

    @pytest.mark.asyncio
    async def test_lead_run_moves_to_next_kickoff_once_inside_lead(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At the lead run itself, the next lead run belongs to a later kickoff."""
        first = NOW + timedelta(minutes=5)
        second = NOW + timedelta(minutes=15)
        self._kickoffs(monkeypatch, [first, second])

        decision = await self._decide()

        assert decision.next_execution == second - timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_lead_run_within_a_minute_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run that started just before its lead run does not book a second one seconds later."""
        self._kickoffs(monkeypatch, [NOW + timedelta(minutes=5, seconds=30)])

        decision = await self._decide()

        assert decision.next_execution == NOW + timedelta(minutes=15)

    @pytest.mark.asyncio
    async def test_earliest_of_several_leads_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A decision lead of 65 minutes runs before the 5-minute close run of the same kickoff."""
        kickoff = NOW + timedelta(minutes=70)
        self._kickoffs(monkeypatch, [kickoff])

        decision = await self._decide(
            leads=[Lead(timedelta(minutes=5)), Lead(timedelta(minutes=65))]
        )

        assert decision.next_execution == kickoff - timedelta(minutes=65)

    @pytest.mark.asyncio
    async def test_lead_run_overrides_overnight_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = datetime(2026, 6, 2, 21, 50, tzinfo=UTC)
        kickoff = datetime(2026, 6, 2, 22, 30, tzinfo=UTC)
        self._kickoffs(monkeypatch, [kickoff])

        decision = await self._decide(now=now, overnight=(22, 6))

        assert decision.next_execution == kickoff - timedelta(minutes=5)

    @pytest.mark.asyncio
    async def test_a_lead_kickoff_lookup_replaces_the_stored_kickoffs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kickoff = NOW + timedelta(minutes=50)
        betfair_start = NOW + timedelta(minutes=12)
        self._kickoffs(monkeypatch, [kickoff])

        async def earlier(sport_key: str, *, now: datetime) -> datetime | None:
            return betfair_start if betfair_start > now else None

        decision = await self._decide(lead_kickoff_lookup=earlier)

        assert decision.next_execution == betfair_start - timedelta(minutes=5)

    DECISION_LEAD = Lead(timedelta(minutes=65), until=timedelta(minutes=60))

    @pytest.mark.asyncio
    async def test_a_run_that_started_before_the_lead_time_does_not_cancel_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kickoff = NOW + timedelta(minutes=66)
        self._kickoffs(monkeypatch, [kickoff])

        decision = await self._decide(
            leads=[self.DECISION_LEAD], run_started=kickoff - timedelta(minutes=67, seconds=30)
        )

        assert decision.next_execution == kickoff - timedelta(minutes=65)

    @pytest.mark.asyncio
    async def test_a_passed_lead_time_runs_now_until_its_deadline(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kickoff = NOW + timedelta(minutes=64)
        self._kickoffs(monkeypatch, [kickoff])
        started = kickoff - timedelta(minutes=67, seconds=30)

        decision = await self._decide(leads=[self.DECISION_LEAD], run_started=started)

        assert decision.next_execution == NOW

    @pytest.mark.asyncio
    async def test_a_lead_time_past_its_deadline_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kickoff = NOW + timedelta(minutes=59)
        self._kickoffs(monkeypatch, [kickoff])
        started = kickoff - timedelta(minutes=67, seconds=30)

        decision = await self._decide(leads=[self.DECISION_LEAD], run_started=started)

        assert decision.next_execution == NOW + timedelta(minutes=15)

    @pytest.mark.asyncio
    async def test_a_run_that_started_at_the_lead_time_serves_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        kickoff = NOW + timedelta(minutes=64)
        self._kickoffs(monkeypatch, [kickoff])
        started = kickoff - timedelta(minutes=65, seconds=30)

        decision = await self._decide(leads=[self.DECISION_LEAD], run_started=started)

        assert decision.next_execution == NOW + timedelta(minutes=15)

    @pytest.mark.asyncio
    async def test_without_leads_cadence_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._kickoffs(monkeypatch, [NOW + timedelta(minutes=12)])

        decision = await self._decide(leads=[])

        assert _hours_after(NOW, decision.next_execution) == self.LEAD_CADENCE.closing


class TestDecideBackward:
    """Recent-games gate for scores / status jobs."""

    @pytest.mark.asyncio
    async def test_runs_when_non_final_game_in_window(
        self, test_session, mock_session_factory
    ) -> None:
        await add_event(
            test_session,
            Event(
                id="recent_live",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=NOW - timedelta(hours=1),
                home_team="A",
                away_team="B",
                status=EventStatus.LIVE,
            ),
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
        await add_event(
            test_session,
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
            ),
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
        await add_event(
            test_session,
            Event(
                id="already_live",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=NOW - timedelta(hours=1),
                home_team="A",
                away_team="B",
                status=EventStatus.LIVE,
            ),
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
        await add_event(
            test_session,
            Event(
                id="mlb_live",
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=NOW - timedelta(hours=1),
                home_team="A",
                away_team="B",
                status=EventStatus.LIVE,
            ),
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


class TestDecideBackwardResilient:
    """Any-sport recent-games gate with a DB-failure fallback cadence."""

    @pytest.mark.asyncio
    async def test_runs_when_any_sport_needs_update(self, monkeypatch) -> None:
        """Returns the first should_execute=True across sports."""

        async def fake_backward(sport_key: str, **_kw) -> ScheduleDecision:
            execute = sport_key == "baseball_mlb"
            return ScheduleDecision(
                should_execute=execute,
                reason=f"{sport_key}",
                next_execution=NOW + timedelta(hours=6 if execute else 12),
                tier=None,
            )

        monkeypatch.setattr("odds_lambda.scheduling.decision.decide_backward", fake_backward)

        decision = await decide_backward_resilient(
            ["soccer_epl", "baseball_mlb"],
            window=timedelta(days=3),
            active_interval=6.0,
            idle_interval=12.0,
            now=NOW,
        )
        assert decision.should_execute is True
        assert "baseball_mlb" in decision.reason

    @pytest.mark.asyncio
    async def test_db_failure_falls_back(self, monkeypatch) -> None:
        """A recent-games query failure yields should_execute=True at db_fallback."""

        async def boom(sport_key: str, **_kw) -> ScheduleDecision:
            raise RuntimeError("DB connection refused")

        monkeypatch.setattr("odds_lambda.scheduling.decision.decide_backward", boom)

        decision = await decide_backward_resilient(
            ["soccer_epl"],
            window=timedelta(days=3),
            active_interval=6.0,
            idle_interval=12.0,
            db_fallback=1.0,
            now=NOW,
        )
        assert decision.should_execute is True
        assert decision.tier is None
        assert _hours_after(NOW, decision.next_execution) == 1.0
