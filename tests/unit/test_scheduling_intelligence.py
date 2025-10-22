"""Unit tests for scheduling intelligence."""

from datetime import datetime, timedelta

import pytest

from core.models import Event, EventStatus
from core.scheduling.intelligence import FetchTier, SchedulingIntelligence
from storage.writers import OddsWriter


class TestFetchTier:
    """Tests for FetchTier enum."""

    def test_tier_interval_hours(self):
        """Test that each tier has correct interval."""
        assert FetchTier.CLOSING.interval_hours == 0.5  # 30 minutes
        assert FetchTier.PREGAME.interval_hours == 3.0
        assert FetchTier.SHARP.interval_hours == 12.0
        assert FetchTier.EARLY.interval_hours == 24.0
        assert FetchTier.OPENING.interval_hours == 48.0


class TestSchedulingIntelligence:
    """Tests for SchedulingIntelligence decision logic."""

    @pytest.mark.asyncio
    async def test_no_games_scheduled(self, mock_session_factory):
        """When no games exist, should not execute and check again in 24h."""
        intelligence = SchedulingIntelligence(
            lookahead_days=7, session_factory=mock_session_factory
        )

        decision = await intelligence.should_execute_fetch()

        assert decision.should_execute is False
        assert "No games scheduled" in decision.reason
        assert decision.tier is None
        # Next execution should be ~24 hours from now
        assert decision.next_execution is not None
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 23.9 < hours_until_next < 24.1

    @pytest.mark.asyncio
    async def test_closing_tier_game(self, test_session, mock_session_factory):
        """Game 2 hours away should trigger CLOSING tier (30min interval)."""
        # Create game 2 hours in future
        writer = OddsWriter(test_session)
        commence_time = datetime.utcnow() + timedelta(hours=2)

        event = Event(
            id="closing_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_fetch()

        assert decision.should_execute is True
        assert decision.tier == FetchTier.CLOSING
        assert "closing" in decision.reason.lower()
        # Next execution should be ~30 minutes (0.5 hours)
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 0.45 < hours_until_next < 0.55

    @pytest.mark.asyncio
    async def test_pregame_tier_game(self, test_session, mock_session_factory):
        """Game 8 hours away should trigger PREGAME tier (3h interval)."""
        writer = OddsWriter(test_session)
        commence_time = datetime.utcnow() + timedelta(hours=8)

        event = Event(
            id="pregame_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team="Warriors",
            away_team="Heat",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_fetch()

        assert decision.should_execute is True
        assert decision.tier == FetchTier.PREGAME
        assert "pregame" in decision.reason.lower()
        # Next execution should be ~3 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 2.9 < hours_until_next < 3.1

    @pytest.mark.asyncio
    async def test_sharp_tier_game(self, test_session, mock_session_factory):
        """Game 18 hours away should trigger SHARP tier (12h interval)."""
        writer = OddsWriter(test_session)
        commence_time = datetime.utcnow() + timedelta(hours=18)

        event = Event(
            id="sharp_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team="Bucks",
            away_team="Nets",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_fetch()

        assert decision.should_execute is True
        assert decision.tier == FetchTier.SHARP
        # Next execution should be ~12 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 11.9 < hours_until_next < 12.1

    @pytest.mark.asyncio
    async def test_early_tier_game(self, test_session, mock_session_factory):
        """Game 2 days away should trigger EARLY tier (24h interval)."""
        writer = OddsWriter(test_session)
        commence_time = datetime.utcnow() + timedelta(days=2)

        event = Event(
            id="early_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team="Suns",
            away_team="Mavericks",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_fetch()

        assert decision.should_execute is True
        assert decision.tier == FetchTier.EARLY
        # Next execution should be ~24 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 23.9 < hours_until_next < 24.1

    @pytest.mark.asyncio
    async def test_opening_tier_game(self, test_session, mock_session_factory):
        """Game 5 days away should trigger OPENING tier (48h interval)."""
        writer = OddsWriter(test_session)
        commence_time = datetime.utcnow() + timedelta(days=5)

        event = Event(
            id="opening_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team="76ers",
            away_team="Knicks",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_fetch()

        assert decision.should_execute is True
        assert decision.tier == FetchTier.OPENING
        # Next execution should be ~48 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 47.9 < hours_until_next < 48.1

    @pytest.mark.asyncio
    async def test_game_already_started(self, test_session, mock_session_factory):
        """Game that already started is not found (filtered out by date range query)."""
        writer = OddsWriter(test_session)
        # Game started 1 hour ago
        commence_time = datetime.utcnow() - timedelta(hours=1)

        event = Event(
            id="started_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team="Clippers",
            away_team="Jazz",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_fetch()

        # Game in the past won't be found by get_closest_game (queries from now forward)
        # So it behaves same as no games scheduled
        assert decision.should_execute is False
        assert "No games scheduled" in decision.reason
        assert decision.tier is None
        # Should check again in 24 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 23.9 < hours_until_next < 24.1

    @pytest.mark.asyncio
    async def test_closest_game_selection(self, test_session, mock_session_factory):
        """Should select closest game when multiple exist."""
        writer = OddsWriter(test_session)

        # Create 3 games at different times
        far_game = Event(
            id="far_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow() + timedelta(days=5),
            home_team="Team A",
            away_team="Team B",
            status=EventStatus.SCHEDULED,
        )
        closest_game = Event(
            id="closest_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow() + timedelta(hours=4),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
        )
        middle_game = Event(
            id="middle_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow() + timedelta(days=2),
            home_team="Team C",
            away_team="Team D",
            status=EventStatus.SCHEDULED,
        )

        await writer.upsert_event(far_game)
        await writer.upsert_event(closest_game)
        await writer.upsert_event(middle_game)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_fetch()

        # Should use closest game (4 hours = PREGAME tier)
        assert decision.should_execute is True
        assert decision.tier == FetchTier.PREGAME
        assert "Lakers" in decision.reason or "Celtics" in decision.reason

    @pytest.mark.asyncio
    async def test_should_execute_scores_with_incomplete_games(
        self, test_session, mock_session_factory
    ):
        """Should execute scores fetch when games need updates."""
        writer = OddsWriter(test_session)

        # Create game from yesterday without final status
        yesterday = datetime.utcnow() - timedelta(days=1)
        event = Event(
            id="incomplete_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=yesterday,
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.LIVE,  # Not FINAL
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_scores()

        assert decision.should_execute is True
        assert "need score updates" in decision.reason
        assert decision.tier is None
        # Should check again in 6 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 5.9 < hours_until_next < 6.1

    @pytest.mark.asyncio
    async def test_should_execute_scores_all_final(self, test_session, mock_session_factory):
        """Should not execute scores when all games are final."""
        writer = OddsWriter(test_session)

        # Create game from yesterday with final status
        yesterday = datetime.utcnow() - timedelta(days=1)
        event = Event(
            id="final_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=yesterday,
            home_team="Warriors",
            away_team="Heat",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_scores()

        assert decision.should_execute is False
        assert "No games need score updates" in decision.reason
        # Should check again in 12 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 11.9 < hours_until_next < 12.1

    @pytest.mark.asyncio
    async def test_should_execute_status_update_with_scheduled_games(
        self, test_session, mock_session_factory
    ):
        """Should execute status update when scheduled games may have started."""
        writer = OddsWriter(test_session)

        # Create game that started 2 hours ago but still marked scheduled
        two_hours_ago = datetime.utcnow() - timedelta(hours=2)
        event = Event(
            id="should_be_live",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=two_hours_ago,
            home_team="Bucks",
            away_team="Nets",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_status_update()

        assert decision.should_execute is True
        assert "may have started" in decision.reason
        # Should check again in 1 hour
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 0.9 < hours_until_next < 1.1

    @pytest.mark.asyncio
    async def test_should_execute_status_update_no_games(self, test_session, mock_session_factory):
        """Should not execute status update when no games need updating."""
        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        decision = await intelligence.should_execute_status_update()

        assert decision.should_execute is False
        assert "No games to update" in decision.reason
        # Should check again in 6 hours
        hours_until_next = (decision.next_execution - datetime.utcnow()).total_seconds() / 3600
        assert 5.9 < hours_until_next < 6.1

    @pytest.mark.asyncio
    async def test_is_nba_season_with_games(self, test_session, mock_session_factory):
        """Should return True when games scheduled in next 30 days."""
        writer = OddsWriter(test_session)

        # Create game in 15 days
        future_game = Event(
            id="future_game",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow() + timedelta(days=15),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(future_game)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        is_season = await intelligence.is_nba_season()

        assert is_season is True

    @pytest.mark.asyncio
    async def test_is_nba_season_no_games(self, test_session, mock_session_factory):
        """Should return False when no games scheduled (off-season)."""
        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        is_season = await intelligence.is_nba_season()

        assert is_season is False

    @pytest.mark.asyncio
    async def test_get_closest_game_none(self, test_session, mock_session_factory):
        """Should return None when no games scheduled."""
        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        closest = await intelligence.get_closest_game()

        assert closest is None

    @pytest.mark.asyncio
    async def test_get_closest_game_returns_earliest(self, test_session, mock_session_factory):
        """Should return earliest game among multiple."""
        writer = OddsWriter(test_session)

        game1 = Event(
            id="game1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow() + timedelta(days=3),
            home_team="Team A",
            away_team="Team B",
            status=EventStatus.SCHEDULED,
        )
        game2 = Event(
            id="game2",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow() + timedelta(hours=6),  # Earliest
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
        )
        game3 = Event(
            id="game3",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow() + timedelta(days=1),
            home_team="Team C",
            away_team="Team D",
            status=EventStatus.SCHEDULED,
        )

        await writer.upsert_event(game1)
        await writer.upsert_event(game2)
        await writer.upsert_event(game3)
        await test_session.commit()

        intelligence = SchedulingIntelligence(session_factory=mock_session_factory)
        closest = await intelligence.get_closest_game()

        assert closest is not None
        assert closest.id == "game2"
        assert closest.home_team == "Lakers"
