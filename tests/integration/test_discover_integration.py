"""Integration tests for discover command with database."""

from datetime import UTC, datetime
from unittest.mock import patch

import pytest
from odds_core.models import Event, EventExternalId, EventStatus
from odds_core.vocabulary import Source
from sqlalchemy import select


class TestDiscoverIntegration:
    """Integration tests for discover command with real database."""

    @pytest.mark.asyncio
    async def test_discover_stores_events_in_database(
        self, test_session, mock_historical_events_response, mock_api_client_factory
    ):
        """Test that discover command stores events in database with SCHEDULED status."""
        from unittest.mock import AsyncMock

        from odds_cli.commands.discover import _discover_games

        mock_client = mock_api_client_factory(mock_historical_events_response)

        # Mock async_session_maker to return test_session
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_context.__aexit__ = AsyncMock()

        with patch("odds_cli.commands.discover.TheOddsAPIClient", return_value=mock_client):
            with patch(
                "odds_cli.commands.discover.async_session_maker",
                return_value=mock_session_context,
            ):
                # Run discover command
                await _discover_games(
                    start_date_str="2024-10-15",
                    end_date_str="2024-10-15",
                    sport="basketball_nba",
                )

            # Verify events were stored in database
            result = await test_session.execute(select(Event))
            events = result.scalars().all()

            assert len(events) == 2

            # Verify event details
            external_ids = (
                await test_session.execute(select(EventExternalId.external_id))
            ).scalars()
            assert set(external_ids) == {"event1", "event2"}

            # Verify all events have SCHEDULED status (not FINAL)
            for event in events:
                assert event.status == EventStatus.SCHEDULED
                assert event.home_score is None
                assert event.away_score is None
                assert event.completed_at is None

            # Verify team names
            lakers_event = next(e for e in events if e.home_team == "Lakers")
            assert lakers_event.home_team == "Lakers"
            assert lakers_event.away_team == "Celtics"

    @pytest.mark.asyncio
    async def test_discover_updates_existing_events(
        self, test_session, mock_historical_events_response, mock_api_client_factory
    ):
        """Test that re-running discover finds existing events (idempotency)."""
        from unittest.mock import AsyncMock

        from odds_cli.commands.discover import _discover_games

        # First, create an existing event with different data
        existing_event = Event(
            id=1,
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 10, 14, 19, 0, 0, tzinfo=UTC),  # Different time
            home_team="OldTeam1",  # Different team
            away_team="OldTeam2",
            status=EventStatus.SCHEDULED,
        )
        test_session.add(existing_event)
        await test_session.flush()
        test_session.add(
            EventExternalId(
                source=Source.ODDS_API,
                external_id="event1",
                event_id=existing_event.id,
            )
        )
        await test_session.commit()

        mock_client = mock_api_client_factory(mock_historical_events_response)

        # Mock async_session_maker to return test_session
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_context.__aexit__ = AsyncMock()

        with patch("odds_cli.commands.discover.TheOddsAPIClient", return_value=mock_client):
            with patch(
                "odds_cli.commands.discover.async_session_maker",
                return_value=mock_session_context,
            ):
                # Run discover command
                await _discover_games(
                    start_date_str="2024-10-15",
                    end_date_str="2024-10-15",
                    sport="basketball_nba",
                )

        # Refresh test_session to get updated data from database
        test_session.expire_all()  # Force SQLAlchemy to re-fetch from database

        # Verify events in database
        result = await test_session.execute(select(Event))
        events = result.scalars().all()

        # Should still have 2 events (1 found, 1 inserted)
        assert len(events) == 2

        # The event the API's id names is found, not duplicated, and keeps its teams.
        updated_event = next(e for e in events if e.id == 1)
        assert updated_event.home_team == "OldTeam1"
        assert updated_event.away_team == "OldTeam2"
        # A historical start is in the past, so it is not offered as a kick-off move.
        assert updated_event.commence_time == datetime(2024, 10, 14, 19, 0, 0, tzinfo=UTC)

    @pytest.mark.asyncio
    async def test_discover_multiple_days(self, test_session, mock_api_client_factory):
        """Test discovering events across multiple days."""
        from unittest.mock import AsyncMock

        from odds_cli.commands.discover import _discover_games

        # Mock responses for different days
        def get_events_for_date(sport, date):
            # Parse date to determine which events to return
            if "2024-10-15" in date:
                return {
                    "data": [
                        {
                            "id": "event_oct15_1",
                            "sport_key": "basketball_nba",
                            "sport_title": "NBA",
                            "commence_time": "2024-10-15T19:00:00Z",
                            "home_team": "Lakers",
                            "away_team": "Celtics",
                        }
                    ],
                    "quota_remaining": 19990,
                    "timestamp": datetime.now(UTC),
                }
            elif "2024-10-16" in date:
                return {
                    "data": [
                        {
                            "id": "event_oct16_1",
                            "sport_key": "basketball_nba",
                            "sport_title": "NBA",
                            "commence_time": "2024-10-16T19:00:00Z",
                            "home_team": "Warriors",
                            "away_team": "Heat",
                        }
                    ],
                    "quota_remaining": 19989,
                    "timestamp": datetime.now(UTC),
                }
            else:
                return {"data": [], "quota_remaining": 19988, "timestamp": datetime.now(UTC)}

        mock_client = mock_api_client_factory(side_effect=get_events_for_date)

        # Mock async_session_maker to return test_session
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_context.__aexit__ = AsyncMock()

        with patch("odds_cli.commands.discover.TheOddsAPIClient", return_value=mock_client):
            with patch(
                "odds_cli.commands.discover.async_session_maker",
                return_value=mock_session_context,
            ):
                # Run discover command for 2-day range
                await _discover_games(
                    start_date_str="2024-10-15",
                    end_date_str="2024-10-16",
                    sport="basketball_nba",
                )

            # Verify events in database
            result = await test_session.execute(select(Event))
            events = result.scalars().all()

            # Should have 2 events (1 per day)
            assert len(events) == 2

            external_ids = (
                await test_session.execute(select(EventExternalId.external_id))
            ).scalars()
            assert set(external_ids) == {"event_oct15_1", "event_oct16_1"}

            # Verify API was called twice (once per day)
            assert mock_client.get_historical_events.call_count == 2

    @pytest.mark.asyncio
    async def test_discover_handles_empty_days(self, test_session, mock_api_client_factory):
        """Test that discover handles days with no games gracefully."""
        from unittest.mock import AsyncMock

        from odds_cli.commands.discover import _discover_games

        mock_client = mock_api_client_factory()  # Uses default empty response

        # Mock async_session_maker to return test_session
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_context.__aexit__ = AsyncMock()

        with patch("odds_cli.commands.discover.TheOddsAPIClient", return_value=mock_client):
            with patch(
                "odds_cli.commands.discover.async_session_maker",
                return_value=mock_session_context,
            ):
                # Run discover command (should not crash)
                await _discover_games(
                    start_date_str="2024-07-01",  # Off-season
                    end_date_str="2024-07-05",
                    sport="basketball_nba",
                )

            # Verify no events in database
            result = await test_session.execute(select(Event))
            events = result.scalars().all()

            assert len(events) == 0

    @pytest.mark.asyncio
    async def test_discover_preserves_existing_scores(self, test_session, mock_api_client_factory):
        """Test that discover doesn't overwrite final scores if event already completed."""
        from unittest.mock import AsyncMock

        from odds_cli.commands.discover import _discover_games

        # Create an existing event with final scores
        completed_event = Event(
            id=1,
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 10, 15, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=108,
            away_score=105,
            completed_at=datetime(2024, 10, 15, 22, 0, 0, tzinfo=UTC),
        )
        test_session.add(completed_event)
        await test_session.commit()

        mock_response = {
            "data": [
                {
                    "id": "event1",
                    "sport_key": "basketball_nba",
                    "sport_title": "NBA",
                    "commence_time": "2024-10-15T19:00:00Z",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                }
            ],
            "quota_remaining": 19990,
            "timestamp": datetime.now(UTC),
        }

        mock_client = mock_api_client_factory(mock_response)

        # Mock async_session_maker to return test_session
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=test_session)
        mock_session_context.__aexit__ = AsyncMock()

        with patch("odds_cli.commands.discover.TheOddsAPIClient", return_value=mock_client):
            with patch(
                "odds_cli.commands.discover.async_session_maker",
                return_value=mock_session_context,
            ):
                # Run discover command
                await _discover_games(
                    start_date_str="2024-10-15",
                    end_date_str="2024-10-15",
                    sport="basketball_nba",
                )

            # Refresh the event from database
            result = await test_session.execute(select(Event).where(Event.id == 1))
            event = result.scalar_one()

            assert event.status == EventStatus.FINAL
            assert (event.home_score, event.away_score) == (108, 105)
