"""Unit tests for BackfillExecutor."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from odds_analytics.backfill_executor import BackfillExecutor
from odds_core.api_models import HistoricalOddsResponse

from tests.test_helpers import StubHistoricalOddsClient


class TestBackfillExecutor:
    """Test BackfillExecutor functionality."""

    @pytest.mark.asyncio
    async def test_dry_run_mode(self, sample_backfill_plan):
        """Test dry run mode doesn't make API calls and counts as success."""
        # Create mock client to verify no calls made
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock()

        progress_updates = []

        async with BackfillExecutor(
            client=mock_client, dry_run=True, rate_limit_seconds=0
        ) as executor:
            result = await executor.execute_plan(
                sample_backfill_plan, progress_callback=lambda p: progress_updates.append(p)
            )

            # All snapshots processed as dry run
            assert len(progress_updates) == 4
            assert all(p.status == "skipped" for p in progress_updates)
            assert result.successful_snapshots == 4
            assert result.failed_snapshots == 0
            assert result.total_quota_used == 0

            # No API calls made
            mock_client.get_historical_odds.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_plan_raises_error(self):
        """Test that empty plan raises ValueError."""
        # Client not used in this test, just needs to exist
        mock_client = AsyncMock()

        async with BackfillExecutor(client=mock_client) as executor:
            with pytest.raises(ValueError, match="Plan contains no games"):
                await executor.execute_plan({"games": []})

    @pytest.mark.asyncio
    async def test_api_error_handling(self, sample_backfill_plan):
        """Test that API errors are properly caught and counted as failures."""
        from unittest.mock import AsyncMock

        # Create client that always raises errors
        error_client = AsyncMock()
        error_client.get_historical_odds = AsyncMock(side_effect=Exception("API Error"))

        progress_updates = []

        async with BackfillExecutor(
            client=error_client, skip_existing=False, rate_limit_seconds=0
        ) as executor:
            result = await executor.execute_plan(
                sample_backfill_plan, progress_callback=lambda p: progress_updates.append(p)
            )

            # All snapshots failed
            assert result.failed_snapshots == 4
            assert result.successful_snapshots == 0
            assert all(p.status == "failed" for p in progress_updates)

    @pytest.mark.asyncio
    async def test_progress_callback(self, sample_backfill_plan):
        """Test that progress callback is called for each snapshot."""
        # Client not actually called in dry run mode
        mock_client = AsyncMock()

        progress_updates = []

        async with BackfillExecutor(client=mock_client, dry_run=True) as executor:
            await executor.execute_plan(
                sample_backfill_plan, progress_callback=lambda p: progress_updates.append(p)
            )

            # One update per snapshot
            assert len(progress_updates) == 4

            # Check both events are represented
            event_ids = {p.event_id for p in progress_updates}
            assert event_ids == {"test_event_1", "test_event_2"}

            # Each event has 2 snapshots
            event1_updates = [p for p in progress_updates if p.event_id == "test_event_1"]
            event2_updates = [p for p in progress_updates if p.event_id == "test_event_2"]
            assert len(event1_updates) == 2
            assert len(event2_updates) == 2

    @pytest.mark.asyncio
    async def test_rate_limiting(self, sample_backfill_plan):
        """Test that rate limiting is applied after successful requests."""
        from unittest.mock import MagicMock, patch

        from odds_core.api_models import api_dict_to_event

        # First part: dry run mode (no sleeps)
        mock_client = AsyncMock()

        with patch("asyncio.sleep") as mock_sleep:
            async with BackfillExecutor(
                client=mock_client, dry_run=True, rate_limit_seconds=1.5
            ) as executor:
                await executor.execute_plan(sample_backfill_plan)

                # In dry run, no sleep is called (no actual API requests)
                mock_sleep.assert_not_called()

        # Second part: actual API calls with rate limiting
        # Create responses for the two events
        event1_dict = {
            "id": "test_event_1",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-01-15T19:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "bookmakers": [],
        }
        event2_dict = {
            "id": "test_event_2",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-01-15T21:00:00Z",
            "home_team": "Warriors",
            "away_team": "Heat",
            "bookmakers": [],
        }

        # Create responses - first 2 calls for event1, next 2 for event2
        responses = [
            HistoricalOddsResponse(
                events=[api_dict_to_event(event1_dict)],
                raw_events_data=[event1_dict],
                response_time_ms=100,
                quota_remaining=19950,
                timestamp=datetime(2024, 1, 15, 18, 0, 0, tzinfo=UTC),
            ),
            HistoricalOddsResponse(
                events=[api_dict_to_event(event1_dict)],
                raw_events_data=[event1_dict],
                response_time_ms=100,
                quota_remaining=19940,
                timestamp=datetime(2024, 1, 15, 18, 30, 0, tzinfo=UTC),
            ),
            HistoricalOddsResponse(
                events=[api_dict_to_event(event2_dict)],
                raw_events_data=[event2_dict],
                response_time_ms=100,
                quota_remaining=19930,
                timestamp=datetime(2024, 1, 15, 20, 0, 0, tzinfo=UTC),
            ),
            HistoricalOddsResponse(
                events=[api_dict_to_event(event2_dict)],
                raw_events_data=[event2_dict],
                response_time_ms=100,
                quota_remaining=19920,
                timestamp=datetime(2024, 1, 15, 20, 30, 0, tzinfo=UTC),
            ),
        ]

        stub_client = StubHistoricalOddsClient(responses)

        with patch("asyncio.sleep") as mock_sleep:
            # Need to mock session factory to avoid database access
            mock_session = MagicMock()
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock()
            mock_session.flush = AsyncMock()
            mock_session.commit = AsyncMock()

            def mock_factory():
                return mock_session

            async with BackfillExecutor(
                client=stub_client,
                session_factory=mock_factory,
                skip_existing=False,
                rate_limit_seconds=1.5,
            ) as executor:
                await executor.execute_plan(sample_backfill_plan)

                # Should sleep after each successful API call
                assert mock_sleep.call_count == 4
                mock_sleep.assert_called_with(1.5)

    @pytest.mark.asyncio
    async def test_event_not_found_in_response(self, sample_backfill_plan):
        """Test handling when event is not in API response."""
        from datetime import datetime
        from unittest.mock import AsyncMock

        from odds_core.api_models import HistoricalOddsResponse

        # Client returns empty response
        empty_client = AsyncMock()
        empty_client.get_historical_odds = AsyncMock(
            return_value=HistoricalOddsResponse(
                events=[],
                raw_events_data=[],
                response_time_ms=100,
                quota_remaining=19000,
                timestamp=datetime(2024, 1, 15, 18, 0, 0),
            )
        )

        progress_updates = []

        async with BackfillExecutor(
            client=empty_client, skip_existing=False, rate_limit_seconds=0
        ) as executor:
            result = await executor.execute_plan(
                sample_backfill_plan, progress_callback=lambda p: progress_updates.append(p)
            )

            # All should be skipped (not found)
            assert result.skipped_snapshots == 4
            assert result.successful_snapshots == 0
            assert all(p.status == "skipped" for p in progress_updates)
