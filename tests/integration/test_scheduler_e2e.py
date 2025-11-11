"""
End-to-end test for the scheduler system.

This test suite provides TWO complementary E2E tests:

1. **test_scheduler_end_to_end**: Tests the complete production flow including
   APScheduler executing jobs (using real-time scheduling, no time mocking)

2. **test_job_self_scheduling_chain**: Tests the critical self-scheduling loop
   (Job → SchedulingIntelligence → Backend) with time mocking for all tiers

Together these provide confidence the scheduler will work in production.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from odds_core.api_models import OddsResponse, api_dict_to_event
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.ingestion import OddsIngestionService
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter

# Test constants
GAME_TIME = datetime(2025, 1, 15, 19, 0, 0, tzinfo=UTC)
EVENT_ID = "e2e_test_game"
HOME_TEAM = "Los Angeles Lakers"
AWAY_TEAM = "Boston Celtics"


@pytest.fixture
def synthetic_event():
    """Create a synthetic event for testing."""
    # Use a game that's 2 hours in the future (CLOSING tier) for E2E test
    return Event(
        id=EVENT_ID,
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime.now(UTC) + timedelta(hours=2),
        home_team=HOME_TEAM,
        away_team=AWAY_TEAM,
        status=EventStatus.SCHEDULED,
    )


@pytest.fixture
def mock_odds_data():
    """Create realistic mock odds data."""

    def _create() -> dict:
        return {
            "id": EVENT_ID,
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            "home_team": HOME_TEAM,
            "away_team": AWAY_TEAM,
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": datetime.now(UTC).isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": HOME_TEAM, "price": -120},
                                {"name": AWAY_TEAM, "price": 100},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": HOME_TEAM, "price": -110, "point": -3.5},
                                {"name": AWAY_TEAM, "price": -110, "point": 3.5},
                            ],
                        },
                    ],
                },
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": datetime.now(UTC).isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": HOME_TEAM, "price": -115},
                                {"name": AWAY_TEAM, "price": -105},
                            ],
                        },
                    ],
                },
            ],
        }

    return _create


@pytest.fixture
async def setup_test_event(test_session, synthetic_event):
    """Set up the initial event in the database."""
    writer = OddsWriter(test_session)
    await writer.upsert_event(synthetic_event)
    await test_session.commit()
    return synthetic_event


@pytest.mark.asyncio
async def test_scheduler_end_to_end(
    test_session, mock_session_factory, setup_test_event, mock_odds_data
):
    """
    TRUE end-to-end test of the scheduler system.

    This test runs the scheduler as close to production as possible:
    1. Starts LocalSchedulerBackend (APScheduler)
    2. Schedules fetch-odds job to run in 2 seconds (REAL time, no mocking)
    3. Job executes via APScheduler (not a direct call)
    4. Job fetches data and stores it in database
    5. We verify all data was persisted correctly

    NOTE: This uses REAL time (no freezegun) because APScheduler runs in real-time.
    The game is 2 hours in the future, so we're in CLOSING tier.
    """
    from odds_lambda.scheduling.backends.local import LocalSchedulerBackend

    # Track execution
    execution_happened = {"fetch_odds": False}

    # Create mock API client
    mock_client = AsyncMock()

    async def mock_get_odds(*args, **kwargs):
        """Return realistic odds data."""
        odds_data = mock_odds_data()
        event = api_dict_to_event(odds_data)
        return OddsResponse(
            events=[event],
            raw_events_data=[odds_data],
            response_time_ms=100,
            quota_remaining=19900,
            timestamp=datetime.now(UTC),
        )

    mock_client.get_odds = mock_get_odds

    # Wrap the actual job to inject our mocks
    from odds_lambda.jobs import fetch_odds

    original_main = fetch_odds.main

    def build_service(settings, client=mock_client):
        class _ClientContext:
            async def __aenter__(self):
                return client

            async def __aexit__(self, exc_type, exc, tb):
                return False

        return OddsIngestionService(
            settings=settings,
            session_factory=mock_session_factory,
            client_factory=cast(Any, lambda: _ClientContext()),
        )

    async def wrapped_fetch_odds():
        """Wrapped job with mocked dependencies."""
        with (
            patch("odds_lambda.jobs.fetch_odds.build_ingestion_service", side_effect=build_service),
            patch("odds_lambda.scheduling.intelligence.async_session_maker", mock_session_factory),
        ):
            # Track execution
            execution_happened["fetch_odds"] = True

            # Execute the real job
            await original_main()

    # Start the scheduler backend
    async with LocalSchedulerBackend(dry_run=False) as backend:
        # Mock the job registry to return our wrapped job
        with patch("odds_lambda.scheduling.jobs.get_job_function", return_value=wrapped_fetch_odds):
            # Schedule job to run 2 seconds in the future
            run_time = datetime.now(UTC) + timedelta(seconds=2)
            await backend.schedule_next_execution(job_name="fetch-odds", next_time=run_time)

            print(f"✓ Scheduler: Job scheduled for {run_time}")

            # Verify job is scheduled
            jobs = await backend.get_scheduled_jobs()
            assert len(jobs) == 1
            assert jobs[0].job_name == "fetch-odds"
            print(f"✓ Scheduler: Found {len(jobs)} scheduled job(s)")

        # Wait for the job to execute (2s + buffer)
        print("⏳ Waiting for job to execute via scheduler...")
        await asyncio.sleep(3)

        # Verify the job executed
        assert execution_happened["fetch_odds"], "Job should have executed via scheduler"
        print("✓ Scheduler: Job executed via APScheduler")

    # Verify data persisted
    await test_session.rollback()  # Refresh to see committed data

    # Verify odds snapshot was created
    snapshot_query = select(OddsSnapshot).where(OddsSnapshot.event_id == EVENT_ID)
    result = await test_session.execute(snapshot_query)
    snapshot = result.scalar_one_or_none()

    assert snapshot is not None, "Odds snapshot should exist"
    assert snapshot.bookmaker_count == 2
    print(f"✓ Data persistence: Odds snapshot created (id={snapshot.id})")

    # Verify normalized odds exist
    reader = OddsReader(test_session)
    odds_list = await reader.get_odds_at_time(
        event_id=EVENT_ID, timestamp=datetime.now(UTC), tolerance_minutes=5
    )

    assert len(odds_list) > 0, "Normalized odds should exist"
    bookmakers = {odd.bookmaker_key for odd in odds_list}
    assert "fanduel" in bookmakers
    assert "draftkings" in bookmakers
    print(f"✓ Data persistence: {len(odds_list)} normalized odds records created")

    # Verify event still scheduled
    event = await reader.get_event_by_id(EVENT_ID)
    assert event.status == EventStatus.SCHEDULED
    print(f"✓ Event lifecycle: Event status is {event.status.value}")

    print(
        "\n✓ END-TO-END TEST COMPLETE: APScheduler executed job and persisted data, "
        "exactly as it would in production"
    )


@pytest.mark.asyncio
async def test_job_self_scheduling_chain(test_session, mock_session_factory):
    """
    Test the self-scheduling chain: Job → SchedulingIntelligence → Backend.

    This verifies that:
    1. Job executes
    2. SchedulingIntelligence determines next execution time based on tier
    3. Job calls backend.schedule_next_execution() with correct time
    4. Adaptive intervals work correctly (48h → 24h → 12h → 3h → 30min)

    This is critical for verifying the scheduler will keep running in production.

    Uses time mocking (freezegun) to test different tiers without waiting.
    """
    from freezegun import freeze_time

    from odds_lambda.jobs import fetch_odds

    # Create a test event for each tier
    writer = OddsWriter(test_session)

    # Test multiple tiers to verify adaptive scheduling
    test_cases = [
        (GAME_TIME - timedelta(days=5), FetchTier.OPENING, 48.0, "OPENING"),
        (GAME_TIME - timedelta(hours=18), FetchTier.SHARP, 12.0, "SHARP"),
        (GAME_TIME - timedelta(hours=1), FetchTier.CLOSING, 0.5, "CLOSING"),
    ]

    for test_time, _, expected_hours, tier_name in test_cases:
        # Create event for this tier
        event = Event(
            id=f"e2e_{tier_name.lower()}",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=GAME_TIME,
            home_team=HOME_TEAM,
            away_team=AWAY_TEAM,
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        scheduled_calls = []

        mock_client = AsyncMock()
        mock_client.get_odds = AsyncMock(
            return_value=OddsResponse(
                events=[],
                raw_events_data=[],
                response_time_ms=100,
                quota_remaining=19900,
                timestamp=test_time,
            )
        )

        async def mock_schedule_next(job_name: str, next_time: datetime, _calls=scheduled_calls):
            _calls.append({"job_name": job_name, "next_time": next_time})

        def build_service(settings, client=mock_client):
            class _ClientContext:
                async def __aenter__(self):
                    return client

                async def __aexit__(self, exc_type, exc, tb):
                    return False

            return OddsIngestionService(
                settings=settings,
                session_factory=mock_session_factory,
                client_factory=cast(Any, lambda: _ClientContext()),
            )

        with (
            freeze_time(test_time),
            patch("odds_lambda.jobs.fetch_odds.build_ingestion_service", side_effect=build_service),
            patch("odds_lambda.scheduling.intelligence.async_session_maker", mock_session_factory),
            patch("odds_lambda.jobs.fetch_odds.get_scheduler_backend") as mock_backend_getter,
        ):
            mock_backend = AsyncMock()
            mock_backend.schedule_next_execution = mock_schedule_next
            mock_backend.get_backend_name = lambda: "mock_backend"
            mock_backend_getter.return_value = mock_backend

            # Execute job
            await fetch_odds.main()

        # Verify self-scheduling
        assert len(scheduled_calls) == 1, f"Expected 1 schedule call for {tier_name}"
        scheduled = scheduled_calls[0]

        expected_next_time = test_time + timedelta(hours=expected_hours)
        assert scheduled["next_time"] == expected_next_time

        hours_ahead = (scheduled["next_time"] - test_time).total_seconds() / 3600
        print(f"✓ {tier_name}: Next run in {hours_ahead}h (tier interval={expected_hours}h)")

    print(
        "\n✓ SELF-SCHEDULING VERIFIED: Jobs correctly schedule next execution "
        "with adaptive intervals"
    )
