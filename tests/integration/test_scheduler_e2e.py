"""
End-to-end test for the scheduler system.

This test suite provides TWO complementary E2E tests:

1. **test_scheduler_end_to_end**: Tests the complete production flow including
   APScheduler executing jobs (using real-time scheduling, no time mocking)

2. **test_job_self_scheduling_chain**: Tests the critical self-scheduling loop
   (Job -> SchedulingIntelligence -> Backend) with time mocking for all tiers

Together these provide confidence the scheduler will work in production.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from apscheduler import AsyncScheduler
from odds_core.api_models import OddsResponse, api_dict_to_event
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.ingestion import OddsIngestionService
from odds_lambda.scheduling.jobs import JobContext
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select

from tests.test_helpers import StubOddsClient

# Test constants
GAME_TIME = datetime(2025, 1, 15, 19, 0, 0, tzinfo=UTC)
EVENT_ID = "e2e_test_game"
HOME_TEAM = "Los Angeles Lakers"
AWAY_TEAM = "Boston Celtics"

_BUILD_PATCH = "odds_lambda.scheduling.backends.local.build_scheduler"

# Module-level state shared with the e2e job function.
_e2e_execution_happened: dict[str, bool] = {}
_e2e_stub_client: StubOddsClient | None = None
_e2e_session_factory = None
_e2e_original_main = None


def _in_memory_scheduler(**kwargs) -> tuple[AsyncScheduler, AsyncMock]:
    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()
    return AsyncScheduler(**kwargs), mock_engine


async def _wrapped_fetch_odds(ctx: JobContext) -> None:
    """Module-level wrapper for the e2e test job."""
    mock_event_sync = AsyncMock()
    mock_event_sync.sync_sports = AsyncMock(return_value=[])

    def build_service(client_arg, settings, _client=_e2e_stub_client):
        return OddsIngestionService(
            client=_client,  # type: ignore
            settings=settings,
            session_factory=_e2e_session_factory,
        )

    with (
        patch("odds_lambda.jobs.fetch_odds.build_ingestion_service", side_effect=build_service),
        patch("odds_lambda.jobs.fetch_odds.build_event_sync_service", return_value=mock_event_sync),
        patch("odds_lambda.scheduling.intelligence.async_session_maker", _e2e_session_factory),
    ):
        _e2e_execution_happened["fetch_odds"] = True
        await _e2e_original_main(JobContext(sport="basketball_nba"))


@pytest.fixture
def synthetic_event():
    """Create a synthetic event for testing."""
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
    1. Starts LocalSchedulerBackend (APScheduler 4)
    2. Schedules fetch-odds job to run in 2 seconds (REAL time, no mocking)
    3. Job executes via APScheduler (not a direct call)
    4. Job fetches data and stores it in database
    5. We verify all data was persisted correctly

    NOTE: This uses REAL time (no freezegun) because APScheduler runs in real-time.
    The game is 2 hours in the future, so we're in CLOSING tier.
    """
    global _e2e_execution_happened, _e2e_stub_client, _e2e_session_factory, _e2e_original_main
    from odds_lambda.jobs import fetch_odds
    from odds_lambda.scheduling.backends.local import LocalSchedulerBackend

    # Set up module-level state for the wrapped job function
    _e2e_execution_happened = {"fetch_odds": False}
    _e2e_original_main = fetch_odds.main
    _e2e_session_factory = mock_session_factory

    odds_data = mock_odds_data()
    event = api_dict_to_event(odds_data)
    response = OddsResponse(
        events=[event],
        raw_events_data=[odds_data],
        response_time_ms=100,
        quota_remaining=19900,
        timestamp=datetime.now(UTC),
    )
    _e2e_stub_client = StubOddsClient(response)

    # Start the scheduler backend (in-memory for tests)
    with patch(_BUILD_PATCH, side_effect=_in_memory_scheduler):
        async with LocalSchedulerBackend(dry_run=False) as backend:
            with patch(
                "odds_lambda.scheduling.jobs.get_job_function",
                return_value=_wrapped_fetch_odds,
            ):
                run_time = datetime.now(UTC) + timedelta(seconds=2)
                await backend.schedule_next_execution(job_name="fetch-odds", next_time=run_time)

                print(f"Scheduler: Job scheduled for {run_time}")

                jobs = await backend.get_scheduled_jobs()
                assert len(jobs) == 1
                assert jobs[0].job_name == "fetch-odds"

            # Wait for the job to execute (2s + buffer)
            print("Waiting for job to execute via scheduler...")
            await asyncio.sleep(4)

            assert _e2e_execution_happened["fetch_odds"], "Job should have executed via scheduler"
            print("Scheduler: Job executed via APScheduler")

    # Verify data persisted
    await test_session.rollback()

    snapshot_query = select(OddsSnapshot).where(OddsSnapshot.event_id == EVENT_ID)
    result = await test_session.execute(snapshot_query)
    snapshot = result.scalar_one_or_none()

    assert snapshot is not None, "Odds snapshot should exist"
    assert snapshot.bookmaker_count == 2

    reader = OddsReader(test_session)
    odds_list = await reader.get_odds_at_time(
        event_id=EVENT_ID, timestamp=datetime.now(UTC), tolerance_minutes=5
    )

    assert len(odds_list) > 0, "Normalized odds should exist"
    bookmakers = {odd.bookmaker_key for odd in odds_list}
    assert "fanduel" in bookmakers
    assert "draftkings" in bookmakers

    event = await reader.get_event_by_id(EVENT_ID)
    assert event.status == EventStatus.SCHEDULED


@pytest.mark.asyncio
async def test_job_self_scheduling_chain(test_session, mock_session_factory):
    """
    Test the self-scheduling chain: Job -> SchedulingIntelligence -> Backend.

    This verifies that:
    1. Job executes
    2. SchedulingIntelligence determines next execution time based on tier
    3. Job calls backend.schedule_next_execution() with correct time
    4. Adaptive intervals work correctly (48h -> 24h -> 12h -> 3h -> 30min)

    Uses time mocking (freezegun) to test different tiers without waiting.
    """
    from freezegun import freeze_time
    from odds_lambda.jobs import fetch_odds

    writer = OddsWriter(test_session)

    test_cases = [
        (GAME_TIME - timedelta(days=5), FetchTier.OPENING, 48.0, "OPENING"),
        (GAME_TIME - timedelta(hours=18), FetchTier.SHARP, 12.0, "SHARP"),
        (GAME_TIME - timedelta(hours=1), FetchTier.CLOSING, 0.5, "CLOSING"),
    ]

    for test_time, _, expected_hours, tier_name in test_cases:
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

        response = OddsResponse(
            events=[],
            raw_events_data=[],
            response_time_ms=100,
            quota_remaining=19900,
            timestamp=test_time,
        )
        stub_client = StubOddsClient(response)

        async def mock_schedule_next(
            job_name: str,
            next_time: datetime,
            payload: dict[str, object] | None = None,
            _calls=scheduled_calls,
        ):
            _calls.append({"job_name": job_name, "next_time": next_time, "payload": payload})

        def build_service(client_arg, settings, _client=stub_client):
            return OddsIngestionService(
                client=_client,  # type: ignore
                settings=settings,
                session_factory=mock_session_factory,
            )

        mock_event_sync = AsyncMock()
        mock_event_sync.sync_sports = AsyncMock(return_value=[])
        with (
            freeze_time(test_time),
            patch("odds_lambda.jobs.fetch_odds.build_ingestion_service", side_effect=build_service),
            patch(
                "odds_lambda.jobs.fetch_odds.build_event_sync_service", return_value=mock_event_sync
            ),
            patch("odds_lambda.scheduling.intelligence.async_session_maker", mock_session_factory),
            patch("odds_lambda.jobs.fetch_odds.get_scheduler_backend") as mock_backend_getter,
        ):
            mock_backend = AsyncMock()
            mock_backend.schedule_next_execution = mock_schedule_next
            mock_backend.get_backend_name = lambda: "mock_backend"
            mock_backend_getter.return_value = mock_backend

            await fetch_odds.main(JobContext(sport="basketball_nba"))

        assert len(scheduled_calls) == 1, f"Expected 1 schedule call for {tier_name}"
        scheduled = scheduled_calls[0]

        expected_next_time = test_time + timedelta(hours=expected_hours)
        assert scheduled["next_time"] == expected_next_time

        hours_ahead = (scheduled["next_time"] - test_time).total_seconds() / 3600
        print(f"{tier_name}: Next run in {hours_ahead}h (tier interval={expected_hours}h)")
