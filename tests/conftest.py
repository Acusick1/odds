"""Pytest configuration and fixtures."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel

# Set required environment variables for testing BEFORE any imports of Settings
os.environ.setdefault("ODDS_API_KEY", "test_api_key")
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://postgres:dev_password@localhost:5432/odds_test"
)

# Test database URL - use PostgreSQL for timezone-aware datetime support
# Can be overridden with TEST_DATABASE_URL environment variable
TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://postgres:dev_password@localhost:5432/odds_test"
)


@pytest.fixture
def sample_odds_data():
    """Load sample odds response from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_odds_response.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def sample_scores_data():
    """Load sample scores response from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_scores_response.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    yield engine

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def test_session(test_engine):
    """Create test database session."""
    async_session_maker = async_sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_maker() as session:
        yield session


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    from odds_core.config import (
        APIConfig,
        DatabaseConfig,
        DataCollectionConfig,
        DataQualityConfig,
        Settings,
    )

    return Settings(
        api=APIConfig(key="test_api_key", base_url="https://api.test.com/v4"),
        database=DatabaseConfig(url=TEST_DATABASE_URL),
        data_collection=DataCollectionConfig(
            sports=["basketball_nba"],
            bookmakers=["fanduel", "draftkings"],
            markets=["h2h", "spreads", "totals"],
            regions=["us"],
        ),
        data_quality=DataQualityConfig(enable_validation=True),
    )


# Backfill test fixtures
@pytest.fixture
def sample_backfill_plan():
    """Sample backfill plan for testing."""
    return {
        "total_games": 2,
        "total_snapshots": 4,
        "estimated_quota_usage": 120,
        "games": [
            {
                "event_id": "test_event_1",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "commence_time": "2024-01-15T19:00:00Z",
                "snapshots": [
                    "2024-01-12T19:00:00Z",
                    "2024-01-15T18:30:00Z",
                ],
                "snapshot_count": 2,
            },
            {
                "event_id": "test_event_2",
                "home_team": "Warriors",
                "away_team": "Heat",
                "commence_time": "2024-01-16T20:00:00Z",
                "snapshots": [
                    "2024-01-13T20:00:00Z",
                    "2024-01-16T19:30:00Z",
                ],
                "snapshot_count": 2,
            },
        ],
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-02-01T00:00:00Z",
    }


@pytest.fixture
def mock_api_response_factory():
    """Factory to create mock API responses for different events."""
    from odds_core.api_models import HistoricalOddsResponse, api_dict_to_event

    def _create_response(event_id="test_event_1", home_team="Lakers", away_team="Celtics"):
        # Create raw event data dict
        event_dict = {
            "id": event_id,
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-01-15T19:00:00Z",
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [
                {
                    "key": "fanduel",
                    "title": "FanDuel",
                    "last_update": "2024-01-15T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home_team, "price": -150},
                                {"name": away_team, "price": 130},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": home_team, "price": -110, "point": -3.5},
                                {"name": away_team, "price": -110, "point": 3.5},
                            ],
                        },
                    ],
                },
                {
                    "key": "draftkings",
                    "title": "DraftKings",
                    "last_update": "2024-01-15T18:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home_team, "price": -145},
                                {"name": away_team, "price": 125},
                            ],
                        }
                    ],
                },
            ],
        }

        # Convert to Event instance
        event = api_dict_to_event(event_dict)

        # Return HistoricalOddsResponse
        return HistoricalOddsResponse(
            events=[event],
            raw_events_data=[event_dict],
            response_time_ms=100,
            quota_remaining=19950,
            timestamp=datetime(2024, 1, 15, 18, 0, 0, tzinfo=UTC),
        )

    return _create_response


@pytest.fixture
def mock_api_client(mock_api_response_factory):
    """Mock API client with configurable responses."""
    from unittest.mock import AsyncMock

    client = AsyncMock()

    # Default behavior: return appropriate response based on call count
    call_count = {"count": 0}

    async def get_historical_odds(*args, **kwargs):
        call_count["count"] += 1
        if call_count["count"] <= 2:
            return mock_api_response_factory("test_event_1", "Lakers", "Celtics")
        else:
            return mock_api_response_factory("test_event_2", "Warriors", "Heat")

    client.get_historical_odds = AsyncMock(side_effect=get_historical_odds)
    return client


@pytest.fixture
async def mock_session_factory(test_engine):
    """Create a session factory for testing that uses the test engine."""
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    return factory
