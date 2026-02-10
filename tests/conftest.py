"""Pytest configuration and fixtures."""

import json
import os
from collections.abc import AsyncGenerator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Set required environment variables for testing BEFORE any imports of Settings
os.environ.setdefault("ODDS_API_KEY", "test_api_key")


@pytest.fixture(scope="session")
def pglite_config():
    """
    Override pglite config to force TCP mode.

    This prevents intermittent Unix socket connection errors by ensuring
    all tests use TCP connections on a dedicated port.
    """
    from py_pglite.config import PGliteConfig

    return PGliteConfig(
        use_tcp=True,
        tcp_port=5434,
        cleanup_on_exit=True,
        timeout=30,
    )


@pytest.fixture
def sample_odds_data() -> list[dict[str, Any]]:
    """Load sample odds response from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_odds_response.json"
    with open(fixture_path) as f:
        return json.load(f)


@pytest.fixture
def sample_scores_data() -> list[dict[str, Any]]:
    """Load sample scores response from fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "sample_scores_response.json"
    with open(fixture_path) as f:
        return json.load(f)


# Override pglite_async_engine to create SQLModel tables
@pytest.fixture(scope="session")
async def pglite_async_engine(pglite_async_sqlalchemy_manager) -> AsyncEngine:
    """Override py-pglite engine to create SQLModel tables."""
    from sqlmodel import SQLModel

    engine = pglite_async_sqlalchemy_manager.get_engine()

    # Create all tables once at session level
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    return engine


# Override pglite_async_session to use expire_on_commit=False
@pytest.fixture
async def pglite_async_session(
    pglite_async_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession, None]:
    """Override py-pglite session to use expire_on_commit=False."""
    from sqlalchemy import text

    # Clean up data before test starts
    async with pglite_async_engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_type = 'BASE TABLE'
            """
            )
        )
        table_names = [row[0] for row in result]

        if table_names:
            await conn.execute(text("SET session_replication_role = replica;"))
            for table_name in table_names:
                await conn.execute(text(f'TRUNCATE TABLE "{table_name}" RESTART IDENTITY CASCADE;'))
            await conn.execute(text("SET session_replication_role = DEFAULT;"))
            await conn.commit()

    # Create session with expire_on_commit=False using SQLAlchemy's AsyncSession
    async with AsyncSession(pglite_async_engine, expire_on_commit=False) as session:
        yield session


# Alias py-pglite fixtures to match existing test code
@pytest.fixture
def test_engine(pglite_async_engine: AsyncEngine) -> AsyncEngine:
    """Alias for pglite_async_engine to match existing tests."""
    return pglite_async_engine


@pytest.fixture
def test_session(pglite_async_session: AsyncSession) -> AsyncSession:
    """Alias for pglite_async_session to match existing tests."""
    return pglite_async_session


@pytest.fixture
def mock_settings() -> Any:
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
        database=DatabaseConfig(url="postgresql+asyncpg://localhost/test"),
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
def sample_backfill_plan() -> dict[str, Any]:
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
def mock_api_response_factory() -> Callable[[str, str, str], Any]:
    """Factory to create mock API responses for different events."""
    from odds_core.api_models import HistoricalOddsResponse, api_dict_to_event

    def _create_response(
        event_id: str = "test_event_1", home_team: str = "Lakers", away_team: str = "Celtics"
    ) -> HistoricalOddsResponse:
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
def mock_session_factory(test_engine):
    """Create a session factory for testing that uses the test engine."""
    factory = sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    return factory


# Discover command test fixtures
@pytest.fixture
def mock_api_client_factory():
    """Factory to create mock TheOddsAPIClient with custom response."""
    from unittest.mock import AsyncMock

    def _create_mock_client(get_historical_events_response=None, side_effect=None):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        if side_effect:
            mock_client.get_historical_events = AsyncMock(side_effect=side_effect)
        elif get_historical_events_response:
            mock_client.get_historical_events = AsyncMock(
                return_value=get_historical_events_response
            )
        else:
            # Default empty response
            mock_client.get_historical_events = AsyncMock(
                return_value={
                    "data": [],
                    "quota_remaining": 19990,
                    "timestamp": datetime.now(UTC),
                }
            )

        return mock_client

    return _create_mock_client


@pytest.fixture
def mock_db_session():
    """Create mock database session for unit tests."""
    from unittest.mock import AsyncMock, MagicMock

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    mock_session.commit = AsyncMock()

    # Mock execute to return a result object with fetchall
    mock_execute_result = MagicMock()
    mock_execute_result.fetchall.return_value = []
    mock_session.execute = AsyncMock(return_value=mock_execute_result)

    return mock_session


@pytest.fixture
def mock_historical_events_response():
    """Standard mock response from get_historical_events for testing."""
    return {
        "data": [
            {
                "id": "event1",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": "2024-10-15T19:00:00Z",
                "home_team": "Lakers",
                "away_team": "Celtics",
            },
            {
                "id": "event2",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": "2024-10-15T20:00:00Z",
                "home_team": "Warriors",
                "away_team": "Heat",
            },
        ],
        "quota_remaining": 19990,
        "timestamp": datetime(2024, 10, 15, 12, 0, 0, tzinfo=UTC),
    }


@pytest.fixture
def sample_training_data():
    """
    Create sample training data for ML tests.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, feature_names)
    """
    import numpy as np

    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.randn(n_samples).astype(np.float32)
    X_val = np.random.randn(20, n_features).astype(np.float32)
    y_val = np.random.randn(20).astype(np.float32)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    return X_train, y_train, X_val, y_val, feature_names


# Polymarket test fixtures (shared across polymarket unit tests)


@pytest.fixture
def mock_polymarket_client():
    """Create mock PolymarketClient for testing."""
    from unittest.mock import AsyncMock

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.fixture
def mock_async_session():
    """Create mock async database session for testing."""
    from unittest.mock import AsyncMock

    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    return mock_session


@pytest.fixture
def mock_polymarket_reader():
    """Create mock PolymarketReader for testing."""
    from unittest.mock import AsyncMock

    mock_reader = AsyncMock()
    mock_reader.get_backfilled_market_ids.return_value = set()
    return mock_reader


@pytest.fixture
def mock_polymarket_writer():
    """Create mock PolymarketWriter for testing."""
    from unittest.mock import AsyncMock

    return AsyncMock()


@pytest.fixture
def sample_polymarket_event():
    """Create sample PolymarketEvent for testing."""
    from odds_core.polymarket_models import PolymarketEvent

    return PolymarketEvent(
        id=1,
        pm_event_id="test-event-123",
        ticker="nba-test-game",
        slug="test-game-slug",
        title="Test Team vs Test Team",
        start_date=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
        end_date=datetime(2024, 1, 15, 22, 0, 0, tzinfo=UTC),
        active=False,
        closed=True,
        volume=10000.0,
        liquidity=5000.0,
        markets_count=3,
    )


@pytest.fixture
def sample_polymarket_market():
    """Create sample PolymarketMarket for testing."""
    from odds_core.polymarket_models import PolymarketMarket, PolymarketMarketType

    return PolymarketMarket(
        id=1,
        polymarket_event_id=1,
        pm_market_id="test-market-123",
        condition_id="test-condition",
        question="Test Team vs Test Team",
        clob_token_ids=["test-token-id-1", "test-token-id-2"],
        outcomes=["Test Team", "Test Team"],
        market_type=PolymarketMarketType.MONEYLINE,
        group_item_title=None,
        point=None,
        active=False,
        closed=True,
        accepting_orders=False,
    )


@pytest.fixture
def sample_event_data():
    """Sample Gamma API event response data."""
    return {
        "id": "event-123",
        "ticker": "nba-lal-bos-2024-01-15",
        "slug": "lakers-vs-celtics",
        "title": "Lakers vs Celtics",
        "startDate": "2024-01-15T19:00:00Z",
        "endDate": "2024-01-15T22:00:00Z",
        "active": False,
        "closed": True,
        "volume": "10000",
        "liquidity": "5000",
        "markets": [
            {
                "id": "market-ml-123",
                "conditionId": "condition-ml-123",
                "question": "Lakers vs Celtics",
                "clobTokenIds": ["token-lal", "token-bos"],
                "outcomes": ["Lakers", "Celtics"],
                "active": False,
                "closed": True,
                "acceptingOrders": False,
            }
        ],
    }


@pytest.fixture
def sample_price_history():
    """Sample CLOB API price history response."""
    return [
        {"t": 1705339200, "p": "0.48"},
        {"t": 1705339500, "p": "0.49"},
        {"t": 1705339800, "p": "0.50"},
        {"t": 1705340100, "p": "0.51"},
        {"t": 1705340400, "p": "0.52"},
    ]


@pytest.fixture
def mock_polymarket_settings():
    """Mock Settings for Polymarket tests."""
    from unittest.mock import MagicMock

    mock = MagicMock()
    mock.polymarket.enabled = True
    return mock
