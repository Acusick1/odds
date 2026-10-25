"""The Odds API client for fetching odds and scores data."""

import time
from datetime import UTC, datetime
from typing import Any

import aiohttp
import structlog
from odds_core.api_models import (
    EventsResponse,
    HistoricalOddsResponse,
    OddsResponse,
    ScoresResponse,
    create_scheduled_event,
)
from odds_core.bookmakers import ODDS_API_BOOKMAKER_KEY_MAP
from odds_core.config import get_settings
from odds_core.utils import relabel_three_way_markets
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from odds_lambda.api_key_manager import AllKeysExhaustedError, APIKeyManager

logger = structlog.get_logger()


def canonicalize_bookmaker_keys(raw_events_data: list[dict[str, Any]]) -> None:
    for event in raw_events_data:
        for bookmaker in event.get("bookmakers", []):
            bookmaker["key"] = ODDS_API_BOOKMAKER_KEY_MAP.get(bookmaker["key"], bookmaker["key"])


def _is_transient_error(exc: BaseException) -> bool:
    """Connection failures, timeouts and 5xx are worth retrying; other HTTP errors are not.

    401, 429 and other 4xx responses cannot succeed on an immediate retry.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    return isinstance(exc, aiohttp.ClientError | TimeoutError)


class TheOddsAPIClient:
    """Client for interacting with The Odds API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize API client.

        Args:
            api_key: API key (defaults to settings)
            base_url: Base URL (defaults to settings)
        """
        app_settings = get_settings()
        self._key_manager: APIKeyManager | None = None

        if api_key:
            self.api_key = api_key
        elif app_settings.api.keys:
            keys = [k.strip() for k in app_settings.api.keys.split(",") if k.strip()]
            self._key_manager = APIKeyManager(keys)
            self.api_key = self._key_manager.get_active_key()
        elif app_settings.api.key:
            self.api_key = app_settings.api.key
        else:
            raise ValueError("No Odds API key configured: set ODDS_API_KEYS or ODDS_API_KEY")

        self.base_url = base_url or app_settings.api.base_url
        self.session: aiohttp.ClientSession | None = None
        self._quota_remaining: int | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    @property
    def quota_remaining(self) -> int | None:
        """Get remaining API quota from last request."""
        return self._quota_remaining

    async def _make_request(
        self, endpoint: str, params: dict | None = None
    ) -> tuple[dict | list, int]:
        """
        Make an HTTP request, rotating the API key on 401/429 when a key manager is configured.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Tuple of (response data, response time in ms)

        Raises:
            aiohttp.ClientResponseError: On a non-retryable HTTP error, or a 401/429 with no
                further key to rotate to
            AllKeysExhaustedError: When every configured key has returned 401/429
            aiohttp.ClientError, TimeoutError: On transient failures that outlast the retries
        """
        while True:
            try:
                return await self._send(endpoint, {**(params or {}), "apiKey": self.api_key})
            except aiohttp.ClientResponseError as e:
                if e.status not in (401, 429) or not self._key_manager:
                    logger.error(
                        "api_request_failed",
                        endpoint=endpoint,
                        status=e.status,
                        message=str(e),
                    )
                    raise
                try:
                    self.api_key = self._key_manager.rotate_key()
                except AllKeysExhaustedError:
                    logger.error("all_api_keys_exhausted", endpoint=endpoint)
                    raise
                logger.info("api_key_rotated_on_error", status=e.status, endpoint=endpoint)
            except Exception as e:
                logger.error("api_request_error", endpoint=endpoint, error=str(e))
                raise

    @retry(
        retry=retry_if_exception(_is_transient_error),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _send(self, endpoint: str, params: dict) -> tuple[dict | list, int]:
        """Send one request, retrying only transient failures."""
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/{endpoint}"
        start_time = time.time()
        timeout = aiohttp.ClientTimeout(total=30)
        async with self.session.get(url, params=params, timeout=timeout) as response:
            if "x-requests-remaining" in response.headers:
                self._quota_remaining = int(float(response.headers["x-requests-remaining"]))

            response.raise_for_status()

            data = await response.json()
            elapsed_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "api_request_success",
                endpoint=endpoint,
                status=response.status,
                elapsed_ms=elapsed_ms,
                quota_remaining=self._quota_remaining,
            )

            return data, elapsed_ms

    async def get_odds(
        self,
        sport: str,
        regions: list[str] | None = None,
        markets: list[str] | None = None,
        odds_format: str = "american",
        bookmakers: list[str] | None = None,
    ) -> OddsResponse:
        """
        Fetch current odds for a sport.

        Args:
            sport: Sport key (e.g., 'basketball_nba')
            regions: List of regions (defaults to settings)
            markets: List of markets (defaults to settings)
            odds_format: Odds format ('american', 'decimal', 'fractional')
            bookmakers: List of bookmakers (defaults to settings)

        Returns:
            OddsResponse with parsed Event instances and raw data

        Example:
            async with TheOddsAPIClient() as client:
                response = await client.get_odds('basketball_nba')
                for event in response.events:
                    print(f"{event.home_team} vs {event.away_team}")
        """
        app_settings = get_settings()
        regions = regions or app_settings.data_collection.regions
        markets = markets or app_settings.data_collection.markets
        bookmakers = bookmakers or app_settings.data_collection.bookmakers_for(sport)

        params = {
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "bookmakers": ",".join(bookmakers),
        }

        data, response_time = await self._make_request(f"sports/{sport}/odds", params=params)

        # Ensure data is a list
        raw_events_data = [
            relabel_three_way_markets(event_dict, sport)
            for event_dict in (data if isinstance(data, list) else [data] if data else [])
        ]
        canonicalize_bookmaker_keys(raw_events_data)

        # Convert API dicts to Event instances (scheduled games only)
        events = [create_scheduled_event(event_dict) for event_dict in raw_events_data]

        logger.info(
            "odds_fetched",
            sport=sport,
            events_count=len(events),
            response_time_ms=response_time,
        )

        return OddsResponse(
            events=events,
            raw_events_data=raw_events_data,
            response_time_ms=response_time,
            quota_remaining=self._quota_remaining,
            timestamp=datetime.now(UTC),
        )

    async def get_events(self, sport: str) -> EventsResponse:
        """
        Fetch upcoming events for a sport using the free events endpoint.

        Args:
            sport: Sport key (e.g., 'basketball_nba')

        Returns:
            EventsResponse with parsed Event instances (no odds data)

        Note:
            This endpoint costs 0 quota units. Use it for event discovery
            instead of get_odds() which costs 3+ units.
        """
        data, response_time = await self._make_request(f"sports/{sport}/events")

        raw_events_data = data if isinstance(data, list) else [data] if data else []
        events = [create_scheduled_event(event_dict) for event_dict in raw_events_data]

        logger.info(
            "events_fetched",
            sport=sport,
            events_count=len(events),
            response_time_ms=response_time,
        )

        return EventsResponse(
            events=events,
            response_time_ms=response_time,
            quota_remaining=self._quota_remaining,
            timestamp=datetime.now(UTC),
        )

    async def get_scores(self, sport: str, days_from: int = 1) -> ScoresResponse:
        """
        Fetch scores for completed games.

        Args:
            sport: Sport key (e.g., 'basketball_nba')
            days_from: Number of days from present to fetch scores

        Returns:
            ScoresResponse with scores data

        Note:
            Scores are kept as dicts since we only extract scores, not full Event objects
        """
        params = {"daysFrom": days_from}

        data, response_time = await self._make_request(f"sports/{sport}/scores", params=params)

        scores_data = data if isinstance(data, list) else [data] if data else []

        logger.info(
            "scores_fetched",
            sport=sport,
            events_count=len(scores_data),
            response_time_ms=response_time,
        )

        return ScoresResponse(
            scores_data=scores_data,
            response_time_ms=response_time,
            quota_remaining=self._quota_remaining,
            timestamp=datetime.now(UTC),
        )

    async def get_historical_odds(
        self,
        sport: str,
        date: str,
        regions: list[str] | None = None,
        markets: list[str] | None = None,
        odds_format: str = "american",
        bookmakers: list[str] | None = None,
    ) -> HistoricalOddsResponse:
        """
        Fetch historical odds for a specific date.

        Args:
            sport: Sport key (e.g., 'basketball_nba')
            date: ISO date string (e.g., '2024-10-15T12:00:00Z')
            regions: List of regions (defaults to settings)
            markets: List of markets (defaults to settings)
            odds_format: Odds format ('american', 'decimal', 'fractional')
            bookmakers: List of bookmakers (defaults to settings)

        Returns:
            HistoricalOddsResponse with parsed Event instances and raw data
        """
        app_settings = get_settings()
        regions = regions or app_settings.data_collection.regions
        markets = markets or app_settings.data_collection.markets
        bookmakers = bookmakers or app_settings.data_collection.bookmakers_for(sport)

        params = {
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "bookmakers": ",".join(bookmakers),
            "date": date,
        }

        data, response_time = await self._make_request(
            f"historical/sports/{sport}/odds", params=params
        )

        # Historical endpoint returns {"data": [...]}
        response_data = data.get("data", []) if isinstance(data, dict) else []
        raw_events_data = [
            relabel_three_way_markets(event_dict, sport)
            for event_dict in (response_data if isinstance(response_data, list) else [])
        ]
        canonicalize_bookmaker_keys(raw_events_data)

        # Convert API dicts to Event instances (scheduled games only)
        events = [create_scheduled_event(event_dict) for event_dict in raw_events_data]

        logger.info(
            "historical_odds_fetched",
            sport=sport,
            date=date,
            events_count=len(events),
            response_time_ms=response_time,
        )

        return HistoricalOddsResponse(
            events=events,
            raw_events_data=raw_events_data,
            response_time_ms=response_time,
            quota_remaining=self._quota_remaining,
            timestamp=datetime.now(UTC),
        )

    async def get_historical_events(
        self,
        sport: str,
        date: str,
    ) -> dict:
        """
        Fetch list of events at a specific historical date.

        Args:
            sport: Sport key (e.g., 'basketball_nba')
            date: ISO date string (e.g., '2024-10-15T12:00:00Z')

        Returns:
            API response with event list (no odds, just event metadata)

        Note:
            This costs 1 request vs get_historical_odds which costs 10x per region per market.
            Use this to discover which games were happening at a given time.
        """
        params = {"date": date}

        data, response_time = await self._make_request(
            f"historical/sports/{sport}/events", params=params
        )

        logger.info(
            "historical_events_fetched",
            sport=sport,
            date=date,
            events_count=len(data) if isinstance(data, list) else 0,
            response_time_ms=response_time,
        )

        return {
            "data": data,
            "response_time_ms": response_time,
            "quota_remaining": self._quota_remaining,
            "timestamp": datetime.now(UTC),
        }
