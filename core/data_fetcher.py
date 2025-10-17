"""The Odds API client for fetching odds and scores data."""

import time
from datetime import datetime

import aiohttp
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import settings

logger = structlog.get_logger()


class TheOddsAPIClient:
    """Client for interacting with The Odds API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize API client.

        Args:
            api_key: API key (defaults to settings)
            base_url: Base URL (defaults to settings)
        """
        self.api_key = api_key or settings.odds_api_key
        self.base_url = base_url or settings.odds_api_base_url
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

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _make_request(
        self, endpoint: str, params: dict | None = None
    ) -> tuple[dict | list, int]:
        """
        Make HTTP request with retry logic.

        Args:
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Tuple of (response data, response time in ms)

        Raises:
            aiohttp.ClientError: On request failure after retries
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{self.base_url}/{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        start_time = time.time()

        try:
            async with self.session.get(url, params=params, timeout=30) as response:
                response.raise_for_status()

                # Track quota from headers
                if "x-requests-remaining" in response.headers:
                    self._quota_remaining = int(response.headers["x-requests-remaining"])

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

        except aiohttp.ClientResponseError as e:
            logger.error(
                "api_request_failed",
                endpoint=endpoint,
                status=e.status,
                message=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "api_request_error",
                endpoint=endpoint,
                error=str(e),
            )
            raise

    async def get_odds(
        self,
        sport: str,
        regions: list[str] | None = None,
        markets: list[str] | None = None,
        odds_format: str = "american",
        bookmakers: list[str] | None = None,
    ) -> dict:
        """
        Fetch current odds for a sport.

        Args:
            sport: Sport key (e.g., 'basketball_nba')
            regions: List of regions (defaults to settings)
            markets: List of markets (defaults to settings)
            odds_format: Odds format ('american', 'decimal', 'fractional')
            bookmakers: List of bookmakers (defaults to settings)

        Returns:
            API response with odds data

        Example:
            async with TheOddsAPIClient() as client:
                odds = await client.get_odds('basketball_nba')
        """
        regions = regions or settings.regions
        markets = markets or settings.markets
        bookmakers = bookmakers or settings.bookmakers

        params = {
            "regions": ",".join(regions),
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "bookmakers": ",".join(bookmakers),
        }

        data, response_time = await self._make_request(f"sports/{sport}/odds", params=params)

        logger.info(
            "odds_fetched",
            sport=sport,
            events_count=len(data) if isinstance(data, list) else 0,
            response_time_ms=response_time,
        )

        return {
            "data": data,
            "response_time_ms": response_time,
            "quota_remaining": self._quota_remaining,
            "timestamp": datetime.utcnow(),
        }

    async def get_scores(self, sport: str, days_from: int = 1) -> dict:
        """
        Fetch scores for completed games.

        Args:
            sport: Sport key (e.g., 'basketball_nba')
            days_from: Number of days from present to fetch scores

        Returns:
            API response with scores data
        """
        params = {"daysFrom": days_from}

        data, response_time = await self._make_request(f"sports/{sport}/scores", params=params)

        logger.info(
            "scores_fetched",
            sport=sport,
            events_count=len(data) if isinstance(data, list) else 0,
            response_time_ms=response_time,
        )

        return {
            "data": data,
            "response_time_ms": response_time,
            "quota_remaining": self._quota_remaining,
            "timestamp": datetime.utcnow(),
        }

    async def get_historical_odds(
        self,
        sport: str,
        date: str,
        regions: list[str] | None = None,
        markets: list[str] | None = None,
        odds_format: str = "american",
        bookmakers: list[str] | None = None,
    ) -> dict:
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
            API response with historical odds data
        """
        regions = regions or settings.regions
        markets = markets or settings.markets
        bookmakers = bookmakers or settings.bookmakers

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

        logger.info(
            "historical_odds_fetched",
            sport=sport,
            date=date,
            events_count=len(data) if isinstance(data, list) else 0,
            response_time_ms=response_time,
        )

        return {
            "data": data,
            "response_time_ms": response_time,
            "quota_remaining": self._quota_remaining,
            "timestamp": datetime.utcnow(),
        }

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
            "timestamp": datetime.utcnow(),
        }
