"""The Odds API client for fetching odds and scores data."""

import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

import aiohttp
import structlog
from odds_core.api_models import (
    EventsResponse,
    HistoricalOddsResponse,
    OddsResponse,
    ScoresResponse,
    parse_api_event,
)
from odds_core.bookmakers import ODDS_API_BOOKMAKER_KEY_MAP
from odds_core.config import get_settings
from odds_core.time import parse_api_datetime
from odds_core.utils import relabel_three_way_markets
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

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


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    """A numeric quota header of this response, None when the response lacks it."""
    return int(float(headers[name])) if name in headers else None


class TheOddsAPIClient:
    """Client for interacting with The Odds API."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        """
        Initialize API client.

        Args:
            api_key: API key for every sport (defaults to the sport's key in settings)
            base_url: Base URL (defaults to settings)
        """
        app_settings = get_settings()
        self._explicit_key = api_key
        self._api = app_settings.api
        if not api_key and not self._api.key_list:
            raise ValueError("No Odds API key configured: set ODDS_API_KEYS or ODDS_API_KEY")

        self.base_url = base_url or app_settings.api.base_url
        self.session: aiohttp.ClientSession | None = None
        self._quota_remaining: int | None = None
        self._quota_used: int | None = None
        self._units_used: int | None = None

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

    @property
    def quota_used(self) -> int | None:
        """Credits used since the last quota reset, from the last ``x-requests-used`` header."""
        return self._quota_used

    @property
    def units_used(self) -> int | None:
        """Units the last request cost, from its ``x-requests-last`` header."""
        return self._units_used

    async def _make_request(
        self, endpoint: str, sport: str, params: dict | None = None
    ) -> tuple[dict | list, int]:
        """
        Make an HTTP request with the key ``sport`` spends.

        Args:
            endpoint: API endpoint path
            sport: Sport key whose configured key pays for the call
            params: Query parameters

        Returns:
            Tuple of (response data, response time in ms)

        Raises:
            aiohttp.ClientResponseError: On a non-retryable HTTP error (a 401 or 429 is not
                retried on another key: each sport spends its own)
            aiohttp.ClientError, TimeoutError: On transient failures that outlast the retries
        """
        api_key = self._explicit_key or self._api.key_for(sport)
        try:
            return await self._send(endpoint, {**(params or {}), "apiKey": api_key})
        except aiohttp.ClientResponseError as e:
            logger.error(
                "api_request_failed",
                endpoint=endpoint,
                status=e.status,
                message=str(e),
            )
            raise
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
            self._quota_remaining = _header_int(response.headers, "x-requests-remaining")
            self._quota_used = _header_int(response.headers, "x-requests-used")
            self._units_used = _header_int(response.headers, "x-requests-last")

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
            OddsResponse with the listed games and raw data

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

        data, response_time = await self._make_request(f"sports/{sport}/odds", sport, params=params)

        # Ensure data is a list
        raw_events_data = [
            relabel_three_way_markets(event_dict, sport)
            for event_dict in (data if isinstance(data, list) else [data] if data else [])
        ]
        canonicalize_bookmaker_keys(raw_events_data)

        events = [parse_api_event(event_dict) for event_dict in raw_events_data]

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
            quota_used=self._quota_used,
            timestamp=datetime.now(UTC),
        )

    async def get_events(self, sport: str) -> EventsResponse:
        """
        Fetch upcoming events for a sport using the free events endpoint.

        Args:
            sport: Sport key (e.g., 'basketball_nba')

        Returns:
            EventsResponse with the listed games (no odds data)

        Note:
            This endpoint costs 0 quota units. Use it for event discovery
            instead of get_odds() which costs 3+ units.
        """
        data, response_time = await self._make_request(f"sports/{sport}/events", sport)

        raw_events_data = data if isinstance(data, list) else [data] if data else []
        events = [parse_api_event(event_dict) for event_dict in raw_events_data]

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
            quota_used=self._quota_used,
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
            Scores are kept as dicts since only the scores are extracted
        """
        params = {"daysFrom": days_from}

        data, response_time = await self._make_request(
            f"sports/{sport}/scores", sport, params=params
        )

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
            quota_used=self._quota_used,
            timestamp=datetime.now(UTC),
        )

    async def get_historical_odds(
        self,
        sport: str,
        date: str,
        markets: list[str],
        regions: list[str] | None = None,
        bookmakers: list[str] | None = None,
        odds_format: str = "american",
    ) -> HistoricalOddsResponse:
        """
        Fetch the latest historical snapshot at or before ``date``.

        Args:
            sport: Sport key (e.g., 'basketball_nba')
            date: ISO date string (e.g., '2024-10-15T12:00:00Z')
            markets: Markets to request
            regions: Regions to request; give this or ``bookmakers``, not both
            bookmakers: Bookmakers to request; give this or ``regions``, not both
            odds_format: Odds format ('american', 'decimal', 'fractional')

        Returns:
            HistoricalOddsResponse with the listed games, raw data and the snapshot's own time

        Raises:
            ValueError: If both or neither of ``regions`` and ``bookmakers`` are given
        """
        if bool(regions) == bool(bookmakers):
            raise ValueError("give exactly one of regions or bookmakers")

        params = {
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
            "date": date,
        }
        if regions:
            params["regions"] = ",".join(regions)
        else:
            params["bookmakers"] = ",".join(bookmakers or [])

        data, response_time = await self._make_request(
            f"historical/sports/{sport}/odds", sport, params=params
        )
        if not isinstance(data, dict):
            raise ValueError(f"historical odds response is not an object: {type(data).__name__}")

        response_data = data.get("data", [])
        raw_events_data = [
            relabel_three_way_markets(event_dict, sport)
            for event_dict in (response_data if isinstance(response_data, list) else [])
        ]
        canonicalize_bookmaker_keys(raw_events_data)

        events = [parse_api_event(event_dict) for event_dict in raw_events_data]
        snapshot_time = parse_api_datetime(data["timestamp"])

        logger.info(
            "historical_odds_fetched",
            sport=sport,
            date=date,
            snapshot_time=snapshot_time.isoformat(),
            events_count=len(events),
            units_used=self._units_used,
            response_time_ms=response_time,
        )

        return HistoricalOddsResponse(
            events=events,
            raw_events_data=raw_events_data,
            response_time_ms=response_time,
            quota_remaining=self._quota_remaining,
            quota_used=self._quota_used,
            snapshot_time=snapshot_time,
            units_used=self._units_used,
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
            f"historical/sports/{sport}/events", sport, params=params
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
            "quota_used": self._quota_used,
            "timestamp": datetime.now(UTC),
        }
