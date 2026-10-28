"""Polymarket API client for Gamma (market discovery) and CLOB (price history)."""

import time
from datetime import UTC, datetime, timedelta

import aiohttp
import structlog
from odds_core.config import get_settings
from odds_core.sports import SportKey
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from odds_lambda.polymarket_sports import POLYMARKET_SPORTS

logger = structlog.get_logger()

# Request timeout in seconds
REQUEST_TIMEOUT = 30

# Maximum concurrent requests for batch operations
MAX_CONCURRENT_REQUESTS = 10

GAMMA_PAGE_SIZE = 100

# CLOB rejects startTs/endTs windows of 20 days and accepts 14.
MAX_HISTORY_WINDOW = timedelta(days=14)


class PolymarketClient:
    """Client for Polymarket Gamma + CLOB APIs. No authentication required."""

    def __init__(self, gamma_url: str | None = None, clob_url: str | None = None):
        """
        Initialize Polymarket client.

        Args:
            gamma_url: Gamma API base URL (defaults to config)
            clob_url: CLOB API base URL (defaults to config)
        """
        app_settings = get_settings()
        polymarket_config = app_settings.polymarket
        self.gamma_url = gamma_url or polymarket_config.gamma_base_url
        self.clob_url = clob_url or polymarket_config.clob_base_url
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "PolymarketClient":
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    @retry(
        retry=retry_if_exception_type((aiohttp.ClientError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _make_request(
        self, base_url: str, endpoint: str, params: dict | None = None
    ) -> tuple[dict | list, int]:
        """
        Make HTTP request with retry logic.

        Args:
            base_url: Base URL (gamma or clob)
            endpoint: API endpoint path
            params: Query parameters

        Returns:
            Tuple of (response data, response time in ms)

        Raises:
            aiohttp.ClientError: On request failure after retries
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        url = f"{base_url}{endpoint}"
        params = params or {}

        start_time = time.time()

        try:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
            async with self.session.get(url, params=params, timeout=timeout) as response:
                response.raise_for_status()

                data = await response.json()
                elapsed_ms = int((time.time() - start_time) * 1000)

                logger.info(
                    "polymarket_request_success",
                    endpoint=endpoint,
                    status=response.status,
                    elapsed_ms=elapsed_ms,
                )

                return data, elapsed_ms

        except aiohttp.ClientResponseError as e:
            logger.error(
                "polymarket_request_failed",
                endpoint=endpoint,
                status=e.status,
                message=str(e),
            )
            raise
        except Exception as e:
            logger.error(
                "polymarket_request_error",
                endpoint=endpoint,
                error=str(e),
            )
            raise

    async def get_events(
        self,
        sport_key: SportKey,
        active: bool | None = True,
        closed: bool = False,
        limit: int = GAMMA_PAGE_SIZE,
        offset: int = 0,
    ) -> list[dict]:
        """Fetch one page of a sport's events from the Gamma API (at most 100 per page).

        Closed events keep ``active=true``, so pass ``active=None`` to list them.
        """
        sport = POLYMARKET_SPORTS[sport_key]
        params: dict[str, str | int] = {
            "series_id": sport.series_id,
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        if active is not None:
            params["active"] = str(active).lower()
        if sport.tag_id is not None:
            params["tag_id"] = sport.tag_id

        data, response_time = await self._make_request(self.gamma_url, "/events", params=params)

        events = data if isinstance(data, list) else []

        logger.info(
            "polymarket_events_fetched",
            sport=sport_key,
            active=active,
            closed=closed,
            events_count=len(events),
            response_time_ms=response_time,
        )

        return events

    async def get_all_events(
        self, sport_key: SportKey, active: bool | None = True, closed: bool = False
    ) -> list[dict]:
        """Fetch every page of a sport's events, ending on the first empty page.

        Offset paging can repeat an event when the listing shifts mid-fetch, so
        events are keyed by their Polymarket id.
        """
        events: dict[str, dict] = {}
        offset = 0
        while page := await self.get_events(sport_key, active=active, closed=closed, offset=offset):
            offset += len(page)
            events.update({str(event["id"]): event for event in page})
        return list(events.values())

    async def get_price_history(
        self,
        token_id: str,
        interval: str | None = "max",
        fidelity: int = 5,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> list[dict]:
        """
        Fetch price history for a token from CLOB API.

        Args:
            token_id: CLOB token ID (as string)
            interval: Time interval ('max', '1d', '1h', etc.); None to use start_ts/end_ts only
            fidelity: Data point density (1-10)
            start_ts: Start timestamp (Unix seconds)
            end_ts: End timestamp (Unix seconds)

        Returns:
            List of price history dicts [{t: unix_timestamp, p: price}]
        """
        params: dict[str, str | int] = {"market": token_id, "fidelity": fidelity}
        if interval is not None:
            params["interval"] = interval
        if start_ts is not None:
            params["startTs"] = start_ts
        if end_ts is not None:
            params["endTs"] = end_ts

        data, response_time = await self._make_request(
            self.clob_url, "/prices-history", params=params
        )

        history = data.get("history", []) if isinstance(data, dict) else []

        logger.info(
            "price_history_fetched",
            token_id=token_id,
            interval=interval,
            data_points=len(history),
            response_time_ms=response_time,
        )

        return history

    async def get_price_series(
        self, token_id: str, start: datetime, end: datetime, fidelity: int = 1
    ) -> list[tuple[datetime, float]]:
        """Midpoint series for ``[start, end]`` from explicit windows, oldest first.

        ``interval=max`` returns nothing for old markets, so the range is split
        into windows of at most ``MAX_HISTORY_WINDOW``.
        """
        series: list[tuple[datetime, float]] = []
        window_start = start
        while window_start < end:
            window_end = min(window_start + MAX_HISTORY_WINDOW, end)
            history = await self.get_price_history(
                token_id,
                interval=None,
                fidelity=fidelity,
                start_ts=int(window_start.timestamp()),
                end_ts=int(window_end.timestamp()),
            )
            series.extend(
                (datetime.fromtimestamp(point["t"], UTC), point["p"]) for point in history
            )
            window_start = window_end
        series.sort(key=lambda point: point[0])
        return list(dict(series).items())
