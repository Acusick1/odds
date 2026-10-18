"""Polymarket API client for Gamma (market discovery) and CLOB (prices/order books)."""

import asyncio
import re
import time
from datetime import UTC, datetime, timedelta

import aiohttp
import structlog
from odds_core.config import get_settings
from odds_core.polymarket_models import PolymarketMarketType
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

    async def get_event_by_id(self, event_id: str) -> dict | None:
        """
        Fetch specific event by ID from Gamma API.

        Args:
            event_id: Polymarket event ID

        Returns:
            Event dict or None if not found
        """
        params = {"id": event_id}

        data, response_time = await self._make_request(self.gamma_url, "/events", params=params)

        # API returns list with single event
        events = data if isinstance(data, list) else []
        event = events[0] if events else None

        logger.info(
            "event_fetched",
            event_id=event_id,
            found=event is not None,
            response_time_ms=response_time,
        )

        return event

    async def get_price(self, token_id: str, side: str = "buy") -> float | None:
        """
        Fetch price for a token from CLOB API.

        Args:
            token_id: CLOB token ID (as string)
            side: Order side ('buy' or 'sell')

        Returns:
            Price as float (0.0-1.0) or None if price unavailable

        Note:
            Returns None (not 0.0) when price is missing, since 0.0 is a valid
            price representing an impossible event in prediction markets.
        """
        params = {"token_id": token_id, "side": side}

        data, response_time = await self._make_request(self.clob_url, "/price", params=params)

        if not isinstance(data, dict) or "price" not in data:
            logger.warning(
                "price_missing",
                token_id=token_id,
                side=side,
                response_time_ms=response_time,
            )
            return None

        price = float(data["price"])

        logger.debug(
            "price_fetched",
            token_id=token_id,
            side=side,
            price=price,
            response_time_ms=response_time,
        )

        return price

    async def get_midpoint(self, token_id: str) -> float | None:
        """
        Fetch midpoint price for a token from CLOB API.

        Args:
            token_id: CLOB token ID (as string)

        Returns:
            Midpoint price as float (0.0-1.0) or None if midpoint unavailable

        Note:
            Returns None (not 0.0) when midpoint is missing, since 0.0 is a valid
            price representing an impossible event in prediction markets.
        """
        params = {"token_id": token_id}

        data, response_time = await self._make_request(self.clob_url, "/midpoint", params=params)

        if not isinstance(data, dict) or "mid" not in data:
            logger.warning(
                "midpoint_missing",
                token_id=token_id,
                response_time_ms=response_time,
            )
            return None

        midpoint = float(data["mid"])

        logger.debug(
            "midpoint_fetched",
            token_id=token_id,
            midpoint=midpoint,
            response_time_ms=response_time,
        )

        return midpoint

    async def get_order_book(self, token_id: str) -> dict:
        """
        Fetch order book for a token from CLOB API.

        Args:
            token_id: CLOB token ID (as string)

        Returns:
            Order book dict with {bids: [{price, size}], asks: [{price, size}]}

        Note:
            Order book is NOT sorted by API - caller must sort
        """
        params = {"token_id": token_id}

        data, response_time = await self._make_request(self.clob_url, "/book", params=params)

        logger.debug(
            "order_book_fetched",
            token_id=token_id,
            response_time_ms=response_time,
        )

        return data if isinstance(data, dict) else {"bids": [], "asks": []}

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

    async def get_prices_batch(
        self, token_ids: list[str], max_concurrent: int = MAX_CONCURRENT_REQUESTS
    ) -> dict[str, float | None]:
        """
        Fetch prices for multiple tokens concurrently with rate limiting.

        Args:
            token_ids: List of CLOB token IDs
            max_concurrent: Maximum concurrent requests (default: 10)

        Returns:
            Dict mapping token_id -> price (excludes failed fetches, includes None for missing prices)
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _fetch_with_limit(token_id: str) -> float | None:
            async with sem:
                return await self.get_price(token_id)

        tasks = [_fetch_with_limit(token_id) for token_id in token_ids]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        prices = {}
        for token_id, result in zip(token_ids, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("price_fetch_failed", token_id=token_id, error=str(result))
            else:
                prices[token_id] = result

        logger.info("prices_batch_fetched", requested=len(token_ids), successful=len(prices))

        return prices

    async def get_order_books_batch(
        self, token_ids: list[str], max_concurrent: int = MAX_CONCURRENT_REQUESTS
    ) -> dict[str, dict]:
        """
        Fetch order books for multiple tokens concurrently with rate limiting.

        Args:
            token_ids: List of CLOB token IDs
            max_concurrent: Maximum concurrent requests (default: 10)

        Returns:
            Dict mapping token_id -> order_book (excludes failed fetches)
        """
        sem = asyncio.Semaphore(max_concurrent)

        async def _fetch_with_limit(token_id: str) -> dict:
            async with sem:
                return await self.get_order_book(token_id)

        tasks = [_fetch_with_limit(token_id) for token_id in token_ids]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        books = {}
        for token_id, result in zip(token_ids, results, strict=True):
            if isinstance(result, Exception):
                logger.warning("order_book_fetch_failed", token_id=token_id, error=str(result))
            else:
                books[token_id] = result

        logger.info("order_books_batch_fetched", requested=len(token_ids), successful=len(books))

        return books


def process_order_book(raw_book: dict) -> dict | None:
    """
    Extract derived metrics from raw order book.

    Args:
        raw_book: Raw order book from API {bids: [{price, size}], asks: [{price, size}]}

    Returns:
        Dict with derived metrics or None if either side is empty or book is crossed:
        {
            best_bid: float,
            best_ask: float,
            spread: float,
            midpoint: float,
            bid_depth_total: float,
            ask_depth_total: float,
            imbalance: float,
            weighted_mid: float,
            bid_levels: int,
            ask_levels: int,
        }
    """
    bids = raw_book.get("bids", [])
    asks = raw_book.get("asks", [])

    # Return None if either side is empty
    if not bids or not asks:
        return None

    # Sort bids descending by price, asks ascending (API returns unsorted)
    sorted_bids = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
    sorted_asks = sorted(asks, key=lambda x: float(x["price"]))

    # Extract best prices
    best_bid = float(sorted_bids[0]["price"])
    best_ask = float(sorted_asks[0]["price"])

    # Check for crossed book (best_bid >= best_ask)
    if best_bid >= best_ask:
        logger.warning(
            "crossed_order_book_detected",
            best_bid=best_bid,
            best_ask=best_ask,
            spread=best_ask - best_bid,
        )
        return None

    # Calculate basic metrics
    spread = best_ask - best_bid
    midpoint = (best_bid + best_ask) / 2

    # Calculate depth totals
    bid_depth_total = sum(float(level["size"]) for level in sorted_bids)
    ask_depth_total = sum(float(level["size"]) for level in sorted_asks)

    # Calculate imbalance: (bid_depth - ask_depth) / (bid_depth + ask_depth)
    total_depth = bid_depth_total + ask_depth_total
    imbalance = (bid_depth_total - ask_depth_total) / total_depth if total_depth > 0 else 0.0

    # Calculate weighted midpoint using top-of-book
    bid_size = float(sorted_bids[0]["size"])
    ask_size = float(sorted_asks[0]["size"])
    total_size = bid_size + ask_size
    weighted_mid = (
        (best_bid * ask_size + best_ask * bid_size) / total_size if total_size > 0 else midpoint
    )

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "midpoint": midpoint,
        "bid_depth_total": bid_depth_total,
        "ask_depth_total": ask_depth_total,
        "imbalance": imbalance,
        "weighted_mid": weighted_mid,
        "bid_levels": len(sorted_bids),
        "ask_levels": len(sorted_asks),
    }


def classify_market(question: str, event_title: str) -> tuple[PolymarketMarketType, float | None]:
    """
    Classify Polymarket market type from question and event title.

    Args:
        question: Market question text
        event_title: Event title

    Returns:
        Tuple of (market_type, point_value)
        - Moneyline: (MONEYLINE, None) when question == event_title
        - Spread: (SPREAD, point_value) when matches spread pattern
        - Total: (TOTAL, total_value) when matches O/U pattern
        - Player prop: (PLAYER_PROP, None) when contains stat keywords
        - Other: (OTHER, None) fallback

    Examples:
        >>> classify_market("Lakers vs Celtics", "Lakers vs Celtics")
        (PolymarketMarketType.MONEYLINE, None)

        >>> classify_market("Lakers Spread: -6.5 (+110)", "Lakers vs Celtics")
        (PolymarketMarketType.SPREAD, -6.5)

        >>> classify_market("O/U 215.5", "Lakers vs Celtics")
        (PolymarketMarketType.TOTAL, 215.5)

        >>> classify_market("LeBron James: Points Over 25.5", "Lakers vs Celtics")
        (PolymarketMarketType.PLAYER_PROP, None)
    """
    # Moneyline: exact match
    if question == event_title:
        return PolymarketMarketType.MONEYLINE, None

    # Spread: regex pattern for "Spread: ±X.X"
    spread_match = re.search(r"Spread:\s*([+-]?\d+\.?\d*)", question)
    if spread_match:
        point_value = float(spread_match.group(1))
        return PolymarketMarketType.SPREAD, point_value

    # Total: regex pattern for "O/U X.X"
    total_match = re.search(r"O/U\s+(\d+\.?\d*)", question, re.IGNORECASE)
    if total_match:
        total_value = float(total_match.group(1))
        return PolymarketMarketType.TOTAL, total_value

    # Player prop: stat keyword AFTER colon AND contains Over/Under
    if ":" in question:
        stat_keywords = [
            "points",
            "rebounds",
            "assists",
            "steals",
            "blocks",
            "threes",
            "turnovers",
            "pts",
            "reb",
            "ast",
            "stl",
            "blk",
        ]
        # Split on colon and check if stat keyword is in the part after colon
        colon_index = question.index(":")
        after_colon = question[colon_index:].lower()
        question_lower = question.lower()

        # Check if stat keyword appears after colon AND question contains over/under
        has_stat_after_colon = any(keyword in after_colon for keyword in stat_keywords)
        has_over_under = "over" in question_lower or "under" in question_lower

        if has_stat_after_colon and has_over_under:
            return PolymarketMarketType.PLAYER_PROP, None

    # Fallback
    return PolymarketMarketType.OTHER, None
