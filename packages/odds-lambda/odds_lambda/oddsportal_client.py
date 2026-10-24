"""HTTP client for OddsPortal's listing pages and encrypted odds feed.

No browser is involved: event listings come from the React Server Components
payload embedded in the listing HTML, and odds come from the JSON feed the
site's own pages call. See docs/adr/0009-oddsportal-feed-client.md.
"""

from __future__ import annotations

import asyncio
import base64
import functools
import gzip
import hashlib
import json
import math
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import TracebackType
from typing import Any
from urllib.parse import urldefrag

import aiohttp
import structlog
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger()

BASE_URL = "https://www.oddsportal.com"

# Constants from the site's JS bundle; see docs/adr/0009-oddsportal-feed-client.md.
_PASSWORD = base64.b64decode("Sio4c1EhcCQ3YURfZlIyeVdAZ0huKjNiVnAjc0FkTGRfaw==")
_SALT = base64.b64decode("NWI5YThmMmMzZTZkMWE0YjdjOGU5ZDBmMWEyYjNjNGQ=")
_PBKDF2_ITERATIONS = 1000

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
)
_BASE_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Language": "en-GB"}
_XHR_HEADERS = {
    **_BASE_HEADERS,
    "X-Requested-With": "XMLHttpRequest",
    "x-client-platform": "next-op",
    "x-language": "en",
    "Accept": "*/*",
}

_REQUEST_TIMEOUT_SECONDS = 30
_RETRY_ATTEMPTS = 3
_FEED_ERROR_MAX_RETRIES = 2
_FEED_ERROR_DEFAULT_REFRESH_SECONDS = 20
_FEED_ERROR_MAX_REFRESH_SECONDS = 30

_PUSH_RE = re.compile(r"self\.__next_f\.push\((\[.*?\])\)</script>", re.DOTALL)
_EVENT_KEYS = ("encodeEventId", "home-name", "away-name", "date-start-timestamp", "url", "sport-id")
_ARCHIVE_URL_RE = re.compile(
    r'"url":"' + re.escape(BASE_URL) + r'/(ajax-sport-country-tournament-archive_/[^"]+)"'
)
_PROXY_ERROR_RE = re.compile(r"\AURL:\S+ Status: (\d{3})\s*\Z")
_STATUS_FINISHED = 3
_STATUS_CANCELED = 5


class OddsPortalError(Exception):
    """Raised when OddsPortal returns something the client cannot use."""


class _TransientError(Exception):
    """Retryable HTTP status (429 or 5xx), possibly wrapped in a proxy error body."""


@dataclass(frozen=True)
class ListingEvent:
    event_id: str
    sport_id: int
    home_team: str
    away_team: str
    start_time: datetime
    url: str


@dataclass(frozen=True)
class FinishedMatch:
    home_team: str
    away_team: str
    start_time: datetime
    home_score: int
    away_score: int


@dataclass(frozen=True)
class CancelledMatch:
    home_team: str
    away_team: str
    start_time: datetime


@dataclass(frozen=True)
class ResultsPage:
    matches: list[FinishedMatch]
    cancelled: list[CancelledMatch]
    rows_seen: int
    oldest_start: datetime | None
    page_count: int


@dataclass(frozen=True)
class FetchedResults:
    matches: list[FinishedMatch]
    rows_seen: int
    complete: bool
    cancelled: list[CancelledMatch] = field(default_factory=list)


@dataclass(frozen=True)
class BookmakerContext:
    hashed_bookmakers: str
    geo: str
    names: dict[str, str]


@dataclass(frozen=True)
class FeedMarketSpec:
    """One market's feed coordinates and its pipeline representation.

    ``handicap`` is the string as it appears in the feed key. ``outcomes`` are
    labels in feed order: home, draw, away, over, under.
    """

    betting_type: int
    scope: int
    handicap: str
    pipeline_key: str
    outcomes: tuple[str, ...]
    point: float | None = None


@dataclass(frozen=True)
class BookmakerPrices:
    bookmaker_id: str
    name: str
    prices: tuple[float, ...]
    volumes: tuple[int, ...] | None
    last_change: datetime | None


# Keys are the LeagueSpec market keys.
FEED_MARKETS: dict[str, FeedMarketSpec] = {
    "1x2": FeedMarketSpec(
        betting_type=1,
        scope=2,
        handicap="0",
        pipeline_key="1x2",
        outcomes=("home", "draw", "away"),
    ),
    "over_under_2_5": FeedMarketSpec(
        betting_type=2,
        scope=2,
        handicap="2.5",
        pipeline_key="totals",
        outcomes=("over", "under"),
        point=2.5,
    ),
    "home_away": FeedMarketSpec(
        betting_type=3,
        scope=1,
        handicap="0",
        pipeline_key="h2h",
        outcomes=("home", "away"),
    ),
}


@functools.cache
def _key() -> bytes:
    return hashlib.pbkdf2_hmac("sha256", _PASSWORD, _SALT, _PBKDF2_ITERATIONS, 32)


def decrypt_payload(body: str) -> Any:
    """Decrypt a feed response body and parse it as JSON.

    Bodies that are already plain JSON pass through. Raises ``OddsPortalError``
    for an empty body or any decoding failure.
    """
    stripped = body.strip()
    if not stripped:
        raise OddsPortalError("empty response body")
    try:
        if stripped[0] in "{[":
            return json.loads(stripped)
        outer = base64.b64decode(stripped).decode("latin1")
        ciphertext_b64, iv_hex = outer.rsplit(":", 1)
        decryptor = Cipher(algorithms.AES(_key()), modes.CBC(bytes.fromhex(iv_hex))).decryptor()
        padded = decryptor.update(base64.b64decode(ciphertext_b64)) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        data = unpadder.update(padded) + unpadder.finalize()
        for _ in range(3):
            if data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
        return json.loads(data.decode())
    except OddsPortalError:
        raise
    except Exception as e:
        raise OddsPortalError(
            f"could not decrypt payload ({type(e).__name__}: {e}); "
            "the site's encryption constants may have changed"
        ) from e


def _iter_dicts(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _iter_dicts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_dicts(value)


def _rsc_payload(html: str) -> str:
    chunks: list[str] = []
    for match in _PUSH_RE.finditer(html):
        try:
            item = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if len(item) >= 2 and item[0] == 1 and isinstance(item[1], str):
            chunks.append(item[1])
    return "".join(chunks)


def parse_listing(html: str) -> list[ListingEvent]:
    """Extract upcoming events from a listing page's RSC payload."""
    payload = _rsc_payload(html)
    events: dict[str, ListingEvent] = {}
    for line in payload.split("\n"):
        _, sep, rest = line.partition(":")
        if not sep:
            continue
        try:
            parsed = json.loads(rest)
        except json.JSONDecodeError:
            continue
        for node in _iter_dicts(parsed):
            if not all(key in node for key in _EVENT_KEYS):
                continue
            event_id = str(node["encodeEventId"])
            if event_id in events:
                continue
            events[event_id] = ListingEvent(
                event_id=event_id,
                sport_id=int(node["sport-id"]),
                home_team=str(node["home-name"]),
                away_team=str(node["away-name"]),
                start_time=datetime.fromtimestamp(int(node["date-start-timestamp"]), tz=UTC),
                url=BASE_URL + str(node["url"]),
            )

    if not events:
        error_match = re.search(r'"productErrorType"\s*:\s*"([^"]*)"', payload)
        if error_match is None:
            raise OddsPortalError(
                "listing has no events and no empty-state marker; page format may have changed"
            )
        logger.warning("oddsportal_listing_empty", product_error_type=error_match.group(1))
    return list(events.values())


def parse_archive_path(html: str) -> str:
    """The proxy path of a results page's paginated archive feed, without a page suffix."""
    match = _ARCHIVE_URL_RE.search(_rsc_payload(html))
    if match is None:
        raise OddsPortalError("results page has no archive feed URL; page format may have changed")
    return f"/proxy/{match.group(1)}"


def parse_results_page(feed: Any) -> ResultsPage:
    """Finished and cancelled matches of one archive feed page, with the page's oldest kick-off."""
    data = feed.get("d") if isinstance(feed, dict) else None
    if not isinstance(data, dict):
        raise OddsPortalError("unexpected archive feed shape")
    try:
        rows = data["rows"]
        page_count = math.ceil(int(data["total"]) / int(data["onePage"]))
        starts = [datetime.fromtimestamp(int(row["date-start-timestamp"]), tz=UTC) for row in rows]
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        raise OddsPortalError(f"unexpected archive feed shape: {e!r}") from e

    matches: list[FinishedMatch] = []
    cancelled: list[CancelledMatch] = []
    for row, start in zip(rows, starts, strict=True):
        if row.get("status-id") == _STATUS_CANCELED:
            cancelled.append(
                CancelledMatch(
                    home_team=str(row["home-name"]),
                    away_team=str(row["away-name"]),
                    start_time=start,
                )
            )
            continue
        home, away = str(row.get("homeResult", "")), str(row.get("awayResult", ""))
        if row.get("status-id") != _STATUS_FINISHED or not (home.isdigit() and away.isdigit()):
            continue
        matches.append(
            FinishedMatch(
                home_team=str(row["home-name"]),
                away_team=str(row["away-name"]),
                start_time=start,
                home_score=int(home),
                away_score=int(away),
            )
        )
    return ResultsPage(
        matches=matches,
        cancelled=cancelled,
        rows_seen=len(rows),
        oldest_start=min(starts, default=None),
        page_count=page_count,
    )


def _outcome_values(raw: Any, count: int) -> list[Any] | None:
    """Normalise a per-outcome list or index-keyed dict to a list of ``count`` items."""
    if isinstance(raw, list):
        return raw if len(raw) == count else None
    if isinstance(raw, dict):
        if set(raw) != {str(i) for i in range(count)}:
            return None
        return [raw[str(i)] for i in range(count)]
    return None


def _decimal_prices(raw: Any, count: int) -> tuple[float, ...] | None:
    values = _outcome_values(raw, count)
    if values is None:
        return None
    try:
        prices = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    return prices if all(p > 1.0 for p in prices) else None


def _volumes(raw: Any, count: int) -> tuple[int, ...] | None:
    values = _outcome_values(raw, count)
    if values is None:
        return None
    try:
        return tuple(int(float(v)) for v in values)
    except (TypeError, ValueError):
        return None


def _last_change(raw: Any) -> datetime | None:
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        return None
    try:
        stamps = [int(v) for v in raw if v]
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(max(stamps), tz=UTC) if stamps else None


def extract_market(
    feed: dict[str, Any], spec: FeedMarketSpec, names: dict[str, str]
) -> list[BookmakerPrices]:
    """Read per-bookmaker prices for one market from a decrypted feed.

    Skips suspended bookmakers (``act`` false), unknown bookmaker ids and rows
    with missing, wrongly sized or non-decimal (at or below 1.0) prices.
    """
    key = f"E-{spec.betting_type}-{spec.scope}-0-{spec.handicap}-0"
    data = feed.get("d")
    if not isinstance(data, dict):
        raise OddsPortalError("unexpected feed shape: 'd' is not an object")
    # The site encodes an empty map as [] (or omits it) before any odds are posted.
    oddsdata = data.get("oddsdata")
    if oddsdata in (None, []):
        return []
    if not isinstance(oddsdata, dict):
        raise OddsPortalError("unexpected feed shape: 'd.oddsdata' is not an object")
    back = oddsdata.get("back")
    if back in (None, []):
        return []
    if not isinstance(back, dict):
        raise OddsPortalError("unexpected feed shape: 'd.oddsdata.back' is not an object")
    market = back.get(key)
    if not isinstance(market, dict):
        return []

    odds, active, volume, change_time = (
        v if isinstance(v, dict) else {}
        for v in (market.get(f) for f in ("odds", "act", "volume", "changeTime"))
    )

    result: list[BookmakerPrices] = []
    for bookmaker_id, raw_prices in odds.items():
        bookmaker_id = str(bookmaker_id)
        if active.get(bookmaker_id) is False:
            continue
        name = names.get(bookmaker_id)
        if name is None:
            logger.debug("oddsportal_unknown_bookmaker", bookmaker_id=bookmaker_id)
            continue
        prices = _decimal_prices(raw_prices, len(spec.outcomes))
        if prices is None:
            continue
        raw_volume = volume.get(bookmaker_id)
        result.append(
            BookmakerPrices(
                bookmaker_id=bookmaker_id,
                name=name,
                prices=prices,
                volumes=_volumes(raw_volume, len(spec.outcomes)) if raw_volume else None,
                last_change=_last_change(change_time.get(bookmaker_id)),
            )
        )
    return result


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class OddsPortalClient:
    """Async client owning one HTTP session. Use as ``async with``."""

    def __init__(self, request_delay: float = 1.0) -> None:
        self._request_delay = request_delay
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> OddsPortalClient:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _fetch_text(self, url: str, headers: dict[str, str]) -> tuple[int, str]:
        if self._session is None:
            raise RuntimeError("OddsPortalClient must be used as an async context manager")
        async with self._session.get(url, headers=headers) as response:
            return response.status, await response.text()

    async def _get(self, url: str, headers: dict[str, str]) -> str:
        retrying = AsyncRetrying(
            stop=stop_after_attempt(_RETRY_ATTEMPTS),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(
                (aiohttp.ClientError, asyncio.TimeoutError, _TransientError)
            ),
            sleep=_sleep,
            reraise=True,
        )
        try:
            async for attempt in retrying:
                with attempt:
                    status, text = await self._fetch_text(url, headers)
                    if status == 429 or status >= 500:
                        raise _TransientError(f"HTTP {status}")
                    if status != 200:
                        raise OddsPortalError(f"HTTP {status} for {url}")
                    wrapped = _PROXY_ERROR_RE.match(text)
                    if wrapped is not None:
                        upstream = int(wrapped.group(1))
                        if upstream == 429 or upstream >= 500:
                            raise _TransientError(f"upstream status {upstream}")
                        raise OddsPortalError(f"upstream status {upstream} for {url}")
                    return text
        except OddsPortalError:
            raise
        except (TimeoutError, aiohttp.ClientError, _TransientError) as e:
            raise OddsPortalError(
                f"request failed after {_RETRY_ATTEMPTS} attempts for {url}: "
                f"{type(e).__name__}: {e}"
            ) from e
        raise AssertionError("unreachable")  # pragma: no cover

    async def _get_json(self, url: str, headers: dict[str, str]) -> Any:
        text = await self._get(url, headers)
        try:
            return decrypt_payload(text)
        except OddsPortalError as e:
            raise OddsPortalError(f"{e} (url={url})") from e

    async def fetch_listing(self, path: str) -> list[ListingEvent]:
        html = await self._get(
            BASE_URL + path, {**_BASE_HEADERS, "Accept": "text/html,application/xhtml+xml"}
        )
        return parse_listing(html)

    async def fetch_results(self, path: str, since: datetime) -> FetchedResults:
        """Finished and cancelled matches on a results page, newest first, back to the page holding ``since``.

        A page that fails after the first keeps the earlier pages and marks the result
        incomplete; a failure on the first page raises.
        """
        html = await self._get(
            BASE_URL + path, {**_BASE_HEADERS, "Accept": "text/html,application/xhtml+xml"}
        )
        archive = parse_archive_path(html)
        matches: list[FinishedMatch] = []
        cancelled: list[CancelledMatch] = []
        rows_seen = 0
        page = 1
        while True:
            try:
                feed = await self._get_json(f"{BASE_URL}{archive}/page/{page}/", _XHR_HEADERS)
                results = parse_results_page(feed)
            except OddsPortalError as e:
                if page == 1:
                    raise
                logger.error("oddsportal_results_page_failed", page=page, error=str(e))
                return FetchedResults(matches, rows_seen, complete=False, cancelled=cancelled)
            matches.extend(results.matches)
            cancelled.extend(results.cancelled)
            rows_seen += results.rows_seen
            if (
                results.oldest_start is None
                or results.oldest_start < since
                or page >= results.page_count
            ):
                return FetchedResults(matches, rows_seen, complete=True, cancelled=cancelled)
            page += 1
            await _sleep(self._request_delay)

    async def fetch_bookmaker_context(self) -> BookmakerContext:
        home = await self._get_json(f"{BASE_URL}/proxy/api/home-data", _XHR_HEADERS)
        providers = await self._get_json(
            f"{BASE_URL}/proxy/ajax-providers-bonus-data/0/?logged=0", _XHR_HEADERS
        )
        try:
            return BookmakerContext(
                hashed_bookmakers=str(home["userData"]["hashedBookmakers"]),
                geo=str(home["geoIPcode"]),
                names={str(k): str(v) for k, v in providers["d"]["providersNames"].items()},
            )
        except (KeyError, TypeError, AttributeError) as e:
            raise OddsPortalError(f"unexpected bookmaker context shape: {e!r}") from e

    async def fetch_market(
        self, event: ListingEvent, market_key: str, context: BookmakerContext
    ) -> list[BookmakerPrices]:
        spec = FEED_MARKETS.get(market_key)
        if spec is None:
            raise OddsPortalError(f"unsupported market '{market_key}'")
        url = (
            f"{BASE_URL}/proxy/match-event/1-{event.sport_id}-{event.event_id}-"
            f"{spec.betting_type}-{spec.scope}-{context.hashed_bookmakers}.dat"
            f"?geo={context.geo}&lang=en"
        )
        headers = {**_XHR_HEADERS, "Referer": urldefrag(event.url).url}

        for attempt in range(_FEED_ERROR_MAX_RETRIES + 1):
            await _sleep(self._request_delay)
            feed = await self._get_json(url, headers)
            data = feed.get("d") if isinstance(feed, dict) else None
            if not isinstance(data, dict):
                raise OddsPortalError(f"unexpected feed shape for {url}")
            if "E" not in data:
                return extract_market(feed, spec, context.names)
            refresh = feed.get("refresh", _FEED_ERROR_DEFAULT_REFRESH_SECONDS)
            delay = min(
                refresh
                if isinstance(refresh, int | float)
                else _FEED_ERROR_DEFAULT_REFRESH_SECONDS,
                _FEED_ERROR_MAX_REFRESH_SECONDS,
            )
            logger.warning(
                "oddsportal_feed_error",
                event_id=event.event_id,
                market=market_key,
                error=data["E"],
                attempt=attempt + 1,
            )
            if attempt < _FEED_ERROR_MAX_RETRIES:
                await _sleep(delay)
        raise OddsPortalError(
            f"feed error '{data['E']}' persisted after {_FEED_ERROR_MAX_RETRIES} retries for {url}"
        )
