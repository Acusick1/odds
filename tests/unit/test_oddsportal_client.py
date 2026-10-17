"""Tests for the OddsPortal HTTP client, using captured fixtures and synthetic feeds."""

from __future__ import annotations

import base64
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from odds_lambda import oddsportal_client
from odds_lambda.oddsportal_client import (
    FEED_MARKETS,
    BookmakerContext,
    ListingEvent,
    OddsPortalClient,
    OddsPortalError,
    decrypt_payload,
    extract_market,
    parse_listing,
)
from structlog.testing import capture_logs

Encrypt = Callable[..., str]

NAMES = {"16": "bet365", "44": "Betfair Exchange", "417": "Betway"}
CONTEXT = BookmakerContext(hashed_bookmakers="a" * 32, geo="GB", names=NAMES)
EVENT = ListingEvent(
    event_id="2wo3m634",
    sport_id=1,
    home_team="Everton",
    away_team="Chelsea",
    start_time=datetime(2026, 10, 17, 11, 30, tzinfo=UTC),
    url="https://www.oddsportal.com/football/h2h/chelsea-x/everton-y/#2wo3m634",
)
KEY_1X2 = "E-1-2-0-0-0"
EMPTY_LISTING = (
    '<script>self.__next_f.push([1,"1:{\\"productErrorType\\":\\"no_data_now\\"}\\n"])</script>'
)


def _feed(market: dict[str, Any], key: str = KEY_1X2) -> dict[str, Any]:
    return {"s": 1, "d": {"oddsdata": {"back": {key: market}, "lay": {}}}}


def _read(fixtures: Path, name: str) -> str:
    return (fixtures / name).read_text()


class TestDecryptPayload:
    def test_round_trip(self, encrypt_payload: Encrypt) -> None:
        assert decrypt_payload(encrypt_payload({"a": [1, 2]})) == {"a": [1, 2]}

    def test_gzipped_round_trip(self, encrypt_payload: Encrypt) -> None:
        assert decrypt_payload(encrypt_payload({"a": 1}, gzipped=True)) == {"a": 1}

    def test_plain_json_passes_through(self) -> None:
        assert decrypt_payload('{"a": 1}') == {"a": 1}
        assert decrypt_payload("[1]") == [1]

    @pytest.mark.parametrize("body", ["", "   \n"])
    def test_empty_body_raises(self, body: str) -> None:
        with pytest.raises(OddsPortalError, match="empty"):
            decrypt_payload(body)

    def test_garbage_raises(self) -> None:
        with pytest.raises(OddsPortalError, match="could not decrypt"):
            decrypt_payload(base64.b64encode(b"not:valid").decode())


class TestFixtures:
    def test_home_data(self, oddsportal_fixtures: Path) -> None:
        home = decrypt_payload(_read(oddsportal_fixtures, "home_data.enc"))

        assert home["userData"]["hashedBookmakers"] == "2f97c08752e51a3424ea2efa16c1cc0e"
        assert home["geoIPcode"] == "GB"

    def test_providers(self, oddsportal_fixtures: Path) -> None:
        names = decrypt_payload(_read(oddsportal_fixtures, "providers.enc"))["d"]["providersNames"]

        assert names["16"] == "bet365"
        assert names["44"] == "Betfair Exchange"

    @pytest.mark.parametrize(
        ("fixture", "market", "outcomes"),
        [
            ("feed_epl_1x2.enc", "1x2", 3),
            ("feed_epl_over_under.enc", "over_under_2_5", 2),
            ("feed_mlb_home_away.enc", "home_away", 2),
        ],
    )
    def test_feed_extraction(
        self, oddsportal_fixtures: Path, fixture: str, market: str, outcomes: int
    ) -> None:
        names = decrypt_payload(_read(oddsportal_fixtures, "providers.enc"))["d"]["providersNames"]
        feed = decrypt_payload(_read(oddsportal_fixtures, fixture))

        bookmakers = extract_market(feed, FEED_MARKETS[market], names)

        assert bookmakers
        for bk in bookmakers:
            assert len(bk.prices) == outcomes
            assert all(p > 1.0 for p in bk.prices)

    def test_epl_listing(self, oddsportal_fixtures: Path) -> None:
        events = parse_listing(_read(oddsportal_fixtures, "epl_listing.html"))

        assert len(events) == 5
        assert {e.sport_id for e in events} == {1}
        first = events[0]
        assert first.home_team == "Everton"
        assert first.away_team == "Chelsea"
        assert first.start_time.tzinfo is not None
        assert first.url.startswith("https://www.oddsportal.com/football/h2h/")
        assert first.url.endswith(f"#{first.event_id}")

    def test_mlb_listing(self, oddsportal_fixtures: Path) -> None:
        events = parse_listing(_read(oddsportal_fixtures, "mlb_listing.html"))

        assert len(events) == 2
        assert {e.sport_id for e in events} == {6}

    def test_no_data_listing_is_empty_and_warns(self, oddsportal_fixtures: Path) -> None:
        with capture_logs() as logs:
            events = parse_listing(_read(oddsportal_fixtures, "listing_no_data.html"))

        assert events == []
        assert any(
            log["event"] == "oddsportal_listing_empty"
            and log["product_error_type"] == "no_data_now"
            for log in logs
        )

    @pytest.mark.parametrize(
        "html",
        [
            "<html><body>no push scripts</body></html>",
            '<script>self.__next_f.push([1,"1:{\\"foo\\":1}\\n"])</script>',
        ],
    )
    def test_listing_without_events_or_marker_raises(self, html: str) -> None:
        with pytest.raises(OddsPortalError, match="page format may have changed"):
            parse_listing(html)

    def test_listing_events_are_unique(self, oddsportal_fixtures: Path) -> None:
        events = parse_listing(_read(oddsportal_fixtures, "epl_listing.html"))

        assert len({e.event_id for e in events}) == len(events)


class TestExtractMarket:
    def test_dict_and_list_price_shapes(self) -> None:
        feed = _feed(
            {
                "odds": {"16": [2.0, 3.0, 4.0], "417": {"0": 2.1, "1": 3.1, "2": 4.1}},
                "act": {"16": True, "417": True},
            }
        )

        result = extract_market(feed, FEED_MARKETS["1x2"], NAMES)

        assert {b.name: b.prices for b in result} == {
            "bet365": (2.0, 3.0, 4.0),
            "Betway": (2.1, 3.1, 4.1),
        }

    def test_inactive_bookmaker_skipped(self) -> None:
        feed = _feed(
            {
                "odds": {"16": [2.0, 3.0, 4.0], "417": [2.1, 3.1, 4.1]},
                "act": {"16": False, "417": True},
            }
        )

        assert [b.name for b in extract_market(feed, FEED_MARKETS["1x2"], NAMES)] == ["Betway"]

    def test_unknown_bookmaker_skipped(self) -> None:
        feed = _feed({"odds": {"9999": [2.0, 3.0, 4.0]}, "act": {"9999": True}})

        assert extract_market(feed, FEED_MARKETS["1x2"], NAMES) == []

    def test_missing_market_returns_empty(self) -> None:
        feed = _feed({"odds": {"16": [2.0, 2.0]}}, key="E-2-2-0-2.5-0")

        assert extract_market(feed, FEED_MARKETS["1x2"], NAMES) == []

    @pytest.mark.parametrize(
        "feed",
        [
            {"d": {}},
            {"d": {"oddsdata": None}},
            {"d": {"oddsdata": []}},
            {"d": {"oddsdata": {"back": None}}},
            {"d": {"oddsdata": {"back": []}}},
        ],
    )
    def test_no_odds_posted_yet_is_empty(self, feed: dict[str, Any]) -> None:
        assert extract_market(feed, FEED_MARKETS["1x2"], NAMES) == []

    @pytest.mark.parametrize(
        "feed",
        [
            {"d": None},
            {"d": []},
            {"d": {"oddsdata": "x"}},
            {"d": {"oddsdata": {"back": 3}}},
        ],
    )
    def test_structurally_wrong_feed_raises(self, feed: dict[str, Any]) -> None:
        with pytest.raises(OddsPortalError, match="unexpected feed shape"):
            extract_market(feed, FEED_MARKETS["1x2"], NAMES)

    @pytest.mark.parametrize(
        "prices", [[2.0, 3.0], [2.0, 0, 4.0], [2.0, None, 4.0], {"0": 2.0, "1": 3.0}, None]
    )
    def test_bad_prices_skipped(self, prices: Any) -> None:
        feed = _feed({"odds": {"16": prices}, "act": {"16": True}})

        assert extract_market(feed, FEED_MARKETS["1x2"], NAMES) == []

    def test_volumes_and_change_time(self) -> None:
        feed = _feed(
            {
                "odds": {"44": [2.0, 3.0, 4.0], "16": [2.0, 3.0, 4.0]},
                "act": {"44": True, "16": True},
                "volume": {"44": {"0": 100, "1": 200, "2": 300}},
                "changeTime": {"44": {"0": 1_760_000_000, "1": 1_760_000_100, "2": 1_760_000_050}},
            }
        )

        by_name = {b.name: b for b in extract_market(feed, FEED_MARKETS["1x2"], NAMES)}

        assert by_name["Betfair Exchange"].volumes == (100, 200, 300)
        assert by_name["Betfair Exchange"].last_change == datetime.fromtimestamp(
            1_760_000_100, tz=UTC
        )
        assert by_name["bet365"].volumes is None
        assert by_name["bet365"].last_change is None

    def test_empty_list_volume_is_ignored(self) -> None:
        feed = _feed({"odds": {"16": [2.0, 3.0, 4.0]}, "volume": [], "act": []})

        assert [b.volumes for b in extract_market(feed, FEED_MARKETS["1x2"], NAMES)] == [None]


class FakeClient(OddsPortalClient):
    """Client whose HTTP layer replays scripted ``(status, body)`` responses."""

    def __init__(self, responses: list[Any], request_delay: float = 0.0) -> None:
        super().__init__(request_delay=request_delay)
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, str]]] = []

    async def _fetch_text(self, url: str, headers: dict[str, str]) -> tuple[int, str]:
        self.requests.append((url, headers))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(oddsportal_client, "_sleep", fake_sleep)
    return recorded


class TestClientRequests:
    async def test_fetch_market_request_shape(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        feed = _feed({"odds": {"16": [2.0, 3.0, 4.0]}, "act": {"16": True}})
        client = FakeClient([(200, encrypt_payload(feed))], request_delay=1.5)

        result = await client.fetch_market(EVENT, "1x2", CONTEXT)

        assert [b.name for b in result] == ["bet365"]
        url, headers = client.requests[0]
        assert url == (
            "https://www.oddsportal.com/proxy/match-event/"
            f"1-1-2wo3m634-1-2-{'a' * 32}.dat?geo=GB&lang=en"
        )
        assert headers["Referer"] == (
            "https://www.oddsportal.com/football/h2h/chelsea-x/everton-y/"
        )
        assert headers["X-Requested-With"] == "XMLHttpRequest"
        assert headers["x-client-platform"] == "next-op"
        assert headers["x-language"] == "en"
        assert headers["Accept"] == "*/*"
        assert headers["Accept-Language"] == "en-GB"
        assert "Chrome/" in headers["User-Agent"]
        assert sleeps == [1.5]

    async def test_over_under_url(self, encrypt_payload: Encrypt, sleeps: list[float]) -> None:
        feed = _feed({"odds": {}}, key="E-2-2-0-2.5-0")
        client = FakeClient([(200, encrypt_payload(feed))])

        assert await client.fetch_market(EVENT, "over_under_2_5", CONTEXT) == []
        assert "-2-2-" in client.requests[0][0]

    async def test_error_envelope_then_success(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        envelope = {"s": 1, "d": {"E": "dcapiexc"}, "refresh": 20}
        feed = _feed({"odds": {"16": [2.0, 3.0, 4.0]}, "act": {"16": True}})
        client = FakeClient([(200, encrypt_payload(envelope)), (200, encrypt_payload(feed))])

        result = await client.fetch_market(EVENT, "1x2", CONTEXT)

        assert len(result) == 1
        assert len(client.requests) == 2
        assert 20 in sleeps

    async def test_error_envelope_refresh_is_capped(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        envelope = {"s": 1, "d": {"E": "x"}, "refresh": 600}
        feed = _feed({"odds": {}})
        client = FakeClient([(200, encrypt_payload(envelope)), (200, encrypt_payload(feed))])

        await client.fetch_market(EVENT, "1x2", CONTEXT)

        assert 30 in sleeps
        assert 600 not in sleeps

    async def test_persistent_error_envelope_raises_after_two_retries(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        envelope = encrypt_payload({"s": 1, "d": {"E": "dcapiexc"}})
        client = FakeClient([(200, envelope)] * 3)

        with pytest.raises(OddsPortalError, match="dcapiexc"):
            await client.fetch_market(EVENT, "1x2", CONTEXT)

        assert len(client.requests) == 3

    async def test_unsupported_market_raises(self) -> None:
        with pytest.raises(OddsPortalError, match="unsupported market"):
            await FakeClient([]).fetch_market(EVENT, "spreads", CONTEXT)

    async def test_503_is_retried(self, sleeps: list[float]) -> None:
        client = FakeClient([(503, ""), (503, ""), (200, EMPTY_LISTING)])

        assert await client.fetch_listing("/football/england/premier-league/") == []
        assert len(client.requests) == 3

    async def test_exhausted_retries_raise(self, sleeps: list[float]) -> None:
        client = FakeClient([(503, "")] * 3)

        with pytest.raises(OddsPortalError, match="after 3 attempts"):
            await client.fetch_listing("/x/")

    async def test_connection_errors_are_retried(self, sleeps: list[float]) -> None:
        client = FakeClient(
            [aiohttp.ClientConnectionError("boom"), TimeoutError(), (200, EMPTY_LISTING)]
        )

        await client.fetch_listing("/x/")

        assert len(client.requests) == 3

    async def test_429_is_retried(self, sleeps: list[float]) -> None:
        client = FakeClient([(429, ""), (200, EMPTY_LISTING)])

        await client.fetch_listing("/x/")

        assert len(client.requests) == 2

    async def test_404_raises_without_retry(self, sleeps: list[float]) -> None:
        client = FakeClient([(404, "")])

        with pytest.raises(OddsPortalError, match="HTTP 404"):
            await client.fetch_listing("/x/")

        assert len(client.requests) == 1

    async def test_undecryptable_feed_raises_with_url(self, sleeps: list[float]) -> None:
        client = FakeClient([(200, "")])

        with pytest.raises(OddsPortalError, match="match-event"):
            await client.fetch_market(EVENT, "1x2", CONTEXT)

    async def test_bookmaker_context(self, oddsportal_fixtures: Path) -> None:
        client = FakeClient(
            [
                (200, _read(oddsportal_fixtures, "home_data.enc")),
                (200, _read(oddsportal_fixtures, "providers.enc")),
            ]
        )

        context = await client.fetch_bookmaker_context()

        assert context.hashed_bookmakers == "2f97c08752e51a3424ea2efa16c1cc0e"
        assert context.geo == "GB"
        assert context.names["16"] == "bet365"
        assert client.requests[0][0].endswith("/proxy/api/home-data")
        assert client.requests[1][0].endswith("/proxy/ajax-providers-bonus-data/0/?logged=0")
        assert client.requests[0][1]["X-Requested-With"] == "XMLHttpRequest"

    async def test_bookmaker_context_bad_shape_raises(self, encrypt_payload: Encrypt) -> None:
        client = FakeClient([(200, encrypt_payload({})), (200, encrypt_payload({}))])

        with pytest.raises(OddsPortalError, match="context"):
            await client.fetch_bookmaker_context()

    async def test_fetch_listing_parses_html(self, oddsportal_fixtures: Path) -> None:
        client = FakeClient([(200, _read(oddsportal_fixtures, "epl_listing.html"))])

        events = await client.fetch_listing("/football/england/premier-league/")

        assert len(events) == 5
        assert (
            client.requests[0][0] == "https://www.oddsportal.com/football/england/premier-league/"
        )

    async def test_fetch_text_requires_context_manager(self) -> None:
        with pytest.raises(RuntimeError):
            await OddsPortalClient()._fetch_text("http://x", {})

    async def test_context_manager_opens_and_closes_session(self) -> None:
        async with OddsPortalClient() as client:
            assert client._session is not None
            session = client._session
        assert session.closed
