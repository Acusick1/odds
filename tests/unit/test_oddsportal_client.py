"""Tests for the OddsPortal HTTP client, using captured fixtures and synthetic feeds."""

from __future__ import annotations

import base64
import json
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
    BookmakerPrices,
    ListingEvent,
    OddsPortalClient,
    OddsPortalError,
    decrypt_payload,
    extract_market,
    parse_archive_path,
    parse_listing,
    parse_results_page,
    select_main_line_window,
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


PROXY_503 = "URL:/ajax-sport-country-tournament-archive_/1/SY30SsKF/X0X1/1/0/page/1/ Status: 503"


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

    async def test_proxy_wrapped_503_is_retried(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        feed = _feed({"odds": {"16": [2.0, 3.0, 4.0]}, "act": {"16": True}})
        client = FakeClient([(200, PROXY_503), (200, encrypt_payload(feed))])

        result = await client.fetch_market(EVENT, "1x2", CONTEXT)

        assert [b.name for b in result] == ["bet365"]
        assert len(client.requests) == 2

    async def test_proxy_wrapped_503_exhausts_retries(self, sleeps: list[float]) -> None:
        client = FakeClient([(200, PROXY_503)] * 3)

        with pytest.raises(OddsPortalError, match=r"after 3 attempts.*match-event.*status 503"):
            await client.fetch_market(EVENT, "1x2", CONTEXT)

        assert len(client.requests) == 3

    async def test_proxy_wrapped_404_fails_immediately(self, sleeps: list[float]) -> None:
        client = FakeClient([(200, "URL:/ajax-x_/1/a/ Status: 404")])

        with pytest.raises(OddsPortalError, match="upstream status 404 for .*match-event"):
            await client.fetch_market(EVENT, "1x2", CONTEXT)

        assert len(client.requests) == 1

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


ARCHIVE = "ajax-sport-country-tournament-archive_/1/AbCdEfGh/X1X2/1/0"
RESULTS_HTML = (
    '<script>self.__next_f.push([1,"1:{\\"url\\":\\"https://www.oddsportal.com/'
    + ARCHIVE
    + '\\"}\\n"])</script>'
)


def _ts(day: int) -> int:
    return int(datetime(2026, 10, day, 15, 0, tzinfo=UTC).timestamp())


def _row(
    home: str, away: str, day: int, status: int = 3, home_result: str = "2", away_result: str = "1"
) -> dict[str, Any]:
    return {
        "home-name": home,
        "away-name": away,
        "date-start-timestamp": _ts(day),
        "status-id": status,
        "homeResult": home_result,
        "awayResult": away_result,
    }


def _archive_feed(rows: list[dict[str, Any]], total: int = 4, one_page: int = 2) -> dict[str, Any]:
    return {"s": 1, "d": {"rows": rows, "total": total, "onePage": one_page}}


class TestResultsParsing:
    def test_archive_path_is_the_proxy_path_without_a_page(self) -> None:
        assert parse_archive_path(RESULTS_HTML) == f"/proxy/{ARCHIVE}"

    def test_page_without_archive_url_raises(self) -> None:
        with pytest.raises(OddsPortalError, match="archive"):
            parse_archive_path(EMPTY_LISTING)

    def test_finished_rows_carry_scores(self) -> None:
        page = parse_results_page(_archive_feed([_row("Arsenal", "Leeds", 10, away_result="0")]))

        [match] = page.matches
        assert (match.home_team, match.away_team) == ("Arsenal", "Leeds")
        assert (match.home_score, match.away_score) == (2, 0)
        assert match.start_time == datetime(2026, 10, 10, 15, 0, tzinfo=UTC)

    def test_cancelled_rows_are_collected_apart_from_finished_ones(self) -> None:
        rows = [_row("A", "B", 10, status=5, home_result="", away_result=""), _row("G", "H", 13)]

        page = parse_results_page(_archive_feed(rows))

        assert [m.home_team for m in page.matches] == ["G"]
        [cancelled] = page.cancelled
        assert (cancelled.home_team, cancelled.away_team) == ("A", "B")
        assert cancelled.start_time == datetime(2026, 10, 10, 15, 0, tzinfo=UTC)

    def test_unfinished_and_unscored_rows_are_skipped(self) -> None:
        rows = [
            _row("A", "B", 10, status=1, home_result="", away_result=""),
            _row("E", "F", 12, home_result="", away_result=""),
            _row("G", "H", 13),
        ]

        page = parse_results_page(_archive_feed(rows))

        assert [m.home_team for m in page.matches] == ["G"]
        assert page.cancelled == []
        assert page.oldest_start == datetime(2026, 10, 10, 15, 0, tzinfo=UTC)

    def test_page_count_rounds_up(self) -> None:
        assert parse_results_page(_archive_feed([], total=101, one_page=50)).page_count == 3

    @pytest.mark.parametrize(
        "feed", [{}, {"d": {}}, {"d": {"rows": [], "total": "x", "onePage": 1}}]
    )
    def test_unexpected_shape_raises(self, feed: dict[str, Any]) -> None:
        with pytest.raises(OddsPortalError, match="archive feed"):
            parse_results_page(feed)


class TestFetchResults:
    async def test_pages_until_the_page_holding_since(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        pages = [
            _archive_feed([_row("A", "B", 20), _row("C", "D", 19)], total=6),
            _archive_feed([_row("E", "F", 18), _row("G", "H", 17)], total=6),
            _archive_feed([_row("I", "J", 16), _row("K", "L", 15)], total=6),
        ]
        client = FakeClient(
            [(200, RESULTS_HTML)] + [(200, encrypt_payload(p)) for p in pages], request_delay=0.5
        )

        fetched = await client.fetch_results(
            "/football/england/premier-league/results/",
            since=datetime(2026, 10, 18, 0, 0, tzinfo=UTC),
        )

        assert [m.home_team for m in fetched.matches] == ["A", "C", "E", "G"]
        assert (fetched.rows_seen, fetched.complete) == (4, True)
        urls = [url for url, _ in client.requests]
        assert urls[1:] == [
            f"https://www.oddsportal.com/proxy/{ARCHIVE}/page/1/",
            f"https://www.oddsportal.com/proxy/{ARCHIVE}/page/2/",
        ]
        assert sleeps == [0.5]

    async def test_stops_at_the_last_page(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        client = FakeClient(
            [
                (200, RESULTS_HTML),
                (200, encrypt_payload(_archive_feed([_row("A", "B", 20)], total=1))),
            ]
        )

        fetched = await client.fetch_results("/x/results/", since=datetime(2020, 1, 1, tzinfo=UTC))

        assert [m.home_team for m in fetched.matches] == ["A"]
        assert len(client.requests) == 2

    async def test_a_failing_later_page_keeps_the_earlier_ones(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        client = FakeClient(
            [
                (200, RESULTS_HTML),
                (200, encrypt_payload(_archive_feed([_row("A", "B", 20)], total=6))),
                (404, ""),
            ]
        )

        fetched = await client.fetch_results("/x/results/", since=datetime(2020, 1, 1, tzinfo=UTC))

        assert [m.home_team for m in fetched.matches] == ["A"]
        assert fetched.complete is False

    async def test_a_failing_first_page_raises(self, sleeps: list[float]) -> None:
        client = FakeClient([(200, RESULTS_HTML), (404, "")])

        with pytest.raises(OddsPortalError):
            await client.fetch_results("/x/results/", since=datetime(2020, 1, 1, tzinfo=UTC))


class TestRecordedResultsFeed:
    def test_archive_path_from_a_real_page(self, oddsportal_fixtures: Path) -> None:
        path = parse_archive_path(_read(oddsportal_fixtures, "mlb_results_page.html"))

        assert path.startswith("/proxy/ajax-sport-country-tournament-archive_/6/")

    def test_rows_from_a_real_feed_page(self, oddsportal_fixtures: Path) -> None:
        feed = json.loads(_read(oddsportal_fixtures, "mlb_results_feed.json"))

        page = parse_results_page(feed)

        assert page.rows_seen == 5
        assert page.page_count == 50
        assert [(m.home_team, m.away_team, m.home_score, m.away_score) for m in page.matches] == [
            ("Cleveland Guardians", "Tampa Bay Rays", 1, 3),
            ("Cleveland Guardians", "Tampa Bay Rays", 3, 7),
            ("Los Angeles Dodgers", "Milwaukee Brewers", 4, 6),
        ]


NBA_NAMES = {
    "16": "bet365",
    "15": "William Hill",
    "74": "Skybet",
    "26": "Betway",
    "14": "10bet",
    "44": "Betfair Exchange",
    "76": "BetVictor",
    "60": "Paddy Power",
    "1281": "BetMGM.uk",
    "625": "Unibetuk",
    "2": "bwin",
    "27": "888sport",
    "979": "Betano.uk",
    "263": "BetUK",
    "841": "Midnite",
    "895": "7BetUK",
    "869": "AllBritishCasino",
    "1009": "SpreadEX",
    "21": "Betfred",
}
NBA_CONTEXT = BookmakerContext(hashed_bookmakers="a" * 32, geo="GB", names=NBA_NAMES)
NBA_EVENT = ListingEvent(
    event_id="hdJcPIgS",
    sport_id=3,
    home_team="Utah Jazz",
    away_team="Portland Trail Blazers",
    start_time=datetime(2026, 10, 31, 19, 0, tzinfo=UTC),
    url="https://www.oddsportal.com/basketball/h2h/portland-x/utah-y/#hdJcPIgS",
)


def _nba_feed(fixtures: Path, market: str, scope: int) -> dict[str, Any]:
    return json.loads((fixtures / f"feed_nba_{market}_scope{scope}.json").read_text())


def _prices_at(
    book: str, line: float | None, prices: tuple[float, ...] = (1.9, 1.9)
) -> BookmakerPrices:
    return BookmakerPrices(book, book, prices, None, None, line)


class TestLadderExtraction:
    @pytest.mark.parametrize("market", ["totals", "spreads"])
    def test_every_line_of_the_ladder_is_read_per_scope(
        self, oddsportal_fixtures: Path, market: str
    ) -> None:
        spec = FEED_MARKETS[f"{market}_ladder"]

        rows = extract_market(_nba_feed(oddsportal_fixtures, market, 1), spec, NBA_NAMES, 1)

        assert len({r.line for r in rows}) > 20
        assert all(r.line is not None and len(r.prices) == 2 for r in rows)

    def test_scopes_split_the_bookmakers(self, oddsportal_fixtures: Path) -> None:
        spec = FEED_MARKETS["totals_ladder"]

        scope1 = {
            r.name
            for r in extract_market(_nba_feed(oddsportal_fixtures, "totals", 1), spec, NBA_NAMES, 1)
        }
        scope2 = {
            r.name
            for r in extract_market(_nba_feed(oddsportal_fixtures, "totals", 2), spec, NBA_NAMES, 2)
        }

        assert {"bet365", "10bet"} <= scope1
        assert {"Skybet", "William Hill", "Paddy Power"} <= scope2
        assert not scope1 & scope2

    def test_a_scope_reads_only_its_own_keys(self, oddsportal_fixtures: Path) -> None:
        spec = FEED_MARKETS["totals_ladder"]

        assert extract_market(_nba_feed(oddsportal_fixtures, "totals", 1), spec, NBA_NAMES, 2) == []

    def test_fixed_market_carries_the_spec_point(self) -> None:
        feed = _feed({"odds": {"16": [1.9, 1.9]}, "act": {"16": True}}, key="E-2-2-0-2.5-0")

        rows = extract_market(feed, FEED_MARKETS["over_under_2_5"], NAMES)

        assert [r.line for r in rows] == [2.5]


class TestSelectMainLineWindow:
    def test_main_line_is_the_one_with_most_books_and_its_window_is_kept(self) -> None:
        rows = [_prices_at(b, 220.5) for b in "ab"] + [_prices_at(b, 221.5) for b in "abc"]
        rows += [_prices_at(b, 222.5) for b in "ab"] + [_prices_at(b, 225.5) for b in "abc"]

        kept = select_main_line_window(rows, 1.0)

        assert sorted({r.line for r in kept}) == [220.5, 221.5, 222.5]

    def test_a_tie_goes_to_the_more_balanced_line(self) -> None:
        rows = [_prices_at(b, 220.5, (1.6, 2.4)) for b in "ab"]
        rows += [_prices_at(b, 221.5, (1.9, 1.95)) for b in "ab"]

        kept = select_main_line_window(rows, 0.0)

        assert {r.line for r in kept} == {221.5}

    def test_main_line_counts_a_book_once(self) -> None:
        rows = [_prices_at("a", 220.5)] * 3 + [_prices_at(b, 221.5) for b in "ab"]

        assert {r.line for r in select_main_line_window(rows, 0.0)} == {221.5}

    def test_no_lines_selects_nothing(self) -> None:
        assert select_main_line_window([], 2.0) == []
        assert select_main_line_window([_prices_at("a", None)], 2.0) == []

    def test_real_ladders_centre_on_the_line_most_books_quote(
        self, oddsportal_fixtures: Path
    ) -> None:
        spec = FEED_MARKETS["spreads_ladder"]
        rows = [
            r
            for scope in spec.scopes
            for r in extract_market(
                _nba_feed(oddsportal_fixtures, "spreads", scope), spec, NBA_NAMES, scope
            )
        ]

        kept = select_main_line_window(rows, spec.line_window)

        assert {r.line for r in kept} == {6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5}
        assert len({r.bookmaker_id for r in kept if r.line == 8.5}) == 16


class TestLadderFetch:
    async def test_requests_every_scope_and_merges_the_bookmakers(
        self, oddsportal_fixtures: Path, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        client = FakeClient(
            [
                (200, encrypt_payload(_nba_feed(oddsportal_fixtures, "totals", scope)))
                for scope in (1, 2)
            ]
        )

        rows = await client.fetch_market(NBA_EVENT, "totals_ladder", NBA_CONTEXT)

        assert [url.split("-")[4:6] for url, _ in client.requests] == [["2", "1"], ["2", "2"]]
        main = {r.name for r in rows if r.line == 231.5}
        assert {"10bet", "Skybet", "William Hill"} <= main
        assert {r.line for r in rows} >= {229.5, 231.5, 233.5}
        assert min(r.line for r in rows if r.line is not None) == 229.5

    async def test_a_book_in_both_scopes_keeps_the_first(
        self, encrypt_payload: Encrypt, sleeps: list[float]
    ) -> None:
        def feed(scope: int, price: float) -> dict[str, Any]:
            market = {
                "odds": {"16": [price, price]},
                "act": {"16": True},
                "handicapValue": "220.50",
            }
            return _feed(market, key=f"E-2-{scope}-0-220.5-0")

        client = FakeClient(
            [(200, encrypt_payload(feed(1, 1.9))), (200, encrypt_payload(feed(2, 2.0)))]
        )

        rows = await client.fetch_market(NBA_EVENT, "totals_ladder", NBA_CONTEXT)

        assert [(r.name, r.prices) for r in rows] == [("bet365", (1.9, 1.9))]


class TestLadderLineIsTheHandicapValue:
    """The feed key strips trailing zeros (``E-5-1-0-1-0`` is handicap 10), so the line is read
    from ``handicapValue``."""

    def test_a_spread_of_ten_is_not_read_as_one(self, oddsportal_fixtures: Path) -> None:
        spec = FEED_MARKETS["spreads_ladder"]
        feed = _nba_feed(oddsportal_fixtures, "spreads", 1)
        assert feed["d"]["oddsdata"]["back"]["E-5-1-0-1-0"]["handicapValue"] == "10.00"

        lines = {r.line for r in extract_market(feed, spec, NBA_NAMES, 1)}

        assert 10.0 in lines

    def test_a_negative_spread_of_ten_keeps_its_sign(self, oddsportal_fixtures: Path) -> None:
        spec = FEED_MARKETS["spreads_ladder"]
        feed = _nba_feed(oddsportal_fixtures, "spreads", 2)
        assert feed["d"]["oddsdata"]["back"]["E-5-2-0--1-0"]["handicapValue"] == "-10.00"

        assert -10.0 in {r.line for r in extract_market(feed, spec, NBA_NAMES, 2)}

    def test_a_total_of_230_is_kept(self, oddsportal_fixtures: Path) -> None:
        spec = FEED_MARKETS["totals_ladder"]
        feed = _nba_feed(oddsportal_fixtures, "totals", 1)
        assert feed["d"]["oddsdata"]["back"]["E-2-1-0-23-0"]["handicapValue"] == "230.00"

        lines = {r.line for r in extract_market(feed, spec, NBA_NAMES, 1)}

        assert 230.0 in lines
        assert 23.0 not in lines

    def test_a_market_without_a_handicap_value_is_skipped(self) -> None:
        feed = _feed({"odds": {"16": [1.9, 1.9]}, "act": {"16": True}}, key="E-2-1-0-23-0")

        assert extract_market(feed, FEED_MARKETS["totals_ladder"], NAMES, 1) == []
