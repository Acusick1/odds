"""Unit tests for Odds API response handling in TheOddsAPIClient."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from odds_lambda.data_fetcher import TheOddsAPIClient, canonicalize_bookmaker_keys


def _event(bookmaker_keys: list[str]) -> dict[str, Any]:
    return {
        "id": "evt1",
        "sport_key": "soccer_epl",
        "sport_title": "EPL",
        "commence_time": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
        "home_team": "Arsenal",
        "away_team": "Everton",
        "bookmakers": [
            {"key": key, "title": key, "last_update": datetime.now(UTC).isoformat(), "markets": []}
            for key in bookmaker_keys
        ],
    }


class TestCanonicalizeBookmakerKeys:
    def test_maps_odds_api_uk_keys_to_pipeline_keys(self):
        events = [_event(["betfair_ex_uk", "ladbrokes_uk", "betfred_uk", "sport888"])]

        canonicalize_bookmaker_keys(events)

        assert [b["key"] for b in events[0]["bookmakers"]] == [
            "betfair_exchange",
            "ladbrokes",
            "betfred",
            "888sport",
        ]

    def test_leaves_shared_keys_unchanged(self):
        keys = ["pinnacle", "williamhill", "paddypower", "skybet", "coral", "betvictor", "fanduel"]
        events = [_event(keys)]

        canonicalize_bookmaker_keys(events)

        assert [b["key"] for b in events[0]["bookmakers"]] == keys

    def test_event_without_bookmakers(self):
        event = _event([])
        del event["bookmakers"]

        canonicalize_bookmaker_keys([event])

        assert "bookmakers" not in event


class TestGetOddsBookmakers:
    @pytest.mark.asyncio
    async def test_requests_sport_bookmakers_and_returns_pipeline_keys(self, mock_settings):
        mock_settings.data_collection.bookmakers = {
            "soccer_epl": ["pinnacle", "betfair_ex_uk", "sport888"]
        }
        client = TheOddsAPIClient(api_key="test")
        request = AsyncMock(return_value=([_event(["pinnacle", "betfair_ex_uk"])], 10))

        with (
            patch("odds_lambda.data_fetcher.get_settings", return_value=mock_settings),
            patch.object(client, "_make_request", request),
        ):
            response = await client.get_odds("soccer_epl")

        params = request.call_args.kwargs["params"]
        assert params["bookmakers"] == "pinnacle,betfair_ex_uk,sport888"
        assert [b["key"] for b in response.raw_events_data[0]["bookmakers"]] == [
            "pinnacle",
            "betfair_exchange",
        ]


def _response_error(status: int) -> aiohttp.ClientResponseError:
    return aiohttp.ClientResponseError(
        request_info=MagicMock(), history=(), status=status, message=f"HTTP {status}"
    )


class _StubResponse:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status = status
        self.headers: dict[str, str] = {"x-requests-remaining": "100"}
        self._payload = payload if payload is not None else []

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise _response_error(self.status)

    async def json(self) -> Any:
        return self._payload

    async def __aenter__(self) -> "_StubResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _StubSession:
    """Serves queued statuses in order and records the apiKey of every call."""

    def __init__(self, statuses: list[int]) -> None:
        self._statuses = list(statuses)
        self.api_keys: list[str] = []

    def get(self, url: str, params: dict[str, Any], timeout: Any) -> _StubResponse:
        self.api_keys.append(params["apiKey"])
        return _StubResponse(self._statuses.pop(0))


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(TheOddsAPIClient._send.retry, "sleep", AsyncMock())


def _client(statuses: list[int]) -> tuple[TheOddsAPIClient, Any]:
    client = TheOddsAPIClient(api_key="only")
    session = _StubSession(statuses)
    client.session = session  # type: ignore[assignment]
    return client, session


class TestRequestRetries:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [401, 429])
    async def test_auth_and_quota_errors_fail_after_one_call(self, status: int):
        client, session = _client([status])

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await client._make_request("sports", "soccer_epl")

        assert exc_info.value.status == status
        assert len(session.api_keys) == 1

    @pytest.mark.asyncio
    async def test_other_client_errors_are_not_retried(self):
        client, session = _client([422])

        with pytest.raises(aiohttp.ClientResponseError):
            await client._make_request("sports", "soccer_epl")

        assert len(session.api_keys) == 1

    @pytest.mark.asyncio
    async def test_server_errors_retry_then_reraise_original_exception(self):
        client, session = _client([500, 500, 500])

        with pytest.raises(aiohttp.ClientResponseError) as exc_info:
            await client._make_request("sports", "soccer_epl")

        assert exc_info.value.status == 500
        assert len(session.api_keys) == 3

    @pytest.mark.asyncio
    async def test_transient_error_recovers(self):
        client, session = _client([503, 200])

        data, _ = await client._make_request("sports", "soccer_epl")

        assert data == []
        assert len(session.api_keys) == 2


class TestClientKeySource:
    def test_no_key_configured_raises(self, mock_settings):
        mock_settings.api.key = None
        mock_settings.api.keys = None
        with patch("odds_lambda.data_fetcher.get_settings", return_value=mock_settings):
            with pytest.raises(ValueError, match="ODDS_API_KEYS or ODDS_API_KEY"):
                TheOddsAPIClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("sport", "expected"), [("soccer_epl", "a"), ("basketball_nba", "b")])
    async def test_each_sport_spends_its_own_key(self, mock_settings, sport: str, expected: str):
        mock_settings.api.key = None
        mock_settings.api.keys = "a, b"
        with patch("odds_lambda.data_fetcher.get_settings", return_value=mock_settings):
            client = TheOddsAPIClient()
        session = _StubSession([200, 200])
        client.session = session  # type: ignore[assignment]

        await client._make_request(f"sports/{sport}/events", sport)

        assert session.api_keys == [expected]

    @pytest.mark.asyncio
    async def test_a_rejected_key_is_not_replaced_by_another_sports_key(self, mock_settings):
        mock_settings.api.key = None
        mock_settings.api.keys = "a, b"
        with patch("odds_lambda.data_fetcher.get_settings", return_value=mock_settings):
            client = TheOddsAPIClient()
        session = _StubSession([429])
        client.session = session  # type: ignore[assignment]

        with pytest.raises(aiohttp.ClientResponseError):
            await client._make_request("sports/basketball_nba/odds", "basketball_nba")

        assert session.api_keys == ["b"]

    @pytest.mark.asyncio
    async def test_a_sport_whose_key_index_is_not_configured_raises(self, mock_settings):
        mock_settings.api.keys = None
        mock_settings.api.key = "only"
        with patch("odds_lambda.data_fetcher.get_settings", return_value=mock_settings):
            client = TheOddsAPIClient()

        with pytest.raises(ValueError, match="basketball_nba spends Odds API key index 1"):
            await client._make_request("sports/basketball_nba/events", "basketball_nba")


class TestQuotaOnErrors:
    @pytest.mark.asyncio
    async def test_quota_updates_from_error_response_headers(self):
        client, _ = _client([429])

        with pytest.raises(aiohttp.ClientResponseError):
            await client._make_request("sports", "soccer_epl")

        assert client.quota_remaining == 100
