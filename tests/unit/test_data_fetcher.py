"""Unit tests for Odds API response handling in TheOddsAPIClient."""

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

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
