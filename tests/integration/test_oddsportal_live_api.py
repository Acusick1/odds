"""Live-API test that OddsPortal's match feed reaches us past its CDN cache."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPEC_BY_NAME
from odds_lambda.oddsportal_client import MAX_FEED_AGE_SECONDS, OddsPortalClient, _Response

pytestmark = [pytest.mark.integration, pytest.mark.live_api]


class RecordingClient(OddsPortalClient):
    def __init__(self) -> None:
        super().__init__()
        self.responses: list[tuple[str, _Response]] = []

    async def _fetch(self, url: str, headers: dict[str, str]) -> _Response:
        response = await super()._fetch(url, headers)
        self.responses.append((url, response))
        return response


class TestOddsPortalFeedCache:
    async def test_epl_feed_is_fresh_on_first_request(self) -> None:
        async with RecordingClient() as client:
            listed = await client.fetch_listing(
                LEAGUE_SPEC_BY_NAME["england-premier-league"].listing_path
            )
            upcoming = [e for e in listed if e.start_time > datetime.now(UTC)]
            if not upcoming:
                pytest.skip("no upcoming EPL events listed")
            context = await client.fetch_bookmaker_context()
            await client.fetch_market(upcoming[0], "1x2", context)

        feeds = [r for url, r in client.responses if "/match-event/" in url]
        assert feeds[0].age is None or feeds[0].age <= MAX_FEED_AGE_SECONDS
