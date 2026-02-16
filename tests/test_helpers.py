"""Reusable test helpers and stub implementations for API clients."""

from __future__ import annotations

from datetime import UTC, datetime

from odds_core.api_models import EventsResponse, HistoricalOddsResponse, OddsResponse


class StubOddsClient:
    """
    Reusable stub for TheOddsAPIClient with configurable current odds responses.

    This stub implements the async context manager protocol and provides a
    simple, explicit way to mock API client behavior for current odds fetching.

    Example:
        >>> response = OddsResponse(events=[...], raw_events_data=[...], ...)
        >>> client = StubOddsClient(response)
        >>> async with client as c:
        ...     result = await c.get_odds(sport="basketball_nba")
        ...     assert result == response
    """

    def __init__(
        self,
        odds_response: OddsResponse,
        events_response: EventsResponse | None = None,
    ):
        """
        Initialize stub with responses to return.

        Args:
            odds_response: The OddsResponse to return from get_odds()
            events_response: The EventsResponse to return from get_events(). Defaults to empty.
        """
        self._odds_response = odds_response
        self._events_response = events_response or EventsResponse(
            events=[],
            response_time_ms=0,
            quota_remaining=None,
            timestamp=datetime.now(UTC),
        )

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit async context manager."""
        return False

    async def get_odds(self, *args, **kwargs) -> OddsResponse:
        """Return the configured odds response."""
        return self._odds_response

    async def get_events(self, *args, **kwargs) -> EventsResponse:
        """Return the configured events response."""
        return self._events_response


class StubHistoricalOddsClient:
    """
    Stub for historical odds API client with call-based response variation.

    This stub cycles through a list of responses based on call count, useful
    for testing pagination or multiple API calls with different results.

    Example:
        >>> responses = [
        ...     HistoricalOddsResponse(events=[event1], ...),
        ...     HistoricalOddsResponse(events=[event2], ...),
        ... ]
        >>> client = StubHistoricalOddsClient(responses)
        >>> async with client as c:
        ...     result1 = await c.get_historical_odds(...)  # Returns responses[0]
        ...     result2 = await c.get_historical_odds(...)  # Returns responses[1]
        ...     result3 = await c.get_historical_odds(...)  # Returns responses[1] again
    """

    def __init__(self, responses: list[HistoricalOddsResponse]):
        """
        Initialize stub with list of responses.

        Args:
            responses: List of HistoricalOddsResponse objects to return.
                       Cycles through based on call count.
        """
        if not responses:
            raise ValueError("Must provide at least one response")
        self._responses = responses
        self._call_count = 0

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit async context manager."""
        return False

    async def get_historical_odds(self, *args, **kwargs) -> HistoricalOddsResponse:
        """
        Return a response based on call count.

        First call returns responses[0], second call returns responses[1], etc.
        If call count exceeds list length, returns the last response repeatedly.

        Args and kwargs are accepted but ignored for stub simplicity.
        """
        # Use min to avoid index out of bounds - return last response if exceeded
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return response

    def reset_call_count(self):
        """Reset the call counter to 0 (useful for test isolation)."""
        self._call_count = 0
