"""The Odds API's NBA games resolve to the events the other NBA sources created."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.api_models import ApiEvent
from odds_core.models import Event, EventExternalId
from odds_core.vocabulary import Source
from odds_lambda.odds_api_events import odds_api_finder, resolve_odds_api_events
from odds_lambda.storage.event_resolver import ListedGame, resolve_event
from sqlalchemy import func, select

NBA = "basketball_nba"
TIP_OFF = (datetime.now(UTC) + timedelta(days=1)).replace(
    hour=23, minute=10, second=0, microsecond=0
)


def _listing(source: Source, external_id: str | None, home: str, away: str) -> ListedGame:
    return ListedGame(
        source=source,
        sport_key=NBA,
        sport_title="NBA",
        home_team=home,
        away_team=away,
        commence_time=TIP_OFF,
        external_id=external_id,
    )


def _api_event(odds_api_id: str, home: str, away: str, start: datetime) -> ApiEvent:
    return ApiEvent(
        odds_api_id=odds_api_id,
        sport_key=NBA,
        sport_title="NBA",
        commence_time=start,
        home_team=home,
        away_team=away,
    )


class TestOddsApiNbaEvents:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("home", "away"),
        [
            ("Utah Jazz", "Portland Trail Blazers"),
            ("Los Angeles Clippers", "Los Angeles Lakers"),
        ],
    )
    async def test_a_game_the_other_sources_stored_is_one_event(
        self, pglite_async_session, home: str, away: str
    ) -> None:
        for source, external_id in [(Source.ODDSPORTAL, "op-1"), (Source.BETFAIR, "1.234")]:
            await resolve_event(pglite_async_session, _listing(source, external_id, home, away))
        listed = [_api_event("api-1", home, away, TIP_OFF + timedelta(minutes=5))]

        finder = await odds_api_finder(NBA, listed)
        result = await resolve_odds_api_events(pglite_async_session, finder, listed)

        assert (result.created, result.matched, result.errors) == (0, 1, [])
        events = (await pglite_async_session.execute(select(func.count(Event.id)))).scalar_one()
        assert events == 1
        sources = (await pglite_async_session.execute(select(EventExternalId.source))).scalars()
        assert set(sources) == {Source.ODDSPORTAL, Source.BETFAIR, Source.ODDS_API}

    @pytest.mark.asyncio
    async def test_back_to_back_listings_of_one_pair_stay_distinct_games(
        self, pglite_async_session
    ) -> None:
        home, away = "Utah Jazz", "Portland Trail Blazers"
        listed = [
            _api_event("api-1", home, away, TIP_OFF),
            _api_event("api-2", away, home, TIP_OFF + timedelta(days=2)),
        ]

        finder = await odds_api_finder(NBA, listed)
        result = await resolve_odds_api_events(pglite_async_session, finder, listed)

        assert (result.created, result.matched) == (2, 0)
