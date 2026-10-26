"""
Fixtures for integration tests.

This module contains shared fixtures for integration testing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.odds_math import american_to_decimal
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda import collection_calendar
from odds_lambda.jobs import fetch_betfair_exchange, fetch_oddsportal
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import (
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    make_epl_event,
    make_tagged_snapshot,
    store_tagged,
    three_way_bookmaker,
)


@pytest.fixture
def job_session(
    pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> AsyncSession:
    """The test session, handed to the OddsPortal and Betfair jobs as their session."""

    class _Maker:
        def __call__(self) -> _Maker:
            return self

        async def __aenter__(self) -> AsyncSession:
            return pglite_async_session

        async def __aexit__(self, *args: object) -> bool:
            return False

    for module in (fetch_oddsportal, fetch_betfair_exchange, collection_calendar):
        monkeypatch.setattr(module, "async_session_maker", _Maker())
    return pglite_async_session


@pytest.fixture
async def test_events_with_odds(pglite_async_session):
    """
    Create test events with odds snapshots for predictable test data.

    Creates 8 events with:
    - FINAL status and scores
    - Opening odds snapshots (48h before)
    - Closing odds snapshots (0.5h before)
    - Bookmakers: pinnacle, fanduel, draftkings
    """
    events = []
    base_time = datetime(2024, 10, 15, 19, 0, tzinfo=UTC)
    bookmakers = ["pinnacle", "fanduel", "draftkings"]

    # Create 8 test events
    for i in range(8):
        commence_time = base_time + timedelta(days=i)

        event = Event(
            id=f"test_event_{i}",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=commence_time,
            home_team=f"Home Team {i}",
            away_team=f"Away Team {i}",
            status=EventStatus.FINAL,
            home_score=100 + i * 2,
            away_score=95 + i,
            completed_at=commence_time + timedelta(hours=3),
        )
        pglite_async_session.add(event)
        events.append(event)

        # Opening odds base prices (will move by closing)
        opening_home_price = -110 + i * 5
        opening_away_price = -110 - i * 5

        # Closing prices (line movement)
        closing_home_price = opening_home_price - 10  # Line moves toward home
        closing_away_price = opening_away_price + 10

        # Create opening odds snapshot (48h before)
        opening_time = commence_time - timedelta(hours=48)
        opening_raw_data = {
            "bookmakers": [
                {
                    "key": bookmaker,
                    "title": bookmaker.title(),
                    "last_update": opening_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": event.home_team, "price": opening_home_price},
                                {"name": event.away_team, "price": opening_away_price},
                            ],
                        }
                    ],
                }
                for bookmaker in bookmakers
            ]
        }
        opening_snapshot = OddsSnapshot(
            event_id=event.id,
            snapshot_time=opening_time,
            raw_data=opening_raw_data,
            bookmaker_count=3,
            fetch_tier="opening",
            hours_until_commence=48.0,
        )
        pglite_async_session.add(opening_snapshot)

        # Create pregame odds snapshot (7.5h before) - falls in [3h, 12h] decision window
        pregame_time = commence_time - timedelta(hours=7.5)
        pregame_home_price = opening_home_price - 5
        pregame_away_price = opening_away_price + 5
        pregame_raw_data = {
            "bookmakers": [
                {
                    "key": bookmaker,
                    "title": bookmaker.title(),
                    "last_update": pregame_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": event.home_team, "price": pregame_home_price},
                                {"name": event.away_team, "price": pregame_away_price},
                            ],
                        }
                    ],
                }
                for bookmaker in bookmakers
            ]
        }
        pregame_snapshot = OddsSnapshot(
            event_id=event.id,
            snapshot_time=pregame_time,
            raw_data=pregame_raw_data,
            bookmaker_count=3,
            fetch_tier="pregame",
            hours_until_commence=7.5,
        )
        pglite_async_session.add(pregame_snapshot)

        # Create closing odds snapshot (0.5h before)
        closing_time = commence_time - timedelta(hours=0.5)
        closing_raw_data = {
            "bookmakers": [
                {
                    "key": bookmaker,
                    "title": bookmaker.title(),
                    "last_update": closing_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": event.home_team, "price": closing_home_price},
                                {"name": event.away_team, "price": closing_away_price},
                            ],
                        }
                    ],
                }
                for bookmaker in bookmakers
            ]
        }
        closing_snapshot = OddsSnapshot(
            event_id=event.id,
            snapshot_time=closing_time,
            raw_data=closing_raw_data,
            bookmaker_count=3,
            fetch_tier="closing",
            hours_until_commence=0.5,
        )
        pglite_async_session.add(closing_snapshot)

        # Also create Odds records for direct querying (used by some functions)
        for bookmaker in bookmakers:
            # Home outcome - opening
            home_odds_opening = Odds(
                event_id=event.id,
                bookmaker_key=bookmaker,
                bookmaker_title=bookmaker.title(),
                market_key="h2h",
                outcome_name=event.home_team,
                price_decimal=american_to_decimal(opening_home_price),
                point=None,
                odds_timestamp=opening_time,
                last_update=opening_time,
            )
            pglite_async_session.add(home_odds_opening)

            # Away outcome - opening
            away_odds_opening = Odds(
                event_id=event.id,
                bookmaker_key=bookmaker,
                bookmaker_title=bookmaker.title(),
                market_key="h2h",
                outcome_name=event.away_team,
                price_decimal=american_to_decimal(opening_away_price),
                point=None,
                odds_timestamp=opening_time,
                last_update=opening_time,
            )
            pglite_async_session.add(away_odds_opening)

            # Home outcome - closing
            home_odds_closing = Odds(
                event_id=event.id,
                bookmaker_key=bookmaker,
                bookmaker_title=bookmaker.title(),
                market_key="h2h",
                outcome_name=event.home_team,
                price_decimal=american_to_decimal(closing_home_price),
                point=None,
                odds_timestamp=closing_time,
                last_update=closing_time,
            )
            pglite_async_session.add(home_odds_closing)

            # Away outcome - closing
            away_odds_closing = Odds(
                event_id=event.id,
                bookmaker_key=bookmaker,
                bookmaker_title=bookmaker.title(),
                market_key="h2h",
                outcome_name=event.away_team,
                price_decimal=american_to_decimal(closing_away_price),
                point=None,
                odds_timestamp=closing_time,
                last_update=closing_time,
            )
            pglite_async_session.add(away_odds_closing)

    await pglite_async_session.commit()

    # Refresh events to get IDs
    for event in events:
        await pglite_async_session.refresh(event)

    return events


@pytest.fixture
async def live_twin_with_rows(pglite_async_session):
    """An op_live_ event owning a snapshot, odds, paper trade and match brief."""
    from odds_core.match_brief_models import BriefDecision, MatchBrief
    from odds_core.paper_trade_models import PaperTrade

    session = pglite_async_session
    session.add(make_epl_event(TWIN_EVENT_ID))
    await session.flush()

    snapshot = OddsSnapshot(
        event_id=TWIN_EVENT_ID,
        snapshot_time=TWIN_KICKOFF - timedelta(hours=3),
        raw_data={},
        bookmaker_count=1,
    )
    session.add(snapshot)
    await session.flush()

    session.add_all(
        [
            Odds(
                event_id=TWIN_EVENT_ID,
                bookmaker_key="bet365",
                bookmaker_title="bet365",
                market_key="1x2",
                outcome_name="Arsenal",
                price_decimal=american_to_decimal(150),
                odds_timestamp=snapshot.snapshot_time,
                last_update=snapshot.snapshot_time,
            ),
            PaperTrade(
                event_id=TWIN_EVENT_ID,
                market="h2h",
                selection="home",
                bookmaker="bet365",
                price_decimal=2.5,
                stake=10.0,
                bankroll_before=100.0,
            ),
            MatchBrief(
                event_id=TWIN_EVENT_ID,
                decision=BriefDecision.WATCHING,
                summary="s",
                brief_text="t",
            ),
        ]
    )
    await session.flush()
    return session


@pytest.fixture
async def stored_epl_event(pglite_async_session) -> Event:
    """One stored EPL event (Arsenal v Chelsea at ``TWIN_KICKOFF``) with no external ids."""
    event = make_epl_event("odds_api_hex_1")
    pglite_async_session.add(event)
    await pglite_async_session.flush()
    return event


@pytest.fixture
async def epl_snapshot(pglite_async_session, stored_epl_event) -> OddsSnapshot:
    """A stored Betfair 1x2 snapshot of ``stored_epl_event``, in the new shape, without prices."""
    snapshot = OddsSnapshot(
        event_id=stored_epl_event.id,
        snapshot_time=TWIN_KICKOFF - timedelta(hours=1),
        raw_data={"bookmakers": []},
        bookmaker_count=1,
        source=Source.BETFAIR,
        market=Market.ONE_X_TWO,
        time_basis=TimeBasis.OBSERVED,
    )
    pglite_async_session.add(snapshot)
    await pglite_async_session.flush()
    return snapshot


@pytest.fixture
async def mixed_source_event(pglite_async_session):
    """An event with an Odds API snapshot and a later Polymarket snapshot carrying mid, ask and bid."""
    session = pglite_async_session
    kickoff = datetime(2026, 4, 18, 15, 0, tzinfo=UTC)
    event = make_epl_event("epl_test_mixed", kickoff)
    session.add(event)
    api_time = kickoff - timedelta(hours=3)
    await store_tagged(
        session,
        make_tagged_snapshot(
            event.id,
            api_time,
            [
                three_way_bookmaker("bet365", (2.0, 3.5, 4.0)),
                three_way_bookmaker("pinnacle", (2.1, 3.4, 3.9)),
            ],
            None,
        ),
    )
    pm_time = kickoff - timedelta(hours=2)
    probabilities = {
        "yes_probabilities": {"home": 0.5, "draw": 0.25, "away": 0.25},
        "best_ask": {"home": 0.52, "draw": 0.26, "away": 0.26},
        "best_bid": {"home": 0.48, "draw": 0.24, "away": 0.24},
    }
    await store_tagged(
        session,
        make_tagged_snapshot(
            event.id,
            pm_time,
            [three_way_bookmaker("polymarket", (2.0, 4.0, 4.0), polymarket_meta=probabilities)],
            "polymarket",
        ),
    )
    await session.commit()
    return event
