"""
Fixtures for integration tests.

This module contains shared fixtures for integration testing.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot


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
                price=opening_home_price,
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
                price=opening_away_price,
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
                price=closing_home_price,
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
                price=closing_away_price,
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
