"""Integration tests for sequence data loader with database."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from odds_analytics.sequence_loader import (
    load_sequences_for_event,
    prepare_lstm_training_data,
)
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.storage.writers import OddsWriter


@pytest.mark.asyncio
class TestSequenceLoaderIntegration:
    """Integration tests for sequence loader with database."""

    async def test_load_sequences_from_database(self, test_session):
        """Test loading sequences from actual database snapshots."""
        # Create test event
        event = Event(
            id="seq_test_event_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        # Insert event
        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        # Create snapshots with raw_data
        base_time = datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)

        for i in range(3):
            snapshot_time = base_time + timedelta(hours=i * 2)
            raw_data = {
                "id": event.id,
                "sport_key": "basketball_nba",
                "commence_time": event.commence_time.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "last_update": snapshot_time.isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {
                                        "name": event.home_team,
                                        "price": -120 - (i * 2),
                                    },
                                    {
                                        "name": event.away_team,
                                        "price": 100 + (i * 2),
                                    },
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {
                                        "name": event.home_team,
                                        "price": -110,
                                        "point": -5.5,
                                    },
                                    {
                                        "name": event.away_team,
                                        "price": -110,
                                        "point": 5.5,
                                    },
                                ],
                            },
                        ],
                    },
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "last_update": snapshot_time.isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {
                                        "name": event.home_team,
                                        "price": -115 - (i * 2),
                                    },
                                    {
                                        "name": event.away_team,
                                        "price": -105 + (i * 2),
                                    },
                                ],
                            }
                        ],
                    },
                ],
            }

            snapshot = OddsSnapshot(
                event_id=event.id,
                snapshot_time=snapshot_time,
                raw_data=raw_data,
                bookmaker_count=2,
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Load sequences
        sequences = await load_sequences_for_event(event.id, test_session)

        # Verify results
        assert len(sequences) == 3, "Should have 3 snapshots"

        # First snapshot should have odds from both bookmakers for both markets
        # Pinnacle: 2 h2h outcomes + 2 spreads outcomes = 4
        # FanDuel: 2 h2h outcomes = 2
        # Total = 6 odds records
        assert len(sequences[0]) == 6, "First snapshot should have 6 odds records"

        # Verify chronological ordering
        for i in range(len(sequences) - 1):
            assert sequences[i][0].odds_timestamp < sequences[i + 1][0].odds_timestamp

        # Verify line movement (prices should change over time)
        first_lakers_h2h = next(
            o
            for o in sequences[0]
            if o.outcome_name == event.home_team and o.market_key == "h2h" and o.bookmaker_key == "pinnacle"
        )
        last_lakers_h2h = next(
            o
            for o in sequences[2]
            if o.outcome_name == event.home_team and o.market_key == "h2h" and o.bookmaker_key == "pinnacle"
        )

        assert first_lakers_h2h.price != last_lakers_h2h.price, "Line should move over time"

    async def test_load_sequences_multiple_markets(self, test_session):
        """Test loading sequences with multiple markets."""
        event = Event(
            id="seq_test_event_2",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        # Create snapshot with multiple markets
        snapshot_time = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)
        raw_data = {
            "id": event.id,
            "sport_key": "basketball_nba",
            "commence_time": event.commence_time.isoformat(),
            "home_team": event.home_team,
            "away_team": event.away_team,
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": event.home_team, "price": -120},
                                {"name": event.away_team, "price": 100},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": event.home_team, "price": -110, "point": -5.5},
                                {"name": event.away_team, "price": -110, "point": 5.5},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 215.5},
                                {"name": "Under", "price": -110, "point": 215.5},
                            ],
                        },
                    ],
                }
            ],
        }

        snapshot = OddsSnapshot(
            event_id=event.id,
            snapshot_time=snapshot_time,
            raw_data=raw_data,
            bookmaker_count=1,
        )
        test_session.add(snapshot)
        await test_session.commit()

        # Load sequences
        sequences = await load_sequences_for_event(event.id, test_session)

        # Should have all markets
        markets = {odds.market_key for odds in sequences[0]}
        assert markets == {"h2h", "spreads", "totals"}

    async def test_prepare_training_data_from_database(self, test_session):
        """Test preparing LSTM training data from database."""
        # Create multiple test events
        events = []
        for i in range(3):
            event = Event(
                id=f"lstm_event_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 11, i + 1, 19, 0, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.FINAL,
                home_score=110 + i,
                away_score=105,
            )
            events.append(event)

            # Insert event
            writer = OddsWriter(test_session)
            await writer.upsert_event(event)

            # Create snapshots
            base_time = event.commence_time - timedelta(hours=24)
            for j in range(4):
                snapshot_time = base_time + timedelta(hours=j * 6)
                raw_data = {
                    "id": event.id,
                    "sport_key": "basketball_nba",
                    "commence_time": event.commence_time.isoformat(),
                    "home_team": event.home_team,
                    "away_team": event.away_team,
                    "bookmakers": [
                        {
                            "key": "pinnacle",
                            "title": "Pinnacle",
                            "last_update": snapshot_time.isoformat(),
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": event.home_team, "price": -120 - j},
                                        {"name": event.away_team, "price": 100 + j},
                                    ],
                                }
                            ],
                        }
                    ],
                }

                snapshot = OddsSnapshot(
                    event_id=event.id,
                    snapshot_time=snapshot_time,
                    raw_data=raw_data,
                    bookmaker_count=1,
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Prepare training data
        X, y, masks = await prepare_lstm_training_data(
            events=events,
            session=test_session,
            outcome="home",
            market="h2h",
            timesteps=8,
            lookback_hours=48,
        )

        # Verify shapes
        assert X.shape[0] == 3, "Should have 3 samples"
        assert X.shape[1] == 8, "Should have 8 timesteps"
        assert X.shape[2] > 0, "Should have features"

        assert y.shape == (3,), "Should have 3 labels"
        assert masks.shape == (3, 8), "Masks should match shape"

        # Verify labels (all home teams won)
        assert np.all(y == 1), "All home teams won"

        # Verify masks have some valid data
        assert masks.any(), "Should have some valid timesteps"

    async def test_prepare_training_data_mixed_outcomes(self, test_session):
        """Test training data with mixed home/away wins."""
        events = []

        # Event 1: Home wins
        event1 = Event(
            id="mixed_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )
        events.append(event1)

        # Event 2: Away wins
        event2 = Event(
            id="mixed_2",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 2, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=100,
            away_score=105,
        )
        events.append(event2)

        # Insert events and snapshots
        writer = OddsWriter(test_session)
        for event in events:
            await writer.upsert_event(event)

            snapshot_time = event.commence_time - timedelta(hours=12)
            raw_data = {
                "id": event.id,
                "sport_key": "basketball_nba",
                "commence_time": event.commence_time.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "last_update": snapshot_time.isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": event.home_team, "price": -120},
                                    {"name": event.away_team, "price": 100},
                                ],
                            }
                        ],
                    }
                ],
            }

            snapshot = OddsSnapshot(
                event_id=event.id,
                snapshot_time=snapshot_time,
                raw_data=raw_data,
                bookmaker_count=1,
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Prepare training data for home team predictions
        X, y, masks = await prepare_lstm_training_data(
            events=events,
            session=test_session,
            outcome="home",
            market="h2h",
            timesteps=8,
        )

        # Verify labels
        assert y[0] == 1, "First event: home won"
        assert y[1] == 0, "Second event: away won"

    async def test_prepare_training_data_filters_incomplete(self, test_session):
        """Test that incomplete events are filtered."""
        events = []

        # Complete event
        complete_event = Event(
            id="complete",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )
        events.append(complete_event)

        # Incomplete event (no scores)
        incomplete_event = Event(
            id="incomplete",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 2, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
            home_score=None,
            away_score=None,
        )
        events.append(incomplete_event)

        # Insert events
        writer = OddsWriter(test_session)
        for event in events:
            await writer.upsert_event(event)

            # Only add snapshot for complete event
            if event.status == EventStatus.FINAL:
                snapshot_time = event.commence_time - timedelta(hours=12)
                raw_data = {
                    "id": event.id,
                    "sport_key": "basketball_nba",
                    "commence_time": event.commence_time.isoformat(),
                    "home_team": event.home_team,
                    "away_team": event.away_team,
                    "bookmakers": [
                        {
                            "key": "pinnacle",
                            "title": "Pinnacle",
                            "last_update": snapshot_time.isoformat(),
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": event.home_team, "price": -120},
                                        {"name": event.away_team, "price": 100},
                                    ],
                                }
                            ],
                        }
                    ],
                }

                snapshot = OddsSnapshot(
                    event_id=event.id,
                    snapshot_time=snapshot_time,
                    raw_data=raw_data,
                    bookmaker_count=1,
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Prepare training data
        X, y, masks = await prepare_lstm_training_data(
            events=events,
            session=test_session,
            outcome="home",
        )

        # Should only have 1 sample (incomplete event filtered out)
        assert X.shape[0] == 1, "Should filter out incomplete events"

    async def test_load_sequences_chronological_ordering(self, test_session):
        """Test that sequences are returned in chronological order."""
        event = Event(
            id="chrono_test",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        # Create snapshots in non-chronological order
        times = [
            datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC),
            datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC),  # Earlier
            datetime(2024, 11, 1, 15, 0, 0, tzinfo=UTC),  # Middle
        ]

        for snapshot_time in times:
            raw_data = {
                "id": event.id,
                "sport_key": "basketball_nba",
                "commence_time": event.commence_time.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [
                    {
                        "key": "pinnacle",
                        "title": "Pinnacle",
                        "last_update": snapshot_time.isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": event.home_team, "price": -120},
                                    {"name": event.away_team, "price": 100},
                                ],
                            }
                        ],
                    }
                ],
            }

            snapshot = OddsSnapshot(
                event_id=event.id,
                snapshot_time=snapshot_time,
                raw_data=raw_data,
                bookmaker_count=1,
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Load sequences
        sequences = await load_sequences_for_event(event.id, test_session)

        # Verify chronological order (earliest to latest)
        timestamps = [seq[0].odds_timestamp for seq in sequences]
        assert timestamps == sorted(timestamps), "Sequences should be chronologically ordered"
        assert timestamps[0] == datetime(2024, 11, 1, 12, 0, 0, tzinfo=UTC)
        assert timestamps[-1] == datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)
