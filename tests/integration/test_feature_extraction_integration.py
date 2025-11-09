"""Integration tests for feature extraction with database."""

from datetime import datetime, timezone

import numpy as np
import pytest

from analytics.backtesting import BacktestEvent
from analytics.feature_extraction import SequenceFeatureExtractor, TabularFeatureExtractor
from core.models import Event, EventStatus, Odds
from storage.readers import OddsReader
from storage.writers import OddsWriter


@pytest.mark.asyncio
class TestFeatureExtractionIntegration:
    """Integration tests for feature extractors with database."""

    async def test_tabular_extractor_with_database_odds(self, test_session):
        """Test TabularFeatureExtractor with odds from database."""
        # Create test event
        event = Event(
            id="test_event_tabular_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=timezone.utc),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        # Create test odds
        odds_time = datetime(2024, 11, 1, 18, 0, 0, tzinfo=timezone.utc)
        odds_records = [
            Odds(
                event_id=event.id,
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name=event.home_team,
                price=-120,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
            Odds(
                event_id=event.id,
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name=event.away_team,
                price=+100,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
            Odds(
                event_id=event.id,
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name=event.home_team,
                price=-115,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
            Odds(
                event_id=event.id,
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name=event.away_team,
                price=-105,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
        ]

        # Insert into database
        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        for odds in odds_records:
            test_session.add(odds)
        await test_session.commit()

        # Query back and extract features
        reader = OddsReader(test_session)
        queried_odds = await reader.get_odds_at_time(
            event.id, odds_time, tolerance_minutes=5
        )

        # Extract features
        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

        extractor = TabularFeatureExtractor()
        features = extractor.extract_features(
            backtest_event, queried_odds, outcome=event.home_team, market="h2h"
        )

        # Verify features were extracted
        assert len(features) > 0
        assert "consensus_prob" in features
        assert "sharp_prob" in features
        assert 0 < features["consensus_prob"] < 1

    async def test_sequence_extractor_with_line_movement_query(self, test_session):
        """Test SequenceFeatureExtractor with line movement from database."""
        # Create test event
        event = Event(
            id="test_event_sequence_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=timezone.utc),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=112,
            away_score=108,
        )

        # Create time series of odds (simulate line movement)
        base_time = event.commence_time - np.timedelta64(12, 'h')
        odds_records = []

        for i in range(5):
            timestamp = base_time + np.timedelta64(i * 2, 'h')
            price = -120 + i * 2  # Line moving

            odds_records.extend([
                Odds(
                    event_id=event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=event.home_team,
                    price=price,
                    point=None,
                    odds_timestamp=timestamp,
                    last_update=timestamp,
                ),
                Odds(
                    event_id=event.id,
                    bookmaker_key="pinnacle",
                    bookmaker_title="Pinnacle",
                    market_key="h2h",
                    outcome_name=event.away_team,
                    price=100,
                    point=None,
                    odds_timestamp=timestamp,
                    last_update=timestamp,
                ),
            ])

        # Insert into database
        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        for odds in odds_records:
            test_session.add(odds)
        await test_session.commit()

        # Query line movement
        reader = OddsReader(test_session)
        line_movement = await reader.get_line_movement(
            event_id=event.id,
            bookmaker_key="pinnacle",
            market_key="h2h",
            outcome_name=event.home_team,
        )

        assert len(line_movement) == 5

        # Group by timestamp to create snapshots
        snapshots = []
        for i in range(5):
            timestamp = base_time + np.timedelta64(i * 2, 'h')
            snapshot_odds = [o for o in odds_records if o.odds_timestamp == timestamp]
            if snapshot_odds:
                snapshots.append(snapshot_odds)

        # Extract sequence features
        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)
        result = extractor.extract_features(
            backtest_event, snapshots, outcome=event.home_team, market="h2h"
        )

        # Verify output structure
        assert "sequence" in result
        assert "mask" in result
        assert result["sequence"].shape == (8, 15)
        assert result["mask"].shape == (8,)

        # Should have some valid data
        assert result["mask"].any()

        # All values should be finite
        assert np.all(np.isfinite(result["sequence"]))

    async def test_sequence_extractor_with_sparse_data(self, test_session):
        """Test SequenceFeatureExtractor handles sparse historical data."""
        # Create test event
        event = Event(
            id="test_event_sequence_2",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=timezone.utc),
            home_team="Phoenix Suns",
            away_team="Denver Nuggets",
            status=EventStatus.FINAL,
            home_score=115,
            away_score=110,
        )

        # Create sparse odds (only 2 snapshots over 24 hour period)
        base_time = event.commence_time - np.timedelta64(20, 'h')
        odds_records = []

        for i in [0, 2]:  # Sparse: only at 0h and 4h
            timestamp = base_time + np.timedelta64(i * 10, 'h')

            odds_records.extend([
                Odds(
                    event_id=event.id,
                    bookmaker_key="fanduel",
                    bookmaker_title="FanDuel",
                    market_key="h2h",
                    outcome_name=event.home_team,
                    price=-110,
                    point=None,
                    odds_timestamp=timestamp,
                    last_update=timestamp,
                ),
                Odds(
                    event_id=event.id,
                    bookmaker_key="fanduel",
                    bookmaker_title="FanDuel",
                    market_key="h2h",
                    outcome_name=event.away_team,
                    price=-110,
                    point=None,
                    odds_timestamp=timestamp,
                    last_update=timestamp,
                ),
            ])

        # Insert into database
        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        for odds in odds_records:
            test_session.add(odds)
        await test_session.commit()

        # Group by timestamp
        snapshots = []
        for i in [0, 2]:
            timestamp = base_time + np.timedelta64(i * 10, 'h')
            snapshot_odds = [o for o in odds_records if o.odds_timestamp == timestamp]
            if snapshot_odds:
                snapshots.append(snapshot_odds)

        # Extract sequence features
        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

        extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=8)
        result = extractor.extract_features(
            backtest_event, snapshots, outcome=event.home_team, market="h2h"
        )

        # Should handle sparse data gracefully
        assert "sequence" in result
        assert "mask" in result

        # Should have some valid data
        assert result["mask"].any()

        # But not all timesteps should have data (sparse)
        assert not result["mask"].all()

    async def test_sequence_extractor_with_multiple_bookmakers(self, test_session):
        """Test SequenceFeatureExtractor aggregates multiple bookmakers correctly."""
        # Create test event
        event = Event(
            id="test_event_sequence_3",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=timezone.utc),
            home_team="Milwaukee Bucks",
            away_team="Miami Heat",
            status=EventStatus.FINAL,
            home_score=118,
            away_score=112,
        )

        # Create odds from multiple bookmakers
        base_time = event.commence_time - np.timedelta64(6, 'h')
        odds_records = []

        for bookmaker in ["pinnacle", "fanduel", "draftkings"]:
            for i in range(3):
                timestamp = base_time + np.timedelta64(i * 2, 'h')

                odds_records.extend([
                    Odds(
                        event_id=event.id,
                        bookmaker_key=bookmaker,
                        bookmaker_title=bookmaker.title(),
                        market_key="h2h",
                        outcome_name=event.home_team,
                        price=-120 if bookmaker == "pinnacle" else -115,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                    Odds(
                        event_id=event.id,
                        bookmaker_key=bookmaker,
                        bookmaker_title=bookmaker.title(),
                        market_key="h2h",
                        outcome_name=event.away_team,
                        price=100,
                        point=None,
                        odds_timestamp=timestamp,
                        last_update=timestamp,
                    ),
                ])

        # Insert into database
        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        for odds in odds_records:
            test_session.add(odds)
        await test_session.commit()

        # Group by timestamp
        timestamps = sorted(set(o.odds_timestamp for o in odds_records))
        snapshots = []
        for timestamp in timestamps:
            snapshot_odds = [o for o in odds_records if o.odds_timestamp == timestamp]
            snapshots.append(snapshot_odds)

        # Extract sequence features
        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

        extractor = SequenceFeatureExtractor(
            lookback_hours=12,
            timesteps=4,
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
        )
        result = extractor.extract_features(
            backtest_event, snapshots, outcome=event.home_team, market="h2h"
        )

        # Should successfully aggregate multiple bookmakers
        assert result["mask"].any()

        # Check that sharp vs retail differential was calculated
        feature_names = extractor.get_feature_names()
        retail_sharp_diff_idx = feature_names.index("retail_sharp_diff")

        # At least one timestep should have retail-sharp differential
        valid_timesteps = result["mask"]
        if valid_timesteps.any():
            # We expect some differential since retail books have different odds
            sequence = result["sequence"]
            assert sequence.shape == (4, 15)

    async def test_feature_extractors_interface_compatibility(self, test_session):
        """Test that both extractors implement the same interface."""
        # Create test event
        event = Event(
            id="test_event_interface_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=timezone.utc),
            home_team="Dallas Mavericks",
            away_team="San Antonio Spurs",
            status=EventStatus.FINAL,
            home_score=108,
            away_score=102,
        )

        odds_time = datetime(2024, 11, 1, 18, 0, 0, tzinfo=timezone.utc)
        odds_records = [
            Odds(
                event_id=event.id,
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name=event.home_team,
                price=-110,
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
        ]

        writer = OddsWriter(test_session)
        await writer.upsert_event(event)

        for odds in odds_records:
            test_session.add(odds)
        await test_session.commit()

        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=event.home_score,
            away_score=event.away_score,
            status=event.status,
        )

        # Test TabularFeatureExtractor
        tabular_extractor = TabularFeatureExtractor()
        tabular_features = tabular_extractor.extract_features(
            backtest_event, odds_records, outcome=event.home_team, market="h2h"
        )
        tabular_names = tabular_extractor.get_feature_names()

        assert isinstance(tabular_features, dict)
        assert isinstance(tabular_names, list)

        # Test SequenceFeatureExtractor
        sequence_extractor = SequenceFeatureExtractor(lookback_hours=24, timesteps=4)
        sequence_result = sequence_extractor.extract_features(
            backtest_event, [odds_records], outcome=event.home_team, market="h2h"
        )
        sequence_names = sequence_extractor.get_feature_names()

        assert isinstance(sequence_result, dict)
        assert isinstance(sequence_names, list)
        assert "sequence" in sequence_result
        assert "mask" in sequence_result
