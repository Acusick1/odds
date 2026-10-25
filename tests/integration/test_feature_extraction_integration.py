"""Integration tests for feature extraction with database."""

from datetime import UTC, datetime

import pytest
from odds_analytics.feature_extraction import FeatureEvent, TabularFeatureExtractor
from odds_core.models import Event, EventStatus, Odds
from odds_core.odds_math import american_to_decimal
from odds_lambda.storage.readers import OddsReader

from tests.test_helpers import add_event


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
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

        # Create test odds
        odds_time = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)
        odds_records = [
            Odds(
                event_id=event.id,
                bookmaker_key="pinnacle",
                bookmaker_title="Pinnacle",
                market_key="h2h",
                outcome_name=event.home_team,
                price_decimal=american_to_decimal(-120),
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
                price_decimal=american_to_decimal(+100),
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
                price_decimal=american_to_decimal(-115),
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
                price_decimal=american_to_decimal(-105),
                point=None,
                odds_timestamp=odds_time,
                last_update=odds_time,
            ),
        ]

        # Insert into database
        await add_event(test_session, event)

        for odds in odds_records:
            test_session.add(odds)
        await test_session.commit()

        # Query back and extract features
        reader = OddsReader(test_session)
        queried_odds = await reader.get_odds_at_time(event.id, odds_time, tolerance_minutes=5)

        # Extract features
        feature_event = FeatureEvent(
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
            feature_event, queried_odds, outcome=event.home_team, market="h2h"
        )

        # Verify features were extracted
        assert features.consensus_prob is not None
        assert features.sharp_prob is not None
        assert 0 < features.consensus_prob < 1
