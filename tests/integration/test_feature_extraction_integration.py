"""Integration tests for feature extraction with database."""

from datetime import UTC, datetime

import pytest
from odds_analytics.feature_extraction import FeatureEvent, TabularFeatureExtractor
from odds_core.models import Event, EventStatus
from odds_core.quotes import quotes
from odds_core.vocabulary import Market, PriceOutcome

from tests.test_helpers import add_event, make_tagged_snapshot, store_tagged


@pytest.mark.asyncio
class TestFeatureExtractionIntegration:
    """Integration tests for feature extractors with database."""

    async def test_tabular_extractor_with_database_odds(self, test_session):
        """Test TabularFeatureExtractor with quotes from database."""
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

        odds_time = datetime(2024, 11, 1, 18, 0, 0, tzinfo=UTC)
        bookmakers = [
            {
                "key": key,
                "title": key.title(),
                "last_update": odds_time.isoformat(),
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": event.home_team, "price": home_price},
                            {"name": event.away_team, "price": away_price},
                        ],
                    }
                ],
            }
            for key, home_price, away_price in [
                ("pinnacle", -120, 100),
                ("fanduel", -115, -105),
            ]
        ]

        # Insert into database, with price rows as the writers store them
        await add_event(test_session, event)
        await store_tagged(
            test_session, make_tagged_snapshot(event.id, odds_time, bookmakers, source=None)
        )
        await test_session.commit()

        # Query back and extract features
        queried = await quotes(test_session, [event.id], Market.H2H)

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
        features = extractor.extract_features(feature_event, queried, outcome=PriceOutcome.HOME)

        # Verify features were extracted
        assert features.consensus_prob is not None
        assert features.sharp_prob is not None
        assert features.retail_sharp_diff is not None
        assert 0 < features.consensus_prob < 1
