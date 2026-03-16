"""Unit tests for EPL schedule and rest feature extraction."""

from datetime import UTC, datetime, timedelta

import numpy as np
from odds_analytics.epl_schedule_features import (
    EplScheduleFeatures,
    extract_epl_schedule_features,
)
from odds_analytics.fixture_cache import FixtureCache, FixtureRecord
from odds_core.models import Event, EventStatus


def _make_event(
    event_id: str = "test_event",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    commence_time: datetime | None = None,
    home_score: int | None = None,
    away_score: int | None = None,
    status: EventStatus = EventStatus.FINAL,
) -> Event:
    if commence_time is None:
        commence_time = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        home_team=home_team,
        away_team=away_team,
        commence_time=commence_time,
        home_score=home_score,
        away_score=away_score,
        status=status,
    )


class TestEplScheduleFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = EplScheduleFeatures.get_feature_names()
        assert names == [
            "home_rest_days",
            "away_rest_days",
            "rest_advantage",
            "is_midweek",
        ]

    def test_to_array_all_none(self) -> None:
        features = EplScheduleFeatures()
        arr = features.to_array()
        assert arr.shape == (4,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = EplScheduleFeatures(
            home_rest_days=7.0,
            away_rest_days=3.0,
            rest_advantage=4.0,
            is_midweek=0.0,
        )
        arr = features.to_array()
        assert arr.shape == (4,)
        assert arr[0] == 7.0
        assert arr[1] == 3.0
        assert arr[2] == 4.0
        assert arr[3] == 0.0

    def test_feature_count_matches_names(self) -> None:
        assert len(EplScheduleFeatures.get_feature_names()) == len(EplScheduleFeatures().to_array())


class TestExtractEplScheduleFeatures:
    def test_no_prior_events(self) -> None:
        event = _make_event()
        features = extract_epl_schedule_features([], event)
        assert features.home_rest_days is None
        assert features.away_rest_days is None
        assert features.rest_advantage is None
        # is_midweek should still be computed (Wednesday)
        assert features.is_midweek == 1.0

    def test_standard_weekend_rest(self) -> None:
        """Saturday-to-Saturday = 7 days rest for both teams."""
        base = datetime(2025, 1, 4, 15, 0, tzinfo=UTC)  # Saturday
        prior = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Chelsea",
                commence_time=base,
                home_score=2,
                away_score=1,
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Chelsea",
            away_team="Arsenal",
            commence_time=base + timedelta(days=7),  # Next Saturday
        )
        features = extract_epl_schedule_features(prior, event)
        assert features.home_rest_days == 7.0
        assert features.away_rest_days == 7.0
        assert features.rest_advantage == 0.0

    def test_midweek_turnaround(self) -> None:
        """Saturday to Tuesday = ~3 day turnaround."""
        saturday = datetime(2025, 1, 4, 15, 0, tzinfo=UTC)  # Saturday
        tuesday = datetime(2025, 1, 7, 19, 45, tzinfo=UTC)  # Tuesday evening
        prior = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Liverpool",
                commence_time=saturday,
                home_score=1,
                away_score=0,
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=tuesday,
        )
        features = extract_epl_schedule_features(prior, event)
        # ~3.2 days (Saturday 15:00 to Tuesday 19:45)
        assert features.home_rest_days is not None
        assert abs(features.home_rest_days - 3.1979166666666665) < 0.01
        assert features.is_midweek == 1.0

    def test_rest_advantage(self) -> None:
        """Home team played 3 days ago, away played 7 days ago."""
        base = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)  # Saturday
        prior = [
            # Arsenal played last Saturday
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Liverpool",
                commence_time=base - timedelta(days=7),
                home_score=2,
                away_score=0,
            ),
            # Chelsea played on Wednesday (3 days before)
            _make_event(
                event_id="e2",
                home_team="Chelsea",
                away_team="Tottenham",
                commence_time=base - timedelta(days=3),
                home_score=1,
                away_score=1,
            ),
        ]
        event = _make_event(
            event_id="e3",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=base,
        )
        features = extract_epl_schedule_features(prior, event)
        assert features.home_rest_days == 7.0
        assert features.away_rest_days == 3.0
        assert features.rest_advantage == 4.0

    def test_team_as_away_in_prior(self) -> None:
        """Previous match was away — should still count for rest."""
        base = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)
        prior = [
            _make_event(
                event_id="e1",
                home_team="Liverpool",
                away_team="Arsenal",  # Arsenal was away
                commence_time=base - timedelta(days=4),
                home_score=1,
                away_score=2,
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=base,
        )
        features = extract_epl_schedule_features(prior, event)
        assert features.home_rest_days == 4.0
        assert features.away_rest_days is None  # Chelsea not in prior

    def test_season_start_no_prior(self) -> None:
        """First match of season — both teams have no prior match."""
        event = _make_event(
            commence_time=datetime(2024, 8, 17, 15, 0, tzinfo=UTC),  # Saturday
        )
        features = extract_epl_schedule_features([], event)
        assert features.home_rest_days is None
        assert features.away_rest_days is None
        assert features.rest_advantage is None
        assert features.is_midweek == 0.0  # Saturday

    def test_is_midweek_classification(self) -> None:
        """Tuesday and Wednesday are midweek; others are not."""
        # Monday
        event = _make_event(commence_time=datetime(2025, 1, 6, 20, 0, tzinfo=UTC))
        assert extract_epl_schedule_features([], event).is_midweek == 0.0

        # Tuesday
        event = _make_event(commence_time=datetime(2025, 1, 7, 19, 45, tzinfo=UTC))
        assert extract_epl_schedule_features([], event).is_midweek == 1.0

        # Wednesday
        event = _make_event(commence_time=datetime(2025, 1, 8, 19, 45, tzinfo=UTC))
        assert extract_epl_schedule_features([], event).is_midweek == 1.0

        # Thursday
        event = _make_event(commence_time=datetime(2025, 1, 9, 20, 0, tzinfo=UTC))
        assert extract_epl_schedule_features([], event).is_midweek == 0.0

        # Friday
        event = _make_event(commence_time=datetime(2025, 1, 10, 20, 0, tzinfo=UTC))
        assert extract_epl_schedule_features([], event).is_midweek == 0.0

        # Saturday
        event = _make_event(commence_time=datetime(2025, 1, 11, 15, 0, tzinfo=UTC))
        assert extract_epl_schedule_features([], event).is_midweek == 0.0

        # Sunday
        event = _make_event(commence_time=datetime(2025, 1, 12, 14, 0, tzinfo=UTC))
        assert extract_epl_schedule_features([], event).is_midweek == 0.0

    def test_multiple_prior_matches_picks_most_recent(self) -> None:
        """With multiple prior matches, rest is computed from the most recent."""
        base = datetime(2025, 1, 18, 15, 0, tzinfo=UTC)
        prior = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Liverpool",
                commence_time=base - timedelta(days=14),
                home_score=1,
                away_score=0,
            ),
            _make_event(
                event_id="e2",
                home_team="Man City",
                away_team="Arsenal",
                commence_time=base - timedelta(days=7),
                home_score=0,
                away_score=0,
            ),
            _make_event(
                event_id="e3",
                home_team="Arsenal",
                away_team="Brighton",
                commence_time=base - timedelta(days=3),
                home_score=3,
                away_score=1,
            ),
        ]
        event = _make_event(
            event_id="e4",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=base,
        )
        features = extract_epl_schedule_features(prior, event)
        assert features.home_rest_days == 3.0  # Most recent: 3 days ago

    def test_one_team_known_other_unknown(self) -> None:
        """Only one team has prior matches — partial features."""
        base = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)
        prior = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Liverpool",
                commence_time=base - timedelta(days=7),
                home_score=2,
                away_score=1,
            ),
        ]
        # Unknown FC not in prior events at all
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Unknown FC",
            commence_time=base,
        )
        features = extract_epl_schedule_features(prior, event)
        assert features.home_rest_days == 7.0
        assert features.away_rest_days is None
        assert features.rest_advantage is None

    def test_fractional_rest_days(self) -> None:
        """Rest days use commence_time precision, not just date."""
        base = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)  # Saturday 3pm
        prior = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Chelsea",
                commence_time=base,
                home_score=1,
                away_score=0,
            ),
        ]
        # Next match on Tuesday at 7:45pm = 3 days + 4h45m
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Wolves",
            commence_time=datetime(2025, 1, 14, 19, 45, tzinfo=UTC),
        )
        features = extract_epl_schedule_features(prior, event)
        expected = (3 * 24 + 4 + 45 / 60) / 24
        assert features.home_rest_days is not None
        assert abs(features.home_rest_days - expected) < 0.001


class TestFixtureCacheIntegration:
    """Tests for fixture_cache parameter on extract_epl_schedule_features."""

    def test_cl_midweek_reduces_rest_days(self) -> None:
        """EPL Sat -> CL Wed -> EPL Sun: without cache=8d, with cache=4d."""
        saturday_epl = datetime(2025, 1, 4, 15, 0, tzinfo=UTC)
        wednesday_cl = datetime(2025, 1, 8, 20, 0, tzinfo=UTC)
        next_sunday = datetime(2025, 1, 12, 14, 0, tzinfo=UTC)

        prior_events = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Liverpool",
                commence_time=saturday_epl,
                home_score=2,
                away_score=1,
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=next_sunday,
        )

        # Without fixture cache: 8 days rest (Sat to Sun)
        features_no_cache = extract_epl_schedule_features(prior_events, event)
        assert features_no_cache.home_rest_days is not None
        assert features_no_cache.home_rest_days > 7.5

        # With fixture cache showing CL on Wednesday: ~4 days rest (Wed to Sun)
        fixture_cache: FixtureCache = {
            "Arsenal": [
                FixtureRecord(date=saturday_epl, competition="Premier League"),
                FixtureRecord(date=wednesday_cl, competition="Champions League"),
            ],
        }
        features_with_cache = extract_epl_schedule_features(
            prior_events, event, fixture_cache=fixture_cache
        )
        assert features_with_cache.home_rest_days is not None
        assert features_with_cache.home_rest_days < 4.0

    def test_fixture_cache_no_effect_when_epl_is_more_recent(self) -> None:
        """When EPL match is more recent than any cup match, rest stays the same."""
        wednesday_epl = datetime(2025, 1, 8, 19, 45, tzinfo=UTC)
        saturday = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)

        prior_events = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Brighton",
                commence_time=wednesday_epl,
                home_score=1,
                away_score=0,
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=saturday,
        )

        # Fixture cache only has older matches
        fixture_cache: FixtureCache = {
            "Arsenal": [
                FixtureRecord(date=datetime(2025, 1, 1, 15, tzinfo=UTC), competition="FA Cup"),
                FixtureRecord(date=wednesday_epl, competition="Premier League"),
            ],
        }

        no_cache = extract_epl_schedule_features(prior_events, event)
        with_cache = extract_epl_schedule_features(prior_events, event, fixture_cache=fixture_cache)
        assert no_cache.home_rest_days == with_cache.home_rest_days

    def test_fixture_cache_only_no_prior_events(self) -> None:
        """Cache provides rest data even when prior_events is empty."""
        wednesday_cl = datetime(2025, 1, 8, 20, 0, tzinfo=UTC)
        saturday = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)

        event = _make_event(
            event_id="e1",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=saturday,
        )

        fixture_cache: FixtureCache = {
            "Arsenal": [
                FixtureRecord(date=wednesday_cl, competition="Champions League"),
            ],
        }

        features = extract_epl_schedule_features([], event, fixture_cache=fixture_cache)
        assert features.home_rest_days is not None
        assert abs(features.home_rest_days - 2.791666666666) < 0.01
        # Chelsea not in cache, so away rest is None
        assert features.away_rest_days is None
        assert features.rest_advantage is None

    def test_rest_advantage_with_fixture_cache(self) -> None:
        """Home team plays CL midweek, away doesn't -> negative rest advantage."""
        saturday_epl = datetime(2025, 1, 4, 15, 0, tzinfo=UTC)
        wednesday_cl = datetime(2025, 1, 8, 20, 0, tzinfo=UTC)
        next_saturday = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)

        prior_events = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Chelsea",
                commence_time=saturday_epl,
                home_score=1,
                away_score=1,
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=next_saturday,
        )

        fixture_cache: FixtureCache = {
            "Arsenal": [
                FixtureRecord(date=saturday_epl, competition="Premier League"),
                FixtureRecord(date=wednesday_cl, competition="Champions League"),
            ],
            "Chelsea": [
                FixtureRecord(date=saturday_epl, competition="Premier League"),
            ],
        }

        features = extract_epl_schedule_features(prior_events, event, fixture_cache=fixture_cache)
        assert features.home_rest_days is not None
        assert features.away_rest_days is not None
        # Arsenal: ~3 days (Wed to Sat), Chelsea: 7 days (Sat to Sat)
        assert features.home_rest_days < 3.0
        assert features.away_rest_days == 7.0
        assert features.rest_advantage is not None
        assert features.rest_advantage < 0  # Home team has less rest

    def test_backward_compatible_with_none_cache(self) -> None:
        """fixture_cache=None gives identical results to no-argument call."""
        base = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)
        prior = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Liverpool",
                commence_time=base - timedelta(days=7),
                home_score=1,
                away_score=0,
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=base,
        )

        default = extract_epl_schedule_features(prior, event)
        explicit_none = extract_epl_schedule_features(prior, event, fixture_cache=None)
        assert default.home_rest_days == explicit_none.home_rest_days
        assert default.away_rest_days == explicit_none.away_rest_days
        assert default.rest_advantage == explicit_none.rest_advantage
        assert default.is_midweek == explicit_none.is_midweek
