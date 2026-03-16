"""Unit tests for EPL schedule and rest feature extraction."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from odds_analytics.epl_schedule_features import (
    EplScheduleFeatures,
    extract_epl_schedule_features,
)
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


def _make_fixtures_df(rows: list[dict[str, str]]) -> pd.DataFrame:
    """Build a small fixtures DataFrame from dicts."""
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


class TestEplScheduleFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = EplScheduleFeatures.get_feature_names()
        assert names == [
            "home_rest_days",
            "away_rest_days",
            "rest_advantage",
            "is_midweek",
            "home_matches_last_14d",
            "away_matches_last_14d",
            "home_european_last_7d",
            "away_european_last_7d",
        ]

    def test_to_array_all_none(self) -> None:
        features = EplScheduleFeatures()
        arr = features.to_array()
        assert arr.shape == (8,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = EplScheduleFeatures(
            home_rest_days=7.0,
            away_rest_days=3.0,
            rest_advantage=4.0,
            is_midweek=0.0,
            home_matches_last_14d=2.0,
            away_matches_last_14d=4.0,
            home_european_last_7d=0.0,
            away_european_last_7d=1.0,
        )
        arr = features.to_array()
        assert arr.shape == (8,)
        np.testing.assert_array_equal(arr, [7.0, 3.0, 4.0, 0.0, 2.0, 4.0, 0.0, 1.0])

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
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Liverpool",
                commence_time=base - timedelta(days=7),
                home_score=2,
                away_score=0,
            ),
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
        """Previous match was away -- should still count for rest."""
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
        """First match of season -- both teams have no prior match."""
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
        """Only one team has prior matches -- partial features."""
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


class TestFixturesDfIntegration:
    """Tests for fixtures_df parameter on extract_epl_schedule_features."""

    def test_cl_midweek_reduces_rest_days(self) -> None:
        """EPL Sat -> CL Wed -> EPL Sun: without df=8d, with df=4d."""
        saturday_epl = datetime(2025, 1, 4, 15, 0, tzinfo=UTC)
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

        # Without fixtures_df: 8 days rest (Sat to Sun)
        features_no_df = extract_epl_schedule_features(prior_events, event)
        assert features_no_df.home_rest_days is not None
        assert features_no_df.home_rest_days > 7.5

        # With fixtures_df showing CL on Wednesday: ~4 days rest (Wed to Sun)
        df = _make_fixtures_df(
            [
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Liverpool",
                    "competition": "Premier League",
                },
                {
                    "date": "2025-01-08T20:00Z",
                    "team": "Arsenal",
                    "opponent": "PSG",
                    "competition": "Champions League",
                },
            ]
        )
        features_with_df = extract_epl_schedule_features(prior_events, event, fixtures_df=df)
        assert features_with_df.home_rest_days is not None
        assert features_with_df.home_rest_days < 4.0

    def test_fixtures_df_no_effect_when_epl_is_more_recent(self) -> None:
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

        df = _make_fixtures_df(
            [
                {
                    "date": "2025-01-01T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Wolves",
                    "competition": "FA Cup",
                },
                {
                    "date": "2025-01-08T19:45Z",
                    "team": "Arsenal",
                    "opponent": "Brighton",
                    "competition": "Premier League",
                },
            ]
        )

        no_df = extract_epl_schedule_features(prior_events, event)
        with_df = extract_epl_schedule_features(prior_events, event, fixtures_df=df)
        assert no_df.home_rest_days == with_df.home_rest_days

    def test_fixtures_df_only_no_prior_events(self) -> None:
        """DataFrame provides rest data even when prior_events is empty."""
        saturday = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)

        event = _make_event(
            event_id="e1",
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=saturday,
        )

        df = _make_fixtures_df(
            [
                {
                    "date": "2025-01-08T20:00Z",
                    "team": "Arsenal",
                    "opponent": "PSG",
                    "competition": "Champions League",
                },
            ]
        )

        features = extract_epl_schedule_features([], event, fixtures_df=df)
        assert features.home_rest_days is not None
        assert abs(features.home_rest_days - 2.791666666666) < 0.01
        # Chelsea not in df, so away rest is None
        assert features.away_rest_days is None
        assert features.rest_advantage is None

    def test_rest_advantage_with_fixtures_df(self) -> None:
        """Home team plays CL midweek, away doesn't -> negative rest advantage."""
        saturday_epl = datetime(2025, 1, 4, 15, 0, tzinfo=UTC)
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

        df = _make_fixtures_df(
            [
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Chelsea",
                    "competition": "Premier League",
                },
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Chelsea",
                    "opponent": "Arsenal",
                    "competition": "Premier League",
                },
                {
                    "date": "2025-01-08T20:00Z",
                    "team": "Arsenal",
                    "opponent": "PSG",
                    "competition": "Champions League",
                },
            ]
        )

        features = extract_epl_schedule_features(prior_events, event, fixtures_df=df)
        assert features.home_rest_days is not None
        assert features.away_rest_days is not None
        # Arsenal: ~3 days (Wed to Sat), Chelsea: 7 days (Sat to Sat)
        assert features.home_rest_days < 3.0
        assert features.away_rest_days == 7.0
        assert features.rest_advantage is not None
        assert features.rest_advantage < 0  # Home team has less rest

    def test_backward_compatible_with_none_df(self) -> None:
        """fixtures_df=None gives identical results to no-argument call."""
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
        explicit_none = extract_epl_schedule_features(prior, event, fixtures_df=None)
        assert default.home_rest_days == explicit_none.home_rest_days
        assert default.away_rest_days == explicit_none.away_rest_days
        assert default.rest_advantage == explicit_none.rest_advantage
        assert default.is_midweek == explicit_none.is_midweek

    def test_opponent_column_indexes_team(self) -> None:
        """Teams appearing only as opponents are still found in fixture lookups."""
        saturday = datetime(2025, 1, 11, 15, 0, tzinfo=UTC)
        event = _make_event(
            event_id="e1",
            home_team="Arsenal",
            away_team="Ipswich",
            commence_time=saturday,
        )

        # Ipswich only appears in the opponent column
        df = _make_fixtures_df(
            [
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Ipswich",
                    "competition": "Premier League",
                },
                {
                    "date": "2025-01-07T20:00Z",
                    "team": "Liverpool",
                    "opponent": "Ipswich",
                    "competition": "League Cup",
                },
            ]
        )

        features = extract_epl_schedule_features([], event, fixtures_df=df)
        # Ipswich's most recent: Jan 7 Liverpool LC match (found via opponent column)
        assert features.away_rest_days is not None
        assert abs(features.away_rest_days - 3.791666666) < 0.01


class TestCongestionFeatures:
    """Tests for matches_last_14d and european_last_7d features."""

    def test_matches_last_14d(self) -> None:
        """Count all-competition matches in trailing 14-day window."""
        event = _make_event(
            commence_time=datetime(2025, 1, 18, 15, 0, tzinfo=UTC),
        )
        df = _make_fixtures_df(
            [
                # Arsenal: 3 matches in last 14d, 1 outside window
                {
                    "date": "2025-01-01T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Wolves",
                    "competition": "Premier League",
                },
                {
                    "date": "2025-01-05T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Brighton",
                    "competition": "Premier League",
                },
                {
                    "date": "2025-01-08T20:00Z",
                    "team": "Arsenal",
                    "opponent": "PSG",
                    "competition": "Champions League",
                },
                {
                    "date": "2025-01-11T15:00Z",
                    "team": "Arsenal",
                    "opponent": "Everton",
                    "competition": "Premier League",
                },
                # Chelsea: 1 match in last 14d
                {
                    "date": "2025-01-11T15:00Z",
                    "team": "Chelsea",
                    "opponent": "Liverpool",
                    "competition": "Premier League",
                },
            ]
        )

        features = extract_epl_schedule_features([], event, fixtures_df=df)
        # 14d window starts Jan 4 15:00. Arsenal matches on Jan 5, Jan 8, Jan 11 = 3
        assert features.home_matches_last_14d == 3.0
        assert features.away_matches_last_14d == 1.0

    def test_european_last_7d(self) -> None:
        """Detect European competition match in trailing 7-day window."""
        event = _make_event(
            commence_time=datetime(2025, 1, 11, 15, 0, tzinfo=UTC),
        )
        df = _make_fixtures_df(
            [
                # Arsenal played CL on Wednesday (3 days ago)
                {
                    "date": "2025-01-08T20:00Z",
                    "team": "Arsenal",
                    "opponent": "PSG",
                    "competition": "Champions League",
                },
                # Chelsea played only EPL
                {
                    "date": "2025-01-04T15:00Z",
                    "team": "Chelsea",
                    "opponent": "Liverpool",
                    "competition": "Premier League",
                },
            ]
        )

        features = extract_epl_schedule_features([], event, fixtures_df=df)
        assert features.home_european_last_7d == 1.0
        assert features.away_european_last_7d == 0.0

    def test_european_all_types(self) -> None:
        """Europa League and Conference League also count as European."""
        event = _make_event(
            commence_time=datetime(2025, 1, 11, 15, 0, tzinfo=UTC),
        )
        df = _make_fixtures_df(
            [
                {
                    "date": "2025-01-09T20:00Z",
                    "team": "Arsenal",
                    "opponent": "Roma",
                    "competition": "Europa League",
                },
                {
                    "date": "2025-01-09T20:00Z",
                    "team": "Chelsea",
                    "opponent": "Gent",
                    "competition": "Conference League",
                },
            ]
        )

        features = extract_epl_schedule_features([], event, fixtures_df=df)
        assert features.home_european_last_7d == 1.0
        assert features.away_european_last_7d == 1.0

    def test_european_outside_7d_window(self) -> None:
        """European match more than 7 days ago does not count."""
        event = _make_event(
            commence_time=datetime(2025, 1, 18, 15, 0, tzinfo=UTC),
        )
        df = _make_fixtures_df(
            [
                # CL match 10 days ago
                {
                    "date": "2025-01-08T20:00Z",
                    "team": "Arsenal",
                    "opponent": "PSG",
                    "competition": "Champions League",
                },
            ]
        )

        features = extract_epl_schedule_features([], event, fixtures_df=df)
        assert features.home_european_last_7d == 0.0

    def test_congestion_none_without_df(self) -> None:
        """Without fixtures_df, congestion features are None."""
        event = _make_event(commence_time=datetime(2025, 1, 11, 15, 0, tzinfo=UTC))
        features = extract_epl_schedule_features([], event)
        assert features.home_matches_last_14d is None
        assert features.away_matches_last_14d is None
        assert features.home_european_last_7d is None
        assert features.away_european_last_7d is None

    def test_congestion_with_empty_df(self) -> None:
        """Empty DataFrame produces zero counts (not None)."""
        event = _make_event(commence_time=datetime(2025, 1, 11, 15, 0, tzinfo=UTC))
        df = _make_fixtures_df(
            [
                {
                    "date": "2025-06-01T15:00Z",
                    "team": "Other",
                    "opponent": "Other2",
                    "competition": "Premier League",
                },
            ]
        )
        features = extract_epl_schedule_features([], event, fixtures_df=df)
        assert features.home_matches_last_14d == 0.0
        assert features.away_matches_last_14d == 0.0
        assert features.home_european_last_7d == 0.0
        assert features.away_european_last_7d == 0.0
