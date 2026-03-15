"""Unit tests for rolling match stats feature extraction."""

from datetime import UTC, datetime, timedelta

import numpy as np
from odds_analytics.match_stats_features import (
    MatchStatsFeatures,
    _extract_team_stats_from_raw_data,
    _rolling_average,
    _TeamMatchEntry,
    extract_match_stats_features,
    get_prior_match_stats_from_cache,
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


class TestMatchStatsFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = MatchStatsFeatures.get_feature_names()
        assert len(names) == 14
        assert names[0] == "home_avg_shots"
        assert names[1] == "away_avg_shots"
        assert names[-2] == "home_avg_ht_goals"
        assert names[-1] == "away_avg_ht_goals"

    def test_to_array_all_none(self) -> None:
        features = MatchStatsFeatures()
        arr = features.to_array()
        assert arr.shape == (14,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = MatchStatsFeatures(
            home_avg_shots=12.0,
            away_avg_shots=10.0,
            home_avg_corners=5.5,
            away_avg_corners=4.0,
        )
        arr = features.to_array()
        assert arr.shape == (14,)
        assert arr[0] == 12.0  # home_avg_shots
        assert arr[1] == 10.0  # away_avg_shots
        # Corners are at index 4, 5
        assert arr[4] == 5.5
        assert arr[5] == 4.0
        # Others should be NaN
        assert np.isnan(arr[2])  # home_avg_shots_on_target

    def test_feature_count_matches_names(self) -> None:
        assert len(MatchStatsFeatures.get_feature_names()) == len(MatchStatsFeatures().to_array())


class TestExtractTeamStats:
    def test_full_stats(self) -> None:
        raw_data = {
            "match_stats": {
                "home_shots": 15,
                "away_shots": 8,
                "home_shots_on_target": 6,
                "away_shots_on_target": 3,
                "home_corners": 7,
                "away_corners": 4,
                "home_fouls": 10,
                "away_fouls": 12,
                "home_yellow_cards": 2,
                "away_yellow_cards": 3,
                "home_red_cards": 0,
                "away_red_cards": 1,
                "home_ht_goals": 1,
                "away_ht_goals": 0,
                "referee": "Michael Oliver",
            }
        }
        result = _extract_team_stats_from_raw_data(raw_data, "Arsenal", "Chelsea")
        assert result["Arsenal"]["shots"] == 15
        assert result["Chelsea"]["shots"] == 8
        assert result["Arsenal"]["red_cards"] == 0
        assert result["Chelsea"]["red_cards"] == 1

    def test_missing_match_stats(self) -> None:
        raw_data = {"bookmakers": []}
        result = _extract_team_stats_from_raw_data(raw_data, "Arsenal", "Chelsea")
        assert result == {}

    def test_partial_stats(self) -> None:
        raw_data = {
            "match_stats": {
                "home_shots": 10,
                "away_shots": 5,
            }
        }
        result = _extract_team_stats_from_raw_data(raw_data, "Arsenal", "Chelsea")
        assert result["Arsenal"] == {"shots": 10}
        assert result["Chelsea"] == {"shots": 5}


class TestRollingAverage:
    def test_basic_average(self) -> None:
        dummy = datetime(2025, 1, 1, tzinfo=UTC)
        entries = [
            _TeamMatchEntry(dummy, {"shots": 10}),
            _TeamMatchEntry(dummy, {"shots": 12}),
            _TeamMatchEntry(dummy, {"shots": 8}),
        ]
        assert _rolling_average(entries, "shots", window=3) == 10.0

    def test_window_smaller_than_history(self) -> None:
        dummy = datetime(2025, 1, 1, tzinfo=UTC)
        entries = [
            _TeamMatchEntry(dummy, {"shots": 10}),
            _TeamMatchEntry(dummy, {"shots": 12}),
            _TeamMatchEntry(dummy, {"shots": 8}),
            _TeamMatchEntry(dummy, {"shots": 14}),
            _TeamMatchEntry(dummy, {"shots": 6}),
        ]
        # Window of 3: uses last 3 values (8, 14, 6) = 28/3
        result = _rolling_average(entries, "shots", window=3)
        assert result is not None
        assert abs(result - 28.0 / 3) < 0.001

    def test_window_larger_than_history(self) -> None:
        dummy = datetime(2025, 1, 1, tzinfo=UTC)
        entries = [
            _TeamMatchEntry(dummy, {"shots": 10}),
            _TeamMatchEntry(dummy, {"shots": 12}),
        ]
        # Window of 5 but only 2 values: uses all 2 = 22/2
        assert _rolling_average(entries, "shots", window=5) == 11.0

    def test_missing_stat_key(self) -> None:
        dummy = datetime(2025, 1, 1, tzinfo=UTC)
        entries = [_TeamMatchEntry(dummy, {"shots": 10})]
        assert _rolling_average(entries, "corners", window=5) is None

    def test_empty_entries(self) -> None:
        assert _rolling_average([], "shots", window=5) is None


class TestGetPriorMatchStatsFromCache:
    def test_filters_by_time(self) -> None:
        base = datetime(2025, 1, 1, 15, 0, tzinfo=UTC)
        cache = {
            "Arsenal": [
                _TeamMatchEntry(base, {"shots": 10}),
                _TeamMatchEntry(base + timedelta(days=7), {"shots": 12}),
                _TeamMatchEntry(base + timedelta(days=14), {"shots": 8}),
            ],
            "Chelsea": [
                _TeamMatchEntry(base, {"shots": 5}),
                _TeamMatchEntry(base + timedelta(days=7), {"shots": 7}),
            ],
        }
        event = _make_event(
            commence_time=base + timedelta(days=10),
            home_team="Arsenal",
            away_team="Chelsea",
        )
        result = get_prior_match_stats_from_cache(cache, event)
        # Arsenal: only first 2 entries (days 0 and 7 are before day 10)
        assert len(result["Arsenal"]) == 2
        assert result["Arsenal"][0]["shots"] == 10
        assert result["Arsenal"][1]["shots"] == 12
        # Chelsea: only first entry (day 0 before day 10, day 7 before day 10)
        assert len(result["Chelsea"]) == 2

    def test_no_prior_matches(self) -> None:
        base = datetime(2025, 1, 1, 15, 0, tzinfo=UTC)
        cache = {
            "Arsenal": [
                _TeamMatchEntry(base + timedelta(days=7), {"shots": 10}),
            ],
        }
        event = _make_event(
            commence_time=base,
            home_team="Arsenal",
            away_team="Chelsea",
        )
        result = get_prior_match_stats_from_cache(cache, event)
        assert "Arsenal" not in result
        assert "Chelsea" not in result

    def test_team_not_in_cache(self) -> None:
        cache = {}
        event = _make_event()
        result = get_prior_match_stats_from_cache(cache, event)
        assert result == {}


class TestExtractMatchStatsFeatures:
    def test_basic_extraction(self) -> None:
        prior = {
            "Arsenal": [
                {
                    "shots": 10,
                    "shots_on_target": 5,
                    "corners": 6,
                    "fouls": 10,
                    "yellow_cards": 2,
                    "red_cards": 0,
                    "ht_goals": 1,
                },
                {
                    "shots": 14,
                    "shots_on_target": 7,
                    "corners": 8,
                    "fouls": 8,
                    "yellow_cards": 1,
                    "red_cards": 0,
                    "ht_goals": 2,
                },
            ],
            "Chelsea": [
                {
                    "shots": 8,
                    "shots_on_target": 3,
                    "corners": 4,
                    "fouls": 12,
                    "yellow_cards": 3,
                    "red_cards": 1,
                    "ht_goals": 0,
                },
            ],
        }
        event = _make_event()
        features = extract_match_stats_features(prior, event, window=5)

        assert features.home_avg_shots == 12.0  # (10+14)/2
        assert features.home_avg_shots_on_target == 6.0  # (5+7)/2
        assert features.away_avg_shots == 8.0  # single match
        assert features.away_avg_red_cards == 1.0

    def test_empty_prior_stats(self) -> None:
        event = _make_event()
        features = extract_match_stats_features({}, event, window=5)
        assert np.all(np.isnan(features.to_array()))

    def test_one_team_missing(self) -> None:
        prior = {
            "Arsenal": [{"shots": 10, "corners": 6}],
        }
        event = _make_event()
        features = extract_match_stats_features(prior, event, window=5)
        assert features.home_avg_shots == 10.0
        assert features.away_avg_shots is None

    def test_window_respected(self) -> None:
        prior = {
            "Arsenal": [
                {"shots": 100},
                {"shots": 10},
                {"shots": 12},
                {"shots": 14},
            ],
        }
        event = _make_event()
        features = extract_match_stats_features(prior, event, window=3)
        # Last 3: 10, 12, 14 -> avg 12
        assert features.home_avg_shots == 12.0

    def test_fewer_matches_than_window(self) -> None:
        prior = {
            "Arsenal": [{"shots": 10}, {"shots": 14}],
        }
        event = _make_event()
        features = extract_match_stats_features(prior, event, window=5)
        # Only 2 matches, uses all -> avg 12
        assert features.home_avg_shots == 12.0
