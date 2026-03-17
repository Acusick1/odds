"""Unit tests for FPL availability feature extraction."""

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from odds_analytics.fpl_availability_features import (
    FplAvailabilityCache,
    FplAvailabilityFeatures,
    _build_fpl_to_espn_player_map,
    _precompute_cumulative_starts,
    extract_fpl_availability_features,
)
from odds_core.models import Event, EventStatus


def _make_event(
    event_id: str = "test_event",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    commence_time: datetime | None = None,
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
        home_score=None,
        away_score=None,
        status=status,
    )


def _make_fpl_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True)
    df["chance_of_playing"] = df["chance_of_playing"].astype(float)
    return df


def _make_lineup_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "team",
                "player_id",
                "player_name",
                "datetime",
                "match_date",
                "starter",
            ]
        )
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["date"], utc=True)
    df["match_date"] = df["datetime"].dt.date
    df["starter"] = True
    return df


def _make_cache(
    fpl_rows: list[dict],
    lineup_rows: list[dict],
) -> FplAvailabilityCache:
    fpl_df = _make_fpl_df(fpl_rows)
    lineup_df = _make_lineup_df(lineup_rows)
    player_map = _build_fpl_to_espn_player_map(fpl_df, lineup_df)
    starts_lookup = _precompute_cumulative_starts(lineup_df)
    snapshot_times = sorted(fpl_df["snapshot_time"].unique())
    return FplAvailabilityCache(
        fpl_df=fpl_df,
        player_map=player_map,
        starts_lookup=starts_lookup,
        snapshot_times=snapshot_times,
    )


class TestFplAvailabilityFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = FplAvailabilityFeatures.get_feature_names()
        assert names == [
            "home_expected_disruption",
            "away_expected_disruption",
            "diff_expected_disruption",
            "home_expected_disruption_unweighted",
            "away_expected_disruption_unweighted",
            "diff_expected_disruption_unweighted",
            "home_n_flagged_players",
            "away_n_flagged_players",
            "diff_n_flagged_players",
        ]

    def test_to_array_all_none(self) -> None:
        features = FplAvailabilityFeatures()
        arr = features.to_array()
        assert arr.shape == (9,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = FplAvailabilityFeatures(
            home_expected_disruption=5.0,
            away_expected_disruption=2.0,
            diff_expected_disruption=3.0,
            home_expected_disruption_unweighted=0.5,
            away_expected_disruption_unweighted=0.25,
            diff_expected_disruption_unweighted=0.25,
            home_n_flagged_players=1.0,
            away_n_flagged_players=1.0,
            diff_n_flagged_players=0.0,
        )
        arr = features.to_array()
        assert arr.shape == (9,)
        np.testing.assert_array_equal(arr, [5.0, 2.0, 3.0, 0.5, 0.25, 0.25, 1.0, 1.0, 0.0])

    def test_feature_count_matches_names(self) -> None:
        assert len(FplAvailabilityFeatures.get_feature_names()) == len(
            FplAvailabilityFeatures().to_array()
        )


class TestNoneCache:
    def test_none_cache_returns_all_nan(self) -> None:
        event = _make_event()
        features = extract_fpl_availability_features(None, event)
        assert np.all(np.isnan(features.to_array()))


class TestNoSnapshotInWindow:
    def test_no_snapshot_before_event(self) -> None:
        snap_time = "2025-01-10T12:00:00+00:00"  # 5 days before commence
        event = _make_event(
            commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        )

        fpl_rows = [
            {
                "snapshot_time": snap_time,
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Player A",
                "chance_of_playing": 75.0,
            }
        ]
        lineup_rows = [
            {
                "date": "2025-01-04T15:00:00+00:00",
                "team": "Arsenal",
                "player_id": "esp_1",
                "player_name": "Player A",
            }
        ]
        cache = _make_cache(fpl_rows, lineup_rows)
        features = extract_fpl_availability_features(cache, event)
        # No snapshot within 48h window -> all NaN
        assert np.all(np.isnan(features.to_array()))

    def test_snapshot_after_event_ignored(self) -> None:
        snap_time = "2025-01-15T18:00:00+00:00"  # after commence
        event = _make_event(
            commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        )

        fpl_rows = [
            {
                "snapshot_time": snap_time,
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Player A",
                "chance_of_playing": 75.0,
            }
        ]
        lineup_rows = [
            {
                "date": "2025-01-04T15:00:00+00:00",
                "team": "Arsenal",
                "player_id": "esp_1",
                "player_name": "Player A",
            }
        ]
        cache = _make_cache(fpl_rows, lineup_rows)
        features = extract_fpl_availability_features(cache, event)
        assert np.all(np.isnan(features.to_array()))


class TestFitPlayerWithDoubt:
    def test_single_player_50_percent_chance(self) -> None:
        snap_time = "2025-01-14T12:00:00+00:00"
        event = _make_event(
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        )

        fpl_rows = [
            # Arsenal: one player at 50% -> severity 0.5, unmatched in ESPN
            {
                "snapshot_time": snap_time,
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Saka",
                "chance_of_playing": 50.0,
            },
            # Chelsea: no flagged players
            {
                "snapshot_time": snap_time,
                "team": "Chelsea",
                "player_code": 2001,
                "player_name": "Palmer",
                "chance_of_playing": 100.0,
            },
        ]
        lineup_rows = [
            {
                "date": "2025-01-04T15:00:00+00:00",
                "team": "Arsenal",
                "player_id": "esp_1",
                "player_name": "Saka",
            },
            {
                "date": "2025-01-04T15:00:00+00:00",
                "team": "Chelsea",
                "player_id": "esp_2",
                "player_name": "Palmer",
            },
        ]

        cache = _make_cache(fpl_rows, lineup_rows)
        features = extract_fpl_availability_features(cache, event)

        # Arsenal: 1 flagged player, severity=0.5
        assert features.home_n_flagged_players == 1.0
        assert features.home_expected_disruption_unweighted == pytest.approx(0.5)
        # Chelsea: no flagged players
        assert features.away_n_flagged_players == 0.0
        assert features.away_expected_disruption_unweighted == pytest.approx(0.0)
        # Diffs
        assert features.diff_n_flagged_players == pytest.approx(1.0)
        assert features.diff_expected_disruption_unweighted == pytest.approx(0.5)

    def test_player_100_percent_not_flagged(self) -> None:
        snap_time = "2025-01-14T12:00:00+00:00"
        event = _make_event(
            commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        )

        fpl_rows = [
            {
                "snapshot_time": snap_time,
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Player A",
                "chance_of_playing": 100.0,
            },
            {
                "snapshot_time": snap_time,
                "team": "Chelsea",
                "player_code": 2001,
                "player_name": "Player B",
                "chance_of_playing": 100.0,
            },
        ]
        lineup_rows: list[dict] = []
        cache = _make_cache(fpl_rows, lineup_rows)
        features = extract_fpl_availability_features(cache, event)

        assert features.home_n_flagged_players == 0.0
        assert features.away_n_flagged_players == 0.0
        assert features.home_expected_disruption == pytest.approx(0.0)


class TestWeightedDisruption:
    def test_weighted_by_cumulative_starts(self) -> None:
        """Player with more starts produces higher weighted disruption.

        The starts_lookup is keyed by the event match_date, so a lineup entry
        on the event date must exist (representing this match) for the lookup to
        populate prior-start counts via the precompute function.
        """
        snap_time = "2025-01-14T12:00:00+00:00"
        event_date = "2025-01-15T15:00:00+00:00"
        # Prior 4 match dates for Arsenal (Saka starts in all)
        prior_arsenal_dates = [f"2025-01-{d:02d}T15:00:00+00:00" for d in [1, 4, 7, 10]]
        # Prior 1 match date for Chelsea (Palmer starts once)
        prior_chelsea_dates = ["2025-01-01T15:00:00+00:00"]

        lineup_rows = (
            [
                {
                    "date": d,
                    "team": "Arsenal",
                    "player_id": "esp_saka",
                    "player_name": "Saka",
                }
                for d in prior_arsenal_dates
            ]
            + [
                # Event date entry for Arsenal (creates lookup key for Jan 15)
                {
                    "date": event_date,
                    "team": "Arsenal",
                    "player_id": "esp_saka",
                    "player_name": "Saka",
                }
            ]
            + [
                {
                    "date": d,
                    "team": "Chelsea",
                    "player_id": "esp_palmer",
                    "player_name": "Palmer",
                }
                for d in prior_chelsea_dates
            ]
            + [
                # Event date entry for Chelsea
                {
                    "date": event_date,
                    "team": "Chelsea",
                    "player_id": "esp_palmer",
                    "player_name": "Palmer",
                }
            ]
        )

        fpl_rows = [
            {
                "snapshot_time": snap_time,
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Saka",
                "chance_of_playing": 50.0,  # severity 0.5
            },
            {
                "snapshot_time": snap_time,
                "team": "Chelsea",
                "player_code": 2001,
                "player_name": "Palmer",
                "chance_of_playing": 50.0,  # severity 0.5
            },
        ]

        event = _make_event(
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        )

        cache = _make_cache(fpl_rows, lineup_rows)
        features = extract_fpl_availability_features(cache, event)

        # Saka: 4 prior starts, severity 0.5 -> weighted = 4 * 0.5 = 2.0
        assert features.home_expected_disruption == pytest.approx(2.0)
        # Palmer: 1 prior start, severity 0.5 -> weighted = 1 * 0.5 = 0.5
        assert features.away_expected_disruption == pytest.approx(0.5)
        # Diff = 1.5
        assert features.diff_expected_disruption == pytest.approx(1.5)


class TestPlayerMapFuzzyMatching:
    def test_exact_name_match(self) -> None:
        fpl_df = pd.DataFrame(
            [
                {
                    "snapshot_time": "2025-01-01T00:00:00+00:00",
                    "team": "Arsenal",
                    "player_code": 1001,
                    "player_name": "Saka",
                    "chance_of_playing": 75.0,
                }
            ]
        )
        fpl_df["snapshot_time"] = pd.to_datetime(fpl_df["snapshot_time"], utc=True)

        lineup_df = pd.DataFrame(
            [
                {
                    "date": "2025-01-01T15:00:00+00:00",
                    "team": "Arsenal",
                    "player_id": "esp_001",
                    "player_name": "Bukayo Saka",
                }
            ]
        )
        lineup_df["datetime"] = pd.to_datetime(lineup_df["date"], utc=True)
        lineup_df["match_date"] = lineup_df["datetime"].dt.date

        player_map = _build_fpl_to_espn_player_map(fpl_df, lineup_df)
        # "Saka" should match "Bukayo Saka" via last-name comparison
        assert ("Arsenal", 1001) in player_map
        assert player_map[("Arsenal", 1001)] == "esp_001"

    def test_no_espn_team_returns_no_match(self) -> None:
        fpl_df = pd.DataFrame(
            [
                {
                    "snapshot_time": "2025-01-01T00:00:00+00:00",
                    "team": "Arsenal",
                    "player_code": 1001,
                    "player_name": "Saka",
                    "chance_of_playing": 75.0,
                }
            ]
        )
        fpl_df["snapshot_time"] = pd.to_datetime(fpl_df["snapshot_time"], utc=True)

        # Lineup for a different team
        lineup_df = pd.DataFrame(
            [
                {
                    "date": "2025-01-01T15:00:00+00:00",
                    "team": "Chelsea",
                    "player_id": "esp_001",
                    "player_name": "Palmer",
                }
            ]
        )
        lineup_df["datetime"] = pd.to_datetime(lineup_df["date"], utc=True)
        lineup_df["match_date"] = lineup_df["datetime"].dt.date

        player_map = _build_fpl_to_espn_player_map(fpl_df, lineup_df)
        assert ("Arsenal", 1001) not in player_map


class TestPrecomputeCumulativeStarts:
    def test_no_prior_matches(self) -> None:
        lineup_df = _make_lineup_df(
            [
                {
                    "date": "2025-01-15T15:00:00+00:00",
                    "team": "Arsenal",
                    "player_id": "p1",
                    "player_name": "Player 1",
                }
            ]
        )
        starts_lookup = _precompute_cumulative_starts(lineup_df)
        match_date = date(2025, 1, 15)
        assert starts_lookup[("Arsenal", match_date)] == {}

    def test_counts_prior_starts_correctly(self) -> None:
        rows = [
            {
                "date": f"2025-01-{d:02d}T15:00:00+00:00",
                "team": "Arsenal",
                "player_id": "p1",
                "player_name": "Player 1",
            }
            for d in [1, 4, 7, 10]
        ] + [
            {
                "date": "2025-01-15T15:00:00+00:00",
                "team": "Arsenal",
                "player_id": "p2",
                "player_name": "Player 2",
            }
        ]
        lineup_df = _make_lineup_df(rows)
        starts_lookup = _precompute_cumulative_starts(lineup_df)

        # On Jan 15, p1 has 4 prior starts and p2 has 0
        assert starts_lookup[("Arsenal", date(2025, 1, 15))]["p1"] == 4
        assert starts_lookup[("Arsenal", date(2025, 1, 15))].get("p2", 0) == 0


class TestSnapshotWindowBoundary:
    def test_uses_latest_snapshot_in_window(self) -> None:
        event = _make_event(
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        )
        # Two snapshots: one at 12h before (within 48h), one at 24h before
        snap_early = "2025-01-14T15:00:00+00:00"  # 24h before
        snap_late = "2025-01-15T03:00:00+00:00"  # 12h before

        fpl_rows = [
            {
                "snapshot_time": snap_early,
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Player A",
                "chance_of_playing": 50.0,
            },
            {
                "snapshot_time": snap_late,
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Player A",
                "chance_of_playing": 75.0,  # updated to 75%
            },
            {
                "snapshot_time": snap_early,
                "team": "Chelsea",
                "player_code": 2001,
                "player_name": "Player B",
                "chance_of_playing": 100.0,
            },
            {
                "snapshot_time": snap_late,
                "team": "Chelsea",
                "player_code": 2001,
                "player_name": "Player B",
                "chance_of_playing": 100.0,
            },
        ]
        lineup_rows: list[dict] = []
        cache = _make_cache(fpl_rows, lineup_rows)
        features = extract_fpl_availability_features(cache, event)

        # Latest snapshot has 75% -> severity = 0.25
        assert features.home_expected_disruption_unweighted == pytest.approx(0.25)

    def test_snapshot_exactly_48h_before_included(self) -> None:
        commence = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
        snap_at_boundary = commence - timedelta(hours=48)

        event = _make_event(
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=commence,
        )
        fpl_rows = [
            {
                "snapshot_time": snap_at_boundary.isoformat(),
                "team": "Arsenal",
                "player_code": 1001,
                "player_name": "Player A",
                "chance_of_playing": 0.0,
            },
            {
                "snapshot_time": snap_at_boundary.isoformat(),
                "team": "Chelsea",
                "player_code": 2001,
                "player_name": "Player B",
                "chance_of_playing": 100.0,
            },
        ]
        lineup_rows: list[dict] = []
        cache = _make_cache(fpl_rows, lineup_rows)
        features = extract_fpl_availability_features(cache, event)

        # Snapshot at exactly 48h is included
        assert features.home_n_flagged_players == pytest.approx(1.0)
        assert features.home_expected_disruption_unweighted == pytest.approx(1.0)
