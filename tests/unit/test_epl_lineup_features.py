"""Unit tests for EPL lineup-delta feature extraction."""

from datetime import UTC, date, datetime

import numpy as np
from odds_analytics.epl_lineup_features import (
    EplLineupFeatures,
    LineupCache,
    _compute_team_features,
    _TeamMatchXI,
    build_lineup_cache,
    extract_epl_lineup_features,
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


def _make_cache(team_entries: dict[str, list[tuple[date, set[str]]]]) -> LineupCache:
    """Build a LineupCache from a simple dict.

    Keys are team names, values are lists of (match_date, player_ids) tuples
    in chronological order.
    """
    cache: LineupCache = {}
    for team, entries in team_entries.items():
        cache[team] = [_TeamMatchXI(match_date=d, player_ids=pids) for d, pids in entries]
    return cache


class TestEplLineupFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = EplLineupFeatures.get_feature_names()
        assert names == [
            "home_xi_changes",
            "away_xi_changes",
            "diff_xi_changes",
            "home_cumulative_starts_lost",
            "away_cumulative_starts_lost",
            "diff_cumulative_starts_lost",
        ]

    def test_to_array_all_none(self) -> None:
        features = EplLineupFeatures()
        arr = features.to_array()
        assert arr.shape == (6,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = EplLineupFeatures(
            home_xi_changes=3.0,
            away_xi_changes=1.0,
            diff_xi_changes=2.0,
            home_cumulative_starts_lost=50.0,
            away_cumulative_starts_lost=10.0,
            diff_cumulative_starts_lost=40.0,
        )
        arr = features.to_array()
        assert arr.shape == (6,)
        np.testing.assert_array_equal(arr, [3.0, 1.0, 2.0, 50.0, 10.0, 40.0])

    def test_feature_count_matches_names(self) -> None:
        assert len(EplLineupFeatures.get_feature_names()) == len(EplLineupFeatures().to_array())


class TestNormalCase:
    """Two matches with dropped players weighted by cumulative starts."""

    def test_two_matches_dropped_players(self) -> None:
        # Match 1: players {A, B, C, D, E, F, G, H, I, J, K}
        # Match 2: players {A, B, C, D, E, F, G, H, I, L, M}  (J and K dropped)
        xi_match1 = {f"p{i}" for i in range(1, 12)}  # p1..p11
        xi_match2 = (xi_match1 - {"p10", "p11"}) | {"p12", "p13"}

        cache = _make_cache(
            {
                "Arsenal": [
                    (date(2025, 1, 4), xi_match1),
                    (date(2025, 1, 11), xi_match2),
                ],
                "Chelsea": [
                    (date(2025, 1, 4), xi_match1),
                    (date(2025, 1, 11), xi_match1),  # unchanged XI
                ],
            }
        )

        event = _make_event(
            commence_time=datetime(2025, 1, 11, 15, 0, tzinfo=UTC),
        )
        features = extract_epl_lineup_features(cache, event)

        # Arsenal dropped 2 players
        assert features.home_xi_changes == 2.0
        # Each dropped player had 1 start in the window (only match 1)
        assert features.home_cumulative_starts_lost == 2.0

        # Chelsea kept same XI
        assert features.away_xi_changes == 0.0
        assert features.away_cumulative_starts_lost == 0.0

        # Diffs
        assert features.diff_xi_changes == 2.0
        assert features.diff_cumulative_starts_lost == 2.0

    def test_starts_weighted_by_history(self) -> None:
        """Dropped player with many prior starts should have higher cumulative_starts_lost."""
        # Player p11 starts in all 5 matches, then gets dropped in match 6
        base_xi = {f"p{i}" for i in range(1, 12)}  # p1..p11
        match6_xi = (base_xi - {"p11"}) | {"p12"}

        entries: list[tuple[date, set[str]]] = [
            (date(2025, 1, d), base_xi) for d in [1, 4, 7, 10, 13]
        ]
        entries.append((date(2025, 1, 16), match6_xi))

        cache = _make_cache({"Arsenal": entries, "Chelsea": entries})

        event = _make_event(
            commence_time=datetime(2025, 1, 16, 15, 0, tzinfo=UTC),
        )
        features = extract_epl_lineup_features(cache, event)

        assert features.home_xi_changes == 1.0
        # p11 started in all 5 prior matches
        assert features.home_cumulative_starts_lost == 5.0


class TestSeasonOpener:
    """First match returns None features (no prior match to compare against)."""

    def test_first_match_returns_none(self) -> None:
        xi = {f"p{i}" for i in range(1, 12)}
        cache = _make_cache(
            {
                "Arsenal": [(date(2025, 8, 17), xi)],
                "Chelsea": [(date(2025, 8, 17), xi)],
            }
        )
        event = _make_event(
            commence_time=datetime(2025, 8, 17, 15, 0, tzinfo=UTC),
        )
        features = extract_epl_lineup_features(cache, event)

        assert features.home_xi_changes is None
        assert features.away_xi_changes is None
        assert features.diff_xi_changes is None
        assert features.home_cumulative_starts_lost is None
        assert features.away_cumulative_starts_lost is None
        assert features.diff_cumulative_starts_lost is None


class TestMissingTeamInCache:
    """Returns all-None features when teams are not in the cache."""

    def test_both_teams_missing(self) -> None:
        cache: LineupCache = {}
        event = _make_event()
        features = extract_epl_lineup_features(cache, event)
        assert np.all(np.isnan(features.to_array()))

    def test_one_team_missing(self) -> None:
        xi1 = {f"p{i}" for i in range(1, 12)}
        xi2 = (xi1 - {"p11"}) | {"p12"}
        cache = _make_cache(
            {
                "Arsenal": [
                    (date(2025, 1, 4), xi1),
                    (date(2025, 1, 11), xi2),
                ],
            }
        )
        event = _make_event(
            commence_time=datetime(2025, 1, 11, 15, 0, tzinfo=UTC),
        )
        features = extract_epl_lineup_features(cache, event)

        # Arsenal has data
        assert features.home_xi_changes == 1.0
        assert features.home_cumulative_starts_lost == 1.0
        # Chelsea missing from cache
        assert features.away_xi_changes is None
        assert features.away_cumulative_starts_lost is None
        # Diffs are None because away is missing
        assert features.diff_xi_changes is None
        assert features.diff_cumulative_starts_lost is None


class TestWindowBoundary:
    """Verify the 38-match sliding window truncates correctly."""

    def test_window_truncates_at_38(self) -> None:
        """With 40 prior matches, only the last 38 should count for starts."""
        base_xi = {f"p{i}" for i in range(1, 12)}  # p1..p11

        # Build 40 matches where p11 starts in all of them
        entries: list[tuple[date, set[str]]] = []
        for i in range(40):
            d = date(2024, 1, 1 + i) if i < 30 else date(2024, 2, i - 29)
            entries.append((d, base_xi))

        # Match 41: drop p11
        final_date = date(2024, 3, 15)
        final_xi = (base_xi - {"p11"}) | {"p12"}
        entries.append((final_date, final_xi))

        cache = _make_cache({"Arsenal": entries, "Chelsea": entries})

        event = _make_event(
            home_team="Arsenal",
            away_team="Chelsea",
            commence_time=datetime(2024, 3, 15, 15, 0, tzinfo=UTC),
        )
        features = extract_epl_lineup_features(cache, event)

        assert features.home_xi_changes == 1.0
        # Window is 38 matches, so p11's starts should be capped at 38
        assert features.home_cumulative_starts_lost == 38.0

    def test_fewer_matches_than_window(self) -> None:
        """With only 5 prior matches, all should count (no truncation)."""
        base_xi = {f"p{i}" for i in range(1, 12)}

        entries: list[tuple[date, set[str]]] = [
            (date(2025, 1, d), base_xi) for d in [1, 4, 7, 10, 13]
        ]
        # Match 6: drop p11
        entries.append((date(2025, 1, 16), (base_xi - {"p11"}) | {"p12"}))

        cache = _make_cache({"Arsenal": entries, "Chelsea": entries})

        event = _make_event(
            commence_time=datetime(2025, 1, 16, 15, 0, tzinfo=UTC),
        )
        features = extract_epl_lineup_features(cache, event)

        assert features.home_xi_changes == 1.0
        # Only 5 prior matches, all within window -> 5 starts
        assert features.home_cumulative_starts_lost == 5.0


class TestNoneCache:
    """extract_epl_lineup_features handles None cache gracefully."""

    def test_none_cache_returns_all_none(self) -> None:
        event = _make_event()
        features = extract_epl_lineup_features(None, event)
        assert np.all(np.isnan(features.to_array()))


class TestBuildLineupCache:
    """Tests for build_lineup_cache from a DataFrame."""

    def test_builds_cache_from_dataframe(self) -> None:
        import pandas as pd

        df = pd.DataFrame(
            {
                "team": ["Arsenal"] * 3 + ["Chelsea"] * 2,
                "match_date": [
                    date(2025, 1, 4),
                    date(2025, 1, 4),
                    date(2025, 1, 11),
                    date(2025, 1, 4),
                    date(2025, 1, 4),
                ],
                "datetime": pd.to_datetime(
                    [
                        "2025-01-04 15:00",
                        "2025-01-04 15:00",
                        "2025-01-11 15:00",
                        "2025-01-04 15:00",
                        "2025-01-04 15:00",
                    ],
                    utc=True,
                ),
                "player_id": ["p1", "p2", "p3", "p4", "p5"],
            }
        )

        cache = build_lineup_cache(df)

        assert "Arsenal" in cache
        assert "Chelsea" in cache
        assert len(cache["Arsenal"]) == 2  # two match dates
        assert len(cache["Chelsea"]) == 1  # one match date
        assert cache["Arsenal"][0].player_ids == {"p1", "p2"}
        assert cache["Arsenal"][1].player_ids == {"p3"}


class TestComputeTeamFeatures:
    """Direct tests for _compute_team_features."""

    def test_returns_none_for_first_match(self) -> None:
        matches = [_TeamMatchXI(match_date=date(2025, 1, 4), player_ids={"p1", "p2"})]
        result = _compute_team_features(matches, date(2025, 1, 4))
        assert result is None

    def test_returns_none_when_date_not_found(self) -> None:
        matches = [_TeamMatchXI(match_date=date(2025, 1, 4), player_ids={"p1", "p2"})]
        result = _compute_team_features(matches, date(2025, 2, 1))
        assert result is None
