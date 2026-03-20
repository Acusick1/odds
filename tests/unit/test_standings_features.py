"""Unit tests for EPL standings and form feature extraction."""

from datetime import UTC, datetime, timedelta

import numpy as np
from odds_analytics.standings_features import (
    StandingsFeatures,
    TeamRecord,
    build_league_table,
    epl_season_key,
    extract_standings_features,
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


def _make_completed_event(
    event_id: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    commence_time: datetime,
) -> Event:
    return _make_event(
        event_id=event_id,
        home_team=home_team,
        away_team=away_team,
        home_score=home_score,
        away_score=away_score,
        commence_time=commence_time,
        status=EventStatus.FINAL,
    )


class TestStandingsFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = StandingsFeatures.get_feature_names()
        assert names == [
            "home_league_position",
            "away_league_position",
            "points_gap",
            "home_goal_difference",
            "away_goal_difference",
            "home_form_last_5",
            "away_form_last_5",
            "home_goals_scored_rate",
            "home_goals_conceded_rate",
            "away_goals_scored_rate",
            "away_goals_conceded_rate",
        ]

    def test_to_array_all_none(self) -> None:
        features = StandingsFeatures()
        arr = features.to_array()
        assert arr.shape == (11,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = StandingsFeatures(
            home_league_position=1.0,
            away_league_position=5.0,
            points_gap=12.0,
            home_goal_difference=20.0,
            away_goal_difference=3.0,
            home_form_last_5=2.4,
            away_form_last_5=1.6,
            home_goals_scored_rate=2.1,
            home_goals_conceded_rate=0.8,
            away_goals_scored_rate=1.5,
            away_goals_conceded_rate=1.2,
        )
        arr = features.to_array()
        assert arr.shape == (11,)
        assert arr[0] == 1.0
        assert arr[1] == 5.0

    def test_feature_count_matches_names(self) -> None:
        assert len(StandingsFeatures.get_feature_names()) == len(StandingsFeatures().to_array())


class TestTeamRecord:
    def test_points_calculation(self) -> None:
        record = TeamRecord(team="Arsenal")
        record.record_result(2, 1)  # W
        record.record_result(1, 1)  # D
        record.record_result(0, 3)  # L
        assert record.points == 4  # 3 + 1 + 0
        assert record.played == 3
        assert record.won == 1
        assert record.drawn == 1
        assert record.lost == 1

    def test_goal_difference(self) -> None:
        record = TeamRecord(team="Arsenal")
        record.record_result(3, 0)
        record.record_result(1, 2)
        assert record.goal_difference == 2  # (3+1) - (0+2)

    def test_form_last_n(self) -> None:
        record = TeamRecord(team="Arsenal")
        # 6 results: form_last_5 should use only last 5
        record.record_result(0, 1)  # L -> 0 (excluded from window)
        record.record_result(2, 0)  # W -> 3
        record.record_result(1, 1)  # D -> 1
        record.record_result(3, 0)  # W -> 3
        record.record_result(0, 0)  # D -> 1
        record.record_result(2, 1)  # W -> 3
        assert record.form_last_n == (3 + 1 + 3 + 1 + 3) / 5

    def test_form_last_n_fewer_than_5(self) -> None:
        record = TeamRecord(team="Arsenal")
        record.record_result(2, 0)  # W -> 3
        record.record_result(1, 1)  # D -> 1
        assert record.form_last_n == (3 + 1) / 2

    def test_form_last_n_empty(self) -> None:
        record = TeamRecord(team="Arsenal")
        assert record.form_last_n is None

    def test_form_window_custom(self) -> None:
        record = TeamRecord(team="Arsenal", form_window=3)
        record.record_result(0, 1)  # L -> 0
        record.record_result(2, 0)  # W -> 3
        record.record_result(1, 1)  # D -> 1
        record.record_result(3, 0)  # W -> 3
        # Last 3: W(3) D(1) W(3) = 7/3
        assert record.form_last_n == (3 + 1 + 3) / 3

    def test_goals_rates_zero_played(self) -> None:
        record = TeamRecord(team="Arsenal")
        assert record.goals_scored_rate is None
        assert record.goals_conceded_rate is None


class TestEplSeasonKey:
    def test_august_start(self) -> None:
        dt = datetime(2024, 8, 17, 15, 0, tzinfo=UTC)
        assert epl_season_key(dt) == "2024-25"

    def test_january_mid_season(self) -> None:
        dt = datetime(2025, 1, 15, 15, 0, tzinfo=UTC)
        assert epl_season_key(dt) == "2024-25"

    def test_may_end_season(self) -> None:
        dt = datetime(2025, 5, 25, 15, 0, tzinfo=UTC)
        assert epl_season_key(dt) == "2024-25"

    def test_june_end_season(self) -> None:
        dt = datetime(2025, 6, 1, 15, 0, tzinfo=UTC)
        assert epl_season_key(dt) == "2024-25"

    def test_july_prior_season(self) -> None:
        dt = datetime(2025, 7, 1, 15, 0, tzinfo=UTC)
        assert epl_season_key(dt) == "2024-25"


class TestBuildLeagueTable:
    def test_simple_table(self) -> None:
        events = [
            _make_completed_event(
                "e1",
                "Arsenal",
                "Chelsea",
                2,
                0,
                datetime(2024, 8, 17, 15, 0, tzinfo=UTC),
            ),
            _make_completed_event(
                "e2",
                "Liverpool",
                "Arsenal",
                1,
                1,
                datetime(2024, 8, 24, 15, 0, tzinfo=UTC),
            ),
        ]
        table = build_league_table(events)
        assert table["Arsenal"].points == 4  # W + D
        assert table["Chelsea"].points == 0
        assert table["Liverpool"].points == 1

    def test_skips_non_final(self) -> None:
        events = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Chelsea",
                home_score=2,
                away_score=0,
                status=EventStatus.SCHEDULED,
            ),
        ]
        table = build_league_table(events)
        assert len(table) == 0

    def test_custom_form_window(self) -> None:
        events = [
            _make_completed_event(
                "e1", "Arsenal", "Chelsea", 2, 0, datetime(2024, 8, 17, 15, 0, tzinfo=UTC)
            ),
            _make_completed_event(
                "e2", "Chelsea", "Arsenal", 1, 1, datetime(2024, 8, 24, 15, 0, tzinfo=UTC)
            ),
        ]
        table = build_league_table(events, form_window=3)
        assert table["Arsenal"].form_window == 3

    def test_skips_null_scores(self) -> None:
        events = [
            _make_event(
                event_id="e1",
                home_team="Arsenal",
                away_team="Chelsea",
                home_score=None,
                away_score=None,
                status=EventStatus.FINAL,
            ),
        ]
        table = build_league_table(events)
        assert len(table) == 0


class TestExtractStandingsFeatures:
    def test_empty_prior_events(self) -> None:
        event = _make_event()
        features = extract_standings_features([], event)
        assert np.all(np.isnan(features.to_array()))

    def test_mid_season_extraction(self) -> None:
        """After 3 matchdays, check standings are correct."""
        prior = [
            # Matchday 1
            _make_completed_event(
                "e1",
                "Arsenal",
                "Chelsea",
                2,
                0,
                datetime(2024, 8, 17, 15, 0, tzinfo=UTC),
            ),
            _make_completed_event(
                "e2",
                "Liverpool",
                "Man City",
                1,
                1,
                datetime(2024, 8, 17, 17, 30, tzinfo=UTC),
            ),
            # Matchday 2
            _make_completed_event(
                "e3",
                "Chelsea",
                "Liverpool",
                0,
                2,
                datetime(2024, 8, 24, 15, 0, tzinfo=UTC),
            ),
            _make_completed_event(
                "e4",
                "Man City",
                "Arsenal",
                1,
                3,
                datetime(2024, 8, 24, 17, 30, tzinfo=UTC),
            ),
        ]
        # Arsenal: W, W => 6pts, GD=+4, GF=5 GA=1
        # Liverpool: D, W => 4pts, GD=+2, GF=3 GA=1
        # Man City: D, L => 1pt, GD=-2, GF=2 GA=4
        # Chelsea: L, L => 0pts, GD=-4, GF=0 GA=4

        # Matchday 3 event: Liverpool vs Arsenal
        event = _make_event(
            event_id="e5",
            home_team="Liverpool",
            away_team="Arsenal",
            commence_time=datetime(2024, 8, 31, 15, 0, tzinfo=UTC),
        )
        features = extract_standings_features(prior, event)

        # Liverpool is 2nd (4pts), Arsenal is 1st (6pts)
        assert features.home_league_position == 2.0
        assert features.away_league_position == 1.0
        assert features.points_gap == -2.0  # Liverpool 4 - Arsenal 6
        assert features.home_goal_difference == 2.0
        assert features.away_goal_difference == 4.0
        assert features.home_goals_scored_rate == 3.0 / 2
        assert features.away_goals_scored_rate == 5.0 / 2

    def test_season_start_edge_case(self) -> None:
        """First match of the season — no prior events."""
        event = _make_event(
            commence_time=datetime(2024, 8, 17, 15, 0, tzinfo=UTC),
        )
        features = extract_standings_features([], event)
        assert np.all(np.isnan(features.to_array()))

    def test_promoted_team_first_season(self) -> None:
        """Newly promoted team not in prior events gets None features."""
        prior = [
            _make_completed_event(
                "e1",
                "Arsenal",
                "Chelsea",
                2,
                0,
                datetime(2024, 8, 17, 15, 0, tzinfo=UTC),
            ),
        ]
        # Ipswich is promoted, not in prior events
        event = _make_event(
            event_id="e2",
            home_team="Ipswich Town",
            away_team="Arsenal",
            commence_time=datetime(2024, 8, 24, 15, 0, tzinfo=UTC),
        )
        features = extract_standings_features(prior, event)

        # Ipswich has no record, Arsenal does
        assert features.home_league_position is None
        assert features.away_league_position == 1.0
        assert features.points_gap is None
        assert features.home_goal_difference is None
        assert features.away_goal_difference == 2.0

    def test_form_uses_rolling_window(self) -> None:
        """Verify form uses last 5 matches, not cumulative."""
        # Arsenal plays 6 matches before the target event
        prior = []
        base = datetime(2024, 8, 17, 15, 0, tzinfo=UTC)
        opponents = ["Chelsea", "Liverpool", "Man City", "Tottenham", "Everton", "Brighton"]
        # Results: L, W, W, W, D, W
        scores = [(0, 1), (3, 0), (2, 1), (1, 0), (0, 0), (4, 0)]
        for i, (opp, (hs, as_)) in enumerate(zip(opponents, scores, strict=True)):
            prior.append(
                _make_completed_event(
                    f"e{i}",
                    "Arsenal",
                    opp,
                    hs,
                    as_,
                    base + timedelta(days=7 * i),
                )
            )

        event = _make_event(
            event_id="e_target",
            home_team="Arsenal",
            away_team="Wolves",
            commence_time=base + timedelta(days=7 * 6),
        )
        features = extract_standings_features(prior, event)

        # Last 5 results (excluding first L): W(3) W(3) W(3) D(1) W(3) = 13/5 = 2.6
        assert features.home_form_last_5 is not None
        assert abs(features.home_form_last_5 - 2.6) < 0.01

    def test_custom_form_window(self) -> None:
        """Custom form_window=3 uses last 3 matches instead of 5."""
        prior = []
        base = datetime(2024, 8, 17, 15, 0, tzinfo=UTC)
        opponents = ["Chelsea", "Liverpool", "Man City", "Tottenham"]
        # Results: L, W, W, D -> form with window=3: W(3) W(3) D(1) = 7/3
        scores = [(0, 1), (3, 0), (2, 1), (0, 0)]
        for i, (opp, (hs, as_)) in enumerate(zip(opponents, scores, strict=True)):
            prior.append(
                _make_completed_event(
                    f"e{i}", "Arsenal", opp, hs, as_, base + timedelta(days=7 * i)
                )
            )

        event = _make_event(
            event_id="e_target",
            home_team="Arsenal",
            away_team="Wolves",
            commence_time=base + timedelta(days=7 * 4),
        )
        features = extract_standings_features(prior, event, form_window=3)
        assert features.home_form_last_5 is not None
        assert abs(features.home_form_last_5 - (3 + 3 + 1) / 3) < 0.01

    def test_both_teams_unknown(self) -> None:
        """Both teams not in prior events returns all None."""
        prior = [
            _make_completed_event(
                "e1",
                "Arsenal",
                "Chelsea",
                2,
                0,
                datetime(2024, 8, 17, 15, 0, tzinfo=UTC),
            ),
        ]
        event = _make_event(
            event_id="e2",
            home_team="Ipswich Town",
            away_team="Leicester City",
            commence_time=datetime(2024, 8, 24, 15, 0, tzinfo=UTC),
        )
        features = extract_standings_features(prior, event)
        assert np.all(np.isnan(features.to_array()))

    def test_draw_awards_correct_points(self) -> None:
        prior = [
            _make_completed_event(
                "e1",
                "Arsenal",
                "Chelsea",
                1,
                1,
                datetime(2024, 8, 17, 15, 0, tzinfo=UTC),
            ),
        ]
        table = build_league_table(prior)
        assert table["Arsenal"].points == 1
        assert table["Chelsea"].points == 1
