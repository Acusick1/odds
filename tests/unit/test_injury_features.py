"""Unit tests for injury feature extraction."""

from datetime import UTC, datetime

import numpy as np
import pytest
from odds_analytics.injury_features import (
    InjuryFeatures,
    _compute_player_impact,
    date_to_nba_season,
    extract_injury_features,
)
from odds_core.injury_models import InjuryReport, InjuryStatus
from odds_core.models import Event
from odds_core.player_stats_models import NbaPlayerSeasonStats


def _make_event(
    home_team: str = "Los Angeles Lakers",
    away_team: str = "Boston Celtics",
    commence_time: datetime | None = None,
) -> Event:
    if commence_time is None:
        commence_time = datetime(2025, 1, 15, 0, 30, tzinfo=UTC)
    return Event(
        id="test_event_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        home_team=home_team,
        away_team=away_team,
        commence_time=commence_time,
    )


def _make_report(
    team: str = "Los Angeles Lakers",
    player_name: str = "James, LeBron",
    status: InjuryStatus = InjuryStatus.OUT,
    report_time: datetime | None = None,
    event_id: str = "test_event_1",
    reason: str = "Left Ankle; Sprain",
) -> InjuryReport:
    if report_time is None:
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)
    return InjuryReport(
        id=1,
        report_time=report_time,
        game_date=report_time.date(),
        game_time_et="07:30 PM ET",
        matchup="LAL@BOS",
        team=team,
        player_name=player_name,
        status=status,
        reason=reason,
        event_id=event_id,
    )


def _make_player_stats(
    player_name: str = "James, LeBron",
    team_abbreviation: str = "LAL",
    season: str = "2024-25",
    minutes: float = 2520.0,
    games_played: int = 70,
    on_off_rtg: float | None = 118.5,
    on_def_rtg: float | None = 112.0,
) -> NbaPlayerSeasonStats:
    return NbaPlayerSeasonStats(
        id=1,
        player_id=2544,
        player_name=player_name,
        team_id=1610612747,
        team_abbreviation=team_abbreviation,
        season=season,
        minutes=minutes,
        games_played=games_played,
        on_off_rtg=on_off_rtg,
        on_def_rtg=on_def_rtg,
        usage=0.30,
        ts_pct=0.60,
        efg_pct=0.55,
        assists=500,
        turnovers=250,
        rebounds=550,
        steals=80,
        blocks=40,
        points=1800,
        plus_minus=200.0,
    )


class TestInjuryFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = InjuryFeatures.get_feature_names()
        assert names == [
            "impact_out_home",
            "impact_out_away",
            "impact_gtd_home",
            "impact_gtd_away",
            "report_hours_before_game",
            "injury_news_recency",
        ]

    def test_to_array_all_none(self) -> None:
        features = InjuryFeatures()
        arr = features.to_array()
        assert arr.shape == (6,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = InjuryFeatures(
            impact_out_home=2.5,
            impact_out_away=1.2,
            impact_gtd_home=0.8,
            impact_gtd_away=0.0,
            report_hours_before_game=4.5,
            injury_news_recency=1.0,
        )
        arr = features.to_array()
        assert arr.shape == (6,)
        np.testing.assert_array_equal(arr, [2.5, 1.2, 0.8, 0.0, 4.5, 1.0])

    def test_to_array_partial_none(self) -> None:
        features = InjuryFeatures(impact_out_home=3.0)
        arr = features.to_array()
        assert arr[0] == 3.0
        assert np.isnan(arr[1])

    def test_feature_count_matches_names(self) -> None:
        assert len(InjuryFeatures.get_feature_names()) == len(InjuryFeatures().to_array())


class TestComputePlayerImpact:
    def test_none_stats_returns_1(self) -> None:
        assert _compute_player_impact(None) == 1.0

    def test_none_on_off_rtg_returns_1(self) -> None:
        stats = _make_player_stats(on_off_rtg=None)
        assert _compute_player_impact(stats) == 1.0

    def test_none_on_def_rtg_returns_1(self) -> None:
        stats = _make_player_stats(on_def_rtg=None)
        assert _compute_player_impact(stats) == 1.0

    def test_zero_games_returns_1(self) -> None:
        stats = _make_player_stats(games_played=0)
        assert _compute_player_impact(stats) == 1.0

    def test_normal_computation(self) -> None:
        # on_off_rtg=118.5, on_def_rtg=112.0 -> net=6.5
        # minutes=2520, games=70 -> mpg=36.0
        # impact = 6.5 * (36.0 / 48.0) = 6.5 * 0.75 = 4.875
        stats = _make_player_stats(
            on_off_rtg=118.5, on_def_rtg=112.0, minutes=2520.0, games_played=70
        )
        result = _compute_player_impact(stats)
        assert result == pytest.approx(4.875)

    def test_negative_net_rating(self) -> None:
        # Player with negative impact (worse on court)
        # on_off_rtg=108.0, on_def_rtg=115.0 -> net=-7.0
        # minutes=1200, games=60 -> mpg=20.0
        # impact = -7.0 * (20.0/48.0) = -7.0 * 0.4167 = -2.9167
        stats = _make_player_stats(
            on_off_rtg=108.0, on_def_rtg=115.0, minutes=1200.0, games_played=60
        )
        result = _compute_player_impact(stats)
        assert result == pytest.approx(-7.0 * (20.0 / 48.0))


class TestDateToNbaSeason:
    def test_january_game(self) -> None:
        dt = datetime(2025, 1, 15, 0, 30, tzinfo=UTC)
        assert date_to_nba_season(dt) == "2024-25"

    def test_october_game(self) -> None:
        dt = datetime(2024, 10, 22, 19, 0, tzinfo=UTC)
        assert date_to_nba_season(dt) == "2024-25"

    def test_june_game(self) -> None:
        dt = datetime(2025, 6, 15, 20, 0, tzinfo=UTC)
        assert date_to_nba_season(dt) == "2024-25"

    def test_september_preseason(self) -> None:
        dt = datetime(2024, 9, 30, 19, 0, tzinfo=UTC)
        assert date_to_nba_season(dt) == "2023-24"

    def test_december_game(self) -> None:
        dt = datetime(2024, 12, 25, 0, 0, tzinfo=UTC)
        assert date_to_nba_season(dt) == "2024-25"


class TestExtractInjuryFeatures:
    def test_empty_reports_returns_all_none(self) -> None:
        event = _make_event()
        features = extract_injury_features([], event, datetime(2025, 1, 14, 22, 0, tzinfo=UTC))
        assert np.all(np.isnan(features.to_array()))

    def test_no_reports_before_snapshot_returns_all_none(self) -> None:
        event = _make_event()
        report = _make_report(report_time=datetime(2025, 1, 14, 23, 0, tzinfo=UTC))
        # Snapshot is before report
        features = extract_injury_features(
            [report], event, datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        )
        assert np.all(np.isnan(features.to_array()))

    def test_counts_out_players_without_stats(self) -> None:
        """Without player_stats, each OUT player contributes 1.0 (headcount)."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
            _make_report(
                team="Los Angeles Lakers",
                player_name="Davis, Anthony",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
            _make_report(
                team="Boston Celtics",
                player_name="Tatum, Jayson",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        assert features.impact_out_home == 2.0
        assert features.impact_out_away == 1.0

    def test_weighted_extraction_with_stats(self) -> None:
        """Known players with stats produce weighted sums, not headcounts."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]

        # LeBron stats: net=6.5, mpg=36.0 -> impact=4.875
        player_stats = {
            "James, LeBron": _make_player_stats(
                player_name="James, LeBron",
                on_off_rtg=118.5,
                on_def_rtg=112.0,
                minutes=2520.0,
                games_played=70,
            ),
        }
        features = extract_injury_features(reports, event, snapshot_time, player_stats=player_stats)
        assert features.impact_out_home == pytest.approx(4.875)

    def test_fallback_for_unknown_player(self) -> None:
        """Player not in player_stats dict contributes 1.0."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="Unknown, Player",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]

        player_stats = {
            "James, LeBron": _make_player_stats(player_name="James, LeBron"),
        }
        features = extract_injury_features(reports, event, snapshot_time, player_stats=player_stats)
        assert features.impact_out_home == 1.0

    def test_mixed_known_and_unknown(self) -> None:
        """Mix of known (weighted) and unknown (1.0) players."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
            _make_report(
                team="Los Angeles Lakers",
                player_name="Unknown, Bench",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]

        # LeBron: impact=4.875, Unknown: impact=1.0 -> total=5.875
        player_stats = {
            "James, LeBron": _make_player_stats(
                player_name="James, LeBron",
                on_off_rtg=118.5,
                on_def_rtg=112.0,
                minutes=2520.0,
                games_played=70,
            ),
        }
        features = extract_injury_features(reports, event, snapshot_time, player_stats=player_stats)
        assert features.impact_out_home == pytest.approx(5.875)

    def test_gtd_discount(self) -> None:
        """GTD players get impact * 0.5."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.QUESTIONABLE,
                report_time=report_time,
            ),
        ]

        # LeBron impact=4.875, GTD discount=0.5 -> 2.4375
        player_stats = {
            "James, LeBron": _make_player_stats(
                player_name="James, LeBron",
                on_off_rtg=118.5,
                on_def_rtg=112.0,
                minutes=2520.0,
                games_played=70,
            ),
        }
        features = extract_injury_features(reports, event, snapshot_time, player_stats=player_stats)
        assert features.impact_gtd_home == pytest.approx(2.4375)
        assert features.impact_out_home == 0.0

    def test_gtd_without_stats_gets_discount(self) -> None:
        """GTD without stats contributes 1.0 * 0.5 = 0.5."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.QUESTIONABLE,
                report_time=report_time,
            ),
            _make_report(
                team="Los Angeles Lakers",
                player_name="Davis, Anthony",
                status=InjuryStatus.DOUBTFUL,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        assert features.impact_gtd_home == pytest.approx(1.0)  # 0.5 + 0.5

    def test_probable_and_available_not_counted(self) -> None:
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.PROBABLE,
                report_time=report_time,
            ),
            _make_report(
                team="Boston Celtics",
                player_name="Tatum, Jayson",
                status=InjuryStatus.AVAILABLE,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        assert features.impact_out_home == 0.0
        assert features.impact_out_away == 0.0
        assert features.impact_gtd_home == 0.0
        assert features.impact_gtd_away == 0.0

    def test_uses_latest_report_only(self) -> None:
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        early_time = datetime(2025, 1, 14, 18, 0, tzinfo=UTC)
        late_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            # Early report: LeBron OUT
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=early_time,
            ),
            # Later report: LeBron QUESTIONABLE (upgraded)
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.QUESTIONABLE,
                report_time=late_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        # Should use only the latest report (late_time)
        assert features.impact_out_home == 0.0
        assert features.impact_gtd_home == pytest.approx(0.5)  # 1.0 * 0.5 (no stats)

    def test_look_ahead_bias_prevention(self) -> None:
        event = _make_event()
        early_time = datetime(2025, 1, 14, 18, 0, tzinfo=UTC)
        late_time = datetime(2025, 1, 14, 21, 0, tzinfo=UTC)
        snapshot_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.QUESTIONABLE,
                report_time=early_time,
            ),
            # This report is AFTER snapshot_time — must be excluded
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=late_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        # Only the early report (QUESTIONABLE) should be visible
        assert features.impact_out_home == 0.0
        assert features.impact_gtd_home == pytest.approx(0.5)  # 1.0 * 0.5

    def test_timing_features(self) -> None:
        # Game at Jan 15 00:30 UTC, report at Jan 14 20:00 UTC, snapshot at Jan 14 22:00 UTC
        commence_time = datetime(2025, 1, 15, 0, 30, tzinfo=UTC)
        event = _make_event(commence_time=commence_time)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        # report_hours_before_game: (00:30 - 20:00) = 4.5 hours
        assert features.report_hours_before_game == pytest.approx(4.5)
        # injury_news_recency: (22:00 - 20:00) = 2.0 hours
        assert features.injury_news_recency == pytest.approx(2.0)

    def test_team_name_normalization(self) -> None:
        # Event uses short names, reports use full names
        event = _make_event(home_team="Lakers", away_team="Celtics")
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        assert features.impact_out_home == 1.0

    def test_unrecognized_team_ignored(self) -> None:
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Unknown Team FC",
                player_name="Player, Test",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        # Unrecognized team → 0 impact (not an error)
        assert features.impact_out_home == 0.0
        assert features.impact_out_away == 0.0

    def test_report_at_exact_snapshot_time_included(self) -> None:
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=snapshot_time,  # Exactly at snapshot time
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        assert features.impact_out_home == 1.0

    def test_no_player_stats_degrades_to_count(self) -> None:
        """When player_stats=None, each player contributes 1.0 (backward compat)."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="James, LeBron",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
            _make_report(
                team="Boston Celtics",
                player_name="Tatum, Jayson",
                status=InjuryStatus.QUESTIONABLE,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time, player_stats=None)
        assert features.impact_out_home == 1.0
        assert features.impact_gtd_away == pytest.approx(0.5)  # 1.0 * 0.5

    def test_none_on_off_rtg_player_falls_back(self) -> None:
        """Player with stats but None on_off_rtg contributes 1.0."""
        event = _make_event()
        snapshot_time = datetime(2025, 1, 14, 22, 0, tzinfo=UTC)
        report_time = datetime(2025, 1, 14, 20, 0, tzinfo=UTC)

        reports = [
            _make_report(
                team="Los Angeles Lakers",
                player_name="Rookie, Young",
                status=InjuryStatus.OUT,
                report_time=report_time,
            ),
        ]
        player_stats = {
            "Rookie, Young": _make_player_stats(
                player_name="Rookie, Young",
                on_off_rtg=None,
                on_def_rtg=None,
            ),
        }
        features = extract_injury_features(reports, event, snapshot_time, player_stats=player_stats)
        assert features.impact_out_home == 1.0
