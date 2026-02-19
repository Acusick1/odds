"""Unit tests for injury feature extraction."""

from datetime import UTC, datetime

import numpy as np
import pytest
from odds_analytics.injury_features import InjuryFeatures, extract_injury_features
from odds_core.injury_models import InjuryReport, InjuryStatus
from odds_core.models import Event


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


class TestInjuryFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = InjuryFeatures.get_feature_names()
        assert names == [
            "num_out_home",
            "num_out_away",
            "num_gtd_home",
            "num_gtd_away",
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
            num_out_home=2.0,
            num_out_away=1.0,
            num_gtd_home=1.0,
            num_gtd_away=0.0,
            report_hours_before_game=4.5,
            injury_news_recency=1.0,
        )
        arr = features.to_array()
        assert arr.shape == (6,)
        np.testing.assert_array_equal(arr, [2.0, 1.0, 1.0, 0.0, 4.5, 1.0])

    def test_to_array_partial_none(self) -> None:
        features = InjuryFeatures(num_out_home=3.0)
        arr = features.to_array()
        assert arr[0] == 3.0
        assert np.isnan(arr[1])

    def test_feature_count_matches_names(self) -> None:
        assert len(InjuryFeatures.get_feature_names()) == len(InjuryFeatures().to_array())


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

    def test_counts_out_players_by_team(self) -> None:
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
        assert features.num_out_home == 2.0
        assert features.num_out_away == 1.0

    def test_counts_gtd_questionable_and_doubtful(self) -> None:
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
            _make_report(
                team="Boston Celtics",
                player_name="Tatum, Jayson",
                status=InjuryStatus.QUESTIONABLE,
                report_time=report_time,
            ),
        ]
        features = extract_injury_features(reports, event, snapshot_time)
        assert features.num_gtd_home == 2.0
        assert features.num_gtd_away == 1.0
        assert features.num_out_home == 0.0
        assert features.num_out_away == 0.0

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
        assert features.num_out_home == 0.0
        assert features.num_out_away == 0.0
        assert features.num_gtd_home == 0.0
        assert features.num_gtd_away == 0.0

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
        assert features.num_out_home == 0.0
        assert features.num_gtd_home == 1.0

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
        assert features.num_out_home == 0.0
        assert features.num_gtd_home == 1.0

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
        assert features.num_out_home == 1.0

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
        # Unrecognized team → 0 counts (not an error)
        assert features.num_out_home == 0.0
        assert features.num_out_away == 0.0

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
        assert features.num_out_home == 1.0
