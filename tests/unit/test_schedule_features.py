"""Unit tests for rest/schedule feature extraction."""

from datetime import UTC, date, datetime

import numpy as np
from odds_analytics.schedule_features import RestScheduleFeatures, extract_rest_features
from odds_core.game_log_models import NbaTeamGameLog
from odds_core.models import Event


def _make_event(
    event_id: str = "test_event_1",
    home_team: str = "Los Angeles Lakers",
    away_team: str = "Boston Celtics",
    commence_time: datetime | None = None,
) -> Event:
    if commence_time is None:
        commence_time = datetime(2025, 1, 15, 0, 30, tzinfo=UTC)
    return Event(
        id=event_id,
        sport_key="basketball_nba",
        sport_title="NBA",
        home_team=home_team,
        away_team=away_team,
        commence_time=commence_time,
    )


def _make_game_log(
    team_abbreviation: str = "LAL",
    game_date: date = date(2025, 1, 15),
    matchup: str = "LAL vs. BOS",
    event_id: str | None = "test_event_1",
    nba_game_id: str = "0022400100",
    team_id: int = 1610612747,
    season: str = "2024-25",
) -> NbaTeamGameLog:
    return NbaTeamGameLog(
        nba_game_id=nba_game_id,
        team_id=team_id,
        team_abbreviation=team_abbreviation,
        game_date=game_date,
        matchup=matchup,
        season=season,
        event_id=event_id,
    )


class TestRestScheduleFeaturesDataclass:
    def test_get_feature_names(self) -> None:
        names = RestScheduleFeatures.get_feature_names()
        assert names == [
            "home_days_rest",
            "away_days_rest",
            "rest_advantage",
            "home_is_b2b",
            "away_is_b2b",
        ]

    def test_to_array_all_none(self) -> None:
        features = RestScheduleFeatures()
        arr = features.to_array()
        assert arr.shape == (5,)
        assert np.all(np.isnan(arr))

    def test_to_array_with_values(self) -> None:
        features = RestScheduleFeatures(
            home_days_rest=2.0,
            away_days_rest=1.0,
            rest_advantage=1.0,
            home_is_b2b=0.0,
            away_is_b2b=1.0,
        )
        arr = features.to_array()
        assert arr.shape == (5,)
        np.testing.assert_array_equal(arr, [2.0, 1.0, 1.0, 0.0, 1.0])

    def test_to_array_partial_none(self) -> None:
        features = RestScheduleFeatures(home_days_rest=3.0)
        arr = features.to_array()
        assert arr[0] == 3.0
        assert np.isnan(arr[1])

    def test_feature_count_matches_names(self) -> None:
        assert len(RestScheduleFeatures.get_feature_names()) == len(
            RestScheduleFeatures().to_array()
        )


class TestExtractRestFeatures:
    def test_empty_logs_returns_all_none(self) -> None:
        event = _make_event()
        features = extract_rest_features([], event)
        assert np.all(np.isnan(features.to_array()))

    def test_no_event_logs_returns_all_none(self) -> None:
        """Game logs exist but none linked to this event."""
        event = _make_event(event_id="test_event_1")
        unlinked_log = _make_game_log(event_id="other_event")
        features = extract_rest_features([unlinked_log], event)
        assert np.all(np.isnan(features.to_array()))

    def test_no_prior_game_returns_none_for_rest(self) -> None:
        """Season opener — event game log exists but no prior game."""
        event = _make_event()
        event_log_home = _make_game_log(
            team_abbreviation="LAL",
            game_date=date(2025, 1, 15),
            matchup="LAL vs. BOS",
            event_id="test_event_1",
        )
        event_log_away = _make_game_log(
            team_abbreviation="BOS",
            game_date=date(2025, 1, 15),
            matchup="BOS @ LAL",
            event_id="test_event_1",
            nba_game_id="0022400100",
            team_id=1610612738,
        )
        features = extract_rest_features([event_log_home, event_log_away], event)
        assert np.all(np.isnan(features.to_array()))

    def test_normal_rest(self) -> None:
        """Home team rested 3 days, away team rested 2 days."""
        event = _make_event()
        game_logs = [
            # Event's own game logs (Jan 15)
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 15),
                matchup="LAL vs. BOS",
                event_id="test_event_1",
            ),
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 15),
                matchup="BOS @ LAL",
                event_id="test_event_1",
                nba_game_id="0022400100",
                team_id=1610612738,
            ),
            # LAL previous game (Jan 12 → 3 days rest)
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 12),
                matchup="LAL vs. GSW",
                event_id=None,
                nba_game_id="0022400090",
            ),
            # BOS previous game (Jan 13 → 2 days rest)
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 13),
                matchup="BOS @ MIA",
                event_id=None,
                nba_game_id="0022400095",
                team_id=1610612738,
            ),
        ]
        features = extract_rest_features(game_logs, event)
        assert features.home_days_rest == 3.0
        assert features.away_days_rest == 2.0
        assert features.rest_advantage == 1.0
        assert features.home_is_b2b == 0.0
        assert features.away_is_b2b == 0.0

    def test_back_to_back(self) -> None:
        """Both teams on B2B (played yesterday)."""
        event = _make_event()
        game_logs = [
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 15),
                matchup="LAL vs. BOS",
                event_id="test_event_1",
            ),
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 15),
                matchup="BOS @ LAL",
                event_id="test_event_1",
                nba_game_id="0022400100",
                team_id=1610612738,
            ),
            # LAL played yesterday (Jan 14 → 1 day = B2B)
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 14),
                matchup="LAL @ PHX",
                event_id=None,
                nba_game_id="0022400098",
            ),
            # BOS played yesterday (Jan 14 → 1 day = B2B)
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 14),
                matchup="BOS vs. NYK",
                event_id=None,
                nba_game_id="0022400099",
                team_id=1610612738,
            ),
        ]
        features = extract_rest_features(game_logs, event)
        assert features.home_days_rest == 1.0
        assert features.away_days_rest == 1.0
        assert features.rest_advantage == 0.0
        assert features.home_is_b2b == 1.0
        assert features.away_is_b2b == 1.0

    def test_asymmetric_rest(self) -> None:
        """Home on B2B, away well-rested → negative rest advantage."""
        event = _make_event()
        game_logs = [
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 15),
                matchup="LAL vs. BOS",
                event_id="test_event_1",
            ),
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 15),
                matchup="BOS @ LAL",
                event_id="test_event_1",
                nba_game_id="0022400100",
                team_id=1610612738,
            ),
            # LAL B2B (1 day)
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 14),
                matchup="LAL @ PHX",
                event_id=None,
                nba_game_id="0022400098",
            ),
            # BOS well rested (4 days)
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 11),
                matchup="BOS vs. MIA",
                event_id=None,
                nba_game_id="0022400085",
                team_id=1610612738,
            ),
        ]
        features = extract_rest_features(game_logs, event)
        assert features.home_days_rest == 1.0
        assert features.away_days_rest == 4.0
        assert features.rest_advantage == -3.0
        assert features.home_is_b2b == 1.0
        assert features.away_is_b2b == 0.0

    def test_only_home_has_prior_game(self) -> None:
        """Away team has no prior game (e.g. season opener)."""
        event = _make_event()
        game_logs = [
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 15),
                matchup="LAL vs. BOS",
                event_id="test_event_1",
            ),
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 15),
                matchup="BOS @ LAL",
                event_id="test_event_1",
                nba_game_id="0022400100",
                team_id=1610612738,
            ),
            # Only LAL has a prior game
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 12),
                matchup="LAL vs. GSW",
                event_id=None,
                nba_game_id="0022400090",
            ),
        ]
        features = extract_rest_features(game_logs, event)
        assert features.home_days_rest == 3.0
        assert features.away_days_rest is None
        assert features.rest_advantage is None
        assert features.home_is_b2b == 0.0
        assert features.away_is_b2b is None

    def test_uses_most_recent_prior_game(self) -> None:
        """When multiple prior games exist, picks the most recent."""
        event = _make_event()
        game_logs = [
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 15),
                matchup="LAL vs. BOS",
                event_id="test_event_1",
            ),
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 15),
                matchup="BOS @ LAL",
                event_id="test_event_1",
                nba_game_id="0022400100",
                team_id=1610612738,
            ),
            # LAL older game (Jan 10)
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 10),
                matchup="LAL vs. DEN",
                event_id=None,
                nba_game_id="0022400080",
            ),
            # LAL more recent game (Jan 13) — should use this one
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 13),
                matchup="LAL @ SAC",
                event_id=None,
                nba_game_id="0022400092",
            ),
        ]
        features = extract_rest_features(game_logs, event)
        assert features.home_days_rest == 2.0  # Jan 15 - Jan 13

    def test_not_b2b_with_two_days_rest(self) -> None:
        """2 days between games is not B2B."""
        event = _make_event()
        game_logs = [
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 15),
                matchup="LAL vs. BOS",
                event_id="test_event_1",
            ),
            _make_game_log(
                team_abbreviation="BOS",
                game_date=date(2025, 1, 15),
                matchup="BOS @ LAL",
                event_id="test_event_1",
                nba_game_id="0022400100",
                team_id=1610612738,
            ),
            _make_game_log(
                team_abbreviation="LAL",
                game_date=date(2025, 1, 13),
                matchup="LAL @ PHX",
                event_id=None,
                nba_game_id="0022400092",
            ),
        ]
        features = extract_rest_features(game_logs, event)
        assert features.home_days_rest == 2.0
        assert features.home_is_b2b == 0.0
