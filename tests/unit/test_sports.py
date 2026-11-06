"""Sport keys and their league calendars."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest
from odds_core.sports import LEAGUE_TIMEZONES, SUPPORTED_SPORTS, SportKey, game_day_window


class TestLeagueCalendars:
    def test_every_sport_has_a_timezone(self) -> None:
        assert set(LEAGUE_TIMEZONES) == set(SUPPORTED_SPORTS)

    @pytest.mark.parametrize(
        ("sport", "midnight_utc"),
        [
            ("soccer_epl", datetime(2025, 1, 3, 0, 0, tzinfo=UTC)),
            ("soccer_efl_champ", datetime(2025, 1, 3, 0, 0, tzinfo=UTC)),
            ("soccer_spain_la_liga", datetime(2025, 1, 2, 23, 0, tzinfo=UTC)),
            ("soccer_italy_serie_a", datetime(2025, 1, 2, 23, 0, tzinfo=UTC)),
            ("soccer_germany_bundesliga", datetime(2025, 1, 2, 23, 0, tzinfo=UTC)),
            ("soccer_france_ligue_one", datetime(2025, 1, 2, 23, 0, tzinfo=UTC)),
            ("basketball_nba", datetime(2025, 1, 3, 5, 0, tzinfo=UTC)),
        ],
    )
    def test_game_day_starts_at_local_midnight(
        self, sport: SportKey, midnight_utc: datetime
    ) -> None:
        start, end = game_day_window(sport, date(2025, 1, 3))

        assert start == midnight_utc
        assert end == midnight_utc + timedelta(days=1)
