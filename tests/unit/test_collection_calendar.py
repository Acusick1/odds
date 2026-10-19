"""Tests for the active-hours arithmetic behind the per-source stale-data check."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from odds_lambda.collection_calendar import active_hours_between

EPL_OVERNIGHT = (22, 6)
MLB_OVERNIGHT = (5, 14)


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=UTC)


class TestActiveHoursBetween:
    def test_no_window_counts_every_hour(self) -> None:
        assert active_hours_between(_at(18, 21), _at(19, 7), None) == 10.0

    def test_wrapping_window_is_excluded(self) -> None:
        assert active_hours_between(_at(18, 21), _at(19, 7), EPL_OVERNIGHT) == 2.0

    def test_span_inside_the_window_is_zero(self) -> None:
        assert active_hours_between(_at(19, 1), _at(19, 4), EPL_OVERNIGHT) == 0.0

    def test_each_night_of_a_long_span_is_excluded(self) -> None:
        assert active_hours_between(_at(17, 6), _at(19, 6), EPL_OVERNIGHT) == 32.0

    def test_same_day_window_is_excluded(self) -> None:
        assert active_hours_between(_at(19, 3), _at(19, 16), MLB_OVERNIGHT) == 4.0

    def test_span_before_the_window_is_untouched(self) -> None:
        assert active_hours_between(_at(19, 7), _at(19, 12), EPL_OVERNIGHT) == 5.0

    @pytest.mark.parametrize("end", [_at(19, 12), _at(18, 12)])
    def test_empty_or_reversed_span_is_zero(self, end: datetime) -> None:
        assert active_hours_between(_at(19, 12), end, EPL_OVERNIGHT) == 0.0
