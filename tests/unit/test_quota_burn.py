from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_lambda.storage.readers import burn_projection

START = datetime(2026, 11, 1, tzinfo=UTC)


def hourly(*remaining: int, start: datetime = START) -> list[tuple[datetime, int]]:
    return [(start + timedelta(hours=i), r) for i, r in enumerate(remaining)]


class TestBurnProjection:
    def test_spend_rate_and_days_left_from_the_first_and_last_reading(self) -> None:
        readings = [(START, 500), (START + timedelta(days=2), 400)]

        projection = burn_projection(readings)

        assert projection is not None
        assert projection.per_day == pytest.approx(50.0)
        assert projection.days_left == pytest.approx(8.0)
        assert projection.since == START

    def test_readings_in_any_order_give_the_same_projection(self) -> None:
        readings = [(START + timedelta(days=2), 400), (START, 500)]

        assert burn_projection(readings) == burn_projection(sorted(readings))

    def test_under_a_day_of_readings_is_unknown(self) -> None:
        assert burn_projection(hourly(*range(500, 476, -1))) is None
        assert burn_projection([(START, 500)]) is None
        assert burn_projection([]) is None

    def test_only_readings_after_the_latest_reset_count(self) -> None:
        before = [(START, 100), (START + timedelta(days=3), 20)]
        after = [(START + timedelta(days=4), 500), (START + timedelta(days=6), 460)]

        projection = burn_projection(before + after)

        assert projection is not None
        assert projection.per_day == pytest.approx(20.0)
        assert projection.since == START + timedelta(days=4)

    def test_a_reset_too_recent_to_measure_is_unknown(self) -> None:
        readings = [
            (START, 100),
            (START + timedelta(days=3), 20),
            (START + timedelta(days=3, hours=2), 500),
        ]

        assert burn_projection(readings) is None

    def test_a_key_spending_nothing_has_no_days_left(self) -> None:
        projection = burn_projection([(START, 300), (START + timedelta(days=2), 300)])

        assert projection is not None
        assert projection.per_day == 0 and projection.days_left is None
