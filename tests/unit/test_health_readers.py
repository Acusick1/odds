"""The dead-chain judgement the health monitor and the web UI's status strip share."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from odds_core.config import Settings
from odds_lambda.health_readers import SCHEDULE_OVERDUE_GRACE, DeadChain, dead_chains

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
SOON = NOW + timedelta(minutes=10)


@pytest.fixture
def settings() -> Settings:
    return SimpleNamespace(  # type: ignore[return-value]
        betfair=SimpleNamespace(enabled=True), polymarket=SimpleNamespace(enabled=False)
    )


class TestDeadChains:
    def test_a_booked_run_is_alive(self, settings: Settings) -> None:
        assert (
            dead_chains(["fetch-oddsportal-epl"], {"fetch-oddsportal-epl": SOON}, (), settings, NOW)
            == []
        )

    def test_a_job_with_no_schedule_or_a_null_next_run_is_dead(self, settings: Settings) -> None:
        dead = dead_chains(
            ["run-candidates-epl", "fetch-oddsportal-epl"],
            {"run-candidates-epl": None},
            (),
            settings,
            NOW,
        )

        assert dead == [
            DeadChain("fetch-oddsportal-epl", None),
            DeadChain("run-candidates-epl", None),
        ]

    def test_a_next_run_is_overdue_past_the_grace(self, settings: Settings) -> None:
        job = "fetch-oddsportal-epl"
        inside = NOW - SCHEDULE_OVERDUE_GRACE + timedelta(minutes=1)
        outside = NOW - SCHEDULE_OVERDUE_GRACE - timedelta(minutes=1)

        assert dead_chains([job], {job: inside}, (), settings, NOW) == []
        assert dead_chains([job], {job: outside}, (), settings, NOW) == [DeadChain(job, outside)]

    def test_an_outstanding_job_is_alive_without_a_next_run(self, settings: Settings) -> None:
        assert dead_chains(["run-candidates-epl"], {}, {"run-candidates-epl"}, settings, NOW) == []

    def test_a_job_disabled_by_its_setting_is_skipped(self, settings: Settings) -> None:
        names = ["fetch-betfair-exchange-epl", "fetch-polymarket-odds-epl"]

        assert dead_chains(names, {}, (), settings, NOW) == [
            DeadChain("fetch-betfair-exchange-epl", None)
        ]
