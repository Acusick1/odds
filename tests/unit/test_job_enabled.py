"""A job switched off by a setting: its spec's predicate, its main(), and the dead-chain check."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.config import Settings
from odds_core.job_specs import JOB_SPECS
from odds_lambda.health_readers import DeadChain, dead_chains
from odds_lambda.scheduling.jobs import JobContext, get_job_function, make_compound_job_name

# Settings that switch off every job with an enabled predicate.
ALL_OFF = SimpleNamespace(
    betfair=SimpleNamespace(enabled=False),
    polymarket=SimpleNamespace(enabled=False),
    scheduler=SimpleNamespace(dry_run=True),
)
ALL_ON = SimpleNamespace(
    betfair=SimpleNamespace(enabled=True),
    polymarket=SimpleNamespace(enabled=True),
    scheduler=SimpleNamespace(dry_run=True),
)
NOW = datetime(2026, 11, 9, 12, 0, tzinfo=UTC)
GATED_JOBS = {"fetch-betfair-exchange": "soccer_epl", "fetch-polymarket-odds": "soccer_epl"}


def _settings(value: SimpleNamespace) -> Settings:
    return value  # type: ignore[return-value]


class TestEnabledPredicate:
    def test_the_gated_jobs_are_exactly_those_with_a_predicate(self) -> None:
        assert {name for name, spec in JOB_SPECS.items() if spec.enabled} == set(GATED_JOBS)

    @pytest.mark.parametrize("job_name", sorted(GATED_JOBS))
    def test_the_predicate_follows_the_setting(self, job_name: str) -> None:
        enabled = JOB_SPECS[job_name].enabled
        assert enabled is not None
        assert enabled(_settings(ALL_ON)) is True
        assert enabled(_settings(ALL_OFF)) is False


class TestDisabledMain:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("job_name", sorted(GATED_JOBS))
    async def test_main_returns_before_collecting(self, job_name: str) -> None:
        module = JOB_SPECS[job_name].module
        with (
            patch(f"{module}.get_settings", return_value=ALL_OFF),
            patch(f"{module}.run_collection", new_callable=AsyncMock) as run,
        ):
            await get_job_function(job_name)(JobContext(sport=GATED_JOBS[job_name]))
        run.assert_not_awaited()


class TestDeadChainOfAGatedJob:
    @pytest.mark.parametrize("job_name", sorted(GATED_JOBS))
    def test_a_disabled_job_has_no_dead_chain(self, job_name: str) -> None:
        compound = make_compound_job_name(job_name, GATED_JOBS[job_name])
        assert dead_chains([compound], {}, (), _settings(ALL_OFF), NOW) == []

    @pytest.mark.parametrize("job_name", sorted(GATED_JOBS))
    def test_an_enabled_job_whose_chain_ended_is_dead(self, job_name: str) -> None:
        compound = make_compound_job_name(job_name, GATED_JOBS[job_name])
        assert dead_chains([compound], {}, (), _settings(ALL_ON), NOW) == [
            DeadChain(compound, None)
        ]
