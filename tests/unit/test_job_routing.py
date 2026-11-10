"""Tests for job name resolution, compound name construction, and bootstrap functions."""

from importlib import import_module
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.job_specs import JOB_SPECS
from odds_lambda.scheduling.jobs import (
    JobContext,
    expected_compound_job_names,
    get_bootstrap_function,
    get_job_cron,
    get_job_sports,
    make_compound_job_name,
    require_sport,
    resolve_job_name,
    resolve_target_sports,
    sport_key_to_suffix,
)


class TestResolveJobName:
    """Tests for resolve_job_name()."""

    def test_direct_match_global_job(self) -> None:
        base, sport = resolve_job_name("check-health")
        assert base == "check-health"
        assert sport is None

    def test_direct_match_per_sport_job_without_suffix(self) -> None:
        base, sport = resolve_job_name("fetch-odds")
        assert base == "fetch-odds"
        assert sport is None

    def test_compound_name_with_sport_suffix(self) -> None:
        base, sport = resolve_job_name("fetch-odds-epl")
        assert base == "fetch-odds"
        assert sport == "soccer_epl"

    def test_compound_name_fetch_scores(self) -> None:
        base, sport = resolve_job_name("fetch-scores-epl")
        assert base == "fetch-scores"
        assert sport == "soccer_epl"

    def test_compound_name_fetch_oddsportal(self) -> None:
        base, sport = resolve_job_name("fetch-oddsportal-epl")
        assert base == "fetch-oddsportal"
        assert sport == "soccer_epl"

    def test_compound_name_fetch_oddsportal_results(self) -> None:
        base, sport = resolve_job_name("fetch-oddsportal-results-epl")
        assert base == "fetch-oddsportal-results"
        assert sport == "soccer_epl"

    def test_compound_name_mlb(self) -> None:
        base, sport = resolve_job_name("fetch-oddsportal-mlb")
        assert base == "fetch-oddsportal"
        assert sport == "baseball_mlb"

    def test_compound_name_mlb_results(self) -> None:
        base, sport = resolve_job_name("fetch-oddsportal-results-mlb")
        assert base == "fetch-oddsportal-results"
        assert sport == "baseball_mlb"

    def test_compound_name_agent_run_epl(self) -> None:
        base, sport = resolve_job_name("agent-run-epl")
        assert base == "agent-run"
        assert sport == "soccer_epl"

    def test_compound_name_agent_run_mlb(self) -> None:
        base, sport = resolve_job_name("agent-run-mlb")
        assert base == "agent-run"
        assert sport == "baseball_mlb"

    def test_unknown_suffix_returns_no_sport(self) -> None:
        base, sport = resolve_job_name("fetch-odds-nhl")
        assert base == "fetch-odds-nhl"
        assert sport is None

    def test_totally_unknown_name(self) -> None:
        base, sport = resolve_job_name("nonexistent-job")
        assert base == "nonexistent-job"
        assert sport is None

    def test_suffix_on_non_per_sport_job(self) -> None:
        """A global-only job with a sport suffix should not resolve."""
        base, sport = resolve_job_name("check-health-epl")
        assert base == "check-health-epl"
        assert sport is None


class TestSportKeyToSuffix:
    """Tests for sport_key_to_suffix()."""

    def test_known_sport(self) -> None:
        assert sport_key_to_suffix("soccer_epl") == "epl"

    def test_known_sport_mlb(self) -> None:
        assert sport_key_to_suffix("baseball_mlb") == "mlb"

    def test_unknown_sport(self) -> None:
        assert sport_key_to_suffix("icehockey_nhl") is None


class TestMakeCompoundJobName:
    """Tests for make_compound_job_name()."""

    def test_with_sport(self) -> None:
        assert make_compound_job_name("fetch-odds", "soccer_epl") == "fetch-odds-epl"

    def test_with_sport_mlb(self) -> None:
        assert make_compound_job_name("fetch-oddsportal", "baseball_mlb") == "fetch-oddsportal-mlb"

    @pytest.mark.parametrize(
        "job", sorted(name for name, spec in JOB_SPECS.items() if spec.per_sport)
    )
    def test_per_sport_job_without_sport_raises(self, job: str) -> None:
        with pytest.raises(ValueError, match="requires a sport"):
            make_compound_job_name(job, None)

    def test_non_per_sport_job_without_sport_returns_base(self) -> None:
        assert make_compound_job_name("check-health", None) == "check-health"

    def test_unknown_sport_returns_base(self) -> None:
        assert make_compound_job_name("fetch-odds", "icehockey_nhl") == "fetch-odds"

    def test_non_per_sport_job_returns_base(self) -> None:
        assert make_compound_job_name("check-health", "soccer_epl") == "check-health"

    def test_roundtrip_resolve_then_make(self) -> None:
        """make_compound_job_name should produce names that resolve_job_name can parse back."""
        compound = make_compound_job_name("fetch-oddsportal-results", "soccer_epl")
        base, sport = resolve_job_name(compound)
        assert base == "fetch-oddsportal-results"
        assert sport == "soccer_epl"

    def test_roundtrip_mlb(self) -> None:
        compound = make_compound_job_name("fetch-oddsportal", "baseball_mlb")
        base, sport = resolve_job_name(compound)
        assert base == "fetch-oddsportal"
        assert sport == "baseball_mlb"

    def test_roundtrip_mlb_results(self) -> None:
        compound = make_compound_job_name("fetch-oddsportal-results", "baseball_mlb")
        base, sport = resolve_job_name(compound)
        assert base == "fetch-oddsportal-results"
        assert sport == "baseball_mlb"

    def test_roundtrip_agent_run_epl(self) -> None:
        compound = make_compound_job_name("agent-run", "soccer_epl")
        base, sport = resolve_job_name(compound)
        assert base == "agent-run"
        assert sport == "soccer_epl"

    def test_roundtrip_agent_run_mlb(self) -> None:
        compound = make_compound_job_name("agent-run", "baseball_mlb")
        base, sport = resolve_job_name(compound)
        assert base == "agent-run"
        assert sport == "baseball_mlb"


class TestGetBootstrapFunction:
    """Tests for get_bootstrap_function()."""

    def test_agent_run_returns_wrapper_calling_schedule_next(self) -> None:
        mock_schedule_next = AsyncMock()
        with patch("odds_lambda.scheduling.jobs.import_module") as mock_import:
            mock_module = mock_import.return_value
            mock_module.schedule_next = mock_schedule_next

            # Clear cached jobs to force fresh import
            from odds_lambda.scheduling.jobs import _loaded_jobs

            _loaded_jobs.pop("agent-run", None)

            fn = get_bootstrap_function("agent-run")

        # Should accept JobContext and forward sport to schedule_next
        import asyncio

        asyncio.run(fn(JobContext(sport="soccer_epl")))
        mock_schedule_next.assert_called_once_with("soccer_epl")

    def test_regular_job_falls_back_to_main(self) -> None:
        mock_main = AsyncMock()
        with patch("odds_lambda.scheduling.jobs.import_module") as mock_import:
            mock_module = mock_import.return_value
            mock_module.main = mock_main

            from odds_lambda.scheduling.jobs import _loaded_jobs

            _loaded_jobs.pop("fetch-odds", None)

            fn = get_bootstrap_function("fetch-odds")

        import asyncio

        ctx = JobContext(sport="soccer_epl")
        asyncio.run(fn(ctx))
        mock_main.assert_called_once_with(ctx)

    def test_unknown_job_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown job"):
            get_bootstrap_function("nonexistent-job")


class TestJobCronRegistry:
    """Tests for cron schedule + sport-allowlist lookups on ``JobSpec.cron``."""

    def test_get_job_cron_returns_none_for_event_driven_job(self) -> None:
        assert get_job_cron("fetch-odds") is None

    def test_get_job_cron_returns_expression_for_mapped_job(self) -> None:
        assert get_job_cron("fetch-espn-fixtures") == "0 6 * * *"

    def test_get_job_cron_returns_none_for_unknown_job(self) -> None:
        assert get_job_cron("totally-made-up") is None

    def test_get_job_sports_returns_allowlist_when_restricted(self) -> None:
        # fetch-espn-fixtures covers EPL, NBA and the leagues ESPN schedules — the sport
        # allowlist is load-bearing (its main() raises for the other configured sports, MLB
        # included).
        from odds_lambda.espn_football_schedule import LEAGUES

        sports = get_job_sports("fetch-espn-fixtures")
        assert sports is not None
        assert set(sports) == {"soccer_epl", "basketball_nba", *LEAGUES}

    @pytest.mark.parametrize(
        ("job", "sports"),
        [
            ("fetch-odds", ("soccer_epl", "basketball_nba")),
            ("fetch-scores", ("soccer_epl",)),
        ],
    )
    def test_get_job_sports_restricts_event_driven_jobs_to_the_sports_with_a_key(
        self, job: str, sports: tuple[str, ...]
    ) -> None:
        assert get_job_cron(job) is None
        assert get_job_sports(job) == sports

    def test_get_job_sports_returns_none_for_unrestricted_job(self) -> None:
        assert get_job_sports("fetch-oddsportal") is None

    def test_get_job_sports_returns_none_for_unknown_job(self) -> None:
        assert get_job_sports("totally-made-up") is None


class TestResolveTargetSports:
    """Tests for resolve_target_sports()."""

    def test_unrestricted_job_targets_all_configured_sports(self) -> None:
        assert resolve_target_sports("fetch-oddsportal", ["soccer_epl", "baseball_mlb"]) == [
            "soccer_epl",
            "baseball_mlb",
        ]

    def test_cron_job_with_allowlist_ignores_unlisted_sports(self) -> None:
        # fetch-espn-fixtures is restricted to EPL even when MLB is configured.
        assert resolve_target_sports("fetch-espn-fixtures", ["soccer_epl", "baseball_mlb"]) == [
            "soccer_epl"
        ]

    @pytest.mark.parametrize("job", ["fetch-odds", "fetch-scores"])
    def test_event_driven_job_with_allowlist_ignores_unlisted_sports(self, job: str) -> None:
        assert resolve_target_sports(job, ["soccer_epl", "baseball_mlb"]) == ["soccer_epl"]

    def test_allowlisted_sport_not_configured_is_dropped(self) -> None:
        assert resolve_target_sports("fetch-odds", ["baseball_mlb"]) == []

    def test_empty_configured_sports_yields_empty(self) -> None:
        assert resolve_target_sports("fetch-oddsportal", []) == []


class TestExpectedCompoundJobNames:
    """Tests for expected_compound_job_names() — drives stale-schedule pruning."""

    def test_expands_per_sport_and_keeps_global_jobs(self) -> None:
        expected = expected_compound_job_names(
            ["fetch-oddsportal", "fetch-espn-fixtures", "check-health"],
            ["soccer_epl", "baseball_mlb"],
        )
        assert expected == {
            "fetch-oddsportal-epl",
            "fetch-oddsportal-mlb",
            "fetch-espn-fixtures-epl",  # sport allowlist drops MLB
            "check-health",  # global job kept as-is
        }

    def test_prune_drops_compound_for_unconfigured_sport(self) -> None:
        # When MLB is removed from config, its compound becomes stale while
        # the configured EPL schedule is preserved.
        expected = expected_compound_job_names(["fetch-oddsportal"], ["soccer_epl"])
        existing = ["fetch-oddsportal-epl", "fetch-oddsportal-mlb"]
        stale = [name for name in existing if name not in expected]
        assert stale == ["fetch-oddsportal-mlb"]

    def test_allowlisted_event_driven_job_drops_unlisted_sport(self) -> None:
        expected = expected_compound_job_names(["fetch-odds"], ["soccer_epl", "baseball_mlb"])
        assert expected == {"fetch-odds-epl"}

    def test_no_bootstrap_jobs_yields_empty(self) -> None:
        assert expected_compound_job_names([], ["soccer_epl"]) == set()

    def test_default_bootstrap_schedules_polymarket_and_espn_for_the_sports_they_list(self) -> None:
        from odds_core.config import AlertConfig, SchedulerConfig

        expected = expected_compound_job_names(
            SchedulerConfig().bootstrap_jobs, ["soccer_epl", "baseball_mlb", "basketball_nba"]
        )
        assert {n for n in expected if n.startswith("fetch-polymarket")} == {
            "fetch-polymarket-odds-epl",
            "fetch-polymarket-odds-nba",
        }
        assert {n for n in expected if n.startswith("fetch-espn-fixtures")} == {
            "fetch-espn-fixtures-epl",
            "fetch-espn-fixtures-nba",
        }
        heartbeats = AlertConfig().heartbeat_expectations
        assert "fetch-polymarket-odds-epl" in heartbeats
        assert heartbeats["fetch-polymarket-odds-nba"] == 11
        assert "fetch-espn-fixtures-nba" in heartbeats


class TestRequireSport:
    """``require_sport`` guards every per-sport job entry point."""

    def test_returns_sport(self) -> None:
        assert require_sport(JobContext(sport="soccer_epl"), "fetch-odds") == "soccer_epl"

    def test_raises_without_sport(self) -> None:
        with pytest.raises(ValueError, match="fetch-odds"):
            require_sport(JobContext(), "fetch-odds")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "job", sorted(name for name, spec in JOB_SPECS.items() if spec.per_sport)
    )
    async def test_every_per_sport_job_rejects_missing_sport(self, job: str) -> None:
        spec = JOB_SPECS[job]
        module_path, func_name = spec.module, spec.function
        with pytest.raises(ValueError, match="requires a sport"):
            await getattr(import_module(module_path), func_name)(JobContext())


class TestJobSpecConsumers:
    def test_settings_defaults_are_generated_and_independently_overridable(self) -> None:
        from odds_core.config import AlertConfig, SchedulerConfig
        from odds_core.job_specs import default_bootstrap_jobs, default_heartbeat_expectations

        assert SchedulerConfig(_env_file=None).bootstrap_jobs == default_bootstrap_jobs()
        assert (
            AlertConfig(_env_file=None).heartbeat_expectations == default_heartbeat_expectations()
        )
        assert SchedulerConfig(bootstrap_jobs=["check-health"]).bootstrap_jobs == ["check-health"]
        assert AlertConfig(heartbeat_expectations={"custom": 3}).heartbeat_expectations == {
            "custom": 3
        }

    def test_legacy_heartbeat_names_differ_from_persisted_schedule_ids(self) -> None:
        from odds_core.job_specs import heartbeat_name

        assert heartbeat_name("fetch-mlb-probables", "baseball_mlb") == "fetch-mlb-probables"
        assert (
            make_compound_job_name("fetch-mlb-probables", "baseball_mlb")
            == "fetch-mlb-probables-mlb"
        )
        assert heartbeat_name("fetch-espn-fixtures", "soccer_epl") == "fetch-espn-fixtures"
        assert (
            make_compound_job_name("fetch-espn-fixtures", "soccer_epl") == "fetch-espn-fixtures-epl"
        )
        assert heartbeat_name("fetch-espn-fixtures", "basketball_nba") == "fetch-espn-fixtures-nba"

    def test_adapters_support_exactly_the_registered_sports(self) -> None:
        from odds_lambda.betfair.constants import SPORT_CONFIG
        from odds_lambda.polymarket_sports import POLYMARKET_SPORTS

        assert set(get_job_sports("fetch-betfair-exchange")) == set(SPORT_CONFIG)
        assert set(get_job_sports("fetch-polymarket-odds")) == set(POLYMARKET_SPORTS)

    def test_heartbeat_definitions_are_unique(self) -> None:
        names = [beat.name for spec in JOB_SPECS.values() for beat in spec.heartbeats]
        assert len(names) == len(set(names))

    def test_job_loading_stays_lazy_and_cached(self) -> None:
        from odds_lambda.scheduling.jobs import _loaded_jobs, get_job_function

        _loaded_jobs.pop("check-health", None)
        try:
            with patch("odds_lambda.scheduling.jobs.import_module") as load:
                main = load.return_value.main
                assert get_job_function("check-health") is main
                assert get_job_function("check-health") is main
                load.assert_called_once_with(JOB_SPECS["check-health"].module)
        finally:
            _loaded_jobs.pop("check-health", None)
