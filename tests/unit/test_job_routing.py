"""Tests for job name resolution and compound name construction."""

from odds_lambda.scheduling.jobs import (
    make_compound_job_name,
    resolve_job_name,
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

    def test_unknown_suffix_returns_no_sport(self) -> None:
        base, sport = resolve_job_name("fetch-odds-nba")
        assert base == "fetch-odds-nba"
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
        assert sport_key_to_suffix("basketball_nba") is None


class TestMakeCompoundJobName:
    """Tests for make_compound_job_name()."""

    def test_with_sport(self) -> None:
        assert make_compound_job_name("fetch-odds", "soccer_epl") == "fetch-odds-epl"

    def test_with_sport_mlb(self) -> None:
        assert make_compound_job_name("fetch-oddsportal", "baseball_mlb") == "fetch-oddsportal-mlb"

    def test_without_sport(self) -> None:
        assert make_compound_job_name("fetch-odds", None) == "fetch-odds"

    def test_unknown_sport_returns_base(self) -> None:
        assert make_compound_job_name("fetch-odds", "basketball_nba") == "fetch-odds"

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
