from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from odds_core.sports import SportKey


@dataclass(frozen=True, slots=True)
class Cadence:
    closing: float
    pregame: float
    sharp: float
    early: float
    opening: float
    no_game: float
    db_fallback: float = 1.0


@dataclass(frozen=True, slots=True)
class Heartbeat:
    name: str
    hours: float
    sport: SportKey | None = None


@dataclass(frozen=True, slots=True)
class SportMap:
    module: str
    attribute: str


@dataclass(frozen=True, slots=True)
class JobSpec:
    module: str
    function: str = "main"
    per_sport: bool = False
    sports: tuple[SportKey, ...] | SportMap | None = None
    bootstrap: bool = True
    bootstrap_function: str | None = None
    agent_trades: bool = False
    candidate_trades: bool = False
    paid_api: bool = False
    cron: str | None = None
    heartbeats: tuple[Heartbeat, ...] = ()
    cadence: Cadence | None = None
    sport_cadences: tuple[tuple[SportKey, Cadence], ...] = ()


JOB_SPECS: Mapping[str, JobSpec] = MappingProxyType(
    {
        "fetch-odds": JobSpec(
            module="odds_lambda.jobs.fetch_odds",
            per_sport=True,
            sports=("soccer_epl", "basketball_nba"),
            paid_api=True,
            heartbeats=(
                Heartbeat("fetch-odds-epl", 49, "soccer_epl"),
                Heartbeat("fetch-odds-nba", 35, "basketball_nba"),
            ),
            cadence=Cadence(
                closing=2.0, pregame=3.0, sharp=12.0, early=24.0, opening=48.0, no_game=24.0
            ),
            sport_cadences=(
                (
                    "basketball_nba",
                    Cadence(
                        closing=12.0,
                        pregame=12.0,
                        sharp=12.0,
                        early=24.0,
                        opening=24.0,
                        no_game=24.0,
                    ),
                ),
            ),
        ),
        "fetch-scores": JobSpec(
            module="odds_lambda.jobs.fetch_scores",
            per_sport=True,
            sports=("soccer_epl",),
            paid_api=True,
            heartbeats=(Heartbeat("fetch-scores-epl", 13, "soccer_epl"),),
        ),
        "update-status": JobSpec(module="odds_lambda.jobs.update_status"),
        "check-health": JobSpec(
            module="odds_lambda.jobs.check_health", heartbeats=(Heartbeat("check-health", 2, None),)
        ),
        "fetch-oddsportal": JobSpec(
            module="odds_lambda.jobs.fetch_oddsportal",
            per_sport=True,
            heartbeats=(
                Heartbeat("fetch-oddsportal-epl", 11, "soccer_epl"),
                Heartbeat("fetch-oddsportal-mlb", 12, "baseball_mlb"),
                Heartbeat("fetch-oddsportal-nba", 11, "basketball_nba"),
            ),
            cadence=Cadence(
                closing=0.5,
                pregame=1.0,
                sharp=2.0,
                early=2.0,
                opening=2.0,
                no_game=6.0,
                db_fallback=1.0,
            ),
        ),
        "fetch-oddsportal-results": JobSpec(
            module="odds_lambda.jobs.fetch_oddsportal_results",
            per_sport=True,
            sports=("soccer_epl",),
            heartbeats=(Heartbeat("fetch-oddsportal-results-epl", 26, "soccer_epl"),),
        ),
        "fetch-mlb-results": JobSpec(
            module="odds_lambda.jobs.fetch_mlb_results",
            heartbeats=(Heartbeat("fetch-mlb-results", 26, None),),
        ),
        "agent-run": JobSpec(
            module="odds_lambda.jobs.agent_run",
            per_sport=True,
            bootstrap=False,
            bootstrap_function="schedule_next",
            agent_trades=True,
            cadence=Cadence(
                closing=1.0, pregame=4.0, sharp=4.0, early=12.0, opening=24.0, no_game=12.0
            ),
        ),
        "fetch-espn-fixtures": JobSpec(
            module="odds_lambda.jobs.fetch_espn_fixtures",
            per_sport=True,
            sports=("soccer_epl", "basketball_nba"),
            cron="0 6 * * *",
            heartbeats=(
                Heartbeat("fetch-espn-fixtures", 26, "soccer_epl"),
                Heartbeat("fetch-espn-fixtures-nba", 26, "basketball_nba"),
            ),
        ),
        "fetch-betfair-exchange": JobSpec(
            module="odds_lambda.jobs.fetch_betfair_exchange",
            per_sport=True,
            sports=SportMap("odds_lambda.betfair.constants", "SPORT_CONFIG"),
            heartbeats=(
                Heartbeat("fetch-betfair-exchange-epl", 10, "soccer_epl"),
                Heartbeat("fetch-betfair-exchange-mlb", 11, "baseball_mlb"),
                Heartbeat("fetch-betfair-exchange-nba", 11, "basketball_nba"),
            ),
            cadence=Cadence(
                closing=0.25,
                pregame=0.5,
                sharp=2.0,
                early=2.0,
                opening=2.0,
                no_game=6.0,
                db_fallback=1.0,
            ),
        ),
        "fetch-polymarket-odds": JobSpec(
            module="odds_lambda.jobs.fetch_polymarket_odds",
            per_sport=True,
            sports=SportMap("odds_lambda.polymarket_sports", "POLYMARKET_SPORTS"),
            heartbeats=(
                Heartbeat("fetch-polymarket-odds-epl", 11, "soccer_epl"),
                Heartbeat("fetch-polymarket-odds-nba", 11, "basketball_nba"),
            ),
            cadence=Cadence(
                closing=1 / 12, pregame=0.25, sharp=0.25, early=0.5, opening=1, no_game=6
            ),
        ),
        "fetch-mlb-probables": JobSpec(
            module="odds_lambda.jobs.fetch_mlb_probables",
            per_sport=True,
            sports=("baseball_mlb",),
            cron="0 6 * * *",
            heartbeats=(Heartbeat("fetch-mlb-probables", 30, "baseball_mlb"),),
        ),
        "fetch-injuries": JobSpec(
            module="odds_lambda.jobs.fetch_injuries",
            per_sport=True,
            sports=("basketball_nba",),
            heartbeats=(Heartbeat("fetch-injuries-nba", 15, "basketball_nba"),),
            cadence=Cadence(
                closing=6.0,
                pregame=6.0,
                sharp=6.0,
                early=6.0,
                opening=6.0,
                no_game=6.0,
                db_fallback=1.0,
            ),
        ),
        "run-candidates": JobSpec(
            module="odds_lambda.jobs.run_candidates", per_sport=True, candidate_trades=True
        ),
        "backup-database": JobSpec(
            module="odds_lambda.jobs.backup_database",
            cron="0 5 * * *",
            heartbeats=(Heartbeat("backup-database", 26, None),),
        ),
    }
)


def job_cadence(job_name: str, sport: SportKey | None = None) -> Cadence:
    spec = JOB_SPECS[job_name]
    for key, cadence in spec.sport_cadences:
        if sport == key:
            return cadence
    if spec.cadence is None:
        raise ValueError(f"No forward cadence for {job_name}")
    return spec.cadence


def default_bootstrap_jobs() -> list[str]:
    return [name for name, spec in JOB_SPECS.items() if spec.bootstrap]


def default_heartbeat_expectations() -> dict[str, float]:
    return {beat.name: beat.hours for spec in JOB_SPECS.values() for beat in spec.heartbeats}


def heartbeat_name(job_name: str, sport: SportKey | None = None) -> str:
    for beat in JOB_SPECS[job_name].heartbeats:
        if beat.sport == sport:
            return beat.name
    raise ValueError(f"No heartbeat for {job_name}, {sport}")
