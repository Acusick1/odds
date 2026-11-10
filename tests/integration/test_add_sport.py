"""A sport added through the add-sport guide reaches the collectors, the status strip and the
candidate rules."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest
from odds_core.candidates import CandidateStatus, MechanicalCandidate
from odds_core.close import close_spec
from odds_core.config import SchedulerConfig, Settings, get_settings
from odds_core.models import Event, EventStatus
from odds_core.vocabulary import Market
from odds_lambda.betfair import constants as betfair_constants
from odds_lambda.betfair.constants import EVENT_TYPE_SOCCER, BetfairMarketSpec, SportBetfairConfig
from odds_lambda.candidate_rules import resolve_rule, rule_sha256
from odds_lambda.jobs import fetch_betfair_exchange, fetch_oddsportal
from odds_lambda.jobs.fetch_betfair_exchange import betfair_collection
from odds_lambda.jobs.fetch_oddsportal import LeagueSpec, league_collection
from odds_lambda.scheduling import decision, jobs
from odds_web.status import ChipState, load_status
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import create_scheduler_tables

SPORT = "soccer_spain_la_liga"
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)

LEAGUE = LeagueSpec(
    league="laliga",
    listing_path="/football/spain/laliga/",
    sport_key=SPORT,
    sport_title="La Liga",
    fixture_horizon=timedelta(hours=48),
    markets=["1x2", "over_under_2_5"],
)
BETFAIR = SportBetfairConfig(
    sport_key=SPORT,
    sport_title="La Liga",
    event_type_id=EVENT_TYPE_SOCCER,
    match_odds=BetfairMarketSpec("MATCH_ODDS", Market.ONE_X_TWO),
    competition_id=None,
    fixture_horizon=timedelta(hours=72),
    competition_name_hints=("Spanish La Liga",),
)


@pytest.fixture
def la_liga(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guide's registrations for La Liga, which its SportKey, timezone, kick-off authority,
    candidate markets and closes already have."""
    monkeypatch.setitem(jobs._SPORT_SUFFIX_MAP, "laliga", SPORT)
    monkeypatch.setitem(decision.OVERNIGHT_WINDOWS, SPORT, (22, 6))
    monkeypatch.setitem(betfair_constants.SPORT_CONFIG, SPORT, BETFAIR)
    for module, collection in (
        (fetch_oddsportal, league_collection(LEAGUE)),
        (fetch_betfair_exchange, betfair_collection(BETFAIR)),
    ):
        source = module.SOURCE
        monkeypatch.setattr(
            module,
            "SOURCE",
            dataclasses.replace(source, sports={**source.sports, SPORT: collection}),
        )


def collecting(*jobs: str) -> Settings:
    settings = get_settings()
    return settings.model_copy(
        update={
            "scheduler": SchedulerConfig(bootstrap_jobs=list(jobs)),
            "data_collection": settings.data_collection.model_copy(update={"sports": [SPORT]}),
        }
    )


@pytest.mark.usefixtures("la_liga")
class TestAddedSport:
    async def test_the_status_strip_shows_each_collected_market(
        self, pglite_async_session: AsyncSession
    ) -> None:
        await create_scheduler_tables(pglite_async_session)
        pglite_async_session.add(
            Event(
                id=1,
                sport_key=SPORT,
                sport_title="La Liga",
                commence_time=NOW + timedelta(hours=24),
                home_team="Real Madrid",
                away_team="Barcelona",
                status=EventStatus.SCHEDULED,
            )
        )
        await pglite_async_session.commit()

        strip = await load_status(
            pglite_async_session,
            collecting("fetch-oddsportal", "fetch-betfair-exchange"),
            NOW,
        )

        assert {(s.label, s.state) for s in strip.sources} == {
            ("oddsportal laliga 1x2", ChipState.UNKNOWN),
            ("oddsportal laliga totals", ChipState.UNKNOWN),
            ("betfair_exchange laliga 1x2", ChipState.UNKNOWN),
        }

    def test_its_candidates_resolve_and_have_a_close(self) -> None:
        candidate = MechanicalCandidate(
            name="exchange-gap-laliga",
            description="d",
            sport=SPORT,
            version=1,
            issue=1,
            start_date="2026-01-01",
            status=CandidateStatus.ACTIVE,
            kind="mechanical",
            market="1x2",
            rule="exchange_gap",
            rule_sha256=rule_sha256("exchange_gap"),
            params={"threshold": 0.01, "minutes_before_kickoff": [60], "max_input_age_minutes": 10},
        )

        resolve_rule(candidate)
        assert close_spec(SPORT, "1x2") is not None
