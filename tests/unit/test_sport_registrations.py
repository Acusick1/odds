"""Every sport is registered everywhere the add-sport guide says, so a missing entry is named."""

from __future__ import annotations

import pytest
from odds_core.close import CLOSES
from odds_core.sports import LEAGUE_TIMEZONES, SUPPORTED_SPORTS
from odds_lambda.candidate_rules.rules import SPORT_MARKETS
from odds_lambda.collection_calendar import CollectionSource
from odds_lambda.health_readers import collection_sources
from odds_lambda.scheduling.decision import OVERNIGHT_WINDOWS
from odds_lambda.scheduling.jobs import _SPORT_SUFFIX_MAP
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY

COLLECTED = [
    pytest.param(source, sport_key, id=f"{source.key}-{sport_key}")
    for source in collection_sources()
    for sport_key in source.sports
]


class TestSportRegistrations:
    @pytest.mark.parametrize("sport_key", SUPPORTED_SPORTS)
    def test_every_sport_has_a_timezone_kickoff_authority_and_candidate_markets(
        self, sport_key: str
    ) -> None:
        assert sport_key in LEAGUE_TIMEZONES, f"{sport_key} has no LEAGUE_TIMEZONES entry"
        assert sport_key in KICKOFF_AUTHORITY, f"{sport_key} has no KICKOFF_AUTHORITY entry"
        assert sport_key in SPORT_MARKETS, f"{sport_key} has no SPORT_MARKETS entry"

    def test_candidate_markets_name_only_supported_sports(self) -> None:
        assert set(SPORT_MARKETS) <= set(SUPPORTED_SPORTS)

    def test_every_candidate_market_has_a_close_entry(self) -> None:
        decidable = {
            (sport, market) for sport, markets in SPORT_MARKETS.items() for market in markets
        }
        missing = decidable - set(CLOSES)
        unknown = set(CLOSES) - decidable
        assert not missing, f"no CLOSES entry (a CloseSpec, or None for no close) for {missing}"
        assert not unknown, f"CLOSES entries for markets SPORT_MARKETS lacks: {unknown}"

    @pytest.mark.parametrize(("source", "sport_key"), COLLECTED)
    def test_every_collected_sport_is_registered(
        self, source: CollectionSource, sport_key: str
    ) -> None:
        assert sport_key in SUPPORTED_SPORTS, f"{sport_key} is not a SportKey"
        assert sport_key in _SPORT_SUFFIX_MAP.values(), (
            f"{sport_key} has no _SPORT_SUFFIX_MAP suffix"
        )
        if source.skips_overnight:
            assert sport_key in OVERNIGHT_WINDOWS, f"{sport_key} has no OVERNIGHT_WINDOWS entry"
