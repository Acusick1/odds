"""The per-sport kick-off authority order."""

from __future__ import annotations

import pytest
from odds_core.sports import SportKey
from odds_core.vocabulary import Source
from odds_lambda.storage.event_kickoff import KICKOFF_AUTHORITY, PLAYED_START_SOURCES


class TestKickoffAuthorityOrder:
    @pytest.mark.parametrize(
        ("sport", "order"),
        [
            ("baseball_mlb", (Source.MLB_STATS_API,)),
            ("soccer_epl", (Source.ODDSPORTAL, Source.ODDS_API, Source.FOOTBALL_DATA_UK)),
            ("basketball_nba", (Source.ODDSPORTAL, Source.ODDS_API)),
        ],
    )
    def test_order_per_sport(self, sport: SportKey, order: tuple[Source, ...]) -> None:
        assert KICKOFF_AUTHORITY[sport] == order

    @pytest.mark.parametrize("sport", ["soccer_epl", "basketball_nba"])
    def test_sources_without_a_stored_kickoff_are_left_out(self, sport: SportKey) -> None:
        unranked = {Source.BETFAIR, Source.POLYMARKET, Source.ESPN}
        assert not unranked & set(KICKOFF_AUTHORITY[sport])

    def test_football_data_uk_ranks_last_for_epl_only(self) -> None:
        assert KICKOFF_AUTHORITY["soccer_epl"][-1] is Source.FOOTBALL_DATA_UK
        assert Source.FOOTBALL_DATA_UK not in KICKOFF_AUTHORITY["basketball_nba"]

    def test_no_source_repeats_within_an_order(self) -> None:
        for order in KICKOFF_AUTHORITY.values():
            assert len(set(order)) == len(order)


class TestPlayedStartSources:
    def test_the_mlb_stats_api_and_football_data_uk_date_games_by_when_played(self) -> None:
        assert PLAYED_START_SOURCES == {Source.MLB_STATS_API, Source.FOOTBALL_DATA_UK}
