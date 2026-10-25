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
            ("soccer_epl", (Source.ODDSPORTAL, Source.ODDS_API)),
            ("basketball_nba", (Source.ODDSPORTAL, Source.ODDS_API)),
        ],
    )
    def test_order_per_sport(self, sport: SportKey, order: tuple[Source, ...]) -> None:
        assert KICKOFF_AUTHORITY[sport] == order

    @pytest.mark.parametrize("sport", ["soccer_epl", "basketball_nba"])
    def test_sources_without_a_stored_kickoff_are_left_out(self, sport: SportKey) -> None:
        unranked = {Source.BETFAIR, Source.POLYMARKET, Source.ESPN, Source.FOOTBALL_DATA_UK}
        assert not unranked & set(KICKOFF_AUTHORITY[sport])

    def test_no_source_repeats_within_an_order(self) -> None:
        for order in KICKOFF_AUTHORITY.values():
            assert len(set(order)) == len(order)


class TestPlayedStartSources:
    def test_only_the_mlb_stats_api_dates_games_by_when_played(self) -> None:
        assert PLAYED_START_SOURCES == {Source.MLB_STATS_API}
