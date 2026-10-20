"""Unit tests for Betfair Exchange adapter logic."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from odds_lambda.betfair.adapter import (
    _is_draw,
    _normalize_runner_name,
    betfair_book_to_bookmaker_entry,
    canonical_teams,
    resolve_teams,
    strip_pitcher_annotation,
)
from odds_lambda.betfair.client import BetfairBook, BetfairRunner
from odds_lambda.betfair.constants import SPORT_CONFIG


def _runner(
    selection_id: int,
    name: str,
    back: float | None,
    lay: float | None = None,
    *,
    back_size: float | None = None,
    lay_size: float | None = None,
    ltp: float | None = None,
) -> BetfairRunner:
    return BetfairRunner(
        selection_id=selection_id,
        runner_name=name,
        best_back=back,
        best_lay=lay,
        back_size=back_size,
        lay_size=lay_size,
        last_price_traded=ltp,
    )


def _book(
    *,
    market_id: str,
    event_id: str,
    event_name: str,
    runners: list[BetfairRunner],
    market_status: str = "OPEN",
    inplay: bool = False,
    total_matched: float | None = 1234.5,
    market_start_time: datetime | None = None,
) -> BetfairBook:
    return BetfairBook(
        market_id=market_id,
        betfair_event_id=event_id,
        betfair_event_name=event_name,
        market_start_time=market_start_time or datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
        market_status=market_status,
        inplay=inplay,
        total_matched=total_matched,
        runners=runners,
    )


class TestStripPitcherAnnotation:
    def test_tbd_suffix_removed(self) -> None:
        assert strip_pitcher_annotation("Colorado Rockies (TBD)") == "Colorado Rockies"

    def test_pitcher_name_suffix_removed(self) -> None:
        assert strip_pitcher_annotation("New York Yankees (Cole)") == "New York Yankees"

    def test_truncated_suffix_without_closing_paren_removed(self) -> None:
        assert strip_pitcher_annotation("Tampa Bay Rays (D Rasm") == "Tampa Bay Rays"
        assert strip_pitcher_annotation("Chicago White Sox (TBD") == "Chicago White Sox"
        assert strip_pitcher_annotation("Los Angeles Dodgers (") == "Los Angeles Dodgers"

    def test_no_paren_unchanged(self) -> None:
        assert strip_pitcher_annotation("Boston Red Sox") == "Boston Red Sox"

    def test_paren_in_middle_preserved(self) -> None:
        # Defensive: only trailing parens are stripped
        assert strip_pitcher_annotation("Some (mid) Team") == "Some (mid) Team"


class TestNormalizeRunnerName:
    def test_betfair_alias_short_form(self) -> None:
        assert _normalize_runner_name("Man Utd") == "Manchester Utd"

    def test_betfair_alias_long_form(self) -> None:
        assert _normalize_runner_name("Manchester United") == "Manchester Utd"

    def test_betfair_short_mlb(self) -> None:
        assert _normalize_runner_name("LA Angels") == "Los Angeles Angels"

    def test_pitcher_annotation_then_alias(self) -> None:
        assert _normalize_runner_name("NY Yankees (TBD)") == "New York Yankees"

    def test_canonical_passthrough(self) -> None:
        assert _normalize_runner_name("Liverpool") == "Liverpool"

    def test_athletics_canonical(self) -> None:
        # Athletics is a single-word canonical name that maps from "Oakland Athletics".
        assert _normalize_runner_name("Oakland Athletics") == "Athletics"


class TestIsDraw:
    @pytest.mark.parametrize("name", ["The Draw", "draw", "DRAW", "the draw"])
    def test_draw_variants(self, name: str) -> None:
        assert _is_draw(name)

    def test_team_name_is_not_draw(self) -> None:
        assert not _is_draw("Liverpool")


class TestResolveTeamsSoccer:
    def test_epl_home_first(self) -> None:
        cfg = SPORT_CONFIG["soccer_epl"]
        book = _book(
            market_id="1.1",
            event_id="35480901",
            event_name="Liverpool v Crystal Palace",
            runners=[],
        )
        assert resolve_teams(book, cfg) == ("Liverpool", "Crystal Palace")

    def test_epl_short_name(self) -> None:
        cfg = SPORT_CONFIG["soccer_epl"]
        book = _book(
            market_id="1.2",
            event_id="x",
            event_name="Man Utd v Spurs",
            runners=[],
        )
        assert resolve_teams(book, cfg) == ("Manchester Utd", "Tottenham")

    def test_epl_unparseable(self) -> None:
        cfg = SPORT_CONFIG["soccer_epl"]
        book = _book(
            market_id="1.3",
            event_id="x",
            event_name="Some Friendly Match",
            runners=[],
        )
        assert resolve_teams(book, cfg) is None


class TestResolveTeamsBaseball:
    def test_mlb_away_first_swap(self) -> None:
        cfg = SPORT_CONFIG["baseball_mlb"]
        book = _book(
            market_id="1.4",
            event_id="x",
            event_name="Boston Red Sox @ Baltimore Orioles",
            runners=[],
        )
        # Betfair: Away @ Home → home first in returned tuple
        assert resolve_teams(book, cfg) == ("Baltimore Orioles", "Boston Red Sox")

    def test_mlb_with_truncated_pitcher_annotations(self) -> None:
        cfg = SPORT_CONFIG["baseball_mlb"]
        book = _book(
            market_id="1.6",
            event_id="x",
            event_name="Cleveland Guardians (S @ Tampa Bay Rays (N Marti",
            runners=[],
        )
        assert resolve_teams(book, cfg) == ("Tampa Bay Rays", "Cleveland Guardians")

    def test_mlb_with_pitchers(self) -> None:
        cfg = SPORT_CONFIG["baseball_mlb"]
        book = _book(
            market_id="1.5",
            event_id="x",
            event_name="Colorado Rockies (TBD) @ New York Mets (TBD)",
            runners=[],
        )
        assert resolve_teams(book, cfg) == ("New York Mets", "Colorado Rockies")


class TestBookToBookmakerEntry:
    @pytest.fixture
    def epl_book(self) -> BetfairBook:
        return _book(
            market_id="1.123",
            event_id="35480901",
            event_name="Liverpool v Crystal Palace",
            runners=[
                _runner(1, "Liverpool", back=1.64, lay=1.65, back_size=292, lay_size=314, ltp=1.64),
                _runner(
                    2, "Crystal Palace", back=5.7, lay=5.9, back_size=243, lay_size=165, ltp=5.8
                ),
                _runner(3, "The Draw", back=4.5, lay=4.6, back_size=32, lay_size=214, ltp=4.6),
            ],
            total_matched=385702.0,
        )

    @pytest.fixture
    def mlb_book(self) -> BetfairBook:
        return _book(
            market_id="1.456",
            event_id="35529928",
            event_name="Boston Red Sox @ Baltimore Orioles",
            runners=[
                _runner(10, "Boston Red Sox", back=2.0, lay=2.05),
                _runner(11, "Baltimore Orioles", back=1.95, lay=1.97),
            ],
            total_matched=3500.0,
        )

    def test_epl_entry_shape(self, epl_book: BetfairBook) -> None:
        cfg = SPORT_CONFIG["soccer_epl"]
        snap = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
        entry = betfair_book_to_bookmaker_entry(
            epl_book, cfg, home_team="Liverpool", away_team="Crystal Palace", snapshot_time=snap
        )
        assert entry is not None
        assert entry["key"] == "betfair_exchange"
        assert entry["title"] == "Betfair Exchange"
        assert entry["last_update"] == snap.isoformat()
        markets = entry["markets"]
        assert len(markets) == 1
        assert markets[0]["key"] == "1x2"
        outcome_names = [o["name"] for o in markets[0]["outcomes"]]
        assert sorted(outcome_names) == ["Crystal Palace", "Draw", "Liverpool"]

    def test_epl_prices_are_best_back_decimals(self, epl_book: BetfairBook) -> None:
        cfg = SPORT_CONFIG["soccer_epl"]
        entry = betfair_book_to_bookmaker_entry(
            epl_book, cfg, home_team="Liverpool", away_team="Crystal Palace"
        )
        assert entry is not None
        outcomes = entry["markets"][0]["outcomes"]
        assert all("price" not in o for o in outcomes)
        prices = {o["name"]: o["price_decimal"] for o in outcomes}
        assert prices == {"Liverpool": 1.64, "Crystal Palace": 5.7, "Draw": 4.5}

    def test_epl_betfair_meta_preserved(self, epl_book: BetfairBook) -> None:
        cfg = SPORT_CONFIG["soccer_epl"]
        entry = betfair_book_to_bookmaker_entry(
            epl_book, cfg, home_team="Liverpool", away_team="Crystal Palace"
        )
        assert entry is not None
        meta = entry["betfair_meta"]
        assert meta["market_id"] == "1.123"
        assert meta["market_status"] == "OPEN"
        assert meta["inplay"] is False
        assert meta["total_matched"] == pytest.approx(385702.0)
        assert len(meta["runners"]) == 3

    def test_mlb_entry_shape(self, mlb_book: BetfairBook) -> None:
        cfg = SPORT_CONFIG["baseball_mlb"]
        entry = betfair_book_to_bookmaker_entry(
            mlb_book,
            cfg,
            home_team="Baltimore Orioles",
            away_team="Boston Red Sox",
        )
        assert entry is not None
        assert entry["markets"][0]["key"] == "h2h"
        outcomes = entry["markets"][0]["outcomes"]
        names = [o["name"] for o in outcomes]
        assert sorted(names) == ["Baltimore Orioles", "Boston Red Sox"]

    def test_returns_none_when_outcomes_missing(self) -> None:
        cfg = SPORT_CONFIG["soccer_epl"]
        # Only one of three outcomes has a back price
        book = _book(
            market_id="1.x",
            event_id="x",
            event_name="A v B",
            runners=[
                _runner(1, "A", back=2.0),
                _runner(2, "B", back=None),
                _runner(3, "The Draw", back=None),
            ],
        )
        entry = betfair_book_to_bookmaker_entry(book, cfg, home_team="A", away_team="B")
        assert entry is None


class TestCanonicalTeams:
    RUNNERS = ["Los Angeles Angels", "Kansas City Royals"]

    def test_truncated_half_takes_the_full_runner_name(self) -> None:
        assert canonical_teams("Kansas City Royal", "Los Angeles Angels (P", self.RUNNERS) == (
            "Kansas City Royals",
            "Los Angeles Angels",
        )

    def test_game_suffix_is_dropped_whatever_the_form(self) -> None:
        runners = ["Pittsburgh Pirates", "Milwaukee Brewers"]
        assert canonical_teams("Pittsburgh Pirates - Game 1", "Milwaukee Brewers", runners) == (
            "Pittsburgh Pirates",
            "Milwaukee Brewers",
        )
        assert canonical_teams("Pittsburgh Pirates Game 2", "Milwaukee Brewers", []) == (
            "Pittsburgh Pirates",
            "Milwaukee Brewers",
        )

    def test_runner_names_get_canonical_aliases(self) -> None:
        assert canonical_teams(
            "St. Louis Cardin",
            "Arizona Diamondbacks",
            [
                "Arizona Diamondbacks",
                "St. Louis Cardinals",
            ],
        ) == ("St.Louis Cardinals", "Arizona Diamondbacks")

    def test_ambiguous_half_falls_back_to_the_stripped_name(self) -> None:
        runners = ["Chicago Cubs", "Chicago White Sox"]
        assert canonical_teams("Chicago (TBD", "Chicago White Sox", runners) == (
            "Chicago",
            "Chicago White Sox",
        )

    def test_draw_runner_is_never_a_team(self) -> None:
        assert canonical_teams("Arsenal", "Chelsea", ["Arsenal", "The Draw", "Chelsea"]) == (
            "Arsenal",
            "Chelsea",
        )
