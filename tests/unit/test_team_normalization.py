"""Tests for team name normalization and Event field validator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus
from odds_core.team import normalize_team_name
from odds_core.vocabulary import Source
from odds_lambda.live_events import LiveEventFinder
from odds_lambda.mlb_game_resolver import GameRow, ResolutionRule, ResolvedGame
from odds_lambda.storage.event_resolver import Resolution, ResolvedEvent, StaleListing
from sqlalchemy.ext.asyncio import AsyncSession


class TestNormalizeTeamName:
    """Unit tests for normalize_team_name()."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            # OddsPortal aliases
            ("Wolverhampton Wanderers", "Wolves"),
            ("Tottenham Hotspur", "Tottenham"),
            ("Manchester United", "Manchester Utd"),
            ("Newcastle United", "Newcastle"),
            ("Nottingham Forest", "Nottingham"),
            ("Sheffield United", "Sheffield Utd"),
            ("West Ham United", "West Ham"),
            ("Brighton and Hove Albion", "Brighton"),
            ("Leeds United", "Leeds"),
            # Passthrough for canonical names
            ("Arsenal", "Arsenal"),
            ("Wolves", "Wolves"),
            ("Tottenham", "Tottenham"),
            ("Manchester Utd", "Manchester Utd"),
            ("Brighton", "Brighton"),
            ("Chelsea", "Chelsea"),
            ("Liverpool", "Liverpool"),
            # Whitespace cleanup
            ("  manchester   united ", "Manchester Utd"),
            ("\tArsenal\n", "Arsenal"),
            ("  Wolverhampton   Wanderers  ", "Wolves"),
            # Case insensitivity
            ("wolverhampton wanderers", "Wolves"),
            ("TOTTENHAM HOTSPUR", "Tottenham"),
            ("manchester united", "Manchester Utd"),
            # Other source variants
            ("Brighton & Hove Albion", "Brighton"),
            ("AFC Bournemouth", "Bournemouth"),
            ("Leicester City", "Leicester"),
            ("Cardiff City", "Cardiff"),
            ("Nott'm Forest", "Nottingham"),
            ("West Bromwich Albion", "West Brom"),
            ("Spurs", "Tottenham"),
            ("Coventry City", "Coventry"),
            ("Coventry", "Coventry"),
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_team_name(raw) == expected

    def test_unknown_team_passthrough(self) -> None:
        assert normalize_team_name("Some Random FC") == "Some Random FC"


class TestEventFieldValidator:
    """Test that Event model normalizes team names on construction."""

    def test_normalizes_home_team(self) -> None:
        event = Event(
            id="test1",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 1, 1, tzinfo=UTC),
            home_team="Wolverhampton Wanderers",
            away_team="Arsenal",
        )
        assert event.home_team == "Wolves"

    def test_normalizes_away_team(self) -> None:
        event = Event(
            id="test2",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 1, 1, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Tottenham Hotspur",
        )
        assert event.away_team == "Tottenham"

    def test_normalizes_both_teams(self) -> None:
        event = Event(
            id="test3",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 1, 1, tzinfo=UTC),
            home_team="Manchester United",
            away_team="Newcastle United",
        )
        assert event.home_team == "Manchester Utd"
        assert event.away_team == "Newcastle"

    def test_canonical_names_unchanged(self) -> None:
        event = Event(
            id="test4",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 1, 1, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
        )
        assert event.home_team == "Arsenal"
        assert event.away_team == "Chelsea"

    def test_whitespace_cleaned(self) -> None:
        event = Event(
            id="test5",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 1, 1, tzinfo=UTC),
            home_team="  Manchester   City ",
            away_team="Liverpool",
        )
        assert event.home_team == "Manchester City"


async def _resolve(
    session: AsyncSession,
    home_team: str,
    away_team: str,
    start: datetime,
    sport_key: str = "soccer_epl",
    game_pk: int | None = None,
) -> tuple[str, bool]:
    game = GameRow(home_team, away_team, start)
    resolutions = (
        {} if game_pk is None else {game: ResolvedGame(game_pk, ResolutionRule.SINGLE_GAME)}
    )
    finder = LiveEventFinder(sport_key, sport_key, resolutions)
    found = await finder.resolve(session, game, Source.ODDSPORTAL, None)
    assert isinstance(found, ResolvedEvent)
    await session.flush()
    return found.event.id, found.resolution is Resolution.CREATED


class TestLiveEventFinderNormalization:
    @pytest.mark.asyncio
    async def test_finds_existing_event_with_alias_input(self, test_session) -> None:
        test_session.add(
            Event(
                id="existing_wolves",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
                home_team="Wolves",
                away_team="Arsenal",
                status=EventStatus.SCHEDULED,
            )
        )
        await test_session.flush()

        event_id, created = await _resolve(
            test_session,
            "Wolverhampton Wanderers",
            "Arsenal",
            datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
        )

        assert (event_id, created) == ("existing_wolves", False)


class TestLiveEventFinderCreates:
    @pytest.mark.asyncio
    async def test_creates_an_op_live_event_when_none_exists(self, test_session) -> None:
        event_id, created = await _resolve(
            test_session, "Arsenal", "Chelsea", datetime.now(UTC) + timedelta(days=7)
        )

        assert created is True
        assert event_id.startswith("op_live_ARS_CHE_")

    @pytest.mark.asyncio
    async def test_re_listing_is_idempotent(self, test_session) -> None:
        start = datetime.now(UTC) + timedelta(days=7)

        first = await _resolve(test_session, "Arsenal", "Chelsea", start)
        second = await _resolve(test_session, "Arsenal", "Chelsea", start + timedelta(minutes=35))

        assert first == (first[0], True)
        assert second == (first[0], False)


class TestLiveEventFinderMatchWindow:
    """MLB games stored as one event each: back-to-back series games and doubleheaders."""

    @pytest.mark.asyncio
    async def test_same_teams_consecutive_days_are_distinct_events(self, test_session) -> None:
        day_one = (datetime.now(UTC) + timedelta(days=1)).replace(
            hour=23, minute=10, second=0, microsecond=0
        )

        first = await _resolve(
            test_session, "New York Yankees", "Boston Red Sox", day_one, "baseball_mlb", 1001
        )
        second = await _resolve(
            test_session,
            "New York Yankees",
            "Boston Red Sox",
            day_one + timedelta(days=1),
            "baseball_mlb",
            1002,
        )

        assert first[1] and second[1]
        assert first[0] != second[0]

    @pytest.mark.asyncio
    async def test_same_day_doubleheader_games_are_distinct_events(self, test_session) -> None:
        game_one = (datetime.now(UTC) + timedelta(days=1)).replace(
            hour=13, minute=0, second=0, microsecond=0
        )

        first = await _resolve(
            test_session, "Chicago Cubs", "St. Louis Cardinals", game_one, "baseball_mlb", 1001
        )
        second = await _resolve(
            test_session,
            "Chicago Cubs",
            "St. Louis Cardinals",
            game_one + timedelta(hours=5),
            "baseball_mlb",
            1002,
        )

        assert first[1] and second[1]
        assert first[0] != second[0]


class TestLiveEventFinderStaleListing:
    """A live listing never creates an event whose start is over an hour past."""

    @pytest.mark.asyncio
    async def test_accepts_a_listing_within_the_window(self, test_session) -> None:
        event_id, created = await _resolve(
            test_session, "Arsenal", "Chelsea", datetime.now(UTC) - timedelta(minutes=30)
        )

        assert created is True

    @pytest.mark.asyncio
    async def test_refuses_to_create_a_stale_event(self, test_session) -> None:
        with pytest.raises(StaleListing, match="too far past"):
            await _resolve(
                test_session, "Arsenal", "Chelsea", datetime.now(UTC) - timedelta(hours=2)
            )

    @pytest.mark.asyncio
    async def test_finds_an_existing_stale_event_without_creating(self, test_session) -> None:
        match_date = datetime.now(UTC) - timedelta(hours=5)
        test_session.add(
            Event(
                id="op_live_ARS_CHE_2020-01-01T1500",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=match_date,
                home_team="Arsenal",
                away_team="Chelsea",
                status=EventStatus.SCHEDULED,
            )
        )
        await test_session.flush()

        event_id, created = await _resolve(test_session, "Arsenal", "Chelsea", match_date)

        assert (event_id, created) == ("op_live_ARS_CHE_2020-01-01T1500", False)
