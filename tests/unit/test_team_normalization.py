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


# Each club's spellings in its 2024-26 seasons: football-data.co.uk, OddsPortal results, and the
# Odds API's free /events listing on 2026-11-05 (None for a club it no longer lists).
FOOTBALL_CLUBS: dict[str, list[tuple[str, str, str | None]]] = {
    "soccer_efl_champ": [
        ("Birmingham", "Birmingham", "Birmingham City"),
        ("Blackburn", "Blackburn", "Blackburn Rovers"),
        ("Bristol City", "Bristol City", "Bristol City"),
        ("Burnley", "Burnley", "Burnley"),
        ("Cardiff", "Cardiff", "Cardiff City"),
        ("Charlton", "Charlton", "Charlton Athletic"),
        ("Coventry", "Coventry", "Coventry City"),
        ("Derby", "Derby", "Derby County"),
        ("Hull", "Hull", "Hull City"),
        ("Ipswich", "Ipswich", "Ipswich Town"),
        ("Leeds", "Leeds", "Leeds United"),
        ("Leicester", "Leicester", None),
        ("Luton", "Luton", None),
        ("Middlesbrough", "Middlesbrough", "Middlesbrough"),
        ("Millwall", "Millwall", "Millwall"),
        ("Norwich", "Norwich", "Norwich City"),
        ("Oxford", "Oxford Utd", None),
        ("Plymouth", "Plymouth", None),
        ("Portsmouth", "Portsmouth", "Portsmouth"),
        ("Preston", "Preston", "Preston North End"),
        ("QPR", "QPR", "Queens Park Rangers"),
        ("Sheffield United", "Sheffield Utd", "Sheffield United"),
        ("Sheffield Weds", "Sheffield Wed", None),
        ("Southampton", "Southampton", "Southampton"),
        ("Stoke", "Stoke", "Stoke City"),
        ("Sunderland", "Sunderland", "Sunderland"),
        ("Swansea", "Swansea", "Swansea City"),
        ("Watford", "Watford", "Watford"),
        ("West Brom", "West Brom", "West Bromwich Albion"),
        ("Wrexham", "Wrexham", "Wrexham AFC"),
    ],
    "soccer_spain_la_liga": [
        ("Alaves", "Alaves", "Alavés"),
        ("Ath Bilbao", "Ath Bilbao", "Athletic Bilbao"),
        ("Ath Madrid", "Atl. Madrid", "Atlético Madrid"),
        ("Barcelona", "Barcelona", "Barcelona"),
        ("Betis", "Betis", "Real Betis"),
        ("Celta", "Celta Vigo", "Celta Vigo"),
        ("Elche", "Elche", "Elche CF"),
        ("Espanol", "Espanyol", "Espanyol"),
        ("Getafe", "Getafe", "Getafe"),
        ("Girona", "Girona", "Girona FC"),
        ("Las Palmas", "Las Palmas", "Las Palmas"),
        ("Leganes", "Leganes", "Leganés"),
        ("Levante", "Levante", "Levante"),
        ("Mallorca", "Mallorca", "Mallorca"),
        ("Osasuna", "Osasuna", "CA Osasuna"),
        ("Oviedo", "Oviedo", "Oviedo"),
        ("Real Madrid", "Real Madrid", "Real Madrid"),
        ("Sevilla", "Sevilla", "Sevilla"),
        ("Sociedad", "Real Sociedad", "Real Sociedad"),
        ("Valencia", "Valencia", "Valencia"),
        ("Valladolid", "Valladolid", "Real Valladolid CF"),
        ("Vallecano", "Rayo Vallecano", "Rayo Vallecano"),
        ("Villarreal", "Villarreal", "Villarreal"),
    ],
    "soccer_italy_serie_a": [
        ("Atalanta", "Atalanta", "Atalanta BC"),
        ("Bologna", "Bologna", "Bologna"),
        ("Cagliari", "Cagliari", "Cagliari"),
        ("Como", "Como", "Como"),
        ("Cremonese", "Cremonese", "Cremonese"),
        ("Empoli", "Empoli", "Empoli"),
        ("Fiorentina", "Fiorentina", "Fiorentina"),
        ("Genoa", "Genoa", "Genoa"),
        ("Inter", "Inter", "Inter Milan"),
        ("Juventus", "Juventus", "Juventus"),
        ("Lazio", "Lazio", "Lazio"),
        ("Lecce", "Lecce", "Lecce"),
        ("Milan", "AC Milan", "AC Milan"),
        ("Monza", "Monza", "Monza"),
        ("Napoli", "Napoli", "Napoli"),
        ("Parma", "Parma", "Parma"),
        ("Pisa", "Pisa", "Pisa"),
        ("Roma", "AS Roma", "AS Roma"),
        ("Sassuolo", "Sassuolo", "Sassuolo"),
        ("Torino", "Torino", "Torino"),
        ("Udinese", "Udinese", "Udinese"),
        ("Venezia", "Venezia", "Venezia"),
        ("Verona", "Verona", "Hellas Verona"),
    ],
    "soccer_germany_bundesliga": [
        ("Augsburg", "Augsburg", "Augsburg"),
        ("Bayern Munich", "Bayern Munich", "Bayern Munich"),
        ("Bochum", "Bochum", "VfL Bochum"),
        ("Dortmund", "Dortmund", "Borussia Dortmund"),
        ("Ein Frankfurt", "Eintracht Frankfurt", "Eintracht Frankfurt"),
        ("FC Koln", "FC Koln", "1. FC Köln"),
        ("Freiburg", "Freiburg", "SC Freiburg"),
        ("Hamburg", "Hamburger SV", "Hamburger SV"),
        ("Heidenheim", "Heidenheim", "1. FC Heidenheim"),
        ("Hoffenheim", "Hoffenheim", "TSG Hoffenheim"),
        ("Holstein Kiel", "Holstein Kiel", "Holstein Kiel"),
        ("Leverkusen", "Bayer Leverkusen", "Bayer Leverkusen"),
        ("M'gladbach", "B. Monchengladbach", "Borussia Monchengladbach"),
        ("Mainz", "Mainz", "FSV Mainz 05"),
        ("RB Leipzig", "RB Leipzig", "RB Leipzig"),
        ("St Pauli", "St. Pauli", "FC St. Pauli"),
        ("Stuttgart", "Stuttgart", "VfB Stuttgart"),
        ("Union Berlin", "Union Berlin", "Union Berlin"),
        ("Werder Bremen", "Werder Bremen", "Werder Bremen"),
        ("Wolfsburg", "Wolfsburg", "VfL Wolfsburg"),
    ],
    "soccer_france_ligue_one": [
        ("Angers", "Angers", "Angers"),
        ("Auxerre", "Auxerre", "Auxerre"),
        ("Brest", "Brest", "Brest"),
        ("Le Havre", "Le Havre", "Le Havre"),
        ("Lens", "Lens", "RC Lens"),
        ("Lille", "Lille", "Lille"),
        ("Lorient", "Lorient", "Lorient"),
        ("Lyon", "Lyon", "Lyon"),
        ("Marseille", "Marseille", "Marseille"),
        ("Metz", "Metz", "Metz"),
        ("Monaco", "Monaco", "AS Monaco"),
        ("Montpellier", "Montpellier", "Montpellier"),
        ("Nantes", "Nantes", "Nantes"),
        ("Nice", "Nice", "Nice"),
        ("Paris FC", "Paris FC", "Paris FC"),
        ("Paris SG", "PSG", "Paris Saint Germain"),
        ("Reims", "Reims", "Stade de Reims"),
        ("Rennes", "Rennes", "Rennes"),
        ("St Etienne", "St Etienne", "Saint Etienne"),
        ("Strasbourg", "Strasbourg", "Strasbourg"),
        ("Toulouse", "Toulouse", "Toulouse"),
    ],
}


class TestFootballLeagueAliases:
    @pytest.mark.parametrize("sport", list(FOOTBALL_CLUBS))
    def test_each_club_has_one_name_across_sources(self, sport: str) -> None:
        for spellings in FOOTBALL_CLUBS[sport]:
            names = {normalize_team_name(s) for s in spellings if s is not None}
            assert len(names) == 1, spellings

    @pytest.mark.parametrize("sport", list(FOOTBALL_CLUBS))
    def test_no_two_clubs_share_a_name(self, sport: str) -> None:
        clubs = FOOTBALL_CLUBS[sport]
        assert len({normalize_team_name(fduk) for fduk, _, _ in clubs}) == len(clubs)


class TestEventFieldValidator:
    """Test that Event model normalizes team names on construction."""

    def test_normalizes_home_team(self) -> None:
        event = Event(
            id=1,
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 1, 1, tzinfo=UTC),
            home_team="Wolverhampton Wanderers",
            away_team="Arsenal",
        )
        assert event.home_team == "Wolves"

    def test_normalizes_away_team(self) -> None:
        event = Event(
            id=2,
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 1, 1, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Tottenham Hotspur",
        )
        assert event.away_team == "Tottenham"

    def test_normalizes_both_teams(self) -> None:
        event = Event(
            id=3,
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
            id=4,
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
            id=5,
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
                id=1,
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

        assert (event_id, created) == (1, False)


class TestLiveEventFinderCreates:
    @pytest.mark.asyncio
    async def test_creates_an_event_when_none_exists(self, test_session) -> None:
        event_id, created = await _resolve(
            test_session, "Arsenal", "Chelsea", datetime.now(UTC) + timedelta(days=7)
        )

        assert created is True
        assert isinstance(event_id, int)

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
                id=1,
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

        assert (event_id, created) == (1, False)
