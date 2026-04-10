"""Tests for team name normalization and Event field validator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from odds_core.models import Event, EventStatus
from odds_core.team import normalize_team_name


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
            # Case insensitivity (title-cased before lookup)
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
        ],
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_team_name(raw) == expected

    def test_unknown_team_passthrough(self) -> None:
        assert normalize_team_name("Some Random FC") == "Some Random Fc"


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


class TestFindOrCreateEventOnWriter:
    """Test that find_or_create_event lives on OddsWriter."""

    @pytest.mark.asyncio
    async def test_creates_event_when_none_exists(self, test_session) -> None:
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)

        event_id, created = await writer.find_or_create_event(
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
            sport_key="soccer_epl",
            sport_title="EPL",
        )

        assert created is True
        assert event_id.startswith("op_live_")
        assert "ARS" in event_id
        assert "CHE" in event_id

    @pytest.mark.asyncio
    async def test_finds_existing_event(self, test_session) -> None:
        from odds_lambda.storage.writers import OddsWriter

        # Pre-create an event
        event = Event(
            id="existing_123",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.SCHEDULED,
        )
        test_session.add(event)
        await test_session.flush()

        writer = OddsWriter(test_session)

        event_id, created = await writer.find_or_create_event(
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
            sport_key="soccer_epl",
            sport_title="EPL",
        )

        assert created is False
        assert event_id == "existing_123"

    @pytest.mark.asyncio
    async def test_idempotent_create(self, test_session) -> None:
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)
        match_date = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)

        event_id_1, created_1 = await writer.find_or_create_event(
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=match_date,
            sport_key="soccer_epl",
            sport_title="EPL",
        )
        await test_session.flush()

        event_id_2, created_2 = await writer.find_or_create_event(
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=match_date,
            sport_key="soccer_epl",
            sport_title="EPL",
        )

        assert event_id_1 == event_id_2
        assert created_1 is True
        assert created_2 is False
