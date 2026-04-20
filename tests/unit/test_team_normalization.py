"""Tests for team name normalization and Event field validator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

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


class TestFindOrCreateEventNormalization:
    """Test that find_or_create_event normalizes inputs before querying."""

    @pytest.mark.asyncio
    async def test_finds_existing_event_with_alias_input(self, test_session) -> None:
        """Query with 'Wolverhampton Wanderers' matches DB row with 'Wolves'."""
        from odds_lambda.storage.writers import OddsWriter

        event = Event(
            id="existing_wolves",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=datetime(2026, 4, 15, 15, 0, tzinfo=UTC),
            home_team="Wolves",
            away_team="Arsenal",
            status=EventStatus.SCHEDULED,
        )
        test_session.add(event)
        await test_session.flush()

        writer = OddsWriter(test_session)

        event_id, created = await writer.find_or_create_event(
            home_team="Wolverhampton Wanderers",
            away_team="Arsenal",
            match_date=datetime(2026, 4, 15, 16, 0, tzinfo=UTC),
            sport_key="soccer_epl",
            sport_title="EPL",
        )

        assert created is False
        assert event_id == "existing_wolves"


class TestFindOrCreateEventOnWriter:
    """Test that find_or_create_event lives on OddsWriter."""

    @pytest.mark.asyncio
    async def test_creates_event_when_none_exists(self, test_session) -> None:
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)

        event_id, created = await writer.find_or_create_event(
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=datetime(2030, 4, 15, 15, 0, tzinfo=UTC),
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
        match_date = datetime(2030, 4, 15, 15, 0, tzinfo=UTC)

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


class TestFindOrCreateEventMatchWindow:
    """Regression tests for the ±2h team+date match window.

    The prior ±24h window silently merged back-to-back same-matchup games
    (MLB series pattern) and same-day doubleheaders into a single event row.
    """

    @pytest.mark.asyncio
    async def test_same_teams_consecutive_days_are_distinct_events(self, test_session) -> None:
        """Team A vs B on day N and day N+1 must create two event rows."""
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)

        day_one = datetime(2026, 6, 10, 23, 10, tzinfo=UTC)
        day_two = datetime(2026, 6, 11, 23, 10, tzinfo=UTC)

        event_id_1, created_1 = await writer.find_or_create_event(
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            match_date=day_one,
            sport_key="baseball_mlb",
            sport_title="MLB",
        )
        await test_session.flush()

        event_id_2, created_2 = await writer.find_or_create_event(
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            match_date=day_two,
            sport_key="baseball_mlb",
            sport_title="MLB",
        )

        assert created_1 is True
        assert created_2 is True
        assert event_id_1 != event_id_2

    @pytest.mark.asyncio
    async def test_same_day_doubleheader_games_are_distinct_events(self, test_session) -> None:
        """Team A vs B at 13:00 and 18:00 same day must create two event rows."""
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)

        game_one = datetime(2026, 7, 4, 13, 0, tzinfo=UTC)
        game_two = datetime(2026, 7, 4, 18, 0, tzinfo=UTC)

        event_id_1, created_1 = await writer.find_or_create_event(
            home_team="Chicago Cubs",
            away_team="St. Louis Cardinals",
            match_date=game_one,
            sport_key="baseball_mlb",
            sport_title="MLB",
        )
        await test_session.flush()

        event_id_2, created_2 = await writer.find_or_create_event(
            home_team="Chicago Cubs",
            away_team="St. Louis Cardinals",
            match_date=game_two,
            sport_key="baseball_mlb",
            sport_title="MLB",
        )

        assert created_1 is True
        assert created_2 is True
        assert event_id_1 != event_id_2

    @pytest.mark.asyncio
    async def test_re_scrape_same_match_is_idempotent(self, test_session) -> None:
        """Re-scraping the same match (match_date jittered within ±2h) reuses the row."""
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)

        first_scrape = datetime(2026, 6, 10, 23, 10, tzinfo=UTC)
        # Source occasionally re-reports commence_time within a small window.
        second_scrape = datetime(2026, 6, 10, 23, 45, tzinfo=UTC)

        event_id_1, created_1 = await writer.find_or_create_event(
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            match_date=first_scrape,
            sport_key="baseball_mlb",
            sport_title="MLB",
        )
        await test_session.flush()

        event_id_2, created_2 = await writer.find_or_create_event(
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            match_date=second_scrape,
            sport_key="baseball_mlb",
            sport_title="MLB",
        )

        assert created_1 is True
        assert created_2 is False
        assert event_id_1 == event_id_2


class TestFindOrCreateEventStaleLiveGuardrail:
    """Defense-in-depth: refuse to mint op_live_* events with past commence times."""

    @pytest.mark.asyncio
    async def test_accepts_live_event_within_window(self, test_session) -> None:
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)
        match_date = datetime.now(UTC) + timedelta(minutes=30)

        event_id, created = await writer.find_or_create_event(
            home_team="Arsenal",
            away_team="Chelsea",
            match_date=match_date,
            sport_key="soccer_epl",
            sport_title="EPL",
        )

        assert created is True
        assert event_id.startswith("op_live_")

    @pytest.mark.asyncio
    async def test_rejects_stale_live_event(self, test_session) -> None:
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)
        match_date = datetime.now(UTC) - timedelta(hours=2)

        with pytest.raises(ValueError, match="stale live-scrape event"):
            await writer.find_or_create_event(
                home_team="Arsenal",
                away_team="Chelsea",
                match_date=match_date,
                sport_key="soccer_epl",
                sport_title="EPL",
            )

    @pytest.mark.asyncio
    async def test_historical_ingestion_via_upsert_event_unaffected(self, test_session) -> None:
        """Historical backfill callers go through upsert_event with non-live ids."""
        from odds_lambda.storage.writers import OddsWriter

        writer = OddsWriter(test_session)
        match_date = datetime(2020, 1, 1, 15, 0, tzinfo=UTC)

        event = Event(
            id="fduk_2020_01_01_arsenal_chelsea",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=match_date,
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.FINAL,
        )

        result = await writer.upsert_event(event)

        assert result.id == "fduk_2020_01_01_arsenal_chelsea"
        assert result.commence_time == match_date

    @pytest.mark.asyncio
    async def test_finds_existing_stale_live_event_without_creating(self, test_session) -> None:
        """Guardrail only fires on create — looking up existing stale events still works."""
        from odds_lambda.storage.writers import OddsWriter

        match_date = datetime.now(UTC) - timedelta(hours=5)
        event = Event(
            id="op_live_ARS_CHE_2020-01-01T1500",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=match_date,
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
            match_date=match_date,
            sport_key="soccer_epl",
            sport_title="EPL",
        )

        assert created is False
        assert event_id == "op_live_ARS_CHE_2020-01-01T1500"
