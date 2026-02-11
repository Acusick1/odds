"""Unit tests for Polymarket event matching logic."""

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from odds_lambda.polymarket_matching import (
    NBA_ABBREV_MAP,
    TEAM_ALIASES,
    match_polymarket_event,
    normalize_team,
    parse_ticker,
)


class TestTeamAliases:
    """Tests for TEAM_ALIASES coverage and consistency."""

    def test_all_30_teams_covered(self):
        """All 30 NBA teams have alias entries."""
        assert len(TEAM_ALIASES) == 30

    def test_canonical_name_in_own_aliases(self):
        """Each canonical name appears in its own alias set."""
        for canonical, aliases in TEAM_ALIASES.items():
            assert canonical in aliases, f"{canonical} not in its own alias set"

    def test_no_alias_maps_to_two_teams(self):
        """No alias resolves to more than one canonical name."""
        seen: dict[str, str] = {}
        for canonical, aliases in TEAM_ALIASES.items():
            for alias in aliases:
                key = alias.lower()
                assert (
                    key not in seen
                ), f"Alias '{alias}' appears in both '{seen[key]}' and '{canonical}'"
                seen[key] = canonical

    def test_abbrev_map_covers_all_teams(self):
        """NBA_ABBREV_MAP covers all 30 teams."""
        assert len(NBA_ABBREV_MAP) == 30

    def test_abbrev_map_values_are_canonical(self):
        """Every value in NBA_ABBREV_MAP is a key in TEAM_ALIASES."""
        for abbrev, team in NBA_ABBREV_MAP.items():
            assert team in TEAM_ALIASES, f"Abbrev '{abbrev}' maps to unknown team '{team}'"


class TestNormalizeTeam:
    """Tests for normalize_team()."""

    def test_canonical_name_returns_self(self):
        assert normalize_team("Dallas Mavericks") == "Dallas Mavericks"
        assert normalize_team("Los Angeles Lakers") == "Los Angeles Lakers"
        assert normalize_team("Golden State Warriors") == "Golden State Warriors"

    def test_known_aliases(self):
        assert normalize_team("Mavericks") == "Dallas Mavericks"
        assert normalize_team("Mavs") == "Dallas Mavericks"
        assert normalize_team("Lakers") == "Los Angeles Lakers"
        assert normalize_team("LA Lakers") == "Los Angeles Lakers"
        assert normalize_team("Celtics") == "Boston Celtics"
        assert normalize_team("76ers") == "Philadelphia 76ers"
        assert normalize_team("Sixers") == "Philadelphia 76ers"
        assert normalize_team("Blazers") == "Portland Trail Blazers"
        assert normalize_team("Trail Blazers") == "Portland Trail Blazers"
        assert normalize_team("Cavs") == "Cleveland Cavaliers"
        assert normalize_team("Wolves") == "Minnesota Timberwolves"
        assert normalize_team("OKC") == "Oklahoma City Thunder"

    def test_case_insensitive(self):
        assert normalize_team("LAKERS") == "Los Angeles Lakers"
        assert normalize_team("lakers") == "Los Angeles Lakers"
        assert normalize_team("LaKeRs") == "Los Angeles Lakers"

    def test_leading_trailing_whitespace(self):
        assert normalize_team("  Lakers  ") == "Los Angeles Lakers"

    def test_unknown_name_returns_none(self):
        assert normalize_team("Unknown Team") is None
        assert normalize_team("") is None
        assert normalize_team("NBA") is None

    def test_all_canonical_names_normalize(self):
        """Every canonical team name normalizes to itself."""
        for canonical in TEAM_ALIASES:
            assert normalize_team(canonical) == canonical


class TestParseTicker:
    """Tests for parse_ticker()."""

    def test_valid_ticker(self):
        result = parse_ticker("nba-dal-mil-2026-01-25")
        assert result is not None
        away, home, game_date = result
        assert away == "dal"
        assert home == "mil"
        assert game_date == date(2026, 1, 25)

    def test_valid_ticker_uppercase(self):
        """Ticker parsing is case-insensitive."""
        result = parse_ticker("NBA-DAL-MIL-2026-01-25")
        assert result is not None
        away, home, _ = result
        assert away == "dal"
        assert home == "mil"

    def test_all_abbrevs_parseable(self):
        """Tickers using any known abbreviation pair parse correctly."""
        abbrevs = list(NBA_ABBREV_MAP.keys())
        ticker = f"nba-{abbrevs[0]}-{abbrevs[1]}-2026-02-01"
        result = parse_ticker(ticker)
        assert result is not None

    def test_invalid_format_returns_none(self):
        assert parse_ticker("not-a-ticker") is None
        assert parse_ticker("nba-dal-mil") is None
        assert parse_ticker("nba-dal-mil-2026-01") is None
        assert parse_ticker("soccer-dal-mil-2026-01-25") is None
        assert parse_ticker("") is None

    def test_invalid_date_returns_none(self):
        assert parse_ticker("nba-dal-mil-2026-13-01") is None  # month 13
        assert parse_ticker("nba-dal-mil-2026-02-30") is None  # Feb 30


class TestMatchPolymarketEvent:
    """Tests for match_polymarket_event()."""

    def _make_event(self, event_id: str, away: str, home: str, commence: datetime):
        event = MagicMock()
        event.id = event_id
        event.away_team = away
        event.home_team = home
        event.commence_time = commence
        return event

    def _make_session(self, events: list) -> AsyncMock:
        """Build a mock AsyncSession that returns given events for any scalars() query."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = events

        session = AsyncMock()
        session.execute = AsyncMock(return_value=mock_result)
        return session

    @pytest.mark.asyncio
    async def test_exact_match(self):
        pm_start = datetime(2026, 1, 25, 2, 0, 0, tzinfo=UTC)
        commence = datetime(2026, 1, 25, 2, 0, 0, tzinfo=UTC)

        event = self._make_event("evt_abc123", "Dallas Mavericks", "Milwaukee Bucks", commence)
        session = self._make_session([event])

        result = await match_polymarket_event(session, "nba-dal-mil-2026-01-25", pm_start)
        assert result == "evt_abc123"

    @pytest.mark.asyncio
    async def test_no_matching_events_returns_none(self):
        pm_start = datetime(2026, 1, 25, 2, 0, 0, tzinfo=UTC)
        session = self._make_session([])

        result = await match_polymarket_event(session, "nba-dal-mil-2026-01-25", pm_start)
        assert result is None

    @pytest.mark.asyncio
    async def test_ambiguous_match_returns_none(self):
        """Two candidates → no match to avoid false positives."""
        pm_start = datetime(2026, 1, 25, 2, 0, 0, tzinfo=UTC)
        commence = datetime(2026, 1, 25, 2, 0, 0, tzinfo=UTC)

        event1 = self._make_event("evt_1", "Dallas Mavericks", "Milwaukee Bucks", commence)
        event2 = self._make_event("evt_2", "Dallas Mavericks", "Milwaukee Bucks", commence)
        session = self._make_session([event1, event2])

        result = await match_polymarket_event(session, "nba-dal-mil-2026-01-25", pm_start)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_ticker_returns_none(self):
        session = self._make_session([])
        result = await match_polymarket_event(
            session, "not-a-ticker", datetime(2026, 1, 25, tzinfo=UTC)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_abbreviation_returns_none(self):
        """Ticker with unknown team abbreviation → None."""
        session = self._make_session([])
        result = await match_polymarket_event(
            session, "nba-zzz-mil-2026-01-25", datetime(2026, 1, 25, tzinfo=UTC)
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_called_once(self):
        """match_polymarket_event issues exactly one DB query for a valid ticker."""
        session = self._make_session([])
        await match_polymarket_event(
            session, "nba-dal-mil-2026-01-25", datetime(2026, 1, 25, 2, 0, tzinfo=UTC)
        )
        session.execute.assert_called_once()


class TestMatchPolymarketEventDB:
    """Integration-level tests using a real DB (pglite) to verify window filtering."""

    @pytest.mark.asyncio
    async def test_event_inside_window_matches(self, pglite_async_session):
        from odds_core.models import Event, EventStatus

        game_time = datetime(2026, 1, 25, 2, 0, 0, tzinfo=UTC)
        event = Event(
            id="test_evt_inside",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=game_time,
            home_team="Milwaukee Bucks",
            away_team="Dallas Mavericks",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(event)
        await pglite_async_session.commit()

        pm_start = game_time  # exact match
        result = await match_polymarket_event(
            pglite_async_session, "nba-dal-mil-2026-01-25", pm_start
        )
        assert result == "test_evt_inside"

    @pytest.mark.asyncio
    async def test_event_outside_window_not_matched(self, pglite_async_session):
        from odds_core.models import Event, EventStatus

        # Game is on Jan 27, but ticker says Jan 25 — outside ±24h window
        game_time = datetime(2026, 1, 27, 2, 0, 0, tzinfo=UTC)
        event = Event(
            id="test_evt_outside",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=game_time,
            home_team="Milwaukee Bucks",
            away_team="Dallas Mavericks",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(event)
        await pglite_async_session.commit()

        # Ticker date is Jan 25, game is Jan 27 — >24h apart, should not match
        result = await match_polymarket_event(pglite_async_session, "nba-dal-mil-2026-01-25")
        assert result is None

    @pytest.mark.asyncio
    async def test_wrong_teams_not_matched(self, pglite_async_session):
        from odds_core.models import Event, EventStatus

        game_time = datetime(2026, 1, 25, 2, 0, 0, tzinfo=UTC)
        # Store Lakers vs Celtics but query for dal vs mil
        event = Event(
            id="test_evt_wrong_teams",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=game_time,
            home_team="Boston Celtics",
            away_team="Los Angeles Lakers",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(event)
        await pglite_async_session.commit()

        result = await match_polymarket_event(
            pglite_async_session, "nba-dal-mil-2026-01-25", game_time
        )
        assert result is None
