"""Tests for fetch_oddsportal job utilities."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest
from odds_lambda.jobs.fetch_oddsportal import (
    LEAGUE_SPECS,
    LeagueScrape,
    LeagueSpec,
    ingest_league,
    scrape_league,
)
from odds_lambda.oddsportal_adapter import MatchOdds
from odds_lambda.oddsportal_client import (
    BookmakerContext,
    BookmakerPrices,
    ListingEvent,
    OddsPortalError,
)
from odds_lambda.scheduling.helpers import apply_overnight_skip


def _event(event_id: str, start: datetime) -> ListingEvent:
    return ListingEvent(
        event_id=event_id,
        sport_id=1,
        home_team=f"Home {event_id}",
        away_team=f"Away {event_id}",
        start_time=start,
        url=f"https://www.oddsportal.com/football/h2h/a/b/#{event_id}",
        tournament_name="Premier League",
        venue="",
    )


class FakeOddsPortalClient:
    """Stands in for OddsPortalClient: scripted listing and per-event feed results."""

    def __init__(self, events: list[ListingEvent], failing: set[tuple[str, str]]) -> None:
        self.events = events
        self.failing = failing
        self.context_fetches = 0

    async def __aenter__(self) -> FakeOddsPortalClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def fetch_listing(self, path: str) -> list[ListingEvent]:
        return self.events

    async def fetch_bookmaker_context(self) -> BookmakerContext:
        self.context_fetches += 1
        return BookmakerContext("h", "GB", {"16": "bet365"})

    async def fetch_market(
        self, event: ListingEvent, market: str, context: BookmakerContext
    ) -> list[BookmakerPrices]:
        if (event.event_id, market) in self.failing:
            raise OddsPortalError("boom")
        prices = (2.0, 3.0, 4.0) if market == "1x2" else (2.0, 1.8)
        return [BookmakerPrices("16", "bet365", prices, None, None)]


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(
        events: list[ListingEvent], failing: set[tuple[str, str]] | None = None
    ) -> FakeOddsPortalClient:
        client = FakeOddsPortalClient(events, failing or set())
        monkeypatch.setattr("odds_lambda.jobs.fetch_oddsportal.OddsPortalClient", lambda: client)
        return client

    return install


EPL = LEAGUE_SPECS[0]


class TestLeagueSpecs:
    def test_listing_paths(self) -> None:
        paths = {s.league: s.listing_path for s in LEAGUE_SPECS}

        assert paths == {
            "england-premier-league": "/football/england/premier-league/",
            "mlb": "/baseball/usa/mlb/",
        }


class TestScrapeLeague:
    async def test_converts_each_market_for_upcoming_events(self, fake_client: Any) -> None:
        future = datetime.now(UTC) + timedelta(days=1)
        fake_client([_event("a", future), _event("b", future)])

        scrape = await scrape_league(EPL)

        assert scrape.events_listed == 2
        assert [m for m, _ in scrape.matches] == ["1x2", "over_under_2_5"] * 2
        assert scrape.errors == []

    async def test_past_events_are_dropped_and_skip_context(self, fake_client: Any) -> None:
        client = fake_client([_event("a", datetime.now(UTC) - timedelta(hours=1))])

        scrape = await scrape_league(EPL)

        assert scrape.events_listed == 1
        assert scrape.matches == []
        assert client.context_fetches == 0

    async def test_market_failure_is_recorded_and_scrape_continues(self, fake_client: Any) -> None:
        future = datetime.now(UTC) + timedelta(days=1)
        fake_client([_event("a", future), _event("b", future)], failing={("a", "1x2")})

        scrape = await scrape_league(EPL)

        assert len(scrape.errors) == 1
        assert "Home a vs Away a (1x2)" in scrape.errors[0]
        assert len(scrape.matches) == 3


class TestIngestLeague:
    def _match(self, event_id: str) -> MatchOdds:
        now = datetime.now(UTC)
        return MatchOdds(
            home_team=f"Home {event_id}",
            away_team=f"Away {event_id}",
            match_date=now + timedelta(days=1),
            match_link=f"https://www.oddsportal.com/#{event_id}",
            scraped_date=now,
            raw_data={"bookmakers": [], "source": "oddsportal_live"},
            bookmaker_count=1,
        )

    async def test_dry_run_counts_distinct_events(self) -> None:
        scraped = LeagueScrape(
            matches=[("1x2", self._match("a")), ("over_under_2_5", self._match("a"))],
            events_listed=3,
            errors=["x"],
        )

        stats = await ingest_league(EPL, scraped=scraped, dry_run=True)

        assert stats.matches_scraped == 1
        assert stats.matches_converted == 2
        assert stats.errors == ["x"]
        assert stats.snapshots_stored == 0

    async def test_empty_scrape_returns_zero_stats(self) -> None:
        stats = await ingest_league(EPL, scraped=LeagueScrape(events_listed=0))

        assert stats.matches_scraped == 0

    async def test_scrapes_when_not_given_data(self) -> None:
        spec = LeagueSpec(
            sport="football",
            league="x",
            listing_path="/x/",
            sport_key="soccer_x",
            sport_title="X",
        )
        with patch(
            "odds_lambda.jobs.fetch_oddsportal.scrape_league",
            return_value=LeagueScrape(matches=[("1x2", self._match("a"))]),
        ) as mock_scrape:
            stats = await ingest_league(spec, dry_run=True)

        mock_scrape.assert_awaited_once_with(spec)
        assert stats.matches_scraped == 1


class TestApplyOvernightSkip:
    def test_epl_defaults_during_day(self) -> None:
        dt = datetime(2026, 4, 13, 14, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == dt

    def test_epl_defaults_late_night(self) -> None:
        dt = datetime(2026, 4, 13, 23, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt)
        assert result == datetime(2026, 4, 14, 6, 0, tzinfo=UTC)

    def test_epl_defaults_early_morning(self) -> None:
        dt = datetime(2026, 4, 13, 3, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt)
        assert result == datetime(2026, 4, 13, 6, 0, tzinfo=UTC)

    def test_epl_defaults_boundary_start(self) -> None:
        dt = datetime(2026, 4, 13, 22, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt)
        assert result == datetime(2026, 4, 14, 6, 0, tzinfo=UTC)

    def test_epl_defaults_boundary_resume(self) -> None:
        dt = datetime(2026, 4, 13, 6, 0, tzinfo=UTC)
        assert apply_overnight_skip(dt) == dt

    def test_mlb_hours_during_games(self) -> None:
        dt = datetime(2026, 6, 15, 23, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt

    def test_mlb_hours_overnight(self) -> None:
        dt = datetime(2026, 6, 16, 7, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == datetime(2026, 6, 16, 14, 0, tzinfo=UTC)

    def test_mlb_hours_boundary_start(self) -> None:
        dt = datetime(2026, 6, 16, 5, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == datetime(2026, 6, 16, 14, 0, tzinfo=UTC)

    def test_mlb_hours_boundary_resume(self) -> None:
        dt = datetime(2026, 6, 16, 14, 0, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt

    def test_mlb_hours_before_overnight(self) -> None:
        dt = datetime(2026, 6, 16, 4, 30, tzinfo=UTC)
        result = apply_overnight_skip(dt, overnight_start_utc=5, overnight_resume_utc=14)
        assert result == dt
