"""Unit tests for EspnFixtureFetcher and the current_season helper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from odds_lambda.espn_fixture_fetcher import (
    LEAGUE_SLUGS,
    EspnFixtureFetcher,
    EspnUpcomingFetchError,
    _months_between,
    current_season,
    season_label,
)


class TestCurrentSeason:
    def test_august_flips_to_new_season(self) -> None:
        # 1 Aug 2025 → 2025-26 season.
        assert current_season(datetime(2025, 8, 1, tzinfo=UTC)) == 2025

    def test_july_still_previous_season(self) -> None:
        # 31 Jul 2025 → 2024-25 season (still).
        assert current_season(datetime(2025, 7, 31, tzinfo=UTC)) == 2024

    def test_january_previous_year(self) -> None:
        # 15 Jan 2026 → 2025-26 season (start year 2025).
        assert current_season(datetime(2026, 1, 15, tzinfo=UTC)) == 2025

    def test_december_current_year(self) -> None:
        # 15 Dec 2025 → 2025-26 season (start year 2025).
        assert current_season(datetime(2025, 12, 15, tzinfo=UTC)) == 2025

    def test_default_uses_now(self) -> None:
        # Just exercise the default path; value depends on clock.
        assert isinstance(current_season(), int)


class TestSeasonLabel:
    def test_known_season(self) -> None:
        assert season_label(2024) == "2024-25"
        assert season_label(2025) == "2025-26"

    def test_derived_future_season(self) -> None:
        # Seasons past the static table still get a sensible label.
        assert season_label(2099) == "2099-00"


def _team_list_payload(teams: list[tuple[str, str]]) -> dict[str, Any]:
    """Build a minimal ESPN `/teams` response with the given (id, name) pairs."""
    return {
        "sports": [
            {
                "leagues": [
                    {
                        "teams": [
                            {"team": {"id": team_id, "displayName": name}}
                            for team_id, name in teams
                        ]
                    }
                ]
            }
        ]
    }


def _schedule_payload(
    *,
    team_id: str,
    opponent_id: str,
    team_name: str,
    opponent_name: str,
    date: str,
    home_away: str = "home",
    score_team: str | None = None,
    score_opponent: str | None = None,
    status: str = "Scheduled",
    state: str = "pre",
    season_name: str = "Regular Season",
) -> dict[str, Any]:
    """Build a minimal ESPN `/teams/{id}/schedule` response containing one event."""
    opponent_home_away = "away" if home_away == "home" else "home"
    team_competitor: dict[str, Any] = {
        "team": {"id": team_id, "displayName": team_name},
        "homeAway": home_away,
    }
    opponent_competitor: dict[str, Any] = {
        "team": {"id": opponent_id, "displayName": opponent_name},
        "homeAway": opponent_home_away,
    }
    if score_team is not None:
        team_competitor["score"] = {"displayValue": score_team}
    if score_opponent is not None:
        opponent_competitor["score"] = {"displayValue": score_opponent}

    return {
        "events": [
            {
                "date": date,
                "seasonType": {"name": season_name},
                "competitions": [
                    {
                        "competitors": [team_competitor, opponent_competitor],
                        "status": {"type": {"description": status, "state": state}},
                    }
                ],
            }
        ]
    }


class TestEspnFixtureFetcher:
    """End-to-end tests for EspnFixtureFetcher with a mock httpx client."""

    @pytest.fixture
    def transport_responses(self) -> dict[str, dict[str, Any]]:
        """URL prefix -> JSON response body. Matched by URL `startswith` on handler."""
        return {}

    @pytest.fixture
    def mock_client(self, transport_responses: dict[str, dict[str, Any]]) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            # Match by longest prefix first to avoid `/teams` shadowing
            # `/teams/{id}/schedule`.
            best_match: str | None = None
            for prefix in transport_responses:
                if url.startswith(prefix) and (best_match is None or len(prefix) > len(best_match)):
                    best_match = prefix
            if best_match is not None:
                return httpx.Response(200, json=transport_responses[best_match])
            # No match — return 404 so tests surface unexpected URLs loudly.
            return httpx.Response(404, json={"error": "unexpected", "url": url})

        transport = httpx.MockTransport(handler)
        return httpx.AsyncClient(transport=transport)

    @pytest.mark.asyncio
    async def test_fetch_season_produces_normalised_records(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        # Arrange: 2 EPL teams, each with one PL fixture; other competitions empty.
        transport_responses["http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams"] = (
            _team_list_payload([("1", "Arsenal"), ("2", "Wolverhampton Wanderers")])
        )

        # Premier League schedules: one match between Arsenal and Wolves.
        # Each team's feed returns the same match mirrored (home_away swapped),
        # which should be deduplicated by the fetcher.
        transport_responses[
            "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/1/schedule"
        ] = _schedule_payload(
            team_id="1",
            opponent_id="2",
            team_name="Arsenal",
            opponent_name="Wolverhampton Wanderers",
            date="2025-08-17T15:00:00Z",
            home_away="home",
            score_team="2",
            score_opponent="1",
            status="Full Time",
            state="post",
        )
        transport_responses[
            "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/2/schedule"
        ] = _schedule_payload(
            team_id="2",
            opponent_id="1",
            team_name="Wolverhampton Wanderers",
            opponent_name="Arsenal",
            date="2025-08-17T15:00:00Z",
            home_away="away",
            score_team="1",
            score_opponent="2",
            status="Full Time",
            state="post",
        )

        # All other competition endpoints: empty events list.
        for slug in (
            "eng.fa",
            "eng.league_cup",
            "uefa.champions",
            "uefa.europa",
            "uefa.europa.conf",
        ):
            for team_id in ("1", "2"):
                transport_responses[
                    f"http://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/teams/{team_id}/schedule"
                ] = {"events": []}

        # Act
        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_season(2025)

        # Assert: two distinct rows (one per team) for the same match.
        assert len(records) == 2

        arsenal_row = next(r for r in records if r.team == "Arsenal")
        wolves_row = next(r for r in records if r.team == "Wolves")

        # Normalisation worked (Wolverhampton Wanderers → Wolves)
        assert arsenal_row.opponent == "Wolves"
        assert wolves_row.opponent == "Arsenal"

        # Home/away is team-relative
        assert arsenal_row.home_away == "home"
        assert wolves_row.home_away == "away"

        # Scores team-relative
        assert arsenal_row.score_team == "2"
        assert arsenal_row.score_opponent == "1"
        assert wolves_row.score_team == "1"
        assert wolves_row.score_opponent == "2"

        # Status + competition + state propagated
        assert arsenal_row.status == "Full Time"
        assert arsenal_row.state == "post"
        assert arsenal_row.competition == "Premier League"

        # Date parsed as UTC-aware datetime
        assert arsenal_row.date == datetime(2025, 8, 17, 15, 0, tzinfo=UTC)

        # Season label uses the display form
        assert arsenal_row.season == "2025-26"

    @pytest.mark.asyncio
    async def test_fetch_season_swallows_single_competition_failure(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        # Only one team; FA Cup endpoint returns 404 (unmatched). Other
        # competitions should still fetch normally and the job must not abort.
        transport_responses["http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams"] = (
            _team_list_payload([("1", "Arsenal")])
        )

        # PL endpoint works
        transport_responses[
            "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/teams/1/schedule"
        ] = _schedule_payload(
            team_id="1",
            opponent_id="99",
            team_name="Arsenal",
            opponent_name="Chelsea",
            date="2025-09-01T15:00:00Z",
            status="Full Time",
            state="post",
            score_team="3",
            score_opponent="0",
        )
        # Remaining competitions have no schedule registered → handler returns 404.

        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_season(2025)

        # PL fixture still ingested even though the other endpoints 404'd.
        assert len(records) == 1
        assert records[0].team == "Arsenal"
        assert records[0].opponent == "Chelsea"


def _scoreboard_payload(
    *,
    home_id: str,
    home_name: str,
    away_id: str,
    away_name: str,
    date: str,
    state: str = "pre",
    status: str = "Scheduled",
    home_score: str | None = None,
    away_score: str | None = None,
    season_name: str = "Regular Season",
) -> dict[str, Any]:
    """Build a minimal ESPN ``/scoreboard`` response with one event."""
    home_comp: dict[str, Any] = {
        "homeAway": "home",
        "team": {"id": home_id, "displayName": home_name},
    }
    away_comp: dict[str, Any] = {
        "homeAway": "away",
        "team": {"id": away_id, "displayName": away_name},
    }
    if home_score is not None:
        home_comp["score"] = {"displayValue": home_score}
    if away_score is not None:
        away_comp["score"] = {"displayValue": away_score}

    return {
        "events": [
            {
                "date": date,
                "seasonType": {"name": season_name},
                "competitions": [
                    {
                        "competitors": [home_comp, away_comp],
                        "status": {"type": {"state": state, "description": status}},
                    }
                ],
            }
        ]
    }


class TestMonthsBetween:
    def test_single_month(self) -> None:
        start = datetime(2026, 10, 20, tzinfo=UTC)
        assert _months_between(start, start + timedelta(days=5)) == [(2026, 10)]

    def test_crosses_month_boundary(self) -> None:
        start = datetime(2026, 10, 20, tzinfo=UTC)
        assert _months_between(start, start + timedelta(days=30)) == [(2026, 10), (2026, 11)]

    def test_crosses_year_boundary(self) -> None:
        start = datetime(2026, 12, 20, tzinfo=UTC)
        assert _months_between(start, start + timedelta(days=30)) == [(2026, 12), (2027, 1)]


class TestFetchUpcoming:
    """Unit tests for fetch_upcoming using the month-keyed scoreboard endpoint."""

    @pytest.fixture
    def requested_urls(self) -> list[str]:
        return []

    @pytest.fixture
    def failing_leagues(self) -> set[str]:
        return set()

    @pytest.fixture
    def transport_responses(self) -> dict[str, dict[str, Any]]:
        return {}

    @pytest.fixture
    def mock_client(
        self,
        transport_responses: dict[str, dict[str, Any]],
        requested_urls: list[str],
        failing_leagues: set[str],
    ) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            requested_urls.append(url)
            if any(f"/{league}/" in url for league in failing_leagues):
                return httpx.Response(400, json={"error": "bad dates"})
            for prefix, payload in transport_responses.items():
                if url.startswith(prefix):
                    return httpx.Response(200, json=payload)
            return httpx.Response(200, json={"events": []})

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @pytest.fixture(autouse=True)
    def no_retry_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("odds_lambda.espn_fixture_fetcher._RETRY_BACKOFF_SECONDS", 0.0)

    @staticmethod
    def _kickoff(days: int) -> str:
        return (datetime.now(UTC) + timedelta(days=days)).strftime("%Y-%m-%dT%H:%MZ")

    @pytest.mark.asyncio
    async def test_scoreboard_event_produces_two_records(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        """One scoreboard event becomes two anchored records (home + away)."""
        kickoff = self._kickoff(2)
        transport_responses[
            "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
        ] = _scoreboard_payload(
            home_id="1",
            home_name="Arsenal",
            away_id="2",
            away_name="Wolverhampton Wanderers",
            date=kickoff,
            state="pre",
            status="Scheduled",
        )

        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_upcoming(days_ahead=7)

        assert len(records) == 2

        home_row = next(r for r in records if r.team == "Arsenal")
        away_row = next(r for r in records if r.team == "Wolves")

        assert home_row.opponent == "Wolves"
        assert home_row.home_away == "home"
        assert home_row.state == "pre"
        assert home_row.status == "Scheduled"
        assert home_row.score_team == ""
        assert home_row.competition == "Premier League"
        assert home_row.date == datetime.fromisoformat(kickoff.replace("Z", "+00:00"))

        assert away_row.opponent == "Arsenal"
        assert away_row.home_away == "away"
        assert away_row.state == "pre"
        assert away_row.score_team == ""

    @pytest.mark.asyncio
    async def test_requests_months_not_ranges(
        self,
        mock_client: httpx.AsyncClient,
        requested_urls: list[str],
    ) -> None:
        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            await fetcher.fetch_upcoming(days_ahead=30)

        dates = {url.split("dates=")[1].split("&")[0] for url in requested_urls}
        assert dates
        assert all(len(d) == 6 and d.isdigit() for d in dates)
        assert all("limit=" in url for url in requested_urls)

    @pytest.mark.asyncio
    async def test_events_outside_window_are_dropped(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        transport_responses[
            "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
        ] = _scoreboard_payload(
            home_id="1",
            home_name="Arsenal",
            away_id="2",
            away_name="Chelsea",
            date=self._kickoff(20),
        )

        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_upcoming(days_ahead=7)

        assert records == []

    @pytest.mark.asyncio
    async def test_empty_scoreboards_return_no_records(
        self,
        mock_client: httpx.AsyncClient,
    ) -> None:
        """All-empty scoreboards produce an empty result without raising."""
        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_upcoming(days_ahead=7)

        assert records == []

    @pytest.mark.asyncio
    async def test_single_league_failure_is_skipped(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
        failing_leagues: set[str],
    ) -> None:
        failing_leagues.add("eng.fa")
        transport_responses[
            "http://site.api.espn.com/apis/site/v2/sports/soccer/eng.1/scoreboard"
        ] = _scoreboard_payload(
            home_id="1",
            home_name="Arsenal",
            away_id="2",
            away_name="Chelsea",
            date=self._kickoff(2),
        )

        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_upcoming(days_ahead=7)

        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_all_leagues_failing_raises(
        self,
        mock_client: httpx.AsyncClient,
        failing_leagues: set[str],
    ) -> None:
        failing_leagues.update(LEAGUE_SLUGS)

        async with EspnFixtureFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            with pytest.raises(EspnUpcomingFetchError, match="every competition"):
                await fetcher.fetch_upcoming(days_ahead=7)
