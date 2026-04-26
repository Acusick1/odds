"""Unit tests for MlbStatsFetcher and the dates_for_window helper."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import httpx
import pytest
from odds_lambda.mlb_stats_fetcher import MlbStatsFetcher, dates_for_window


class TestDatesForWindow:
    def test_zero_lookahead_single_day(self) -> None:
        now = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
        assert dates_for_window(now, 0) == [date(2026, 4, 26)]

    def test_lookahead_within_same_utc_day(self) -> None:
        now = datetime(2026, 4, 26, 6, 0, tzinfo=UTC)
        # 6 hours later is still 2026-04-26 UTC.
        assert dates_for_window(now, 6) == [date(2026, 4, 26)]

    def test_lookahead_crosses_midnight(self) -> None:
        now = datetime(2026, 4, 26, 22, 0, tzinfo=UTC)
        # 6 hours later is 2026-04-27 04:00 UTC.
        assert dates_for_window(now, 6) == [date(2026, 4, 26), date(2026, 4, 27)]

    def test_three_day_lookahead(self) -> None:
        now = datetime(2026, 4, 26, 0, 0, tzinfo=UTC)
        assert dates_for_window(now, 24 * 3) == [
            date(2026, 4, 26),
            date(2026, 4, 27),
            date(2026, 4, 28),
            date(2026, 4, 29),
        ]

    def test_negative_lookahead_raises(self) -> None:
        now = datetime(2026, 4, 26, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError):
            dates_for_window(now, -1)


def _game_payload(
    *,
    game_pk: int,
    game_date: str,
    home_team: str,
    away_team: str,
    home_pitcher: dict[str, Any] | None = None,
    away_pitcher: dict[str, Any] | None = None,
    game_type: str = "R",
) -> dict[str, Any]:
    home_team_block: dict[str, Any] = {"team": {"name": home_team}}
    away_team_block: dict[str, Any] = {"team": {"name": away_team}}
    if home_pitcher is not None:
        home_team_block["probablePitcher"] = home_pitcher
    if away_pitcher is not None:
        away_team_block["probablePitcher"] = away_pitcher
    return {
        "gamePk": game_pk,
        "gameDate": game_date,
        "gameType": game_type,
        "teams": {"home": home_team_block, "away": away_team_block},
    }


def _schedule_payload(games_by_date: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "dates": [{"date": iso_date, "games": games} for iso_date, games in games_by_date.items()]
    }


class TestMlbStatsFetcher:
    """End-to-end tests for MlbStatsFetcher with a mock httpx client."""

    @pytest.fixture
    def request_log(self) -> list[httpx.Request]:
        return []

    @pytest.fixture
    def transport_responses(self) -> dict[str, dict[str, Any]]:
        """``date param value`` -> JSON response body."""
        return {}

    @pytest.fixture
    def mock_client(
        self,
        request_log: list[httpx.Request],
        transport_responses: dict[str, dict[str, Any]],
    ) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            request_log.append(request)
            params = dict(request.url.params)
            if request.url.path != "/api/v1/schedule":
                return httpx.Response(404, json={"error": "unexpected path"})
            d = params.get("date")
            payload = transport_responses.get(d, {"dates": []})
            return httpx.Response(200, json=payload)

        transport = httpx.MockTransport(handler)
        return httpx.AsyncClient(transport=transport, base_url="https://statsapi.mlb.com")

    @pytest.mark.asyncio
    async def test_full_payload_both_pitchers(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        transport_responses["2026-04-26"] = _schedule_payload(
            {
                "2026-04-26": [
                    _game_payload(
                        game_pk=777001,
                        game_date="2026-04-26T23:05:00Z",
                        home_team="Boston Red Sox",
                        away_team="New York Yankees",
                        home_pitcher={"id": 1001, "fullName": "Sample Lefty"},
                        away_pitcher={"id": 1002, "fullName": "Sample Righty"},
                    )
                ]
            }
        )

        async with MlbStatsFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_dates([date(2026, 4, 26)])

        assert len(records) == 1
        record = records[0]
        assert record.game_pk == 777001
        assert record.commence_time == datetime(2026, 4, 26, 23, 5, tzinfo=UTC)
        assert record.home_team == "Boston Red Sox"
        assert record.away_team == "New York Yankees"
        assert record.home_pitcher_id == 1001
        assert record.home_pitcher_name == "Sample Lefty"
        assert record.away_pitcher_id == 1002
        assert record.away_pitcher_name == "Sample Righty"
        assert record.game_type == "R"

    @pytest.mark.asyncio
    async def test_missing_probable_pitcher_emits_null_row(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        # Both sides missing ``probablePitcher`` — still emit a row.
        transport_responses["2026-04-27"] = _schedule_payload(
            {
                "2026-04-27": [
                    _game_payload(
                        game_pk=777002,
                        game_date="2026-04-27T23:05:00Z",
                        home_team="Tampa Bay Rays",
                        away_team="Toronto Blue Jays",
                        home_pitcher=None,
                        away_pitcher=None,
                    )
                ]
            }
        )

        async with MlbStatsFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_dates([date(2026, 4, 27)])

        assert len(records) == 1
        record = records[0]
        assert record.game_pk == 777002
        assert record.home_pitcher_name is None
        assert record.home_pitcher_id is None
        assert record.away_pitcher_name is None
        assert record.away_pitcher_id is None

    @pytest.mark.asyncio
    async def test_only_home_announced_away_null(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        transport_responses["2026-04-28"] = _schedule_payload(
            {
                "2026-04-28": [
                    _game_payload(
                        game_pk=777003,
                        game_date="2026-04-28T19:10:00Z",
                        home_team="Atlanta Braves",
                        away_team="Philadelphia Phillies",
                        home_pitcher={"id": 2001, "fullName": "Home Ace"},
                        away_pitcher=None,
                    )
                ]
            }
        )

        async with MlbStatsFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_dates([date(2026, 4, 28)])

        assert len(records) == 1
        record = records[0]
        assert record.home_pitcher_id == 2001
        assert record.home_pitcher_name == "Home Ace"
        assert record.away_pitcher_id is None
        assert record.away_pitcher_name is None

    @pytest.mark.asyncio
    async def test_iterates_each_date_in_window(
        self,
        mock_client: httpx.AsyncClient,
        request_log: list[httpx.Request],
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        # Three sequential dates, each with one game on that date.
        for d, pk, iso in [
            ("2026-04-26", 1, "2026-04-26T23:05:00Z"),
            ("2026-04-27", 2, "2026-04-27T23:05:00Z"),
            ("2026-04-28", 3, "2026-04-28T23:05:00Z"),
        ]:
            transport_responses[d] = _schedule_payload(
                {
                    d: [
                        _game_payload(
                            game_pk=pk,
                            game_date=iso,
                            home_team=f"Home {pk}",
                            away_team=f"Away {pk}",
                        )
                    ]
                }
            )

        target_dates = [date(2026, 4, 26), date(2026, 4, 27), date(2026, 4, 28)]
        async with MlbStatsFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_dates(target_dates)

        assert len(records) == 3
        # One HTTP request per date.
        observed_dates = [dict(r.url.params).get("date") for r in request_log]
        assert observed_dates == ["2026-04-26", "2026-04-27", "2026-04-28"]
        # All requests carry sportId=1 and the probablePitcher hydration.
        for r in request_log:
            params = dict(r.url.params)
            assert params["sportId"] == "1"
            assert params["hydrate"] == "probablePitcher"

    @pytest.mark.asyncio
    async def test_shared_fetched_at_across_dates(
        self,
        mock_client: httpx.AsyncClient,
        transport_responses: dict[str, dict[str, Any]],
    ) -> None:
        for d, pk, iso in [
            ("2026-04-26", 11, "2026-04-26T23:05:00Z"),
            ("2026-04-27", 12, "2026-04-27T23:05:00Z"),
        ]:
            transport_responses[d] = _schedule_payload(
                {
                    d: [
                        _game_payload(
                            game_pk=pk,
                            game_date=iso,
                            home_team=f"Home {pk}",
                            away_team=f"Away {pk}",
                        )
                    ]
                }
            )

        pinned = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
        async with MlbStatsFetcher(client=mock_client, request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_dates(
                [date(2026, 4, 26), date(2026, 4, 27)],
                fetched_at=pinned,
            )

        assert {r.fetched_at for r in records} == {pinned}

    @pytest.mark.asyncio
    async def test_empty_dates_returns_empty(self) -> None:
        async with MlbStatsFetcher(request_delay_seconds=0.0) as fetcher:
            records = await fetcher.fetch_dates([])
        assert records == []
