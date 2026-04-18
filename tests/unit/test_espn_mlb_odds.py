"""Tests for the ESPN MLB featured-totals helper."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from odds_lambda.espn_mlb_odds import (
    MlbGameTotal,
    distinct_market_keys,
    get_mlb_main_totals,
    group_match_links_by_line,
    line_to_market_key,
)


class TestLineToMarketKey:
    def test_half_line(self) -> None:
        assert line_to_market_key(8.5) == "over_under_8_5"

    def test_whole_number(self) -> None:
        assert line_to_market_key(9.0) == "over_under_9_0"

    def test_low_line(self) -> None:
        assert line_to_market_key(6.5) == "over_under_6_5"

    def test_high_line(self) -> None:
        assert line_to_market_key(11.5) == "over_under_11_5"

    def test_rounds_up_to_half(self) -> None:
        # 8.3 is closer to 8.5 than 8.0
        assert line_to_market_key(8.3) == "over_under_8_5"

    def test_rounds_down_to_half(self) -> None:
        # 8.2 rounds to 8.0 (nearest 0.5)
        assert line_to_market_key(8.2) == "over_under_8_0"

    def test_rounds_whole(self) -> None:
        assert line_to_market_key(9.1) == "over_under_9_0"


class TestDistinctMarketKeys:
    def _total(self, line: float) -> MlbGameTotal:
        return MlbGameTotal(
            event_id=f"id_{line}",
            home_team="H",
            away_team="A",
            commence_time=datetime(2026, 4, 18, 20, 0, tzinfo=UTC),
            line=line,
        )

    def test_deduplicates(self) -> None:
        totals = [self._total(8.5), self._total(8.5), self._total(9.0)]
        assert distinct_market_keys(totals) == ["over_under_8_5", "over_under_9_0"]

    def test_sorted_numerically(self) -> None:
        totals = [self._total(9.5), self._total(7.0), self._total(8.5)]
        assert distinct_market_keys(totals) == [
            "over_under_7_0",
            "over_under_8_5",
            "over_under_9_5",
        ]

    def test_two_digit_lines_sort_by_number_not_string(self) -> None:
        """Regression: lexicographic sort would put 10_5 / 11_5 before 9_5."""
        totals = [self._total(11.5), self._total(9.5), self._total(10.5)]
        assert distinct_market_keys(totals) == [
            "over_under_9_5",
            "over_under_10_5",
            "over_under_11_5",
        ]

    def test_empty(self) -> None:
        assert distinct_market_keys([]) == []


def _make_scoreboard_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    return {"events": events}


def _make_event(
    *,
    event_id: str = "401814979",
    home: str = "New York Yankees",
    away: str = "Boston Red Sox",
    commence: str = "2026-04-18T23:05Z",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "date": commence,
        "competitions": [
            {
                "competitors": [
                    {"homeAway": "home", "team": {"displayName": home}},
                    {"homeAway": "away", "team": {"displayName": away}},
                ]
            }
        ],
    }


def _make_odds_payload(over_under: float | None) -> dict[str, Any]:
    items: list[dict[str, Any]] = [{}]
    if over_under is not None:
        items[0]["overUnder"] = over_under
    return {"items": items}


class _MockResponse:
    """Minimal stand-in for aiohttp.ClientResponse usable as an async context manager."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status = status

    async def __aenter__(self) -> _MockResponse:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
            )


class _ResponseMap:
    """Dispatch different responses based on requested URL substrings."""

    def __init__(self, routes: dict[str, dict[str, Any] | Exception]) -> None:
        self._routes = routes

    def get(self, url: str, **_: object) -> _MockResponse:
        for marker, payload in self._routes.items():
            if marker in url:
                if isinstance(payload, Exception):
                    raise payload
                return _MockResponse(payload)
        raise AssertionError(f"Unexpected URL in test: {url}")


class TestGetMlbMainTotals:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        routes = _ResponseMap(
            {
                "scoreboard": _make_scoreboard_payload([_make_event()]),
                "odds": _make_odds_payload(over_under=8.5),
            }
        )
        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = routes.get

        result = await get_mlb_main_totals(session=session)

        assert len(result) == 1
        assert result[0].line == 8.5
        assert result[0].home_team == "New York Yankees"
        assert result[0].away_team == "Boston Red Sox"
        assert result[0].event_id == "401814979"

    @pytest.mark.asyncio
    async def test_missing_over_under_drops_event(self) -> None:
        routes = _ResponseMap(
            {
                "scoreboard": _make_scoreboard_payload([_make_event()]),
                "odds": _make_odds_payload(over_under=None),
            }
        )
        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = routes.get

        result = await get_mlb_main_totals(session=session)
        assert result == []

    @pytest.mark.asyncio
    async def test_event_odds_error_drops_that_event_only(self) -> None:
        event_a = _make_event(event_id="A", home="A Home", away="A Away")
        event_b = _make_event(event_id="B", home="B Home", away="B Away")

        def _get(url: str, **_: object) -> _MockResponse:
            if "scoreboard" in url:
                return _MockResponse(_make_scoreboard_payload([event_a, event_b]))
            if "events/A/" in url:
                raise aiohttp.ClientError("boom")
            if "events/B/" in url:
                return _MockResponse(_make_odds_payload(over_under=9.0))
            raise AssertionError(url)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = _get

        result = await get_mlb_main_totals(session=session)
        assert len(result) == 1
        assert result[0].event_id == "B"
        assert result[0].line == 9.0

    @pytest.mark.asyncio
    async def test_scoreboard_failure_returns_empty(self) -> None:
        def _get(url: str, **_: object) -> _MockResponse:
            raise aiohttp.ClientError("scoreboard down")

        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = _get

        result = await get_mlb_main_totals(session=session)
        assert result == []

    @pytest.mark.asyncio
    async def test_malformed_event_missing_teams_dropped(self) -> None:
        bad_event = {
            "id": "401814979",
            "date": "2026-04-18T23:05Z",
            "competitions": [{"competitors": []}],  # no teams
        }
        routes = _ResponseMap(
            {
                "scoreboard": _make_scoreboard_payload([bad_event]),
                "odds": _make_odds_payload(over_under=8.5),
            }
        )
        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = routes.get

        result = await get_mlb_main_totals(session=session)
        assert result == []

    @pytest.mark.asyncio
    async def test_zero_over_under_drops_event(self) -> None:
        """ESPN sometimes returns overUnder=0.0 for pre-release games; filter it."""
        routes = _ResponseMap(
            {
                "scoreboard": _make_scoreboard_payload([_make_event()]),
                "odds": _make_odds_payload(over_under=0.0),
            }
        )
        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = routes.get

        result = await get_mlb_main_totals(session=session)
        assert result == []

    @pytest.mark.asyncio
    async def test_bool_over_under_drops_event(self) -> None:
        """bool is a subclass of int; exclude it from the numeric check."""
        routes = _ResponseMap(
            {
                "scoreboard": _make_scoreboard_payload([_make_event()]),
                "odds": {"items": [{"overUnder": True}]},
            }
        )
        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = routes.get

        result = await get_mlb_main_totals(session=session)
        assert result == []

    @pytest.mark.asyncio
    async def test_target_date_serialized_as_yyyymmdd(self) -> None:
        """target_date must hit ESPN as dates=YYYYMMDD on the scoreboard call."""
        from datetime import date

        captured: dict[str, Any] = {}

        def _get(url: str, **kwargs: Any) -> _MockResponse:
            if "scoreboard" in url:
                captured["params"] = kwargs.get("params")
                return _MockResponse(_make_scoreboard_payload([]))
            raise AssertionError(url)

        session = MagicMock(spec=aiohttp.ClientSession)
        session.get = _get

        await get_mlb_main_totals(target_date=date(2026, 4, 18), session=session)

        assert captured["params"] == {"dates": "20260418"}

    @pytest.mark.asyncio
    async def test_creates_and_closes_session_when_not_provided(self) -> None:
        """When no session is supplied, the helper manages its own lifecycle."""
        with patch("odds_lambda.espn_mlb_odds.aiohttp.ClientSession") as mock_cls:
            instance = MagicMock()
            instance.close = AsyncMock()
            instance.get = _ResponseMap({"scoreboard": _make_scoreboard_payload([])}).get
            mock_cls.return_value = instance

            result = await get_mlb_main_totals()

            assert result == []
            instance.close.assert_awaited_once()


class TestGroupMatchLinksByLine:
    def _total(self, home: str, away: str, line: float) -> MlbGameTotal:
        return MlbGameTotal(
            event_id=f"{home}-{away}",
            home_team=home,
            away_team=away,
            commence_time=datetime(2026, 4, 18, 20, 0, tzinfo=UTC),
            line=line,
        )

    def _raw(self, home: str, away: str, link: str) -> dict[str, Any]:
        return {"home_team": home, "away_team": away, "match_link": link}

    def test_groups_by_featured_line(self) -> None:
        totals = [
            self._total("New York Yankees", "Boston Red Sox", 8.5),
            self._total("Los Angeles Dodgers", "San Francisco Giants", 9.0),
            self._total("Chicago Cubs", "Milwaukee Brewers", 8.5),
        ]
        raw = [
            self._raw("New York Yankees", "Boston Red Sox", "url-nyy-bos"),
            self._raw("Los Angeles Dodgers", "San Francisco Giants", "url-lad-sf"),
            self._raw("Chicago Cubs", "Milwaukee Brewers", "url-cubs-mil"),
        ]
        groups = group_match_links_by_line(totals, raw)
        assert groups == {
            "over_under_8_5": ["url-nyy-bos", "url-cubs-mil"],
            "over_under_9_0": ["url-lad-sf"],
        }

    def test_case_insensitive_team_match(self) -> None:
        """OddsPortal and ESPN casing may differ slightly."""
        totals = [self._total("New York Yankees", "Boston Red Sox", 8.5)]
        raw = [self._raw("new york yankees", "boston red sox", "url-nyy-bos")]
        groups = group_match_links_by_line(totals, raw)
        assert groups == {"over_under_8_5": ["url-nyy-bos"]}

    def test_drops_espn_games_without_oddsportal_match(self) -> None:
        """Games on ESPN's schedule but missing from OddsPortal are skipped."""
        totals = [
            self._total("New York Yankees", "Boston Red Sox", 8.5),
            self._total("Tampa Bay Rays", "Baltimore Orioles", 9.0),
        ]
        raw = [self._raw("New York Yankees", "Boston Red Sox", "url-nyy-bos")]
        groups = group_match_links_by_line(totals, raw)
        assert groups == {"over_under_8_5": ["url-nyy-bos"]}

    def test_drops_oddsportal_games_without_espn_line(self) -> None:
        """Extra OddsPortal games (e.g. tomorrow's early slate) are skipped."""
        totals = [self._total("New York Yankees", "Boston Red Sox", 8.5)]
        raw = [
            self._raw("New York Yankees", "Boston Red Sox", "url-nyy-bos"),
            self._raw("Houston Astros", "Texas Rangers", "url-hou-tex"),
        ]
        groups = group_match_links_by_line(totals, raw)
        assert groups == {"over_under_8_5": ["url-nyy-bos"]}

    def test_empty_inputs(self) -> None:
        assert group_match_links_by_line([], []) == {}
        totals = [self._total("A", "B", 8.5)]
        assert group_match_links_by_line(totals, []) == {}
        raw = [self._raw("A", "B", "url")]
        assert group_match_links_by_line([], raw) == {}

    def test_skips_raw_matches_with_missing_fields(self) -> None:
        totals = [self._total("A", "B", 8.5)]
        raw = [
            {"home_team": "A", "away_team": "B", "match_link": ""},
            {"home_team": "", "away_team": "B", "match_link": "url"},
        ]
        assert group_match_links_by_line(totals, raw) == {}
