"""Soccer three-way markets are stored under the 1x2 key."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock

import pytest
from odds_core.team import normalize_team_name
from odds_core.utils import relabel_three_way_markets
from odds_lambda.data_fetcher import TheOddsAPIClient

MIGRATION_PATH = (
    Path(__file__).parents[2]
    / "migrations"
    / "versions"
    / "a513c0de3way_relabel_soccer_odds_api_three_way_to_1x2.py"
)


def three_way_event(sport_key: str = "soccer_epl") -> dict[str, Any]:
    return {
        "id": "evt1",
        "sport_key": sport_key,
        "sport_title": "EPL",
        "commence_time": "2026-10-20T19:00:00Z",
        "home_team": "Newcastle",
        "away_team": "Tottenham",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": "2026-10-18T10:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Newcastle United", "price": -120},
                            {"name": "Tottenham Hotspur", "price": 300},
                            {"name": "Draw", "price": 250},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 2.5},
                            {"name": "Under", "price": -110, "point": 2.5},
                        ],
                    },
                ],
            }
        ],
    }


def market_keys(raw_event: dict[str, Any]) -> list[str]:
    return [m["key"] for b in raw_event["bookmakers"] for m in b.get("markets", [])]


class TestRelabelThreeWayMarkets:
    def test_soccer_three_way_becomes_1x2(self) -> None:
        result = relabel_three_way_markets(three_way_event(), "soccer_epl")
        assert market_keys(result) == ["1x2", "totals"]

    def test_team_names_become_canonical(self) -> None:
        result = relabel_three_way_markets(three_way_event(), "soccer_epl")
        names = [o["name"] for o in result["bookmakers"][0]["markets"][0]["outcomes"]]
        assert names == ["Newcastle", "Tottenham", "Draw"]

    def test_input_is_not_mutated(self) -> None:
        event = three_way_event()
        relabel_three_way_markets(event, "soccer_epl")
        assert event == three_way_event()

    def test_is_idempotent(self) -> None:
        once = relabel_three_way_markets(three_way_event(), "soccer_epl")
        twice = relabel_three_way_markets(once, "soccer_epl")
        assert market_keys(twice) == ["1x2", "totals"]

    @pytest.mark.parametrize("sport_key", ["baseball_mlb", "basketball_nba"])
    def test_two_way_sports_keep_h2h(self, sport_key: str) -> None:
        event = three_way_event(sport_key)
        event["bookmakers"][0]["markets"][0]["outcomes"].pop()
        assert market_keys(relabel_three_way_markets(event, sport_key)) == ["h2h", "totals"]

    def test_soccer_market_without_draw_keeps_h2h(self) -> None:
        event = three_way_event()
        event["bookmakers"][0]["markets"][0]["outcomes"].pop()
        assert market_keys(relabel_three_way_markets(event, "soccer_epl")) == ["h2h", "totals"]


class TestClientRelabelsAtIngest:
    @pytest.mark.asyncio
    async def test_get_odds_returns_1x2_for_soccer(self) -> None:
        client = TheOddsAPIClient(api_key="k", base_url="http://example.invalid")
        client._make_request = AsyncMock(return_value=([three_way_event()], 1))  # type: ignore[method-assign]
        response = await client.get_odds(sport="soccer_epl")
        assert market_keys(response.raw_events_data[0]) == ["1x2", "totals"]

    @pytest.mark.asyncio
    async def test_get_historical_odds_returns_1x2_for_soccer(self) -> None:
        client = TheOddsAPIClient(api_key="k", base_url="http://example.invalid")
        client._make_request = AsyncMock(  # type: ignore[method-assign]
            return_value=({"data": [three_way_event()]}, 1)
        )
        response = await client.get_historical_odds(sport="soccer_epl", date="2026-10-18T00:00:00Z")
        assert market_keys(response.raw_events_data[0]) == ["1x2", "totals"]


def load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("relabel_three_way_migration", MIGRATION_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRelabelMigrationNames:
    def test_name_map_matches_normalize_team_name(self) -> None:
        for raw, canonical in json.loads(load_migration().TEAM_NAMES).items():
            assert normalize_team_name(raw) == canonical
