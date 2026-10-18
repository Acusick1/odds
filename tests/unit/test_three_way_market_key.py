"""Soccer three-way markets are stored under the 1x2 key."""

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock

import pytest
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.team import normalize_team_name
from odds_core.utils import relabel_three_way_markets
from odds_lambda.data_fetcher import TheOddsAPIClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

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


class TestRelabelMigration:
    async def seed(self, session: AsyncSession) -> None:
        when = datetime(2026, 10, 18, 10, tzinfo=UTC)
        for event_id, sport_key, three_way, market_key in [
            ("epl", "soccer_epl", True, "h2h"),
            ("epl_1x2_long", "soccer_epl", True, "1x2"),
            ("epl_1x2_canonical", "soccer_epl", True, "1x2"),
            ("mlb", "baseball_mlb", False, "h2h"),
        ]:
            session.add(
                Event(
                    id=event_id,
                    sport_key=sport_key,
                    sport_title=sport_key,
                    commence_time=when,
                    home_team="H",
                    away_team="A",
                    status=EventStatus.SCHEDULED,
                )
            )
            raw = three_way_event(sport_key)
            raw["bookmakers"][0]["markets"][0]["key"] = market_key
            if event_id == "epl_1x2_canonical":
                for outcome in raw["bookmakers"][0]["markets"][0]["outcomes"]:
                    outcome["name"] = normalize_team_name(outcome["name"])
            if not three_way:
                raw["bookmakers"][0]["markets"][0]["outcomes"].pop()
            raw["bookmakers"].append({"key": "empty", "title": "Empty", "markets": []})
            raw["bookmakers"].append({"key": "nomarkets", "title": "No markets"})
            session.add(
                OddsSnapshot(event_id=event_id, snapshot_time=when, raw_data=raw, bookmaker_count=1)
            )
            names = (
                ["Newcastle", "Tottenham", "Draw"]
                if event_id == "epl_1x2_canonical"
                else ["Newcastle United", "Tottenham Hotspur", "Draw"]
                if three_way
                else ["Newcastle United", "Tottenham Hotspur"]
            )
            for name in names:
                session.add(
                    Odds(
                        event_id=event_id,
                        bookmaker_key="pinnacle",
                        bookmaker_title="Pinnacle",
                        market_key=market_key,
                        outcome_name=name,
                        price=100,
                        odds_timestamp=when,
                        last_update=when,
                    )
                )
        await session.commit()

    async def keys(
        self, session: AsyncSession
    ) -> tuple[dict[str, set[tuple[str, str]]], dict[str, dict[str, Any]]]:
        odds: dict[str, set[tuple[str, str]]] = {}
        rows = (
            await session.execute(select(Odds.event_id, Odds.market_key, Odds.outcome_name))
        ).all()
        for event_id, market_key, outcome_name in rows:
            odds.setdefault(event_id, set()).add((market_key, outcome_name))
        snapshots = {
            s.event_id: s.raw_data for s in (await session.execute(select(OddsSnapshot))).scalars()
        }
        return odds, snapshots

    @pytest.mark.asyncio
    async def test_relabels_only_soccer_three_way_and_is_idempotent(
        self, test_session: AsyncSession
    ) -> None:
        migration = load_migration()
        await self.seed(test_session)

        for _ in range(2):
            await test_session.execute(text(migration.RELABEL_ODDS_SQL))
            await test_session.execute(text(migration.RELABEL_SNAPSHOTS_SQL))
            await test_session.commit()
            test_session.expire_all()

            odds, snapshots = await self.keys(test_session)
            assert odds["epl"] == {("1x2", "Newcastle"), ("1x2", "Tottenham"), ("1x2", "Draw")}
            for event_id in ("epl_1x2_long", "epl_1x2_canonical"):
                assert odds[event_id] == {
                    ("1x2", "Newcastle"),
                    ("1x2", "Tottenham"),
                    ("1x2", "Draw"),
                }
                names = [
                    o["name"]
                    for o in snapshots[event_id]["bookmakers"][0]["markets"][0]["outcomes"]
                ]
                assert names == ["Newcastle", "Tottenham", "Draw"]
                assert market_keys(snapshots[event_id]) == ["1x2", "totals"]
            assert odds["mlb"] == {
                ("h2h", "Newcastle United"),
                ("h2h", "Tottenham Hotspur"),
            }
            epl = snapshots["epl"]
            assert market_keys(epl) == ["1x2", "totals"]
            assert [o["name"] for o in epl["bookmakers"][0]["markets"][0]["outcomes"]] == [
                "Newcastle",
                "Tottenham",
                "Draw",
            ]
            assert epl["bookmakers"][1:] == [
                {"key": "empty", "title": "Empty", "markets": []},
                {"key": "nomarkets", "title": "No markets"},
            ]
            mlb = snapshots["mlb"]
            assert market_keys(mlb) == ["h2h", "totals"]
            assert mlb["bookmakers"][0]["markets"][0]["outcomes"][0]["name"] == "Newcastle United"

    def test_name_map_matches_normalize_team_name(self) -> None:
        for raw, canonical in json.loads(load_migration().TEAM_NAMES).items():
            assert normalize_team_name(raw) == canonical
