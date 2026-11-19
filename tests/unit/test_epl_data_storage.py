"""Tests for EPL data storage: ESPN games, ESPN lineups, FPL availability."""

from datetime import UTC, datetime
from typing import Any

import pytest
from odds_core.epl_data_models import (
    EspnGame,
    EspnGameRecord,
    EspnLineup,
    EspnLineupRecord,
    FplAvailability,
    FplAvailabilityRecord,
)
from odds_lambda.storage.espn_game_reader import EspnGameReader
from odds_lambda.storage.espn_game_writer import EspnGameWriter
from odds_lambda.storage.espn_lineup_writer import EspnLineupWriter
from odds_lambda.storage.fpl_availability_writer import FplAvailabilityWriter
from sqlalchemy import select


def _game_record(**overrides: Any) -> EspnGameRecord:
    defaults: dict[str, Any] = {
        "date": datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        "team": "Arsenal",
        "opponent": "Chelsea",
        "competition": "Premier League",
        "game_round": "Regular Season",
        "home_away": "home",
        "score_team": "2",
        "score_opponent": "1",
        "status": "Final",
        "season": "2024-25",
    }
    defaults.update(overrides)
    return EspnGameRecord(**defaults)


def _lineup_record(**overrides: Any) -> EspnLineupRecord:
    defaults: dict[str, Any] = {
        "date": datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "team": "Arsenal",
        "player_id": "12345",
        "player_name": "Bukayo Saka",
        "position": "F",
        "starter": True,
        "formation_place": 7,
        "season": "2024-25",
    }
    defaults.update(overrides)
    return EspnLineupRecord(**defaults)


def _fpl_record(**overrides: Any) -> FplAvailabilityRecord:
    defaults: dict[str, Any] = {
        "snapshot_time": datetime(2025, 1, 14, 18, 0, tzinfo=UTC),
        "gameweek": 21,
        "season": "2024-25",
        "player_code": 438098,
        "player_name": "Saka",
        "team": "Arsenal",
        "position": "MID",
        "chance_of_playing": 75.0,
        "status": "d",
        "news": None,
    }
    defaults.update(overrides)
    return FplAvailabilityRecord(**defaults)


class TestEspnGameWriter:
    @pytest.mark.asyncio
    async def test_upsert_creates_records(self, pglite_async_session):
        writer = EspnGameWriter(pglite_async_session)
        records = [_game_record(), _game_record(team="Chelsea", home_away="away")]
        count = await writer.upsert_games(records)
        await pglite_async_session.commit()

        assert count == 2
        result = await pglite_async_session.execute(select(EspnGame))
        assert len(list(result.scalars().all())) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        writer = EspnGameWriter(pglite_async_session)
        records = [_game_record()]

        await writer.upsert_games(records)
        await pglite_async_session.commit()

        await writer.upsert_games(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(EspnGame))
        assert len(list(result.scalars().all())) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_on_conflict(self, pglite_async_session):
        writer = EspnGameWriter(pglite_async_session)

        await writer.upsert_games([_game_record(score_team="0")])
        await pglite_async_session.commit()

        await writer.upsert_games([_game_record(score_team="2")])
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(EspnGame))
        rows = list(result.scalars().all())
        assert len(rows) == 1
        assert rows[0].score_team == "2"


class TestEspnGameReader:
    @pytest.mark.asyncio
    async def test_get_all_games(self, pglite_async_session):
        writer = EspnGameWriter(pglite_async_session)
        await writer.upsert_games(
            [
                _game_record(),
                _game_record(
                    date=datetime(2025, 1, 22, 15, 0, tzinfo=UTC),
                    opponent="Liverpool",
                ),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnGameReader(pglite_async_session)
        games = await reader.get_all_games()
        assert len(games) == 2

    @pytest.mark.asyncio
    async def test_get_games_by_season(self, pglite_async_session):
        writer = EspnGameWriter(pglite_async_session)
        await writer.upsert_games(
            [
                _game_record(season="2024-25"),
                _game_record(
                    date=datetime(2024, 1, 15, 15, 0, tzinfo=UTC),
                    season="2023-24",
                ),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnGameReader(pglite_async_session)
        games = await reader.get_games_by_season("2024-25")
        assert len(games) == 1
        assert games[0].season == "2024-25"

    @pytest.mark.asyncio
    async def test_get_games_for_team(self, pglite_async_session):
        writer = EspnGameWriter(pglite_async_session)
        await writer.upsert_games(
            [
                _game_record(team="Arsenal"),
                _game_record(team="Chelsea", home_away="away"),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnGameReader(pglite_async_session)
        games = await reader.get_games_for_team("Arsenal")
        assert len(games) == 1


class TestEspnLineupWriter:
    @pytest.mark.asyncio
    async def test_upsert_creates_records(self, pglite_async_session):
        writer = EspnLineupWriter(pglite_async_session)
        records = [
            _lineup_record(),
            _lineup_record(player_id="67890", player_name="Martin Odegaard"),
        ]
        count = await writer.upsert_lineups(records)
        await pglite_async_session.commit()

        assert count == 2
        result = await pglite_async_session.execute(select(EspnLineup))
        assert len(list(result.scalars().all())) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        writer = EspnLineupWriter(pglite_async_session)
        records = [_lineup_record()]

        await writer.upsert_lineups(records)
        await pglite_async_session.commit()

        await writer.upsert_lineups(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(EspnLineup))
        assert len(list(result.scalars().all())) == 1


class TestFplAvailabilityWriter:
    @pytest.mark.asyncio
    async def test_upsert_creates_records(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)
        records = [
            _fpl_record(),
            _fpl_record(player_code=205651, player_name="G.Jesus"),
        ]
        count = await writer.upsert_availability(records)
        await pglite_async_session.commit()

        assert count == 2
        result = await pglite_async_session.execute(select(FplAvailability))
        assert len(list(result.scalars().all())) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)
        records = [_fpl_record()]

        await writer.upsert_availability(records)
        await pglite_async_session.commit()

        await writer.upsert_availability(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(FplAvailability))
        assert len(list(result.scalars().all())) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_on_conflict(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)

        await writer.upsert_availability([_fpl_record(chance_of_playing=50.0)])
        await pglite_async_session.commit()

        await writer.upsert_availability([_fpl_record(chance_of_playing=75.0)])
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(FplAvailability))
        rows = list(result.scalars().all())
        assert len(rows) == 1
        assert rows[0].chance_of_playing == 75.0


class TestLoadGamesDf:
    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self, pglite_async_session):
        from odds_analytics.feature_groups import load_games_df

        df = await load_games_df(pglite_async_session)
        assert df is None

    @pytest.mark.asyncio
    async def test_returns_dataframe_with_expected_columns(self, pglite_async_session):
        from odds_analytics.feature_groups import load_games_df

        writer = EspnGameWriter(pglite_async_session)
        await writer.upsert_games([_game_record(), _game_record(team="Chelsea", home_away="away")])
        await pglite_async_session.commit()

        df = await load_games_df(pglite_async_session)
        assert df is not None
        assert len(df) == 2
        expected_cols = {
            "date",
            "team",
            "opponent",
            "competition",
            "round",
            "home_away",
            "score_team",
            "score_opponent",
            "status",
            "state",
            "season",
        }
        assert set(df.columns) == expected_cols

    @pytest.mark.asyncio
    async def test_dates_are_utc_aware(self, pglite_async_session):
        from odds_analytics.feature_groups import load_games_df

        writer = EspnGameWriter(pglite_async_session)
        await writer.upsert_games([_game_record()])
        await pglite_async_session.commit()

        df = await load_games_df(pglite_async_session)
        assert df is not None
        assert df["date"].dt.tz is not None
