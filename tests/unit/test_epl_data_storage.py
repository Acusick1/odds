"""Tests for EPL data storage: ESPN fixtures, ESPN lineups, FPL availability."""

from datetime import UTC, datetime
from typing import Any

import pytest
from odds_core.epl_data_models import (
    EspnFixture,
    EspnFixtureRecord,
    EspnLineup,
    EspnLineupRecord,
    FplAvailability,
    FplAvailabilityRecord,
)
from odds_lambda.storage.espn_fixture_reader import EspnFixtureReader
from odds_lambda.storage.espn_fixture_writer import EspnFixtureWriter
from odds_lambda.storage.espn_lineup_reader import EspnLineupReader
from odds_lambda.storage.espn_lineup_writer import EspnLineupWriter
from odds_lambda.storage.fpl_availability_reader import FplAvailabilityReader
from odds_lambda.storage.fpl_availability_writer import FplAvailabilityWriter
from sqlalchemy import select


def _fixture_record(**overrides: Any) -> EspnFixtureRecord:
    defaults: dict[str, Any] = {
        "date": datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        "team": "Arsenal",
        "opponent": "Chelsea",
        "competition": "Premier League",
        "match_round": "Regular Season",
        "home_away": "home",
        "score_team": "2",
        "score_opponent": "1",
        "status": "Final",
        "season": "2024-25",
    }
    defaults.update(overrides)
    return EspnFixtureRecord(**defaults)


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


class TestEspnFixtureWriter:
    @pytest.mark.asyncio
    async def test_upsert_creates_records(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)
        records = [_fixture_record(), _fixture_record(team="Chelsea", home_away="away")]
        count = await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        assert count == 2
        result = await pglite_async_session.execute(select(EspnFixture))
        assert len(list(result.scalars().all())) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)
        records = [_fixture_record()]

        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(EspnFixture))
        assert len(list(result.scalars().all())) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_on_conflict(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)

        await writer.upsert_fixtures([_fixture_record(score_team="0")])
        await pglite_async_session.commit()

        await writer.upsert_fixtures([_fixture_record(score_team="2")])
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(EspnFixture))
        rows = list(result.scalars().all())
        assert len(rows) == 1
        assert rows[0].score_team == "2"


class TestEspnFixtureReader:
    @pytest.mark.asyncio
    async def test_get_all_fixtures(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)
        await writer.upsert_fixtures(
            [
                _fixture_record(),
                _fixture_record(
                    date=datetime(2025, 1, 22, 15, 0, tzinfo=UTC),
                    opponent="Liverpool",
                ),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnFixtureReader(pglite_async_session)
        fixtures = await reader.get_all_fixtures()
        assert len(fixtures) == 2

    @pytest.mark.asyncio
    async def test_get_fixtures_by_season(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)
        await writer.upsert_fixtures(
            [
                _fixture_record(season="2024-25"),
                _fixture_record(
                    date=datetime(2024, 1, 15, 15, 0, tzinfo=UTC),
                    season="2023-24",
                ),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnFixtureReader(pglite_async_session)
        fixtures = await reader.get_fixtures_by_season("2024-25")
        assert len(fixtures) == 1
        assert fixtures[0].season == "2024-25"

    @pytest.mark.asyncio
    async def test_get_fixtures_for_team(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)
        await writer.upsert_fixtures(
            [
                _fixture_record(team="Arsenal"),
                _fixture_record(team="Chelsea", home_away="away"),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnFixtureReader(pglite_async_session)
        fixtures = await reader.get_fixtures_for_team("Arsenal")
        assert len(fixtures) == 1


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


class TestEspnLineupReader:
    @pytest.mark.asyncio
    async def test_get_all_lineups(self, pglite_async_session):
        writer = EspnLineupWriter(pglite_async_session)
        await writer.upsert_lineups(
            [
                _lineup_record(starter=True),
                _lineup_record(player_id="67890", player_name="Sub Player", starter=False),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnLineupReader(pglite_async_session)
        lineups = await reader.get_all_lineups()
        assert len(lineups) == 2

    @pytest.mark.asyncio
    async def test_get_starters(self, pglite_async_session):
        writer = EspnLineupWriter(pglite_async_session)
        await writer.upsert_lineups(
            [
                _lineup_record(starter=True),
                _lineup_record(player_id="67890", player_name="Sub Player", starter=False),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnLineupReader(pglite_async_session)
        starters = await reader.get_starters()
        assert len(starters) == 1
        assert starters[0].player_name == "Bukayo Saka"

    @pytest.mark.asyncio
    async def test_get_lineups_by_season(self, pglite_async_session):
        writer = EspnLineupWriter(pglite_async_session)
        await writer.upsert_lineups(
            [
                _lineup_record(season="2024-25"),
                _lineup_record(
                    date=datetime(2024, 1, 15, 15, 0, tzinfo=UTC),
                    player_id="99999",
                    season="2023-24",
                ),
            ]
        )
        await pglite_async_session.commit()

        reader = EspnLineupReader(pglite_async_session)
        lineups = await reader.get_lineups_by_season("2024-25")
        assert len(lineups) == 1


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


class TestFplAvailabilityReader:
    @pytest.mark.asyncio
    async def test_get_all_availability(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)
        await writer.upsert_availability(
            [
                _fpl_record(),
                _fpl_record(player_code=205651, player_name="G.Jesus"),
            ]
        )
        await pglite_async_session.commit()

        reader = FplAvailabilityReader(pglite_async_session)
        records = await reader.get_all_availability()
        assert len(records) == 2

    @pytest.mark.asyncio
    async def test_get_availability_by_season(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)
        await writer.upsert_availability(
            [
                _fpl_record(season="2024-25"),
                _fpl_record(
                    season="2023-24",
                    snapshot_time=datetime(2024, 1, 14, 18, 0, tzinfo=UTC),
                ),
            ]
        )
        await pglite_async_session.commit()

        reader = FplAvailabilityReader(pglite_async_session)
        records = await reader.get_availability_by_season("2024-25")
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_get_availability_for_gameweek(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)
        await writer.upsert_availability(
            [
                _fpl_record(gameweek=21),
                _fpl_record(
                    gameweek=22,
                    snapshot_time=datetime(2025, 1, 21, 18, 0, tzinfo=UTC),
                ),
            ]
        )
        await pglite_async_session.commit()

        reader = FplAvailabilityReader(pglite_async_session)
        records = await reader.get_availability_for_gameweek("2024-25", 21)
        assert len(records) == 1
        assert records[0].gameweek == 21


class TestLoadFixturesDf:
    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self, pglite_async_session):
        from odds_analytics.feature_groups import load_fixtures_df

        df = await load_fixtures_df(pglite_async_session)
        assert df is None

    @pytest.mark.asyncio
    async def test_returns_dataframe_with_expected_columns(self, pglite_async_session):
        from odds_analytics.feature_groups import load_fixtures_df

        writer = EspnFixtureWriter(pglite_async_session)
        await writer.upsert_fixtures(
            [_fixture_record(), _fixture_record(team="Chelsea", home_away="away")]
        )
        await pglite_async_session.commit()

        df = await load_fixtures_df(pglite_async_session)
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
        from odds_analytics.feature_groups import load_fixtures_df

        writer = EspnFixtureWriter(pglite_async_session)
        await writer.upsert_fixtures([_fixture_record()])
        await pglite_async_session.commit()

        df = await load_fixtures_df(pglite_async_session)
        assert df is not None
        assert df["date"].dt.tz is not None


class TestLoadLineupCache:
    @pytest.mark.asyncio
    async def test_returns_none_when_empty(self, pglite_async_session):
        from odds_analytics.feature_groups import load_lineup_cache

        cache = await load_lineup_cache(pglite_async_session)
        assert cache is None

    @pytest.mark.asyncio
    async def test_returns_cache_from_db_records(self, pglite_async_session):
        from odds_analytics.feature_groups import load_lineup_cache

        writer = EspnLineupWriter(pglite_async_session)
        await writer.upsert_lineups(
            [
                _lineup_record(player_id="1", player_name="Player A", starter=True),
                _lineup_record(player_id="2", player_name="Player B", starter=True),
                _lineup_record(player_id="3", player_name="Sub C", starter=False),
            ]
        )
        await pglite_async_session.commit()

        cache = await load_lineup_cache(pglite_async_session)
        assert cache is not None
        assert "Arsenal" in cache
