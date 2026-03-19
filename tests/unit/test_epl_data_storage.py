"""Tests for EPL data storage: ESPN fixtures, ESPN lineups, FPL availability."""

from datetime import UTC, datetime

import pytest
from odds_core.epl_data_models import EspnFixture, EspnLineup, FplAvailability
from odds_lambda.storage.espn_fixture_reader import EspnFixtureReader
from odds_lambda.storage.espn_fixture_writer import EspnFixtureWriter
from odds_lambda.storage.espn_lineup_reader import EspnLineupReader
from odds_lambda.storage.espn_lineup_writer import EspnLineupWriter
from odds_lambda.storage.fpl_availability_reader import FplAvailabilityReader
from odds_lambda.storage.fpl_availability_writer import FplAvailabilityWriter
from sqlalchemy import select


def _fixture_dict(**overrides: object) -> dict:
    defaults = {
        "date": datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        "team": "Arsenal",
        "opponent": "Chelsea",
        "competition": "Premier League",
        "round": "Regular Season",
        "home_away": "home",
        "score_team": "2",
        "score_opponent": "1",
        "status": "Final",
        "season": "2024-25",
    }
    defaults.update(overrides)
    return defaults


def _lineup_dict(**overrides: object) -> dict:
    defaults = {
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
    return defaults


def _fpl_dict(**overrides: object) -> dict:
    defaults = {
        "snapshot_time": datetime(2025, 1, 14, 18, 0, tzinfo=UTC),
        "gameweek": 21,
        "season": "2024-25",
        "player_code": 438098,
        "player_name": "Saka",
        "team": "Arsenal",
        "position": "MID",
        "chance_of_playing": 75.0,
        "status": "d",
    }
    defaults.update(overrides)
    return defaults


class TestEspnFixtureWriter:
    @pytest.mark.asyncio
    async def test_upsert_creates_records(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)
        records = [_fixture_dict(), _fixture_dict(team="Chelsea", home_away="away")]
        count = await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        assert count == 2
        result = await pglite_async_session.execute(select(EspnFixture))
        assert len(list(result.scalars().all())) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)
        records = [_fixture_dict()]

        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(EspnFixture))
        assert len(list(result.scalars().all())) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_on_conflict(self, pglite_async_session):
        writer = EspnFixtureWriter(pglite_async_session)

        await writer.upsert_fixtures([_fixture_dict(score_team="0")])
        await pglite_async_session.commit()

        await writer.upsert_fixtures([_fixture_dict(score_team="2")])
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
                _fixture_dict(),
                _fixture_dict(
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
                _fixture_dict(season="2024-25"),
                _fixture_dict(
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
                _fixture_dict(team="Arsenal"),
                _fixture_dict(team="Chelsea", home_away="away"),
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
            _lineup_dict(),
            _lineup_dict(player_id="67890", player_name="Martin Odegaard"),
        ]
        count = await writer.upsert_lineups(records)
        await pglite_async_session.commit()

        assert count == 2
        result = await pglite_async_session.execute(select(EspnLineup))
        assert len(list(result.scalars().all())) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        writer = EspnLineupWriter(pglite_async_session)
        records = [_lineup_dict()]

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
                _lineup_dict(starter=True),
                _lineup_dict(player_id="67890", player_name="Sub Player", starter=False),
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
                _lineup_dict(starter=True),
                _lineup_dict(player_id="67890", player_name="Sub Player", starter=False),
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
                _lineup_dict(season="2024-25"),
                _lineup_dict(
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
            _fpl_dict(),
            _fpl_dict(player_code=205651, player_name="G.Jesus"),
        ]
        count = await writer.upsert_availability(records)
        await pglite_async_session.commit()

        assert count == 2
        result = await pglite_async_session.execute(select(FplAvailability))
        assert len(list(result.scalars().all())) == 2

    @pytest.mark.asyncio
    async def test_upsert_idempotent(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)
        records = [_fpl_dict()]

        await writer.upsert_availability(records)
        await pglite_async_session.commit()

        await writer.upsert_availability(records)
        await pglite_async_session.commit()

        result = await pglite_async_session.execute(select(FplAvailability))
        assert len(list(result.scalars().all())) == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_on_conflict(self, pglite_async_session):
        writer = FplAvailabilityWriter(pglite_async_session)

        await writer.upsert_availability([_fpl_dict(chance_of_playing=50.0)])
        await pglite_async_session.commit()

        await writer.upsert_availability([_fpl_dict(chance_of_playing=75.0)])
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
                _fpl_dict(),
                _fpl_dict(player_code=205651, player_name="G.Jesus"),
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
                _fpl_dict(season="2024-25"),
                _fpl_dict(
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
                _fpl_dict(gameweek=21),
                _fpl_dict(
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
