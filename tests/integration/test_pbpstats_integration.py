"""Integration tests for PBPStats writer and reader."""

from __future__ import annotations

import pytest
from odds_lambda.pbpstats_fetcher import PlayerSeasonRecord
from odds_lambda.storage.pbpstats_reader import PbpStatsReader
from odds_lambda.storage.pbpstats_writer import PbpStatsWriter


def _make_record(
    player_id: int = 201939,
    player_name: str = "Curry, Stephen",
    team_id: int = 1610612744,
    team_abbreviation: str = "GSW",
    season: str = "2024-25",
    minutes: float = 2252.0,
    games_played: int = 70,
    on_off_rtg: float | None = 120.5,
    on_def_rtg: float | None = 115.5,
    usage: float | None = 29.2,
    ts_pct: float | None = 0.63,
    efg_pct: float | None = 0.57,
    assists: int = 350,
    turnovers: int = 200,
    rebounds: int = 300,
    steals: int = 80,
    blocks: int = 30,
    points: int = 1800,
    plus_minus: float = 250.0,
) -> PlayerSeasonRecord:
    return PlayerSeasonRecord(
        player_id=player_id,
        player_name=player_name,
        team_id=team_id,
        team_abbreviation=team_abbreviation,
        season=season,
        minutes=minutes,
        games_played=games_played,
        on_off_rtg=on_off_rtg,
        on_def_rtg=on_def_rtg,
        usage=usage,
        ts_pct=ts_pct,
        efg_pct=efg_pct,
        assists=assists,
        turnovers=turnovers,
        rebounds=rebounds,
        steals=steals,
        blocks=blocks,
        points=points,
        plus_minus=plus_minus,
    )


class TestPbpStatsWriter:
    @pytest.mark.asyncio
    async def test_upsert_inserts_new_records(self, pglite_async_session) -> None:
        writer = PbpStatsWriter(pglite_async_session)
        records = [
            _make_record(player_id=1, player_name="Curry, Stephen"),
            _make_record(player_id=2, player_name="Thompson, Klay"),
        ]
        count = await writer.upsert_player_stats(records)
        await pglite_async_session.commit()

        assert count == 2

        reader = PbpStatsReader(pglite_async_session)
        curry = await reader.get_player_stats("Curry, Stephen", "2024-25")
        assert curry is not None
        assert curry.player_id == 1

    @pytest.mark.asyncio
    async def test_upsert_updates_on_conflict(self, pglite_async_session) -> None:
        """Same (player_id, season) upserted twice updates rather than duplicates."""
        writer = PbpStatsWriter(pglite_async_session)

        # First insert
        records_v1 = [_make_record(player_id=1, minutes=1000.0, points=500)]
        await writer.upsert_player_stats(records_v1)
        await pglite_async_session.commit()

        # Second insert with updated stats
        records_v2 = [_make_record(player_id=1, minutes=2000.0, points=1200)]
        await writer.upsert_player_stats(records_v2)
        await pglite_async_session.commit()

        reader = PbpStatsReader(pglite_async_session)
        stats = await reader.get_pipeline_stats()
        assert stats["total_rows"] == 1  # Not 2

        curry = await reader.get_player_stats("Curry, Stephen", "2024-25")
        assert curry is not None
        assert curry.minutes == 2000.0
        assert curry.points == 1200

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self, pglite_async_session) -> None:
        writer = PbpStatsWriter(pglite_async_session)
        count = await writer.upsert_player_stats([])
        assert count == 0

    @pytest.mark.asyncio
    async def test_upsert_nullable_advanced_stats(self, pglite_async_session) -> None:
        """Players with None for advanced stats (low minutes) are stored correctly."""
        writer = PbpStatsWriter(pglite_async_session)
        record = _make_record(
            player_id=99,
            player_name="Bench, Player",
            on_off_rtg=None,
            on_def_rtg=None,
            usage=None,
            ts_pct=None,
            efg_pct=None,
        )
        await writer.upsert_player_stats([record])
        await pglite_async_session.commit()

        reader = PbpStatsReader(pglite_async_session)
        result = await reader.get_player_stats("Bench, Player", "2024-25")
        assert result is not None
        assert result.on_off_rtg is None
        assert result.on_def_rtg is None
        assert result.usage is None


class TestPbpStatsReader:
    @pytest.fixture
    async def seeded_stats(self, pglite_async_session):
        """Seed DB with players across two teams and two seasons."""
        writer = PbpStatsWriter(pglite_async_session)
        records = [
            _make_record(
                player_id=1,
                player_name="Curry, Stephen",
                team_abbreviation="GSW",
                season="2024-25",
                minutes=2252.0,
            ),
            _make_record(
                player_id=2,
                player_name="Thompson, Klay",
                team_abbreviation="GSW",
                season="2024-25",
                minutes=1800.0,
            ),
            _make_record(
                player_id=3,
                player_name="Tatum, Jayson",
                team_abbreviation="BOS",
                season="2024-25",
                minutes=2400.0,
            ),
            _make_record(
                player_id=1,
                player_name="Curry, Stephen",
                team_abbreviation="GSW",
                season="2023-24",
                minutes=2100.0,
            ),
        ]
        await writer.upsert_player_stats(records)
        await pglite_async_session.commit()

    @pytest.mark.asyncio
    async def test_get_player_stats_found(self, pglite_async_session, seeded_stats) -> None:
        reader = PbpStatsReader(pglite_async_session)
        result = await reader.get_player_stats("Curry, Stephen", "2024-25")
        assert result is not None
        assert result.player_id == 1
        assert result.minutes == 2252.0

    @pytest.mark.asyncio
    async def test_get_player_stats_not_found(self, pglite_async_session, seeded_stats) -> None:
        reader = PbpStatsReader(pglite_async_session)
        result = await reader.get_player_stats("Nobody, No", "2024-25")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_player_stats_wrong_season(self, pglite_async_session, seeded_stats) -> None:
        reader = PbpStatsReader(pglite_async_session)
        result = await reader.get_player_stats("Thompson, Klay", "2023-24")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_team_players_ordered_by_minutes(
        self, pglite_async_session, seeded_stats
    ) -> None:
        reader = PbpStatsReader(pglite_async_session)
        players = await reader.get_team_players("GSW", "2024-25")
        assert len(players) == 2
        assert players[0].player_name == "Curry, Stephen"  # 2252 > 1800
        assert players[1].player_name == "Thompson, Klay"

    @pytest.mark.asyncio
    async def test_get_team_players_empty(self, pglite_async_session, seeded_stats) -> None:
        reader = PbpStatsReader(pglite_async_session)
        players = await reader.get_team_players("LAL", "2024-25")
        assert players == []

    @pytest.mark.asyncio
    async def test_pipeline_stats(self, pglite_async_session, seeded_stats) -> None:
        reader = PbpStatsReader(pglite_async_session)
        stats = await reader.get_pipeline_stats()
        assert stats["total_rows"] == 4
        assert stats["unique_players"] == 3  # Curry appears in 2 seasons but is 1 player
        assert stats["season_counts"]["2024-25"] == 3
        assert stats["season_counts"]["2023-24"] == 1
