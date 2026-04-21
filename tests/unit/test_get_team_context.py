"""Unit tests for the ``get_team_context`` MCP tool and standings derivation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.epl_data_models import EspnFixtureRecord
from odds_lambda.storage.espn_fixture_writer import EspnFixtureWriter


def _pl_record(
    *,
    team: str,
    opponent: str,
    date: datetime,
    home_away: str,
    score_team: str,
    score_opponent: str,
    status: str = "Final",
    competition: str = "Premier League",
    season: str = "2025-26",
) -> EspnFixtureRecord:
    return EspnFixtureRecord(
        date=date,
        team=team,
        opponent=opponent,
        competition=competition,
        match_round="Regular Season",
        home_away=home_away,
        score_team=score_team,
        score_opponent=score_opponent,
        status=status,
        season=season,
    )


def _match_records(
    home: str,
    away: str,
    *,
    date: datetime,
    score_home: int,
    score_away: int,
    status: str = "Final",
    competition: str = "Premier League",
    season: str = "2025-26",
) -> list[EspnFixtureRecord]:
    """Both sides of a single match as paired fixture records."""
    return [
        _pl_record(
            team=home,
            opponent=away,
            date=date,
            home_away="home",
            score_team=str(score_home),
            score_opponent=str(score_away),
            status=status,
            competition=competition,
            season=season,
        ),
        _pl_record(
            team=away,
            opponent=home,
            date=date,
            home_away="away",
            score_team=str(score_away),
            score_opponent=str(score_home),
            status=status,
            competition=competition,
            season=season,
        ),
    ]


class TestDeriveStandings:
    """Hand-crafted fixture sets → known-good standings."""

    def _build(self, records: list[EspnFixtureRecord]) -> list[Any]:
        # Build fake EspnFixture objects with the fields the deriver reads.
        rows: list[Any] = []
        for r in records:
            row = MagicMock()
            row.team = r.team
            row.opponent = r.opponent
            row.competition = r.competition
            row.status = r.status
            row.score_team = r.score_team
            row.score_opponent = r.score_opponent
            row.date = r.date
            rows.append(row)
        return rows

    def test_three_teams_round_robin(self) -> None:
        from odds_mcp.server import _derive_standings

        # A beats B 2-0, A draws C 1-1, B beats C 3-1
        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "A",
                "B",
                date=datetime(2025, 8, 10, 15, 0, tzinfo=UTC),
                score_home=2,
                score_away=0,
            )
        )
        records.extend(
            _match_records(
                "A",
                "C",
                date=datetime(2025, 8, 17, 15, 0, tzinfo=UTC),
                score_home=1,
                score_away=1,
            )
        )
        records.extend(
            _match_records(
                "B",
                "C",
                date=datetime(2025, 8, 24, 15, 0, tzinfo=UTC),
                score_home=3,
                score_away=1,
            )
        )

        table = _derive_standings(self._build(records))

        # A: played 2, W1 D1 L0 → 4 pts, GF 3, GA 1, GD +2
        a = table["A"]
        assert a["played"] == 2
        assert a["wins"] == 1
        assert a["draws"] == 1
        assert a["losses"] == 0
        assert a["goals_for"] == 3
        assert a["goals_against"] == 1
        assert a["goal_diff"] == 2
        assert a["points"] == 4

        # B: played 2, W1 D0 L1 → 3 pts, GF 3, GA 3, GD 0
        b = table["B"]
        assert b["played"] == 2
        assert b["wins"] == 1
        assert b["losses"] == 1
        assert b["points"] == 3
        assert b["goals_for"] == 3
        assert b["goals_against"] == 3

        # C: played 2, W0 D1 L1 → 1 pt, GF 2, GA 4, GD -2
        c = table["C"]
        assert c["points"] == 1
        assert c["goal_diff"] == -2

        # Position ordering: A (4 pts) → B (3 pts) → C (1 pt)
        assert a["position"] == 1
        assert b["position"] == 2
        assert c["position"] == 3

    def test_position_tiebreak_by_goal_diff_then_gf(self) -> None:
        from odds_mcp.server import _derive_standings

        # X, Y, Z all play 2 games, all end with 4 pts, differentiated by GD/GF.
        # X: beats Y 3-0, draws Z 0-0 → 4 pts, GD +3, GF 3
        # Y: loses to X 0-3, beats Z 2-0 → 3 pts
        # Z: draws X 0-0, loses to Y 0-2 → 1 pt
        # Add W that mirrors X's points with different GD to force tiebreak:
        # Actually let's build a cleaner tie: two teams on same pts & GD, differ on GF.
        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "X",
                "Z",
                date=datetime(2025, 8, 10, 15, 0, tzinfo=UTC),
                score_home=4,
                score_away=2,
            )
        )
        records.extend(
            _match_records(
                "Y",
                "Z",
                date=datetime(2025, 8, 17, 15, 0, tzinfo=UTC),
                score_home=3,
                score_away=1,
            )
        )
        # Now X and Y both have 3 pts, GD +2; X has GF 4 vs Y's GF 3.
        table = _derive_standings(self._build(records))
        assert table["X"]["points"] == 3
        assert table["Y"]["points"] == 3
        assert table["X"]["goal_diff"] == 2
        assert table["Y"]["goal_diff"] == 2
        # X has more goals for → ranks above Y
        assert table["X"]["position"] == 1
        assert table["Y"]["position"] == 2

    def test_only_counts_premier_league(self) -> None:
        from odds_mcp.server import _derive_standings

        # An FA Cup win does NOT count toward the league table.
        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "A",
                "B",
                date=datetime(2025, 8, 10, 15, 0, tzinfo=UTC),
                score_home=3,
                score_away=0,
                competition="FA Cup",
            )
        )
        table = _derive_standings(self._build(records))
        assert table == {}

    def test_only_counts_final_status(self) -> None:
        from odds_mcp.server import _derive_standings

        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "A",
                "B",
                date=datetime(2025, 8, 10, 15, 0, tzinfo=UTC),
                score_home=3,
                score_away=0,
                status="Scheduled",
            )
        )
        table = _derive_standings(self._build(records))
        assert table == {}

    def test_skips_malformed_scores(self) -> None:
        from odds_mcp.server import _derive_standings

        # Final row with blank scores (ESPN quirk) is ignored.
        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "A",
                "B",
                date=datetime(2025, 8, 10, 15, 0, tzinfo=UTC),
                score_home=2,
                score_away=1,
            )
        )
        # Corrupt the second match: Final, but blank scores.
        bad_home = _pl_record(
            team="A",
            opponent="C",
            date=datetime(2025, 8, 17, 15, 0, tzinfo=UTC),
            home_away="home",
            score_team="",
            score_opponent="",
        )
        bad_away = _pl_record(
            team="C",
            opponent="A",
            date=datetime(2025, 8, 17, 15, 0, tzinfo=UTC),
            home_away="away",
            score_team="",
            score_opponent="",
        )
        records.extend([bad_home, bad_away])

        table = _derive_standings(self._build(records))
        # A has played 1, not 2, because the blank-score row was dropped.
        assert table["A"]["played"] == 1
        assert table["B"]["played"] == 1
        assert "C" not in table


class TestGetTeamContextAsOf:
    """Integration-ish: seed ESPN fixtures, call the MCP tool, assert semantics."""

    @pytest.mark.asyncio
    async def test_last_results_respects_as_of_cutoff(
        self, pglite_async_session, test_engine
    ) -> None:
        """as_of must exclude fixtures strictly in the future (no look-ahead)."""
        from odds_mcp import server as mcp_server

        writer = EspnFixtureWriter(pglite_async_session)

        # Arsenal record: one completed PL match in the past (1 Sep), one
        # upcoming (15 Sep). With as_of = 10 Sep, only the 1 Sep result counts.
        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "Arsenal",
                "Chelsea",
                date=datetime(2025, 9, 1, 15, 0, tzinfo=UTC),
                score_home=2,
                score_away=1,
            )
        )
        records.extend(
            _match_records(
                "Arsenal",
                "Liverpool",
                date=datetime(2025, 9, 15, 15, 0, tzinfo=UTC),
                score_home=0,
                score_away=0,
                status="Scheduled",
            )
        )
        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        from sqlalchemy.ext.asyncio import async_sessionmaker

        test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

        with patch.object(mcp_server, "async_session_maker", test_session_maker):
            result = await mcp_server.get_team_context(
                team="Arsenal",
                as_of="2025-09-10T00:00:00Z",
                last_n=5,
                next_n=5,
                include_standings=False,
            )

        assert result["team"] == "Arsenal"
        assert len(result["last_results"]) == 1
        assert result["last_results"][0]["opponent"] == "Chelsea"
        assert result["last_results"][0]["outcome"] == "W"
        # Upcoming: Liverpool fixture still in the future.
        assert len(result["upcoming_fixtures"]) == 1
        assert result["upcoming_fixtures"][0]["opponent"] == "Liverpool"

    @pytest.mark.asyncio
    async def test_in_progress_rows_skipped_from_both_sides(
        self, pglite_async_session, test_engine
    ) -> None:
        from odds_mcp import server as mcp_server
        from sqlalchemy.ext.asyncio import async_sessionmaker

        writer = EspnFixtureWriter(pglite_async_session)

        # A match currently in progress should NOT appear in last_results (no
        # final score yet) nor in upcoming_fixtures (already kicked off).
        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "Arsenal",
                "Chelsea",
                date=datetime(2025, 9, 10, 14, 0, tzinfo=UTC),
                score_home=1,
                score_away=0,
                status="In Progress",
            )
        )
        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

        with patch.object(mcp_server, "async_session_maker", test_session_maker):
            # as_of slightly after kickoff (match is live)
            result = await mcp_server.get_team_context(
                team="Arsenal",
                as_of="2025-09-10T15:00:00Z",
                last_n=5,
                next_n=5,
                include_standings=False,
            )

        assert result["last_results"] == []
        assert result["upcoming_fixtures"] == []

    @pytest.mark.asyncio
    async def test_normalises_team_alias(self, pglite_async_session, test_engine) -> None:
        """Inputs like 'Wolverhampton Wanderers' resolve to canonical 'Wolves'."""
        from odds_mcp import server as mcp_server
        from sqlalchemy.ext.asyncio import async_sessionmaker

        writer = EspnFixtureWriter(pglite_async_session)
        records = _match_records(
            "Wolves",
            "Arsenal",
            date=datetime(2025, 9, 1, 15, 0, tzinfo=UTC),
            score_home=1,
            score_away=0,
        )
        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

        with patch.object(mcp_server, "async_session_maker", test_session_maker):
            result = await mcp_server.get_team_context(
                team="Wolverhampton Wanderers",
                as_of="2025-09-10T00:00:00Z",
                include_standings=False,
            )

        assert result["team"] == "Wolves"
        assert len(result["last_results"]) == 1

    @pytest.mark.asyncio
    async def test_upcoming_includes_cup_competitions(
        self, pglite_async_session, test_engine
    ) -> None:
        from odds_mcp import server as mcp_server
        from sqlalchemy.ext.asyncio import async_sessionmaker

        writer = EspnFixtureWriter(pglite_async_session)
        records: list[EspnFixtureRecord] = []
        records.extend(
            _match_records(
                "Arsenal",
                "Chelsea",
                date=datetime(2025, 9, 15, 15, 0, tzinfo=UTC),
                score_home=0,
                score_away=0,
                status="Scheduled",
            )
        )
        records.extend(
            _match_records(
                "Arsenal",
                "Liverpool",
                date=datetime(2025, 9, 22, 19, 0, tzinfo=UTC),
                score_home=0,
                score_away=0,
                status="Scheduled",
                competition="Champions League",
            )
        )
        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

        with patch.object(mcp_server, "async_session_maker", test_session_maker):
            result = await mcp_server.get_team_context(
                team="Arsenal",
                as_of="2025-09-10T00:00:00Z",
                include_standings=False,
            )

        competitions = {f["competition"] for f in result["upcoming_fixtures"]}
        assert competitions == {"Premier League", "Champions League"}

    @pytest.mark.asyncio
    async def test_standings_returned_from_db(self, pglite_async_session, test_engine) -> None:
        from odds_mcp import server as mcp_server
        from sqlalchemy.ext.asyncio import async_sessionmaker

        writer = EspnFixtureWriter(pglite_async_session)
        records: list[EspnFixtureRecord] = []
        # Arsenal beats Chelsea, then draws Liverpool.
        records.extend(
            _match_records(
                "Arsenal",
                "Chelsea",
                date=datetime(2025, 8, 15, 15, 0, tzinfo=UTC),
                score_home=2,
                score_away=0,
            )
        )
        records.extend(
            _match_records(
                "Arsenal",
                "Liverpool",
                date=datetime(2025, 8, 22, 15, 0, tzinfo=UTC),
                score_home=1,
                score_away=1,
            )
        )
        await writer.upsert_fixtures(records)
        await pglite_async_session.commit()

        test_session_maker = async_sessionmaker(test_engine, expire_on_commit=False)

        with patch.object(mcp_server, "async_session_maker", test_session_maker):
            result = await mcp_server.get_team_context(
                team="Arsenal",
                as_of="2025-09-01T00:00:00Z",
                include_standings=True,
            )

        standings = result["standings"]
        arsenal_row = standings["team_row"]
        assert arsenal_row["points"] == 4  # 1W + 1D
        assert arsenal_row["position"] == 1
        # Full table returned in position order
        assert [r["team"] for r in standings["table"]] == ["Arsenal", "Liverpool", "Chelsea"]


class TestParseScore:
    def test_valid_integer(self) -> None:
        from odds_mcp.server import _parse_score

        assert _parse_score("3") == 3

    def test_blank(self) -> None:
        from odds_mcp.server import _parse_score

        assert _parse_score("") is None

    def test_malformed(self) -> None:
        from odds_mcp.server import _parse_score

        assert _parse_score("abc") is None

    def test_negative_rejected(self) -> None:
        from odds_mcp.server import _parse_score

        assert _parse_score("-1") is None


class TestGetTeamContextMockedDB:
    """Guardrail: `as_of=None` defaults to now; resolved datetime echoes back."""

    @pytest.mark.asyncio
    async def test_default_as_of_returns_now(self) -> None:
        from odds_mcp import server as mcp_server

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch("odds_mcp.server.async_session_maker") as mock_session_maker:
            mock_session_maker.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_maker.return_value.__aexit__ = AsyncMock()

            result = await mcp_server.get_team_context(
                team="Arsenal",
                as_of=None,
                include_standings=False,
            )

        # as_of echoed back as an ISO string
        parsed = datetime.fromisoformat(result["as_of"])
        # Should be within a second of now (test ran milliseconds ago)
        assert abs((parsed - datetime.now(UTC)).total_seconds()) < 5
