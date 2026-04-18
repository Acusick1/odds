"""Tests for the cc937c8cd362_repoint_conflated_oddsportal_snapshots migration.

Exercises the divergent-set and offending-pairs SQL directly against pglite so
we can verify the migration skips `(event_id, snapshot_time)` pairs backed by
multiple snapshots instead of raising, while still repointing the safe
remainder.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

# Import all models the migration touches so pglite's session-scoped
# SQLModel.metadata.create_all picks up their tables. The migration queries
# paper_trades and match_briefs; without these imports the tables may be
# absent when create_all runs.
from odds_core.match_brief_models import MatchBrief  # noqa: F401
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade  # noqa: F401
from odds_core.prediction_models import Prediction  # noqa: F401
from sqlalchemy.ext.asyncio import AsyncSession


def _load_migration() -> ModuleType:
    """Load the migration module from its file path.

    Migration files are not on sys.path (no `__init__.py` under versions/),
    so import it by path to access the module-level SQL constants.
    """
    repo_root = Path(__file__).resolve().parents[2]
    migration_path = (
        repo_root
        / "migrations"
        / "versions"
        / "cc937c8cd362_repoint_conflated_oddsportal_snapshots.py"
    )
    spec = importlib.util.spec_from_file_location("cc937c8cd362_migration", migration_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def migration_module() -> ModuleType:
    return _load_migration()


class TestRepointConflatedQueries:
    """Verify divergent-set and offending-pairs SQL against populated pglite."""

    @pytest.fixture
    async def populated_session(self, pglite_async_session: AsyncSession) -> AsyncSession:
        """Populate events + snapshots to exercise the skip path.

        Layout:
        - event_safe: one event whose commence_time is >2h off from
          (snapshot_time + hours_until_commence). One divergent snapshot
          with a unique (event_id, snapshot_time). Must appear in the
          divergent set.
        - event_dup: one event with TWO divergent snapshots sharing the same
          snapshot_time (the scenario that blew up on prod). Must appear
          in offending pairs and be EXCLUDED from the divergent set.
        - event_clean: one event whose commence_time matches
          (snapshot_time + hours_until_commence). Non-divergent; must not
          appear in either set.
        """
        base = datetime(2026, 3, 15, 20, 0, tzinfo=UTC)

        # Safe divergent event: commence stored as base, snapshot implies
        # base + 24h (offset of 24h, well over the 2h threshold).
        event_safe = Event(
            id="event_safe",
            sport_key="baseball_mlb",
            sport_title="MLB",
            commence_time=base,
            home_team="Safe Home",
            away_team="Safe Away",
            status=EventStatus.SCHEDULED,
        )
        snap_safe_time = base - timedelta(hours=5)
        # 5h + 24h = 29h until commence, but event commences at base (5h later),
        # so implied true_commence = snap_safe_time + 29h = base + 24h.
        snap_safe = OddsSnapshot(
            event_id="event_safe",
            snapshot_time=snap_safe_time,
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=29.0,
        )

        # Duplicate-pair event: two snapshots, same snapshot_time, both
        # divergent, different hours_until_commence (simulates the conflation).
        event_dup = Event(
            id="event_dup",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=base,
            home_team="Dup Home",
            away_team="Dup Away",
            status=EventStatus.SCHEDULED,
        )
        snap_dup_time = base - timedelta(hours=4)
        # hours_until_commence=28 → implies commence = base + 24h, divergent.
        snap_dup_1 = OddsSnapshot(
            event_id="event_dup",
            snapshot_time=snap_dup_time,
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=28.0,
        )
        # hours_until_commence=52 → implies commence = base + 48h, divergent.
        snap_dup_2 = OddsSnapshot(
            event_id="event_dup",
            snapshot_time=snap_dup_time,
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=52.0,
        )

        # Clean event: snapshot implies the stored commence_time (no divergence).
        event_clean = Event(
            id="event_clean",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=base,
            home_team="Clean Home",
            away_team="Clean Away",
            status=EventStatus.SCHEDULED,
        )
        snap_clean = OddsSnapshot(
            event_id="event_clean",
            snapshot_time=base - timedelta(hours=3),
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=3.0,
        )

        pglite_async_session.add_all(
            [
                event_safe,
                event_dup,
                event_clean,
                snap_safe,
                snap_dup_1,
                snap_dup_2,
                snap_clean,
            ]
        )
        await pglite_async_session.commit()

        return pglite_async_session

    async def test_offending_pairs_query_returns_only_duplicates(
        self,
        populated_session: AsyncSession,
        migration_module: ModuleType,
    ) -> None:
        """The offending-pairs query must pick out exactly the dup event's pair."""
        result = await populated_session.execute(migration_module._OFFENDING_PAIRS_QUERY)
        rows = result.fetchall()

        assert len(rows) == 1, f"Expected 1 offending pair, got {len(rows)}: {rows}"
        (event_id, snapshot_time, count) = rows[0]
        assert event_id == "event_dup"
        assert count == 2

    async def test_divergence_query_excludes_offending_pairs(
        self,
        populated_session: AsyncSession,
        migration_module: ModuleType,
    ) -> None:
        """The divergence query must skip offending pairs and return only the safe row."""
        result = await populated_session.execute(migration_module._DIVERGENCE_QUERY)
        rows = result.fetchall()

        event_ids = [r.old_event_id for r in rows]
        assert event_ids == ["event_safe"], (
            f"Expected only event_safe in divergent set, got {event_ids}"
        )

    async def test_breakdown_query_matches_filtered_set(
        self,
        populated_session: AsyncSession,
        migration_module: ModuleType,
    ) -> None:
        """Per-sport breakdown must reflect the filtered set (no offending rows)."""
        result = await populated_session.execute(migration_module._DIVERGENCE_BREAKDOWN_QUERY)
        rows = result.fetchall()

        assert len(rows) == 1
        assert rows[0].sport_key == "baseball_mlb"
        assert rows[0].n == 1

    async def test_queries_are_noop_on_clean_db(
        self,
        pglite_async_session: AsyncSession,
        migration_module: ModuleType,
    ) -> None:
        """With no divergent data, all three queries must return empty results."""
        offending = (
            await pglite_async_session.execute(migration_module._OFFENDING_PAIRS_QUERY)
        ).fetchall()
        divergent = (
            await pglite_async_session.execute(migration_module._DIVERGENCE_QUERY)
        ).fetchall()
        breakdown = (
            await pglite_async_session.execute(migration_module._DIVERGENCE_BREAKDOWN_QUERY)
        ).fetchall()

        assert offending == []
        assert divergent == []
        assert breakdown == []


class TestRepointMigrationUpgrade:
    """End-to-end test of upgrade(): skip path logs + safe rows are moved."""

    async def test_upgrade_skips_duplicates_and_repoints_safe_rows(
        self,
        pglite_async_session: AsyncSession,
        migration_module: ModuleType,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """upgrade() must log skipped pairs, repoint the safe snapshot, and not raise.

        Layout:
        - event_safe_e2e: one safe divergent snapshot + Odds + PaperTrade; all
          three should move to event_target.
        - event_dup_e2e: two snapshots at the same snapshot_time (offending
          pair) + Odds row on that pair + PaperTrade. The Odds and PaperTrade
          must NOT move. Additionally hosts a sibling safe snapshot at a
          different snapshot_time that DOES repoint to event_dup_target, but
          the PaperTrade on event_dup_e2e must still NOT move (MUST-FIX:
          event-level attribution ambiguous while offending pair unhealed).
        """
        from odds_core.match_brief_models import BriefDecision

        base = datetime(2026, 3, 15, 20, 0, tzinfo=UTC)
        true_commence = base + timedelta(hours=24)

        # Safe divergent: snapshot says the true game is 24h after stored commence.
        event_safe = Event(
            id="event_safe_e2e",
            sport_key="baseball_mlb",
            sport_title="MLB",
            commence_time=base,
            home_team="Safe Home",
            away_team="Safe Away",
            status=EventStatus.SCHEDULED,
        )
        snap_safe_time = base - timedelta(hours=5)
        snap_safe = OddsSnapshot(
            event_id="event_safe_e2e",
            snapshot_time=snap_safe_time,
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=29.0,  # implies commence = base + 24h
        )
        odds_safe = Odds(
            event_id="event_safe_e2e",
            bookmaker_key="bet365",
            bookmaker_title="Bet365",
            market_key="h2h",
            outcome_name="Safe Home",
            price=-110,
            odds_timestamp=snap_safe_time,
            last_update=snap_safe_time,
        )
        # PaperTrade on the safe event: should follow the snapshot repoint.
        trade_safe = PaperTrade(
            event_id="event_safe_e2e",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=-110,
            stake=10.0,
            bankroll_before=1000.0,
        )

        # Target event at the true commence: the safe snapshot should be repointed here.
        event_target = Event(
            id="event_target",
            sport_key="baseball_mlb",
            sport_title="MLB",
            commence_time=true_commence,
            home_team="Safe Home",
            away_team="Safe Away",
            status=EventStatus.SCHEDULED,
        )

        # Duplicate-pair event: must be skipped at the snapshot level AND
        # held back from the paper_trades repoint.
        event_dup = Event(
            id="event_dup_e2e",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=base,
            home_team="Dup Home",
            away_team="Dup Away",
            status=EventStatus.SCHEDULED,
        )
        snap_dup_time = base - timedelta(hours=4)
        snap_dup_1 = OddsSnapshot(
            event_id="event_dup_e2e",
            snapshot_time=snap_dup_time,
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=28.0,
        )
        snap_dup_2 = OddsSnapshot(
            event_id="event_dup_e2e",
            snapshot_time=snap_dup_time,
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=52.0,
        )
        # Odds row tied to the offending (event, snapshot_time) pair: must NOT move.
        odds_dup_offending = Odds(
            event_id="event_dup_e2e",
            bookmaker_key="bet365",
            bookmaker_title="Bet365",
            market_key="h2h",
            outcome_name="Dup Home",
            price=-120,
            odds_timestamp=snap_dup_time,
            last_update=snap_dup_time,
        )
        # PaperTrade on the offending event: must NOT move even though the
        # sibling safe snapshot below causes a snapshot-level repoint.
        trade_dup = PaperTrade(
            event_id="event_dup_e2e",
            market="h2h",
            selection="home",
            bookmaker="bet365",
            odds=-120,
            stake=15.0,
            bankroll_before=1000.0,
        )
        # MatchBrief on the offending event: must also NOT move.
        brief_dup = MatchBrief(
            event_id="event_dup_e2e",
            decision=BriefDecision.WATCHING,
            summary="Dup event — awaiting manual review",
            brief_text="Do not move me.",
        )

        # Sibling safe snapshot on the offending event: different snapshot_time,
        # unique pair, so it is in the safe divergent set. Its repoint should
        # succeed at the snapshot level but MUST NOT trigger a paper_trades
        # move on event_dup_e2e (MUST-FIX).
        snap_dup_sibling_time = base - timedelta(hours=10)
        snap_dup_sibling = OddsSnapshot(
            event_id="event_dup_e2e",
            snapshot_time=snap_dup_sibling_time,
            raw_data={},
            bookmaker_count=1,
            hours_until_commence=34.0,  # implies commence = base + 24h
        )
        # Target event for the sibling snapshot's implied true_commence.
        event_dup_target = Event(
            id="event_dup_target",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=true_commence,
            home_team="Dup Home",
            away_team="Dup Away",
            status=EventStatus.SCHEDULED,
        )

        # Flush events first so FK checks on the child rows pass — SQLAlchemy's
        # unit-of-work ordering can still produce a child INSERT before the
        # parent if the objects are added in a single batch with cross-mapper
        # cycles, so force the order explicitly.
        pglite_async_session.add_all([event_safe, event_target, event_dup, event_dup_target])
        await pglite_async_session.flush()
        pglite_async_session.add_all(
            [
                snap_safe,
                snap_dup_1,
                snap_dup_2,
                snap_dup_sibling,
                odds_safe,
                odds_dup_offending,
                trade_safe,
                trade_dup,
                brief_dup,
            ]
        )
        await pglite_async_session.commit()

        # Capture IDs before switching to a sync connection — the async
        # session's identity map is invalidated by the mutations below.
        safe_snap_id = snap_safe.id
        dup_snap_1_id = snap_dup_1.id
        dup_snap_2_id = snap_dup_2.id
        dup_sibling_snap_id = snap_dup_sibling.id
        safe_odds_id = odds_safe.id
        dup_odds_id = odds_dup_offending.id
        safe_trade_id = trade_safe.id
        dup_trade_id = trade_dup.id
        dup_brief_id = brief_dup.id
        assert safe_snap_id is not None
        assert dup_snap_1_id is not None
        assert dup_snap_2_id is not None
        assert dup_sibling_snap_id is not None
        assert safe_odds_id is not None
        assert dup_odds_id is not None
        assert safe_trade_id is not None
        assert dup_trade_id is not None
        assert dup_brief_id is not None

        # Drive upgrade() against a sync connection bridged from the async
        # pglite engine via run_sync — op.get_bind() is the only alembic
        # coupling, so patch it to return the sync connection we hand in.
        engine = pglite_async_session.bind
        assert engine is not None

        def _run_upgrade(sync_conn: object) -> None:
            with patch.object(migration_module.op, "get_bind", return_value=sync_conn):
                migration_module.upgrade()

        async with engine.begin() as async_conn:
            await async_conn.run_sync(_run_upgrade)

        captured = capsys.readouterr().out
        assert "Skipping 1 (event_id, snapshot_time) pairs" in captured, captured
        # Two safe rows now: event_safe_e2e's snapshot + event_dup_e2e's sibling.
        assert "Repointing 2 snapshots" in captured, captured
        # MUST-FIX log: event_dup_e2e was held back from paper_trades repoint.
        assert "Skipping paper_trades/match_briefs repoint for 1 events" in captured, captured

        # Verify final state via raw SQL against a fresh connection — the
        # original session's ORM cache is stale after the sync UPDATE.
        from sqlalchemy import text as sql_text

        async with engine.connect() as verify_conn:
            # Safe snapshot + odds + trade all follow to event_target.
            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM odds_snapshots WHERE id = :id"),
                {"id": safe_snap_id},
            )
            assert result.scalar() == "event_target"

            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM odds WHERE id = :id"),
                {"id": safe_odds_id},
            )
            assert result.scalar() == "event_target"

            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM paper_trades WHERE id = :id"),
                {"id": safe_trade_id},
            )
            assert result.scalar() == "event_target"

            # Offending snapshots stay on event_dup_e2e.
            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM odds_snapshots WHERE id IN (:id1, :id2)"),
                {"id1": dup_snap_1_id, "id2": dup_snap_2_id},
            )
            assert {row[0] for row in result.fetchall()} == {"event_dup_e2e"}

            # Odds row tied to the offending pair stays on event_dup_e2e
            # (unchanged — _REPOINT_ODDS never fires for this pair).
            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM odds WHERE id = :id"),
                {"id": dup_odds_id},
            )
            assert result.scalar() == "event_dup_e2e"

            # Sibling safe snapshot on event_dup_e2e gets repointed.
            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM odds_snapshots WHERE id = :id"),
                {"id": dup_sibling_snap_id},
            )
            assert result.scalar() == "event_dup_target"

            # MUST-FIX: PaperTrade on event_dup_e2e must NOT be moved, even
            # though the sibling snapshot repointed off event_dup_e2e.
            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM paper_trades WHERE id = :id"),
                {"id": dup_trade_id},
            )
            assert result.scalar() == "event_dup_e2e"

            # Same rule for MatchBrief.
            result = await verify_conn.execute(
                sql_text("SELECT event_id FROM match_briefs WHERE id = :id"),
                {"id": dup_brief_id},
            )
            assert result.scalar() == "event_dup_e2e"
