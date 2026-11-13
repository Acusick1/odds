"""Stored backtest runs: what each producer writes and how a re-store replaces a run."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
from odds_analytics import backtest_store as store
from odds_analytics import candidate_backtest as bt
from odds_analytics import cell_selection as cs
from odds_analytics.candidate_report import ScoredBet
from odds_analytics.holdout import FIVE_LEAGUES, Preregistration
from odds_core.candidate_backtest_models import (
    CandidateBacktest,
    CandidateBacktestBet,
    WindowKind,
)
from odds_core.vocabulary import Source
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlmodel import col

from tests.unit.test_holdout import draft, replayed

pytestmark = pytest.mark.integration

NOW = datetime(2026, 11, 1, tzinfo=UTC)
SHA = "a" * 40
START, END = date(2025, 1, 1), date(2025, 6, 30)


def bet(event_id: int, ev: float | None, minutes: int = 0, pnl: float | None = 1.0) -> ScoredBet:
    when = datetime(2025, 3, 1, 12, tzinfo=UTC) + timedelta(minutes=minutes)
    return ScoredBet(
        "back-longest",
        1,
        event_id,
        when + timedelta(hours=2),
        1.0,
        pnl,
        2.0,
        None if ev is None else (1 + ev) / 2.0,
        when,
        timedelta(0),
        "soccer_spain_la_liga",
        "home",
        None,
        "bet365",
        None,
        None if ev is None else Source.BETFAIR,
    )


def result(bets: list[ScoredBet]) -> bt.BacktestResult:
    return bt.score(draft(), START, END, replayed(bets), 1, NOW, 200, 0)


async def runs(session: Any) -> list[CandidateBacktest]:
    return list((await session.execute(select(CandidateBacktest).order_by("id"))).scalars())


async def stored_bets(session: Any, run: CandidateBacktest) -> list[CandidateBacktestBet]:
    found = await session.execute(
        select(CandidateBacktestBet)
        .where(col(CandidateBacktestBet.backtest_id) == run.id)
        .order_by("position")
    )
    return list(found.scalars())


class TestReplayRecord:
    async def test_stores_the_run_with_its_estimate_and_bets_in_placement_order(
        self, pglite_async_session: Any
    ) -> None:
        bets = [bet(i, 0.04 if i % 2 else 0.02, minutes=60 - i) for i in range(1, 41)]
        record = store.replay_record(draft(), result(bets), WindowKind.DEVELOPMENT, SHA)

        await store.store_runs(pglite_async_session, [record])

        (run,) = await runs(pglite_async_session)
        overall = result(bets).overall
        assert overall is not None
        assert (run.candidate_name, run.candidate_version, run.window_kind) == (
            "back-longest",
            1,
            WindowKind.DEVELOPMENT,
        )
        assert (run.start_date, run.end_date, run.leagues) == (START, END, ["soccer_spain_la_liga"])
        assert (run.bets, run.bets_with_close, run.events, run.stake, run.pnl) == (
            40,
            40,
            40,
            40.0,
            40.0,
        )
        assert run.mean_ev == pytest.approx(overall.mean_ev) == pytest.approx(0.03)
        assert (run.ci_lower, run.ci_upper) == tuple(overall.ci95)
        assert run.ci_level == 0.95
        assert (run.cell_index, run.eligible, run.selected) == (None, None, False)
        rows = await stored_bets(pglite_async_session, run)
        assert [r.position for r in rows] == list(range(1, 41))
        assert [r.event_id for r in rows] == list(range(40, 0, -1))
        assert rows[0].price == 2.0 and rows[0].close_source is Source.BETFAIR
        assert rows[0].ev == pytest.approx(0.02)

    async def test_a_run_with_no_bet_with_a_close_has_no_estimate(
        self, pglite_async_session: Any
    ) -> None:
        record = store.replay_record(
            draft(), result([bet(1, None, pnl=-1.0)]), WindowKind.DEVELOPMENT, SHA
        )

        await store.store_runs(pglite_async_session, [record])

        (run,) = await runs(pglite_async_session)
        assert (run.bets, run.bets_with_close, run.pnl) == (1, 0, -1.0)
        assert (run.mean_ev, run.ci_lower, run.ci_upper, run.ci_level) == (None,) * 4
        (row,) = await stored_bets(pglite_async_session, run)
        assert (row.p_close, row.ev, row.close_source) == (None, None, None)

    async def test_storing_again_replaces_the_run_and_its_bets(
        self, pglite_async_session: Any
    ) -> None:
        first = store.replay_record(
            draft(), result([bet(i, 0.02) for i in range(1, 11)]), WindowKind.DEVELOPMENT, SHA
        )
        await store.store_runs(pglite_async_session, [first])
        again = store.replay_record(
            draft(), result([bet(i, 0.05) for i in range(1, 6)]), WindowKind.DEVELOPMENT, SHA
        )

        await store.store_runs(pglite_async_session, [again])

        (run,) = await runs(pglite_async_session)
        assert run.bets == 5 and run.mean_ev == pytest.approx(0.05)
        count = await pglite_async_session.scalar(select(func.count(CandidateBacktestBet.id)))
        assert count == 5

    async def test_another_commit_or_range_is_another_run(self, pglite_async_session: Any) -> None:
        bets = result([bet(1, 0.02)])
        await store.store_runs(
            pglite_async_session,
            [store.replay_record(draft(), bets, WindowKind.DEVELOPMENT, SHA)],
        )
        await store.store_runs(
            pglite_async_session,
            [store.replay_record(draft(), bets, WindowKind.DEVELOPMENT, "b" * 40)],
        )

        assert len(await runs(pglite_async_session)) == 2

    async def test_the_database_refuses_a_second_run_with_the_same_identity(
        self, pglite_async_session: Any
    ) -> None:
        record = store.replay_record(draft(), result([bet(1, 0.02)]), WindowKind.DEVELOPMENT, SHA)
        await store.store_runs(pglite_async_session, [record])
        twin = store.replay_record(draft(), result([bet(1, 0.02)]), WindowKind.DEVELOPMENT, SHA)

        pglite_async_session.add(twin.run)
        with pytest.raises(IntegrityError):
            await pglite_async_session.flush()


PREREG = Preregistration(
    issue=900,
    name="small-grid",
    version=1,
    rule="exchange_gap",
    market="1x2",
    leagues=["soccer_epl", "soccer_spain_la_liga"],
    grid={"threshold": [0.02, 0.05], "minutes_before_kickoff": [[60]]},
    selection_rule=cs.ROUND_ONE_RULE,
)


def cell_bets(ev: float, n: int) -> tuple[ScoredBet, ...]:
    return tuple(bet(i, ev, minutes=i) for i in range(1, n + 1))


def selection_of(*cells: tuple[float, int]) -> cs.GridSelection:
    """A grid whose cells have ``(ev, bets)`` in the EPL 2024-26 block."""
    blocks = tuple(cs.development_blocks(PREREG.leagues))
    replays = [
        cs.CellReplay(
            i,
            {"threshold": 0.02 * (i + 1)},
            tuple((b, cell_bets(ev, n) if b.name == "EPL 2024-26" else ()) for b in blocks),
        )
        for i, (ev, n) in enumerate(cells)
    ]
    scores = tuple(cs.score_cell(r, False) for r in replays)
    return cs.GridSelection(
        PREREG, "f" * 64, blocks, tuple(replays), scores, cs.pick(scores), longshots=False
    )


class TestGridRecords:
    async def test_stores_every_cell_and_the_picked_cells_bets_only(
        self, pglite_async_session: Any
    ) -> None:
        grid = selection_of((0.02, 60), (0.06, 55), (-0.01, 80))
        assert grid.picked is not None and grid.picked.index == 1

        await store.store_runs(
            pglite_async_session, store.grid_records(grid, SHA, date(2026, 11, 1))
        )

        stored = await runs(pglite_async_session)
        assert [(r.cell_index, r.selected, r.eligible) for r in stored] == [
            (0, False, True),
            (1, True, True),
            (2, False, False),
        ]
        picked = stored[1]
        assert (picked.window_kind, picked.ci_level) == (WindowKind.DEVELOPMENT, 0.9)
        assert picked.params == {"threshold": 0.04}
        assert (picked.start_date, picked.end_date) == (date(2022, 8, 1), date(2026, 7, 31))
        assert picked.leagues == ["soccer_epl", "soccer_spain_la_liga"]
        assert picked.mean_ev == pytest.approx(0.06)
        assert grid.picked.lower_bound == pytest.approx(picked.ci_lower)
        assert grid.picked.upper_bound == pytest.approx(picked.ci_upper)
        assert [(b["name"], b["bets"]) for b in picked.blocks or []] == [
            ("EPL 2022-24", 0),
            ("EPL 2024-26", 55),
            ("five leagues 2024-25", 0),
        ]
        assert [len(await stored_bets(pglite_async_session, r)) for r in stored] == [0, 55, 0]
        assert [r.bets for r in stored] == [60, 55, 80]

    async def test_a_killed_grid_selects_nothing_and_keeps_no_bets(
        self, pglite_async_session: Any
    ) -> None:
        grid = selection_of((-0.02, 60), (-0.03, 70))
        assert grid.picked is None

        await store.store_runs(
            pglite_async_session, store.grid_records(grid, SHA, date(2026, 11, 1))
        )

        stored = await runs(pglite_async_session)
        assert [r.selected for r in stored] == [False, False]
        count = await pglite_async_session.scalar(select(func.count(CandidateBacktestBet.id)))
        assert count == 0

    async def test_storing_the_grid_again_replaces_each_cell(
        self, pglite_async_session: Any
    ) -> None:
        for _ in range(2):
            await store.store_runs(
                pglite_async_session,
                store.grid_records(selection_of((0.02, 60), (0.06, 55)), SHA, date(2026, 11, 1)),
            )

        assert len(await runs(pglite_async_session)) == 2
        count = await pglite_async_session.scalar(select(func.count(CandidateBacktestBet.id)))
        assert count == 55


class TestHoldoutRecord:
    async def test_the_holdout_run_is_the_selected_cell_on_the_sealed_windows(
        self, pglite_async_session: Any
    ) -> None:
        from odds_analytics.holdout import windows_of
        from odds_analytics.holdout_run import SpentReplay

        from tests.unit.test_holdout import result_row

        windows = [windows_of(s)[0] for s in FIVE_LEAGUES[:2]]
        pooled = bt.score(
            draft(),
            windows[0].start,
            windows[0].end,
            replayed([bet(i, 0.03) for i in range(1, 31)]),
            1,
            NOW,
            200,
            0,
        )

        await store.store_runs(
            pglite_async_session,
            [store.holdout_record(SpentReplay(result_row(), draft(), windows, pooled), SHA)],
        )

        (run,) = await runs(pglite_async_session)
        assert (run.window_kind, run.selected, run.cell_index) == (WindowKind.HOLDOUT, True, None)
        assert (run.start_date, run.end_date) == (date(2025, 8, 1), date(2026, 7, 31))
        assert run.bets == 30 and run.ci_level == 0.95


class TestReaders:
    """What the web pages read back from the stored runs."""

    async def test_a_selected_cells_runs_come_back_with_their_series_and_the_killed_grid_does_not(
        self, pglite_async_session: Any
    ) -> None:
        from odds_analytics.holdout import windows_of
        from odds_analytics.holdout_run import SpentReplay
        from odds_web import backtests as web

        from tests.unit.test_holdout import result_row

        session = pglite_async_session
        await store.store_runs(
            session,
            store.grid_records(selection_of((0.02, 60), (0.06, 55)), SHA, date(2026, 11, 1)),
        )
        killed = selection_of((-0.02, 60), (-0.03, 70))
        for record in store.grid_records(killed, SHA, date(2026, 11, 1)):
            record.run.candidate_name = "dead-idea"
            await store.store_runs(session, [record])
        windows = [windows_of("soccer_spain_la_liga")[0]]
        pooled = bt.score(
            draft(),
            windows[0].start,
            windows[0].end,
            replayed([bet(i, 0.03) for i in range(1, 31)]),
            1,
            NOW,
            200,
            0,
        )
        await store.store_runs(
            session,
            [store.holdout_record(SpentReplay(result_row(), draft(), windows, pooled), SHA)],
        )
        # The holdout run is of "back-longest"; the grid's runs are of "small-grid".
        holdout_run = (
            await session.execute(
                select(CandidateBacktest).where(
                    col(CandidateBacktest.window_kind) == WindowKind.HOLDOUT
                )
            )
        ).scalar_one()

        page = await web.load_runs(session, "small-grid", 1)
        overview = await web.load_overview(session)

        assert [r.kind for r in page] == [WindowKind.DEVELOPMENT]
        development = page[0]
        assert development.bets == 55 and development.series is not None
        assert len(development.series.running_ev_percent) == 55
        assert [b.name for b in development.blocks][1] == "EPL 2024-26"
        assert [r.kind for r in await web.load_runs(session, holdout_run.candidate_name, 1)] == [
            WindowKind.HOLDOUT
        ]
        assert await web.load_runs(session, "dead-idea", 1) == []
        assert [(k.name, k.cells) for k in overview.killed] == [("dead-idea", 2)]
        assert overview.estimates[("small-grid", 1)].kind is WindowKind.DEVELOPMENT
        assert overview.estimates[("back-longest", 1)].kind is WindowKind.HOLDOUT
