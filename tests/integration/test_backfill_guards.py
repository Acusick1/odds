"""The probe and the run guards, on recorded historical responses replayed through the real
executor under the MLB manifest (10 units per call, Pinnacle required on h2h, fixtures needed
9 h ahead)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_analytics import backfill_executor
from odds_analytics.backfill_checks import ResponseCheck, book_judged_lead
from odds_analytics.backfill_executor import BackfillExecutor, BackfillStopped
from odds_analytics.backfill_plan import GridStep, Manifest
from odds_core.backfill_models import OddsApiSnapshotGap
from odds_core.models import FetchLog, OddsSnapshot
from odds_core.sports import SportKey
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.backfill_replay_helpers import (
    AFTERNOON,
    CARDINALS,
    CUBS,
    MLB,
    BackfillReplay,
    count,
    load_manifest,
    load_mlb,
)
from tests.test_helpers import (
    FakeMlbStatsFetcher,
    odds_api_game,
    odds_api_h2h,
)

STAMP = "2025-06-02T12:00:00Z"
FIRST_PROBE = datetime(2024, 7, 20, 15, 55, 38, tzinfo=UTC)
SECOND_PROBE = datetime(2026, 7, 18, 15, 55, 38, tzinfo=UTC)


def game(api_id: str, start: datetime, *books: str) -> dict[str, Any]:
    """An MLB listing quoting h2h at each of ``books``."""
    markets = [odds_api_h2h({CUBS: -120, CARDINALS: 100}, STAMP)]
    return odds_api_game(
        api_id, MLB, "MLB", CUBS, CARDINALS, start, dict.fromkeys(books, markets), STAMP
    )


def far_game(snapshot: datetime, *books: str, lead_hours: int = 10) -> dict[str, Any]:
    """A listing no stored event takes, which the probe still judges."""
    return game(f"far-{snapshot:%Y%m%d}", snapshot + timedelta(hours=lead_hours), *books)


# A planned MLB grid snapshot two hours before the afternoon game.
PLANNED = datetime(2025, 6, 2, 14, 55, 38, tzinfo=UTC)


def response_check(manifest: Manifest, planned: datetime, returned: datetime) -> ResponseCheck:
    return ResponseCheck(
        manifest=manifest,
        modelled=planned,
        snapshot_time=returned,
        units=10,
        events=1,
        previous_gap=None,
        next_gap=None,
        farthest_lead=None,
        book_events={},
        book_leads={},
    )


def grid_times(count_: int) -> list[datetime]:
    """Consecutive MLB grid snapshots from ``PLANNED``."""
    return [PLANNED + timedelta(minutes=5 * i) for i in range(count_)]


def times(count_: int) -> list[datetime]:
    return [AFTERNOON - timedelta(hours=2, minutes=5 * i) for i in range(count_)]


class TestRunGuards:
    async def test_a_call_billed_other_units_stops_the_run_after_storing_it(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        games = await load_mlb(session, mlb_schedule)
        first, second = times(2)
        for at in (first, second):
            replay.record(at, [g.listing(at) for g in games], billed=35)

        with pytest.raises(BackfillStopped, match="billed 35 units, expected 10") as e:
            await replay.run(
                manifest,
                [replay.call(manifest, at, games[0].event) for at in (first, second)],
                skip_existing=False,
            )

        assert len(replay.requests) == 1
        assert e.value.result.units_used == 35
        assert await count(session, OddsSnapshot) > 0

    async def test_a_call_with_no_billed_units_stops_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        at = times(1)[0]
        replay.record(at, [g.listing(at) for g in games], bill_header=False)

        with pytest.raises(BackfillStopped, match="billed None units"):
            await replay.run(manifest, [replay.call(manifest, at, games[0].event)])

    async def test_a_call_billed_the_manifests_units_does_not_stop_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at in times(3):
            replay.record(at, [g.listing(at) for g in games])
            calls.append(replay.call(manifest, at, games[0].event))

        result = await replay.run(manifest, calls, skip_existing=False)

        assert (result.calls_made, result.units_used) == (3, 30)

    @pytest.mark.parametrize(("cap", "made"), [(20, 2), (29, 2), (30, 3)])
    async def test_the_run_stops_before_the_call_that_would_pass_max_units(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        cap: int,
        made: int,
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at in times(3):
            replay.record(at, [g.listing(at) for g in games])
            calls.append(replay.call(manifest, at, games[0].event))

        if made == len(calls):
            await replay.run(manifest, calls, skip_existing=False, max_units=cap)
        else:
            with pytest.raises(BackfillStopped, match=f"--max-units {cap}") as stop:
                await replay.run(manifest, calls, skip_existing=False, max_units=cap)
            assert stop.value.result.units_used == 10 * made
        assert len(replay.requests) == made

    async def test_a_call_returning_another_snapshot_stops_the_run_after_storing_it(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        games = await load_mlb(session, mlb_schedule)
        first, second = PLANNED, PLANNED + timedelta(minutes=5)
        early = first - timedelta(seconds=39)
        replay.record(first, [g.listing(early) for g in games], snapshot=early)
        replay.record(second, [g.listing(second) for g in games])

        with pytest.raises(
            BackfillStopped,
            match="call for the 2025-06-02T14:55:38Z snapshot returned the "
            "2025-06-02T14:54:59Z snapshot",
        ) as e:
            await replay.run(
                manifest, [replay.call(manifest, at, games[0].event) for at in (first, second)]
            )

        assert len(replay.requests) == 1
        assert e.value.result.fixtures_stored == len(games)
        assert await count(session, OddsSnapshot) > 0

    @pytest.mark.parametrize(
        ("offset", "returned"),
        [
            (timedelta(minutes=2, seconds=30), "2025-06-02T14:53:08Z"),
            (timedelta(minutes=10, seconds=-61), "2025-06-02T14:46:39Z"),
            (timedelta(minutes=-5), "2025-06-02T15:00:38Z"),
        ],
    )
    async def test_a_call_returning_an_off_grid_or_later_snapshot_stops_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        offset: timedelta,
        returned: str,
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        games = await load_mlb(session, mlb_schedule)
        taken = PLANNED - offset
        replay.record(PLANNED, [g.listing(taken) for g in games], snapshot=taken)

        with pytest.raises(BackfillStopped, match=f"returned the {returned} snapshot"):
            await replay.run(manifest, [replay.call(manifest, PLANNED, games[0].event)])

        assert await count(session, OddsApiSnapshotGap) == 0

    @pytest.mark.parametrize("offset_seconds", [-38, 0, 4, 22])
    async def test_a_call_returning_its_snapshot_with_jitter_does_not_stop_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        offset_seconds: int,
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at in (PLANNED, PLANNED + timedelta(minutes=5)):
            taken = at + timedelta(seconds=offset_seconds)
            replay.record(at, [g.listing(taken) for g in games], snapshot=taken)
            calls.append(replay.call(manifest, at, games[0].event))

        result = await replay.run(manifest, calls)

        assert result.calls_made == 2

    async def test_consecutive_empty_calls_stop_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at in times(4):
            replay.record(at, [])
            calls.append(replay.call(manifest, at))

        with pytest.raises(BackfillStopped, match="2 consecutive sparse responses.*no events"):
            await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=2)

        assert len(replay.requests) == 2

    async def test_consecutive_calls_lacking_a_required_book_stop_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at in times(3):
            replay.record(at, [game("mlb-901", AFTERNOON, "williamhill")])
            calls.append(replay.call(manifest, at, games[0].event))

        with pytest.raises(BackfillStopped, match="lacks betfair_exchange on h2h"):
            await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=3)

        assert len(replay.requests) == 3

    async def test_books_are_not_judged_on_a_call_serving_only_distant_fixtures(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for i in range(3):
            at = AFTERNOON - timedelta(days=3, minutes=5 * i)
            replay.record(at, [game("mlb-901", AFTERNOON, "williamhill")])
            calls.append(replay.call(manifest, at, games[0].event))

        result = await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=2)

        assert result.calls_made == 3

    async def test_a_call_storing_none_of_its_fixtures_counts_toward_the_streak(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at in times(4):
            replay.record(at, [far_game(at, "betfair_ex_uk")])
            calls.append(replay.call(manifest, at, games[0].event))

        with pytest.raises(BackfillStopped, match="stored none of the 1 fixtures it serves") as e:
            await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=2)

        assert len(replay.requests) == 2
        assert e.value.result.fixtures_stored == 0

    async def test_an_empty_response_bills_nothing_and_counts_as_sparse(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at, header in zip(times(3), (True, False, True), strict=True):
            replay.record(at, [], bill_header=header)
            calls.append(replay.call(manifest, at))

        with pytest.raises(BackfillStopped, match="3 consecutive sparse responses.*no events") as e:
            await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=3)

        assert e.value.result.units_used == 0

    @pytest.mark.parametrize("billed", [0, 10])
    async def test_an_empty_response_billed_nothing_or_the_manifests_units_does_not_stop_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        billed: int,
    ) -> None:
        manifest = load_manifest(MLB)
        await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at in times(2):
            replay.record(at, [], billed=billed)
            calls.append(replay.call(manifest, at))

        with pytest.raises(BackfillStopped, match="2 consecutive sparse responses.*no events"):
            await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=2)

        assert len(replay.requests) == 2

    async def test_an_empty_response_billed_other_units_stops_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        await load_mlb(pglite_async_session, mlb_schedule)
        at = times(1)[0]
        replay.record(at, [], billed=35)

        with pytest.raises(BackfillStopped, match="billed 35 units, expected 0 or 10"):
            await replay.run(manifest, [replay.call(manifest, at)], skip_existing=False)


class TestProviderGaps:
    async def test_a_gap_is_stored_recorded_and_the_run_continues(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        games = await load_mlb(session, mlb_schedule)
        later = PLANNED + timedelta(minutes=5)
        earlier = PLANNED - timedelta(minutes=10, seconds=1)
        replay.record(
            PLANNED,
            [g.listing(earlier) for g in games],
            snapshot=earlier,
            next_gap=timedelta(minutes=5),
        )
        replay.record(later, [g.listing(later) for g in games])
        calls = [replay.call(manifest, at, games[0].event) for at in (PLANNED, later)]

        result = await replay.run(manifest, calls)

        assert (result.calls_made, result.provider_gaps) == (2, 1)
        times = await session.execute(select(OddsSnapshot.snapshot_time).distinct())
        assert set(times.scalars()) == {earlier, later}
        gap = (await session.execute(select(OddsApiSnapshotGap))).scalars().one()
        assert (gap.sport_key, gap.market, gap.planned_snapshot) == (MLB, "h2h", PLANNED)
        assert (gap.returned_snapshot, gap.next_snapshot) == (
            earlier,
            earlier + timedelta(minutes=5),
        )

    async def test_a_rerun_skips_a_recorded_gap_without_a_request(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        earlier = PLANNED - timedelta(minutes=15)
        replay.record(PLANNED, [g.listing(earlier) for g in games], snapshot=earlier)
        call = replay.call(manifest, PLANNED, games[0].event)
        await replay.run(manifest, [call])

        rerun = await replay.run(manifest, [call])

        assert (rerun.calls_made, rerun.calls_skipped) == (0, 1)
        assert len(replay.requests) == 1

    async def test_a_call_inside_a_recorded_gap_is_skipped_and_one_after_it_is_made(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        earlier = PLANNED - timedelta(minutes=55)
        # The provider has nothing from ``earlier`` until an hour later.
        replay.record(
            PLANNED,
            [g.listing(earlier) for g in games],
            snapshot=earlier,
            next_gap=timedelta(hours=1),
        )
        inside, after = PLANNED + timedelta(minutes=2), PLANNED + timedelta(minutes=5)
        replay.record(after, [g.listing(after) for g in games])
        calls = [replay.call(manifest, at, games[0].event) for at in (PLANNED, inside, after)]

        result = await replay.run(manifest, calls)

        assert (result.calls_made, result.calls_skipped, result.provider_gaps) == (2, 1, 1)
        assert len(replay.requests) == 2

    @pytest.mark.parametrize(
        ("inside", "next_gap"),
        [
            # The next snapshot comes before the gap call's own request time.
            (PLANNED - timedelta(minutes=10), timedelta(minutes=50)),
            # The next snapshot is off the grid's phase.
            (PLANNED + timedelta(minutes=5), timedelta(hours=1, minutes=2, seconds=30)),
        ],
    )
    async def test_an_inconsistent_next_snapshot_skips_only_the_planned_call(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        inside: datetime,
        next_gap: timedelta,
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        earlier = PLANNED - timedelta(minutes=55)
        replay.record(
            PLANNED, [g.listing(earlier) for g in games], snapshot=earlier, next_gap=next_gap
        )
        replay.record(inside, [g.listing(inside) for g in games])
        calls = [replay.call(manifest, at, games[0].event) for at in (PLANNED, inside)]
        await replay.run(manifest, calls[:1])

        result = await replay.run(manifest, calls)

        assert (result.calls_made, result.calls_skipped) == (1, 1)
        assert len(replay.requests) == 2

    async def test_a_check_that_raises_keeps_the_paid_calls_rows_and_gap(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        games = await load_mlb(session, mlb_schedule)
        earlier = PLANNED - timedelta(minutes=10)
        replay.record(PLANNED, [g.listing(earlier) for g in games], snapshot=earlier)
        call = replay.call(manifest, PLANNED, games[0].event)

        def fail(*args: Any, **kwargs: Any) -> ResponseCheck:
            raise KeyError("key")

        with monkeypatch.context() as patched:
            patched.setattr(backfill_executor, "check_response", fail)
            with pytest.raises(KeyError):
                await replay.run(manifest, [call])

        assert await count(session, OddsSnapshot) > 0
        assert await count(session, OddsApiSnapshotGap) == 1
        rerun = await replay.run(manifest, [call])
        assert (rerun.calls_made, rerun.calls_skipped) == (0, 1)
        assert len(replay.requests) == 1

    async def test_the_returned_snapshots_own_grid_time_is_not_inside_the_gap(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        own = PLANNED - timedelta(minutes=10)
        returned = own - timedelta(seconds=1)
        replay.record(
            PLANNED,
            [games[0].listing(returned)],
            snapshot=returned,
            next_gap=timedelta(minutes=15),
        )
        replay.record(own, [g.listing(returned) for g in games], snapshot=returned)
        await replay.run(manifest, [replay.call(manifest, PLANNED, games[0].event)])

        result = await replay.run(manifest, [replay.call(manifest, own, games[1].event)])

        assert (result.calls_made, result.calls_skipped) == (1, 0)

    async def test_consecutive_gaps_stop_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        games = await load_mlb(session, mlb_schedule)
        calls = []
        for at in grid_times(3):
            earlier = at - timedelta(minutes=10)
            replay.record(at, [g.listing(earlier) for g in games], snapshot=earlier)
            calls.append(replay.call(manifest, at, games[0].event))

        with pytest.raises(BackfillStopped, match="2 consecutive provider gaps") as e:
            await replay.run(manifest, calls, max_consecutive_provider_gaps=2)

        assert len(replay.requests) == 2
        assert e.value.result.provider_gaps == 2
        assert await count(session, OddsApiSnapshotGap) == 2

    async def test_a_good_call_resets_the_gap_streak(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        games = await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for i, at in enumerate(grid_times(3)):
            taken = at if i == 1 else at - timedelta(minutes=10)
            replay.record(at, [g.listing(taken) for g in games], snapshot=taken)
            calls.append(replay.call(manifest, at, games[0].event))

        result = await replay.run(manifest, calls, max_consecutive_provider_gaps=2)

        assert (result.calls_made, result.provider_gaps) == (3, 2)


class TestProbe:
    async def test_the_legacy_betfair_key_counts_for_the_required_exchange(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        for at in (FIRST_PROBE, SECOND_PROBE):
            replay.record(at, [far_game(at, "betfair", "williamhill")])

        checks = await replay.probe(manifest)

        assert [c.ok for c in checks] == [True, True]

    async def test_a_conforming_response_passes_every_check(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        for at in (FIRST_PROBE, SECOND_PROBE):
            replay.record(at, [far_game(at, "betfair_ex_uk", "williamhill")])

        checks = await replay.probe(manifest)

        assert [c.ok for c in checks] == [True, True]
        first = checks[0]
        assert first.units == 10
        assert first.lag == timedelta(0)
        assert (first.previous_gap, first.next_gap) == (timedelta(minutes=5),) * 2
        assert first.farthest_lead == timedelta(hours=10)
        assert first.book_events == {("h2h", "betfair_exchange"): 1}
        assert first.book_leads == {("h2h", "betfair_exchange"): timedelta(hours=10)}
        assert sorted(r["date"] for r in replay.requests) == [
            "2024-07-20T15:58:08Z",
            "2026-07-18T15:58:08Z",
        ]

    async def test_a_probe_stores_what_it_returns_as_a_run_does(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        games = await load_mlb(session, mlb_schedule)
        for at in (FIRST_PROBE, SECOND_PROBE):
            replay.record(at, [games[0].listing(at)])

        await replay.probe(manifest)

        assert await count(session, OddsSnapshot) > 0

    @pytest.mark.parametrize(
        ("field", "setup"),
        [
            ("units_ok", {"billed": 35}),
            ("units_ok", {"bill_header": False}),
            ("lag_ok", {"lag_seconds": 39}),
            ("lag_ok", {"lag_seconds": -23}),
            ("lag_ok", {"lag_seconds": 300}),
            ("grid_ok", {"previous_gap": timedelta(minutes=6, seconds=1)}),
            ("grid_ok", {"next_gap": timedelta(minutes=3, seconds=59)}),
            ("grid_ok", {"previous_gap": timedelta(minutes=10)}),
            ("grid_ok", {"next_gap": timedelta(minutes=10)}),
            ("grid_ok", {"previous_gap": None}),
            ("grid_ok", {"next_gap": None}),
            ("lead_ok", {"lead_hours": 8}),
            ("books_ok", {"books": ("williamhill",)}),
        ],
    )
    async def test_each_check_fails_when_the_response_disagrees(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        field: str,
        setup: dict[str, Any],
    ) -> None:
        manifest = load_manifest(MLB)
        for at in (FIRST_PROBE, SECOND_PROBE):
            replay.record(
                at,
                [
                    far_game(
                        at,
                        *setup.get("books", ("betfair_ex_uk",)),
                        lead_hours=setup.get("lead_hours", 10),
                    )
                ],
                snapshot=at - timedelta(seconds=setup.get("lag_seconds", 0)),
                **{
                    k: v
                    for k, v in setup.items()
                    if k.endswith("_gap") or k in ("billed", "bill_header")
                },
            )

        checks = await replay.probe(manifest)

        for check in checks:
            assert getattr(check, field) is False
            assert check.ok is False

    async def test_a_required_book_not_listed_as_far_ahead_as_the_manifest_needs_fails_the_probe(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        for at in (FIRST_PROBE, SECOND_PROBE):
            listings = [
                far_game(at, "williamhill"),
                game("near", at + timedelta(hours=8), "betfair_ex_uk"),
            ]
            replay.record(at, listings)

        checks = await replay.probe(manifest)

        for check in checks:
            assert (check.books_ok, check.lead_ok, check.horizon_ok) == (True, True, False)
            assert check.book_leads[("h2h", "betfair_exchange")] == timedelta(hours=8)
            assert check.ok is False

    @pytest.mark.parametrize("lag_seconds", [38, 1, -4, -22])
    async def test_provider_timestamp_jitter_within_the_snapshots_minute_passes(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        lag_seconds: int,
    ) -> None:
        manifest = load_manifest(MLB)
        for at in (FIRST_PROBE, SECOND_PROBE):
            replay.record(
                at,
                [far_game(at, "betfair_ex_uk")],
                snapshot=at - timedelta(seconds=lag_seconds),
                previous_gap=timedelta(minutes=5, seconds=1),
                next_gap=timedelta(minutes=4, seconds=1),
            )

        checks = await replay.probe(manifest)

        assert [c.ok for c in checks] == [True, True]
        assert checks[0].lag == timedelta(seconds=lag_seconds)

    async def test_an_erroring_call_stops_the_probe_and_logs_the_failure(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        replay.record(FIRST_PROBE, [far_game(FIRST_PROBE, "betfair_ex_uk")], snapshot=FIRST_PROBE)

        with pytest.raises(AssertionError, match="no recorded response for 2026-07-18T15:58:08Z"):
            await replay.probe(manifest)

        assert len(replay.requests) == 1
        assert await count(session, FetchLog) == 2


class TestResponseCheck:
    @pytest.mark.parametrize(
        ("sport", "planned", "returned"),
        [
            ("soccer_epl", "2024-02-25T19:25:38Z", "2024-02-25T19:10:38Z"),
            ("soccer_france_ligue_one", "2024-11-30T02:40:38Z", "2024-11-30T02:35:38Z"),
            ("soccer_italy_serie_a", "2024-11-30T02:40:38Z", "2024-11-30T02:35:38Z"),
            (MLB, "2024-07-21T00:25:38Z", "2024-07-21T00:15:37Z"),
            ("soccer_germany_bundesliga", "2025-04-15T21:25:38Z", "2025-04-15T18:00:38Z"),
            ("soccer_efl_champ", "2025-04-15T18:55:38Z", "2025-04-15T18:00:38Z"),
            ("soccer_spain_la_liga", "2025-04-15T18:55:38Z", "2025-04-15T18:00:38Z"),
        ],
    )
    def test_the_providers_recorded_holes_are_provider_gaps(
        self, sport: SportKey, planned: str, returned: str
    ) -> None:
        check = response_check(
            load_manifest(sport), datetime.fromisoformat(planned), datetime.fromisoformat(returned)
        )

        assert (check.lag_ok, check.provider_gap) == (False, True)

    @pytest.mark.parametrize(
        ("lag", "gap"),
        [
            (timedelta(0), False),
            (timedelta(seconds=39), False),
            (timedelta(minutes=2, seconds=30), False),
            (timedelta(minutes=5, seconds=-60), True),
            (timedelta(minutes=5, seconds=60), True),
            (timedelta(minutes=5, seconds=61), False),
            (timedelta(minutes=-5), False),
        ],
    )
    def test_only_an_earlier_snapshot_on_the_grid_is_a_provider_gap(
        self, lag: timedelta, gap: bool
    ) -> None:
        check = response_check(load_manifest(MLB), PLANNED, PLANNED - lag)

        assert check.provider_gap is gap

    def test_the_grid_step_in_force_at_each_neighbour_is_the_expected_gap(self) -> None:
        boundary = datetime(2022, 9, 1, 0, 0, 38, tzinfo=UTC)
        epl = load_manifest("soccer_epl").model_copy(
            update={
                "snapshot_grid": (
                    GridStep(start=datetime(2020, 6, 6, 0, 0, 38, tzinfo=UTC), step_minutes=10),
                    GridStep(start=boundary, step_minutes=5),
                )
            }
        )

        def check(previous: int, nxt: int) -> ResponseCheck:
            return ResponseCheck(
                manifest=epl,
                modelled=boundary,
                snapshot_time=boundary,
                units=10,
                events=1,
                previous_gap=timedelta(minutes=previous),
                next_gap=timedelta(minutes=nxt),
                farthest_lead=timedelta(hours=174),
                book_events={},
                book_leads={},
            )

        assert check(10, 5).grid_ok
        assert not check(5, 5).grid_ok
        assert not check(10, 10).grid_ok

    @pytest.mark.parametrize(
        ("sport", "hours"),
        [("soccer_epl", 24), ("soccer_efl_champ", 24), (MLB, 9), ("basketball_nba", 9)],
    )
    def test_books_are_judged_as_far_ahead_as_the_manifest_needs_fixtures_up_to_a_day(
        self, sport: SportKey, hours: int
    ) -> None:
        assert book_judged_lead(load_manifest(sport)) == timedelta(hours=hours)


class TestGuardSettings:
    def test_a_streak_limit_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_gaps must be at least 1"):
            BackfillExecutor(load_manifest(MLB), max_consecutive_gaps=0)

    def test_a_provider_gap_limit_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_provider_gaps must be at least 1"):
            BackfillExecutor(load_manifest(MLB), max_consecutive_provider_gaps=0)
