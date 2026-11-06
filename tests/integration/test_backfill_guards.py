"""The probe and the run guards, on recorded historical responses replayed through the real
executor under the MLB manifest (20 units per call, Pinnacle required on both markets)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_analytics.backfill_checks import ResponseCheck
from odds_analytics.backfill_executor import BackfillExecutor, BackfillStopped
from odds_core.models import FetchLog, OddsSnapshot
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import (
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
    odds_api_totals,
)

STAMP = "2025-06-02T12:00:00Z"
FIRST_PROBE = datetime(2024, 7, 20, 16, 0, tzinfo=UTC)
SECOND_PROBE = datetime(2026, 7, 18, 16, 0, tzinfo=UTC)


def game(api_id: str, start: datetime, *books: str) -> dict[str, Any]:
    """An MLB listing quoting h2h and totals at each of ``books``."""
    markets = [
        odds_api_h2h({CUBS: -120, CARDINALS: 100}, STAMP),
        odds_api_totals(8.5, -110, -105, STAMP),
    ]
    return odds_api_game(
        api_id, MLB, "MLB", CUBS, CARDINALS, start, dict.fromkeys(books, markets), STAMP
    )


def far_game(snapshot: datetime, *books: str, lead_hours: int = 174) -> dict[str, Any]:
    """A listing no stored event takes, which the probe still judges."""
    return game(f"far-{snapshot:%Y%m%d}", snapshot + timedelta(hours=lead_hours), *books)


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
            replay.record(at, [g.listing(at) for g in games], snapshot=at, billed=35)

        with pytest.raises(BackfillStopped, match="billed 35 units, expected 20") as e:
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
        replay.record(at, [g.listing(at) for g in games], snapshot=at, bill_header=False)

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
            replay.record(at, [g.listing(at) for g in games], snapshot=at)
            calls.append(replay.call(manifest, at, games[0].event))

        result = await replay.run(manifest, calls, skip_existing=False)

        assert (result.calls_made, result.units_used) == (3, 60)

    @pytest.mark.parametrize(("cap", "made"), [(40, 2), (59, 2), (60, 3)])
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
            replay.record(at, [g.listing(at) for g in games], snapshot=at)
            calls.append(replay.call(manifest, at, games[0].event))

        if made == len(calls):
            await replay.run(manifest, calls, skip_existing=False, max_units=cap)
        else:
            with pytest.raises(BackfillStopped, match=f"--max-units {cap}") as stop:
                await replay.run(manifest, calls, skip_existing=False, max_units=cap)
            assert stop.value.result.units_used == 20 * made
        assert len(replay.requests) == made

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
            replay.record(at, [], snapshot=at)
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
            replay.record(at, [game("mlb-901", AFTERNOON, "williamhill")], snapshot=at)
            calls.append(replay.call(manifest, at, games[0].event))

        with pytest.raises(BackfillStopped, match="lacks pinnacle on h2h, pinnacle on totals"):
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
            replay.record(at, [game("mlb-901", AFTERNOON, "williamhill")], snapshot=at)
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
            replay.record(at, [far_game(at, "pinnacle")], snapshot=at)
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
            replay.record(at, [], snapshot=at, bill_header=header)
            calls.append(replay.call(manifest, at))

        with pytest.raises(BackfillStopped, match="3 consecutive sparse responses.*no events") as e:
            await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=3)

        assert e.value.result.units_used == 0

    async def test_an_empty_response_billed_units_stops_the_run(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        await load_mlb(pglite_async_session, mlb_schedule)
        at = times(1)[0]
        replay.record(at, [], snapshot=at, billed=20)

        with pytest.raises(BackfillStopped, match="billed 20 units, expected 0"):
            await replay.run(manifest, [replay.call(manifest, at)], skip_existing=False)

    async def test_a_good_call_between_sparse_ones_resets_the_streak(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        await load_mlb(pglite_async_session, mlb_schedule)
        calls = []
        for at, listings in zip(
            times(5),
            [[], [], [game("g", AFTERNOON, "pinnacle")], [], []],
            strict=True,
        ):
            replay.record(at, listings, snapshot=at)
            calls.append(replay.call(manifest, at))

        result = await replay.run(manifest, calls, skip_existing=False, max_consecutive_gaps=3)

        assert result.calls_made == 5

    async def test_skipped_calls_do_not_count_toward_the_streak(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        await load_mlb(pglite_async_session, mlb_schedule)
        sparse = times(2)
        for at in sparse:
            replay.record(at, [], snapshot=at)

        result = await replay.run(
            manifest,
            [replay.call(manifest, at) for at in sparse],
            max_consecutive_gaps=2,
        )

        assert (result.calls_made, result.calls_skipped) == (0, 2)


class TestProbe:
    async def test_a_conforming_response_passes_every_check(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        for at in (FIRST_PROBE, SECOND_PROBE):
            replay.record(at, [far_game(at, "pinnacle", "williamhill")], snapshot=at)

        checks = await replay.probe(manifest)

        assert [c.ok for c in checks] == [True, True]
        first = checks[0]
        assert first.units == 20
        assert (first.previous_gap, first.next_gap) == (timedelta(minutes=5),) * 2
        assert first.farthest_lead == timedelta(hours=174)
        assert first.book_events == {("h2h", "pinnacle"): 1, ("totals", "pinnacle"): 1}
        assert sorted(r["date"] for r in replay.requests) == [
            "2024-07-20T16:00:00Z",
            "2026-07-18T16:00:00Z",
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
            replay.record(at, [games[0].listing(at)], snapshot=at)

        await replay.probe(manifest)

        assert await count(session, OddsSnapshot) > 0

    @pytest.mark.parametrize(
        ("field", "setup"),
        [
            ("units_ok", {"billed": 35}),
            ("units_ok", {"bill_header": False}),
            ("lag_ok", {"lag_seconds": 61}),
            ("grid_ok", {"previous_gap": timedelta(minutes=6, seconds=1)}),
            ("grid_ok", {"next_gap": timedelta(minutes=3, seconds=59)}),
            ("grid_ok", {"previous_gap": timedelta(minutes=10)}),
            ("grid_ok", {"next_gap": timedelta(minutes=10)}),
            ("grid_ok", {"previous_gap": None}),
            ("grid_ok", {"next_gap": None}),
            ("lead_ok", {"lead_hours": 173}),
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
                        *setup.get("books", ("pinnacle",)),
                        lead_hours=setup.get("lead_hours", 174),
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

    async def test_provider_timestamp_jitter_within_a_minute_passes(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        manifest = load_manifest(MLB)
        for at in (FIRST_PROBE, SECOND_PROBE):
            replay.record(
                at,
                [far_game(at, "pinnacle")],
                snapshot=at - timedelta(seconds=60),
                previous_gap=timedelta(minutes=5, seconds=1),
                next_gap=timedelta(minutes=4, seconds=1),
            )

        checks = await replay.probe(manifest)

        assert [c.ok for c in checks] == [True, True]
        assert checks[0].lag == timedelta(seconds=60)

    async def test_an_erroring_call_stops_the_probe_and_logs_the_failure(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        replay.record(FIRST_PROBE, [far_game(FIRST_PROBE, "pinnacle")], snapshot=FIRST_PROBE)

        with pytest.raises(AssertionError, match="no recorded response for 2026-07-18"):
            await replay.probe(manifest)

        assert len(replay.requests) == 1
        assert await count(session, FetchLog) == 2


class TestResponseCheck:
    def test_the_grid_step_in_force_at_each_neighbour_is_the_expected_gap(self) -> None:
        epl = load_manifest("soccer_epl")
        boundary = datetime(2022, 9, 1, tzinfo=UTC)

        def check(previous: int, nxt: int) -> ResponseCheck:
            return ResponseCheck(
                manifest=epl,
                lag=timedelta(0),
                snapshot_time=boundary,
                units=20,
                events=1,
                previous_gap=timedelta(minutes=previous),
                next_gap=timedelta(minutes=nxt),
                farthest_lead=timedelta(hours=174),
                book_events={},
            )

        assert check(10, 5).grid_ok
        assert not check(5, 5).grid_ok
        assert not check(10, 10).grid_ok


class TestGuardSettings:
    def test_a_streak_limit_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_consecutive_gaps must be at least 1"):
            BackfillExecutor(load_manifest(MLB), max_consecutive_gaps=0)
