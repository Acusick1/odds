"""Recorded historical Odds API responses replayed through the real executor for every manifest (#786).

Each response mirrors the documented historical shape and goes through the real client and
``BackfillExecutor``, in committed sessions on the test database, under the real manifest.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest
from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    FetchLog,
    OddsSnapshot,
    Price,
)
from odds_core.odds_math import american_to_decimal
from odds_core.sports import SportKey
from odds_core.utils import is_three_way_sport
from odds_core.vocabulary import Market, PriceOutcome, Source, TimeBasis
from odds_lambda.live_events import SkipReason
from odds_lambda.mlb_stats_fetcher import MlbGameState
from odds_lambda.price_rows import RowIssueType
from odds_lambda.storage.shaped_snapshots import SnapshotShape
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import (
    ACQUISITION_KEY,
    AFTERNOON,
    BACKFILLS,
    EVENING,
    HOME_PRICE,
    MANIFEST_FILES,
    MLB,
    NBA,
    RED_SOX,
    STORED_BOOKS,
    YANKEES,
    BackfillReplay,
    Listed,
    Loader,
    count,
    final_event,
    load_football,
    load_manifest,
    load_mlb,
    load_nba,
    totals_lines,
)
from tests.test_helpers import (
    LEAGUE_FIXTURES,
    FakeMlbStatsFetcher,
    LeagueFixture,
    add_event,
    hold_game_pks,
    mlb_game,
    with_game_pk,
)


def football_loader(league: LeagueFixture) -> Loader:
    async def load(session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]) -> list[Listed]:
        return await load_football(session, mlb_schedule, league)

    return load


REPLAYS = [
    *(pytest.param(f.sport, football_loader(f), id=f.sport) for f in LEAGUE_FIXTURES),
    pytest.param(NBA, load_nba, id=NBA),
    pytest.param(MLB, load_mlb, id=MLB),
]


async def stored_rows(
    session: AsyncSession, event: Event
) -> set[tuple[datetime, Market, str, PriceOutcome, Decimal | None, Decimal]]:
    rows = await session.execute(
        select(
            OddsSnapshot.snapshot_time,
            OddsSnapshot.market,
            Price.bookmaker,
            Price.outcome,
            Price.line,
            Price.price,
        )
        .join(Price, Price.snapshot_id == OddsSnapshot.id)  # type: ignore[arg-type]
        .where(OddsSnapshot.event_id == event.id, OddsSnapshot.source == Source.ODDS_API)
    )
    return {(*row,) for row in rows.all()}  # type: ignore[misc]


async def odds_api_ids(session: AsyncSession, event: Event) -> list[str]:
    rows = await session.execute(
        select(EventExternalId.external_id).where(
            EventExternalId.event_id == event.id, EventExternalId.source == Source.ODDS_API
        )
    )
    return list(rows.scalars().all())


async def event_named(session: AsyncSession, event_id: int) -> Event:
    return (await session.execute(select(Event).where(Event.id == event_id))).scalar_one()


class TestManifestsAreReplayed:
    def test_every_manifest_has_a_replay(self) -> None:
        assert {f.name for f in BACKFILLS.glob("*.yaml")} == set(MANIFEST_FILES.values())
        assert {p.values[0] for p in REPLAYS} == set(MANIFEST_FILES)

    @pytest.mark.parametrize("sport", MANIFEST_FILES)
    def test_manifest_requests_h2h_and_totals_from_its_bookmakers(self, sport: SportKey) -> None:
        manifest = load_manifest(sport)
        assert [str(m) for m in manifest.markets] == ["h2h", "totals"]
        assert len(manifest.bookmakers) == 10 and not manifest.regions


@pytest.mark.asyncio
class TestReplayPerSport:
    @pytest.mark.parametrize(("sport", "load"), REPLAYS)
    async def test_prices_attach_to_the_final_events_and_a_rerun_changes_nothing(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        sport: SportKey,
        load: Loader,
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(sport)
        games = await load(session, mlb_schedule)
        events = tuple(g.event for g in games)
        scored = {e.id: (e.status, e.home_score, e.away_score, e.completed_at) for e in events}
        first = min(g.start for g in games) - timedelta(hours=2)
        second = first + timedelta(hours=1)
        replay.record(first, [g.listing(first) for g in games], snapshot=first)
        replay.record(
            second, [g.listing(second, totals_moved=True) for g in games], snapshot=second
        )
        calls = [replay.call(manifest, at, *events) for at in (first, second)]

        result = await replay.run(manifest, calls)

        assert (result.calls_made, result.fixtures_stored, result.dropped_games) == (
            2,
            2 * len(games),
            {},
        )
        assert result.units_used == 2 * manifest.units_per_call
        assert result.quota_remaining == 4900000
        for request in replay.requests:
            assert request["apiKey"] == ACQUISITION_KEY
            assert request["markets"] == "h2h,totals"
            assert request["bookmakers"] == ",".join(manifest.bookmakers)
            assert "regions" not in request
        pinnacle, william_hill, moved = totals_lines(sport)
        three_way = is_three_way_sport(sport)
        outcomes = {PriceOutcome.HOME, PriceOutcome.AWAY}
        if three_way:
            outcomes.add(PriceOutcome.DRAW)
        for game in games:
            event = await event_named(session, game.event.id)
            assert (
                event.status,
                event.home_score,
                event.away_score,
                event.completed_at,
            ) == scored[event.id]
            assert await odds_api_ids(session, event) == [game.api_id]
            rows = await stored_rows(session, event)
            assert {r[1] for r in rows} == {
                Market.ONE_X_TWO if three_way else Market.H2H,
                Market.TOTALS,
            }
            for at, line in ((first, pinnacle), (second, moved)):
                h2h = {r for r in rows if r[0] == at and r[1] != Market.TOTALS}
                assert {(r[2], r[3]) for r in h2h} == {
                    (book, outcome) for book in STORED_BOOKS for outcome in outcomes
                }
                assert {r[5] for r in h2h if r[2] == "pinnacle" and r[3] is PriceOutcome.HOME} == {
                    Decimal(str(american_to_decimal(HOME_PRICE)))
                }
                totals = {r[2:5] for r in rows if r[0] == at and r[1] == Market.TOTALS}
                assert totals == {
                    ("pinnacle", PriceOutcome.OVER, Decimal(str(line))),
                    ("pinnacle", PriceOutcome.UNDER, Decimal(str(line))),
                    ("williamhill", PriceOutcome.OVER, Decimal(str(william_hill))),
                    ("williamhill", PriceOutcome.UNDER, Decimal(str(william_hill))),
                }
        fetch_logs = (await session.execute(select(FetchLog))).scalars().all()
        assert [(f.sport_key, f.events_count, f.api_quota_remaining) for f in fetch_logs] == [
            (sport, len(games), None)
        ] * 2
        snapshots, prices = await count(session, OddsSnapshot), await count(session, Price)

        skipped = await replay.run(manifest, calls)
        assert (skipped.calls_made, skipped.calls_skipped) == (0, 2)
        assert len(replay.requests) == 2

        refetched = await replay.run(manifest, calls, skip_existing=False)
        assert (refetched.calls_made, refetched.fixtures_stored, refetched.snapshots_stored) == (
            2,
            2 * len(games),
            0,
        )
        assert len(replay.requests) == 4
        assert (await count(session, OddsSnapshot), await count(session, Price)) == (
            snapshots,
            prices,
        )
        assert await count(session, DataQualityLog) == 0
        for game in games:
            assert [e.id for e in await session_events(session, sport)].count(game.event.id) == 1
            assert await odds_api_ids(session, game.event) == [game.api_id]


async def session_events(session: AsyncSession, sport: str) -> list[Event]:
    rows = await session.execute(select(Event).where(Event.sport_key == sport))
    return list(rows.scalars().all())


@pytest.mark.asyncio
class TestMlbReplay:
    async def test_a_doubleheader_attaches_each_game_to_its_own_event_in_any_listing_order(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        game_one, game_two, other = await load_mlb(session, mlb_schedule)
        at = AFTERNOON - timedelta(hours=2)
        replay.record(at, [g.listing(at) for g in (game_two, other, game_one)], snapshot=at)

        result = await replay.run(manifest, [replay.call(manifest, at, game_one.event)])

        assert (result.fixtures_stored, result.dropped_games) == (3, {})
        assert mlb_schedule.requests == 1
        for game in (game_one, game_two, other):
            assert await odds_api_ids(session, game.event) == [game.api_id]
            assert {r[0] for r in await stored_rows(session, game.event)} == {at}
        assert len(await session_events(session, MLB)) == 3

    async def test_a_postponed_game_attaches_to_its_gamepk_at_either_listed_start(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        original = EVENING - timedelta(days=1)
        played = with_game_pk(final_event(1, MLB, YANKEES, RED_SOX, EVENING), 813001)
        await add_event(session, played)
        await hold_game_pks(session)
        await session.commit()
        mlb_schedule.schedule = [
            mlb_game(813001, original, YANKEES, RED_SOX, state=MlbGameState.POSTPONED),
            mlb_game(813001, EVENING, YANKEES, RED_SOX, state=MlbGameState.FINAL),
        ]
        before = Listed(played, "mlb-813001", YANKEES, RED_SOX, original, "MLB")
        after = Listed(played, "mlb-813001", YANKEES, RED_SOX, EVENING, "MLB")
        at_before = original - timedelta(hours=3)
        at_after = EVENING - timedelta(hours=3)
        replay.record(at_before, [before.listing(at_before)], snapshot=at_before)
        replay.record(at_after, [after.listing(at_after)], snapshot=at_after)

        result = await replay.run(
            manifest,
            [replay.call(manifest, at, played) for at in (at_before, at_after)],
        )

        assert (result.fixtures_stored, result.dropped_games) == (2, {})
        assert [e.id for e in await session_events(session, MLB)] == [1]
        assert await odds_api_ids(session, played) == ["mlb-813001"]
        assert {r[0] for r in await stored_rows(session, played)} == {at_before, at_after}
        event = await event_named(session, 1)
        assert (event.commence_time, event.status) == (EVENING, EventStatus.FINAL)

    async def test_a_game_the_schedule_lacks_is_dropped_and_creates_no_event(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(MLB)
        game_one, _, _ = await load_mlb(session, mlb_schedule)
        stranger = Listed(
            game_one.event, "mlb-9", "Seattle Mariners", "Texas Rangers", EVENING, "MLB"
        )
        at = AFTERNOON - timedelta(hours=2)
        replay.record(at, [game_one.listing(at), stranger.listing(at)], snapshot=at)

        result = await replay.run(manifest, [replay.call(manifest, at, game_one.event)])

        assert result.fixtures_stored == 1
        [dropped] = result.dropped_games
        assert dropped.game.home_team == "Seattle Mariners"
        assert dropped.reason is SkipReason.UNRESOLVED
        assert len(await session_events(session, MLB)) == 3


@pytest.mark.asyncio
class TestNbaReplay:
    async def test_a_preseason_game_with_no_event_is_dropped(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(NBA)
        regular, _ = await load_nba(session, mlb_schedule)
        preseason = Listed(
            regular.event,
            "nba-pre",
            "Boston Celtics",
            "Charlotte Hornets",
            regular.start + timedelta(hours=6),
            "NBA",
        )
        at = regular.start - timedelta(hours=2)
        replay.record(at, [regular.listing(at), preseason.listing(at)], snapshot=at)

        result = await replay.run(manifest, [replay.call(manifest, at, regular.event)])

        assert result.fixtures_stored == 1
        assert [(d.game.away_team, d.reason) for d in result.dropped_games] == [
            ("Charlotte Hornets", SkipReason.NO_EVENT)
        ]
        assert len(await session_events(session, NBA)) == 2


@pytest.mark.asyncio
class TestRerunSemantics:
    async def test_a_failed_call_stores_nothing_and_its_rerun_stores_everything(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(NBA)
        games = await load_nba(session, mlb_schedule)
        events = tuple(g.event for g in games)
        at = games[0].start - timedelta(hours=2)
        replay.record(at, [g.listing(at) for g in games], snapshot=at)
        call = replay.call(manifest, at, *events)
        store = OddsWriter.store_envelope
        stored: list[int] = []

        async def fail_second(self: OddsWriter, *args: Any, **kwargs: Any) -> Any:
            if stored:
                raise RuntimeError("store failed")
            stored.append(1)
            return await store(self, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(OddsWriter, "store_envelope", fail_second)
            with pytest.raises(RuntimeError, match="store failed"):
                await replay.run(manifest, [call])

        assert (await count(session, OddsSnapshot), await count(session, EventExternalId)) == (
            0,
            0,
        )
        failed = (await session.execute(select(FetchLog.success))).scalars().all()
        assert failed == [False]

        rerun = await replay.run(manifest, [call])

        assert (rerun.calls_made, rerun.calls_skipped, rerun.fixtures_stored) == (1, 0, 2)
        assert len(replay.requests) == 2
        for game in games:
            assert await odds_api_ids(session, game.event) == [game.api_id]
            assert len(await stored_rows(session, game.event)) > 0

    async def test_a_call_is_skipped_when_any_fixture_it_serves_is_stored(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(NBA)
        early, late = await load_nba(session, mlb_schedule)
        at = early.start - timedelta(hours=2)
        replay.record(at, [early.listing(at)], snapshot=at)
        call = replay.call(manifest, at, early.event, late.event)

        first = await replay.run(manifest, [call])
        rerun = await replay.run(manifest, [call])

        assert (first.calls_made, first.fixtures_stored) == (1, 1)
        assert (rerun.calls_made, rerun.calls_skipped) == (0, 1)
        assert len(replay.requests) == 1
        assert await stored_rows(session, late.event) == set()

    async def test_a_resumed_run_skips_the_calls_it_stored_and_fetches_the_rest(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(NBA)
        early, late = await load_nba(session, mlb_schedule)
        times = [early.start - timedelta(hours=3) + timedelta(hours=i) for i in range(3)]
        for at in times:
            replay.record(at, [early.listing(at)], snapshot=at)
        calls = [replay.call(manifest, at, early.event, late.event) for at in times]
        store = OddsWriter.store_envelope
        stored: list[int] = []

        async def fail_on_second_call(self: OddsWriter, *args: Any, **kwargs: Any) -> Any:
            if stored:
                raise RuntimeError("store failed")
            stored.append(1)
            return await store(self, *args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(OddsWriter, "store_envelope", fail_on_second_call)
            await replay.run(manifest, calls[:1])
            with pytest.raises(RuntimeError, match="store failed"):
                await replay.run(manifest, calls[1:2])
        assert len(replay.requests) == 2

        resumed = await replay.run(manifest, calls)

        assert (resumed.calls_made, resumed.calls_skipped) == (2, 1)
        assert len(replay.requests) == 4
        assert {r[0] for r in await stored_rows(session, early.event)} == set(times)
        assert await stored_rows(session, late.event) == set()

    async def test_a_call_missing_a_served_market_is_not_fetched_again(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(NBA)
        [game, _] = await load_nba(session, mlb_schedule)
        at = game.start - timedelta(hours=2)
        listing = game.listing(at)
        for book in listing["bookmakers"]:
            book["markets"] = [m for m in book["markets"] if m["key"] == "h2h"]
        replay.record(at, [listing], snapshot=at)
        call = replay.call(manifest, at, game.event)

        await replay.run(manifest, [call])
        rerun = await replay.run(manifest, [call])

        assert {r[1] for r in await stored_rows(session, game.event)} == {Market.H2H}
        assert (rerun.calls_made, rerun.calls_skipped) == (0, 1)

    async def test_a_live_snapshot_in_the_window_does_not_skip_the_call(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(NBA)
        [game, _] = await load_nba(session, mlb_schedule)
        at = game.start - timedelta(hours=2)
        fetched_at = at - timedelta(minutes=2, seconds=21)
        live = game.listing(fetched_at)
        live["bookmakers"] = [b for b in live["bookmakers"] if b["key"] == "pinnacle"]
        await OddsWriter(session).store_envelope(
            event_id=game.event.id,
            envelope=live,
            snapshot_time=fetched_at,
            shape=SnapshotShape(Source.ODDS_API, TimeBasis.OBSERVED, None),
        )
        await session.commit()
        replay.record(at, [game.listing(at)], snapshot=at)

        result = await replay.run(manifest, [replay.call(manifest, at, game.event)])

        assert (result.calls_made, result.calls_skipped) == (1, 0)
        rows = await stored_rows(session, game.event)
        assert {r[2] for r in rows if r[0] == at} == STORED_BOOKS
        bases = await session.execute(
            select(OddsSnapshot.snapshot_time, OddsSnapshot.time_basis).distinct()
        )
        assert set(bases.all()) == {  # type: ignore[comparison-overlap]
            (fetched_at, TimeBasis.OBSERVED),
            (at, TimeBasis.SOURCE_STAMPED),
        }

    async def test_a_snapshot_stored_by_another_request_is_kept_never_merged(
        self,
        pglite_async_session: AsyncSession,
        replay: BackfillReplay,
        mlb_schedule: type[FakeMlbStatsFetcher],
    ) -> None:
        session = pglite_async_session
        manifest = load_manifest(NBA)
        [game, _] = await load_nba(session, mlb_schedule)
        at = game.start - timedelta(hours=2)
        narrow = game.listing(at)
        narrow["bookmakers"] = [b for b in narrow["bookmakers"] if b["key"] == "pinnacle"]
        replay.record(at, [narrow], snapshot=at)
        await replay.run(manifest, [replay.call(manifest, at, game.event)])
        held = await stored_rows(session, game.event)
        # A later request at another time returns the same provider snapshot with every book.
        later = at + timedelta(minutes=1)
        replay.record(later, [game.listing(at)], snapshot=at)

        skipped = await replay.run(manifest, [replay.call(manifest, later, game.event)])
        forced = await replay.run(
            manifest, [replay.call(manifest, later, game.event)], skip_existing=False
        )

        assert (skipped.calls_made, skipped.calls_skipped) == (0, 1)
        assert (forced.calls_made, forced.snapshots_stored) == (1, 0)
        assert await stored_rows(session, game.event) == held
        logged = await session.execute(
            select(DataQualityLog.issue_type).where(
                DataQualityLog.issue_type == RowIssueType.SNAPSHOT_KEY_CONFLICT.value
            )
        )
        assert len(logged.scalars().all()) == 2
