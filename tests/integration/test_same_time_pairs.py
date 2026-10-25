"""Integration tests for deleting same-time snapshot pairs (#535)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from odds_cli.commands import events as events_cli
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.paper_trade_models import CandidateDecision
from odds_lambda.storage.same_time_pairs import (
    GroupAction,
    event_same_time_pairs,
    find_same_time_groups,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

KICKOFF = datetime(2026, 4, 18, 20, 5, tzinfo=UTC)
AT = KICKOFF - timedelta(hours=1)

Quote = tuple[str, str, float]


def _raw(source: str | None, quotes: list[Quote]) -> dict:
    by_bookmaker: dict[str, list[dict[str, Any]]] = {}
    for bookmaker, outcome, price in quotes:
        by_bookmaker.setdefault(bookmaker, []).append({"name": outcome, "price_decimal": price})
    raw: dict[str, Any] = {
        "bookmakers": [
            {"key": b, "title": b, "markets": [{"key": "h2h", "outcomes": outcomes}]}
            for b, outcomes in by_bookmaker.items()
        ]
    }
    if source is not None:
        raw["source"] = source
    return raw


async def _event(session: AsyncSession, event_id: str, sport_key: str = "baseball_mlb") -> None:
    session.add(
        Event(
            id=event_id,
            sport_key=sport_key,
            sport_title=sport_key,
            commence_time=KICKOFF,
            home_team="Athletics",
            away_team="Chicago White Sox",
            status=EventStatus.SCHEDULED,
        )
    )
    await session.flush()


async def _snapshot(
    session: AsyncSession,
    event_id: str,
    quotes: list[Quote],
    source: str | None = "oddsportal_live",
    linked: list[Quote] | None = None,
    unlinked: list[Quote] | None = None,
) -> int:
    """A stored snapshot with odds rows as the backfill leaves them: ``linked`` rows carry
    its id, ``unlinked`` ones (keys another snapshot also quotes) carry none."""
    snapshot = OddsSnapshot(
        event_id=event_id,
        snapshot_time=AT,
        raw_data=_raw(source, quotes),
        bookmaker_count=len({q[0] for q in quotes}),
    )
    session.add(snapshot)
    await session.flush()
    assert snapshot.id is not None
    for rows, snapshot_id in ((linked or [], snapshot.id), (unlinked or [], None)):
        for bookmaker, outcome, price in rows:
            session.add(
                Odds(
                    event_id=event_id,
                    snapshot_id=snapshot_id,
                    bookmaker_key=bookmaker,
                    bookmaker_title=bookmaker,
                    market_key="h2h",
                    outcome_name=outcome,
                    price_decimal=price,
                    odds_timestamp=AT,
                    last_update=AT,
                )
            )
    await session.flush()
    return snapshot.id


async def _snapshot_ids(session: AsyncSession) -> list[int]:
    return list(
        (await session.execute(select(OddsSnapshot.id).order_by(OddsSnapshot.id))).scalars()
    )


async def _odds(session: AsyncSession) -> list[tuple[str, float, int | None]]:
    rows = await session.execute(
        select(Odds.bookmaker_key, Odds.price_decimal, Odds.snapshot_id).order_by(Odds.id)
    )
    return list(rows.tuples())


def _session_maker(session: AsyncSession) -> Any:
    class _Maker:
        def __call__(self) -> _Maker:
            return self

        async def __aenter__(self) -> AsyncSession:
            return session

        async def __aexit__(self, *args: object) -> bool:
            return False

    return _Maker()


@pytest.fixture
def cli(pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> AsyncSession:
    monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(pglite_async_session))
    return pglite_async_session


async def _two_games_on_one_event(session: AsyncSession) -> tuple[int, int, int]:
    """Two scrapes of one MLB event at one time pricing 7bet differently, each with a book of
    its own, beside a Polymarket snapshot at the same time."""
    await _event(session, "op_live_ATH_CHISOX")
    game_1 = await _snapshot(
        session,
        "op_live_ATH_CHISOX",
        [("7bet", "Athletics", 1.6), ("bet365", "Athletics", 1.62)],
        linked=[("bet365", "Athletics", 1.62)],
        unlinked=[("7bet", "Athletics", 1.6)],
    )
    game_2 = await _snapshot(
        session,
        "op_live_ATH_CHISOX",
        [("7bet", "Athletics", 1.71), ("betway", "Athletics", 1.7)],
        linked=[("betway", "Athletics", 1.7)],
        unlinked=[("7bet", "Athletics", 1.71)],
    )
    polymarket = await _snapshot(
        session,
        "op_live_ATH_CHISOX",
        [("polymarket", "Athletics", 1.65)],
        source="polymarket",
        linked=[("polymarket", "Athletics", 1.65)],
    )
    return game_1, game_2, polymarket


class TestFindGroups:
    async def test_conflicting_identical_and_disjoint_snapshots(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        game_1, game_2, _ = await _two_games_on_one_event(session)
        await _event(session, "op_live_dup", sport_key="soccer_epl")
        first = await _snapshot(session, "op_live_dup", [("bet365", "A", 2.0)])
        second = await _snapshot(session, "op_live_dup", [("bet365", "A", 2.0)])
        # One market per snapshot at one time is not a pair.
        await _event(session, "op_live_split")
        await _snapshot(session, "op_live_split", [("bet365", "A", 2.0)])
        await _snapshot(session, "op_live_split", [("betway", "A", 2.1)])

        groups = {g.event_id: g for g in await find_same_time_groups(session)}

        assert set(groups) == {"op_live_ATH_CHISOX", "op_live_dup"}
        mlb = groups["op_live_ATH_CHISOX"]
        assert (mlb.action, mlb.snapshot_ids, mlb.deleted_ids) == (
            GroupAction.CONFLICTING,
            (game_1, game_2),
            (game_1, game_2),
        )
        dup = groups["op_live_dup"]
        assert (dup.action, dup.deleted_ids) == (GroupAction.DUPLICATE, (second,))
        assert first < second
        pairs = await event_same_time_pairs(session, "op_live_ATH_CHISOX")
        assert (pairs.identical, pairs.conflicting) == (0, 1)

    async def test_conflict_outside_mlb_is_left(self, pglite_async_session: AsyncSession) -> None:
        session = pglite_async_session
        await _event(session, "op_2024_ARS_CHE", sport_key="soccer_epl")
        await _snapshot(session, "op_2024_ARS_CHE", [("bet365", "A", 2.0)], source="oddsportal")
        await _snapshot(session, "op_2024_ARS_CHE", [("bet365", "A", 2.2)], source="oddsportal")

        [group] = await find_same_time_groups(session)

        assert (group.action, group.deleted_ids) == (GroupAction.CONFLICTING_LEFT, ())

    async def test_mlb_conflict_from_another_source_is_left(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        await _event(session, "op_live_ATH_CHISOX")
        for price in (1.6, 1.7):
            await _snapshot(
                session,
                "op_live_ATH_CHISOX",
                [("betfair_exchange", "Athletics", price)],
                source="betfair_api",
            )

        [group] = await find_same_time_groups(session)

        assert (group.action, group.deleted_ids) == (GroupAction.CONFLICTING_LEFT, ())

    async def test_snapshot_a_decision_used_is_refused(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        game_1, _, _ = await _two_games_on_one_event(session)
        session.add(
            CandidateDecision(
                candidate_name="exchange-gap",
                candidate_version=1,
                event_id="op_live_ATH_CHISOX",
                decision_time=AT,
                kickoff_at_decision=KICKOFF,
                known_at=AT,
                input_snapshot_ids=[game_1],
            )
        )
        await session.flush()

        [group] = await find_same_time_groups(session)

        assert (group.action, group.referenced_ids, group.deleted_ids) == (
            GroupAction.REFUSED,
            (game_1,),
            (),
        )


class TestDeleteSameTimePairsCommand:
    async def test_dry_run_writes_nothing(
        self, cli: AsyncSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = cli
        game_1, game_2, polymarket = await _two_games_on_one_event(session)
        await session.commit()
        before = (await _snapshot_ids(session), await _odds(session))

        await events_cli._delete_same_time_pairs(apply=False)

        out = capsys.readouterr().out
        assert "conflicting: 1\n" in out
        assert "snapshots deleted: 2 (conflicting: 2, duplicates: 0)" in out
        assert (
            f"CONFLICTING event=op_live_ATH_CHISOX sport=baseball_mlb time=2026-04-18T19:05Z"
            f" sources=oddsportal_live identical=0 conflicting=1 snapshots={game_1},{game_2}"
            f" deleted={game_1},{game_2} odds_deleted=4 odds_linked=0"
        ) in out.splitlines()
        assert "Dry run: nothing written. Re-run with --apply." in out
        assert (await _snapshot_ids(session), await _odds(session)) == before

    async def test_apply_deletes_both_conflicting_snapshots_and_their_odds(
        self, cli: AsyncSession
    ) -> None:
        session = cli
        _, _, polymarket = await _two_games_on_one_event(session)
        await session.commit()

        await events_cli._delete_same_time_pairs(apply=True)

        assert await _snapshot_ids(session) == [polymarket]
        assert await _odds(session) == [("polymarket", 1.65, polymarket)]

    async def test_apply_keeps_one_identical_snapshot_and_one_copy_of_its_rows(
        self, cli: AsyncSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = cli
        await _event(session, "nba_dup", sport_key="basketball_nba")
        quotes = [("fanduel", "A", 1.8), ("fanduel", "B", 2.05)]
        kept = await _snapshot(session, "nba_dup", quotes, source=None, unlinked=quotes)
        await _snapshot(session, "nba_dup", quotes, source=None, unlinked=quotes)
        await session.commit()

        await events_cli._delete_same_time_pairs(apply=True)

        assert "odds rows without a snapshot: 0" in capsys.readouterr().out
        assert await _snapshot_ids(session) == [kept]
        assert await _odds(session) == [("fanduel", 1.8, kept), ("fanduel", 2.05, kept)]

    async def test_second_apply_finds_nothing_to_delete(
        self, cli: AsyncSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = cli
        await _two_games_on_one_event(session)
        await session.commit()
        await events_cli._delete_same_time_pairs(apply=True)
        capsys.readouterr()

        await events_cli._delete_same_time_pairs(apply=True)

        assert "event-times with same-time pairs: 0" in capsys.readouterr().out

    async def test_unlinked_row_no_snapshot_quotes_is_left_alone(self, cli: AsyncSession) -> None:
        session = cli
        _, _, polymarket = await _two_games_on_one_event(session)
        session.add(
            Odds(
                event_id="op_live_ATH_CHISOX",
                bookmaker_key="unibet",
                bookmaker_title="unibet",
                market_key="h2h",
                outcome_name="Athletics",
                price_decimal=1.66,
                odds_timestamp=AT,
                last_update=AT,
            )
        )
        await session.commit()

        await events_cli._delete_same_time_pairs(apply=True)

        assert await _odds(session) == [
            ("polymarket", 1.65, polymarket),
            ("unibet", 1.66, None),
        ]

    async def test_duplicate_rows_link_despite_float_rounding(self, cli: AsyncSession) -> None:
        session = cli
        await _event(session, "nba_dup", sport_key="basketball_nba")
        quotes = [("fanduel", "A", 1.8)]
        rounded = [("fanduel", "A", 1.8 + 1e-12)]
        kept = await _snapshot(session, "nba_dup", quotes, source=None, unlinked=rounded)
        await _snapshot(session, "nba_dup", quotes, source=None, unlinked=rounded)
        await session.commit()

        await events_cli._delete_same_time_pairs(apply=True)

        assert [(b, s) for b, _, s in await _odds(session)] == [("fanduel", kept)]

    async def test_duplicate_a_decision_used_is_left(self, cli: AsyncSession) -> None:
        session = cli
        await _event(session, "nba_dup", sport_key="basketball_nba")
        quotes = [("fanduel", "A", 1.8)]
        await _snapshot(session, "nba_dup", quotes, source=None, unlinked=quotes)
        used = await _snapshot(session, "nba_dup", quotes, source=None, unlinked=quotes)
        session.add(
            CandidateDecision(
                candidate_name="exchange-gap",
                candidate_version=1,
                event_id="nba_dup",
                decision_time=AT,
                kickoff_at_decision=KICKOFF,
                known_at=AT,
                input_snapshot_ids=[used],
            )
        )
        await session.commit()
        before = await _odds(session)

        await events_cli._delete_same_time_pairs(apply=True)

        assert len(await _snapshot_ids(session)) == 2
        assert await _odds(session) == before

    async def test_orphan_rows_are_counted_apart(
        self, cli: AsyncSession, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = cli
        await _two_games_on_one_event(session)
        session.add(
            Odds(
                event_id="op_live_ATH_CHISOX",
                bookmaker_key="unibet",
                bookmaker_title="unibet",
                market_key="h2h",
                outcome_name="Athletics",
                price_decimal=1.66,
                odds_timestamp=AT,
                last_update=AT,
            )
        )
        await session.commit()

        await events_cli._delete_same_time_pairs(apply=False)

        assert (
            "odds rows without a snapshot: 1 (orphans no snapshot quotes: 1;"
            " quoted by several snapshots: 0)"
        ) in capsys.readouterr().out
