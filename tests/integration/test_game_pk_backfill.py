"""Integration tests for assigning stored MLB events their gamePk and merging duplicates (#612)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pytest
import typer
from odds_cli.commands import events as events_cli
from odds_core.config import get_settings
from odds_core.match_brief_models import BriefDecision, MatchBrief
from odds_core.models import (
    Event,
    EventExternalId,
    EventStatus,
    HealthCheckAcknowledgement,
    OddsSnapshot,
)
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_core.vocabulary import Source
from odds_lambda.health_monitor import HealthMonitor
from odds_lambda.mlb_game_resolver import UnresolvedReason
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState
from odds_lambda.storage.game_pk_backfill import (
    BackfillSkip,
    GamePkRefused,
    OutcomeStatus,
    RefusalReason,
    ScoreCarry,
    Settlement,
    mark_no_game_pk,
    plan_game_pks,
    plan_hand_assignment,
    run_assignments,
)
from odds_lambda.storage.game_pks import game_pks_of
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import hold_game_pks

MLB = "baseball_mlb"
CUBS, CARDINALS = "Chicago Cubs", "St. Louis Cardinals"
YANKEES, RED_SOX = "New York Yankees", "Boston Red Sox"
# 17:05 and 23:10 UTC are 13:05 and 19:10 US Eastern, both on the UTC date.
DAY = datetime(2026, 7, 10, tzinfo=UTC)
AFTERNOON = DAY.replace(hour=17, minute=5)
EVENING = DAY.replace(hour=23, minute=10)
TODAY = date(2026, 10, 25)


def _game(
    pk: int,
    start: datetime,
    game_number: int = 1,
    *,
    home: str = CUBS,
    away: str = CARDINALS,
    state: MlbGameState = MlbGameState.FINAL,
    tbd: bool = False,
    score: tuple[int, int] | None = None,
) -> MlbGameResult:
    return MlbGameResult(
        game_pk=pk,
        start_time=start,
        home_team=home,
        away_team=away,
        is_final=state is MlbGameState.FINAL,
        home_score=score[0] if score else None,
        away_score=score[1] if score else None,
        game_number=game_number,
        start_time_tbd=tbd,
        official_date=start.date(),
        state=state,
        schedule_date=start.date(),
    )


class FakeFetcher:
    def __init__(self, schedule: list[MlbGameResult]) -> None:
        self.schedule = schedule
        self.ranges: list[tuple[date, date]] = []

    async def __aenter__(self) -> FakeFetcher:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def fetch_results(self, start: date, end: date) -> list[MlbGameResult]:
        self.ranges.append((start, end))
        return list(self.schedule)

    async def fetch_game(self, game_pk: int) -> list[MlbGameResult]:
        return [g for g in self.schedule if g.game_pk == game_pk]


def _event(
    event_id: str,
    commence: datetime,
    game_pk: int | None = None,
    *,
    home: str = CUBS,
    away: str = CARDINALS,
    **fields: Any,
) -> Event:
    return Event(
        id=event_id,
        sport_key=MLB,
        sport_title="MLB",
        commence_time=commence,
        home_team=home,
        away_team=away,
        mlb_game_pk=game_pk,
        **fields,
    )


async def _game_pk(session: AsyncSession, event_id: str) -> int | None:
    event = await session.get(Event, event_id)
    assert event is not None
    return (await game_pks_of(session, [event])).get(event_id)


def _snapshot(event_id: str, at: datetime, home_price: float) -> OddsSnapshot:
    return OddsSnapshot(
        event_id=event_id,
        snapshot_time=at,
        raw_data={
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": CUBS, "price_decimal": home_price},
                                {"name": CARDINALS, "price_decimal": 2.1},
                            ],
                        }
                    ],
                }
            ],
            "source": "oddsportal_live",
        },
        bookmaker_count=1,
        hours_until_commence=1.0,
    )


async def _plan_and_apply(session: AsyncSession, fetcher: FakeFetcher, apply: bool = True):
    plan = await plan_game_pks(session, fetcher, None, TODAY)  # type: ignore[arg-type]
    outcomes = await run_assignments(session, plan.assignments, commit=apply)
    return plan, outcomes


async def _count(session: AsyncSession, model: type, event_id: str) -> int:
    return (
        await session.execute(
            select(func.count()).select_from(model).where(model.event_id == event_id)
        )
    ).scalar_one()


async def _acknowledged_reasons(session: AsyncSession, event: Event) -> list[str]:
    rows = await session.execute(
        select(HealthCheckAcknowledgement.reason).where(
            HealthCheckAcknowledgement.event_id == event.surrogate_id
        )
    )
    return list(rows.scalars())


class TestAssignGamePks:
    async def test_single_event_takes_its_game_pk(self, pglite_async_session: AsyncSession) -> None:
        session = pglite_async_session
        session.add(_event("op_live_CHC_STL", AFTERNOON))
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON)])

        plan, outcomes = await _plan_and_apply(session, fetcher)

        assert [(o.game_pk, o.kept_id, o.merged_ids) for o in outcomes] == [
            (824101, "op_live_CHC_STL", ())
        ]
        assert await _game_pk(session, "op_live_CHC_STL") == 824101
        assert fetcher.ranges == [(AFTERNOON.date(), TODAY)]

    async def test_doubleheader_twins_with_split_snapshots_merge_per_game(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        twin_start = EVENING + timedelta(minutes=5)
        session.add_all(
            [
                _event("op_live_g1", AFTERNOON),
                _event("op_live_g2", EVENING),
                _event("op_live_g2_twin", twin_start),
            ]
        )
        await session.flush()
        conflict_at = EVENING - timedelta(hours=1)
        same_at = EVENING - timedelta(hours=2)
        session.add_all(
            [
                _snapshot("op_live_g2", conflict_at, 1.8),
                _snapshot("op_live_g2_twin", conflict_at, 1.6),
                _snapshot("op_live_g2", same_at, 1.9),
                _snapshot("op_live_g2_twin", same_at, 1.9),
                _snapshot("op_live_g2_twin", EVENING - timedelta(hours=3), 1.7),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON, 1), _game(824102, EVENING, 2)])

        plan, outcomes = await _plan_and_apply(session, fetcher)

        merge = next(o for o in outcomes if o.merged_ids)
        assert (merge.game_pk, merge.kept_id, merge.merged_ids) == (
            824102,
            "op_live_g2",
            ("op_live_g2_twin",),
        )
        assert merge.rows_moved == {"odds_snapshots": 3}
        assert (merge.pairs.identical, merge.pairs.conflicting) == (1, 1)
        assert await session.get(Event, "op_live_g2_twin") is None
        assert await _game_pk(session, "op_live_g1") == 824101
        assert await _game_pk(session, "op_live_g2") == 824102
        assert await _count(session, OddsSnapshot, "op_live_g2") == 5

    async def test_twins_of_game_one_are_not_paired_with_game_two(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_g1", AFTERNOON),
                _event("op_live_g1_twin", AFTERNOON + timedelta(minutes=5)),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON, 1), _game(824102, EVENING, 2)])

        plan, outcomes = await _plan_and_apply(session, fetcher)

        assert [(o.game_pk, o.kept_id, o.merged_ids) for o in outcomes] == [
            (824101, "op_live_g1", ("op_live_g1_twin",))
        ]

    async def test_since_keeps_events_from_that_eastern_date_on(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        # 02:00 UTC on the 10th is the evening of the 9th in US Eastern.
        session.add_all(
            [
                _event("op_live_late_9th", DAY.replace(hour=2)),
                _event("op_live_10th", AFTERNOON),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON)])

        plan = await plan_game_pks(session, fetcher, DAY.date(), TODAY)  # type: ignore[arg-type]

        assert plan.pending == 1
        assert [a.kept.id for a in plan.assignments] == ["op_live_10th"]

    async def test_failed_game_rolls_back_alone(self, pglite_async_session: AsyncSession) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_taken", AFTERNOON),
                _event("op_live_free", AFTERNOON, home=YANKEES, away=RED_SOX),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher(
            [_game(824101, AFTERNOON), _game(824102, AFTERNOON, home=YANKEES, away=RED_SOX)]
        )
        plan = await plan_game_pks(session, fetcher, None, TODAY)  # type: ignore[arg-type]
        session.add(_event("op_live_racer", AFTERNOON + timedelta(days=3), 824101))
        await hold_game_pks(session)
        await session.commit()

        outcomes = await run_assignments(session, plan.assignments, commit=True)

        assert [(o.game_pk, o.status) for o in outcomes] == [
            (824101, OutcomeStatus.FAILED),
            (824102, OutcomeStatus.APPLIED),
        ]
        assert await _game_pk(session, "op_live_taken") is None
        assert await _game_pk(session, "op_live_free") == 824102

    async def test_event_holding_the_game_pk_is_kept(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_holder", AFTERNOON + timedelta(minutes=30), 824101),
                _event("op_live_legacy", AFTERNOON),
            ]
        )
        await session.flush()
        session.add(_snapshot("op_live_legacy", AFTERNOON - timedelta(hours=1), 1.8))
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON)])

        plan, outcomes = await _plan_and_apply(session, fetcher)

        assert [(o.kept_id, o.merged_ids) for o in outcomes] == [
            ("op_live_holder", ("op_live_legacy",))
        ]
        holder = await session.get(Event, "op_live_holder")
        assert await _game_pk(session, holder.id) == 824101
        assert holder.commence_time == AFTERNOON
        snapshot = (await session.execute(select(OddsSnapshot))).scalar_one()
        assert snapshot.event_id == "op_live_holder"

    async def test_postponed_slot_moves_to_the_played_start(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        played = AFTERNOON + timedelta(days=1)
        session.add(_event("op_live_postponed", EVENING, status=EventStatus.POSTPONED))
        await session.flush()
        session.add(_snapshot("op_live_postponed", EVENING - timedelta(minutes=30), 1.8))
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher(
            [_game(824101, EVENING, state=MlbGameState.POSTPONED), _game(824101, played)]
        )

        await _plan_and_apply(session, fetcher)

        event = await session.get(Event, "op_live_postponed")
        assert (await _game_pk(session, event.id), event.commence_time, event.status) == (
            824101,
            played,
            EventStatus.SCHEDULED,
        )
        snapshot = (await session.execute(select(OddsSnapshot))).scalar_one()
        assert snapshot.hours_until_commence == pytest.approx(18.0 + 25 / 60)

    async def test_unresolved_events_are_left_alone_and_ghosts_listed(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_no_game", AFTERNOON, home=YANKEES, away=RED_SOX),
                _event("op_live_ghost", AFTERNOON + timedelta(days=2), home=YANKEES, away=RED_SOX),
                _event("op_live_between_games", DAY.replace(hour=20)),
            ]
        )
        await session.flush()
        session.add(_snapshot("op_live_no_game", AFTERNOON - timedelta(hours=1), 1.8))
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON, 1), _game(824102, EVENING, 2)])

        plan, outcomes = await _plan_and_apply(session, fetcher)

        assert outcomes == []
        assert [(u.event_id, u.reason) for u in plan.unresolved] == [
            ("op_live_between_games", UnresolvedReason.AMBIGUOUS_DOUBLEHEADER),
            ("op_live_no_game", UnresolvedReason.NO_GAME),
        ]
        assert plan.ghost_ids == ("op_live_ghost",)
        stored = (await session.execute(select(Event))).scalars().all()
        assert await game_pks_of(session, stored) == {}

    async def test_doubleheader_whose_held_game_resolves_elsewhere_is_left_alone(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all([_event("op_live_g1", AFTERNOON, 824102), _event("op_live_g2", EVENING)])
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON, 1), _game(824102, EVENING, 2)])

        plan, outcomes = await _plan_and_apply(session, fetcher)

        assert outcomes == []
        assert [(u.event_id, u.reason) for u in plan.unresolved] == [
            ("op_live_g2", BackfillSkip.HELD_GAME_MISMATCH)
        ]

    async def test_merge_repoints_paper_trades_and_briefs(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [_event("op_live_a", AFTERNOON), _event("op_live_b", AFTERNOON + timedelta(minutes=20))]
        )
        await session.flush()
        session.add_all(
            [
                PaperTrade(
                    event_id="op_live_b",
                    market="h2h",
                    selection="home",
                    bookmaker="bet365",
                    price_decimal=2.5,
                    stake=10.0,
                    bankroll_before=100.0,
                ),
                MatchBrief(
                    event_id="op_live_b",
                    decision=BriefDecision.WATCHING,
                    summary="s",
                    brief_text="t",
                ),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON)])

        plan, [outcome] = await _plan_and_apply(session, fetcher)

        assert (outcome.kept_id, outcome.merged_ids) == ("op_live_a", ("op_live_b",))
        assert outcome.read_only_rows == {"match_briefs": 1, "paper_trades": 1}
        assert await _count(session, PaperTrade, "op_live_a") == 1
        assert await _count(session, MatchBrief, "op_live_a") == 1


class TestMergeRefusesClashingDecisions:
    async def test_merge_that_would_delete_a_decision_is_refused_and_rolled_back(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [_event("op_live_a", AFTERNOON), _event("op_live_b", AFTERNOON + timedelta(minutes=20))]
        )
        await session.flush()
        decisions = [
            CandidateDecision(
                candidate_name="exchange-gap",
                candidate_version=1,
                event_id=event_id,
                minutes_before_kickoff=60,
                decision_time=AFTERNOON - timedelta(hours=1),
                kickoff_at_decision=AFTERNOON,
                known_at=AFTERNOON - timedelta(hours=1),
                bets_placed=1,
            )
            for event_id in ("op_live_a", "op_live_b")
        ]
        session.add_all(decisions)
        await session.flush()
        session.add(
            PaperTrade(
                event_id="op_live_b",
                market="h2h",
                selection="home",
                bookmaker="bet365",
                price_decimal=2.0,
                stake=1.0,
                candidate_name="exchange-gap",
                candidate_version=1,
                decision_id=decisions[1].id,
            )
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON)])

        plan, [outcome] = await _plan_and_apply(session, fetcher)

        assert (outcome.status, outcome.detail) == (
            OutcomeStatus.REFUSED,
            RefusalReason.PAPER_TRADE_CLASH,
        )
        assert outcome.rows_deleted == {"candidate_decisions": 1, "paper_trades": 1}
        assert await session.get(Event, "op_live_b") is not None
        assert await _game_pk(session, "op_live_a") is None
        assert await _count(session, CandidateDecision, "op_live_b") == 1
        assert await _count(session, PaperTrade, "op_live_b") == 1


class TestTbdReplay:
    """A rainout replayed as a doubleheader's TBD game 2 (gamePk 823637, 2026-04-25 → 04-26)."""

    POSTPONED_AT = datetime(2026, 4, 25, 20, 10, tzinfo=UTC)
    GAME_ONE_AT = datetime(2026, 4, 26, 17, 40, tzinfo=UTC)
    PLACEHOLDER_AT = datetime(2026, 4, 26, 17, 46, tzinfo=UTC)
    LISTED_AT = datetime(2026, 4, 26, 21, 11, tzinfo=UTC)

    def _schedule(self) -> list[MlbGameResult]:
        return [
            _game(823637, self.POSTPONED_AT, state=MlbGameState.POSTPONED),
            _game(823700, self.GAME_ONE_AT, 1),
            _game(823637, self.PLACEHOLDER_AT, 2, tbd=True),
        ]

    async def test_event_on_the_played_date_is_kept_over_the_postponed_slot(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_2026-04-25T2011", self.POSTPONED_AT),
                _event("op_live_2026-04-26T1740", self.GAME_ONE_AT, 823700),
                _event("op_live_2026-04-26T2111", self.LISTED_AT),
            ]
        )
        await session.flush()
        session.add_all(
            [
                _snapshot("op_live_2026-04-26T2111", self.LISTED_AT - timedelta(hours=1), 1.8),
                _snapshot("op_live_2026-04-25T2011", self.POSTPONED_AT - timedelta(hours=1), 1.9),
            ]
        )
        await hold_game_pks(session)
        await session.commit()

        plan, [outcome] = await _plan_and_apply(session, FakeFetcher(self._schedule()))

        assert (outcome.game_pk, outcome.kept_id, outcome.merged_ids) == (
            823637,
            "op_live_2026-04-26T2111",
            ("op_live_2026-04-25T2011",),
        )
        assert (outcome.moved_from, outcome.start_tbd) == (None, True)
        kept = await session.get(Event, "op_live_2026-04-26T2111")
        assert (await _game_pk(session, kept.id), kept.commence_time) == (823637, self.LISTED_AT)
        hours = (
            (
                await session.execute(
                    select(OddsSnapshot.hours_until_commence).order_by(OddsSnapshot.snapshot_time)
                )
            )
            .scalars()
            .all()
        )
        assert hours == [pytest.approx(26.0 + 1 / 60), pytest.approx(1.0)]

    async def test_lone_postponed_slot_event_moves_to_the_placeholder_on_the_played_date(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(_event("op_live_2026-04-25T2011", self.POSTPONED_AT))
        await hold_game_pks(session)
        await session.commit()

        plan, [outcome] = await _plan_and_apply(session, FakeFetcher(self._schedule()))

        assert (outcome.kept_id, outcome.moved_from, outcome.moved_to, outcome.start_tbd) == (
            "op_live_2026-04-25T2011",
            self.POSTPONED_AT,
            self.PLACEHOLDER_AT,
            True,
        )


class TestScoresAndSettlement:
    async def test_postponed_slot_moved_onto_a_final_game_is_settled_from_mlb(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        played = AFTERNOON + timedelta(days=1)
        session.add(_event("op_live_postponed", EVENING, status=EventStatus.POSTPONED))
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher(
            [
                _game(824101, EVENING, state=MlbGameState.POSTPONED),
                _game(824101, played, score=(5, 3)),
            ]
        )

        plan, [outcome] = await _plan_and_apply(session, fetcher)

        assert (outcome.moved_from, outcome.moved_to) == (EVENING, played)
        assert (outcome.score, outcome.settled) == (ScoreCarry.NONE, Settlement.MLB)
        event = await session.get(Event, "op_live_postponed")
        assert (event.status, event.home_score, event.away_score) == (EventStatus.FINAL, 5, 3)

    async def test_stored_score_that_differs_from_mlb_is_left_alone(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(
            _event(
                "op_live_scored", AFTERNOON, status=EventStatus.FINAL, home_score=4, away_score=3
            )
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, AFTERNOON, score=(5, 3))])

        plan, [outcome] = await _plan_and_apply(session, fetcher)

        assert (outcome.settled, outcome.settled_detail) == (
            Settlement.CONFLICT,
            "stored_4-3_vs_mlb_5-3",
        )
        event = await session.get(Event, "op_live_scored")
        assert (event.home_score, event.away_score) == (4, 3)
        assert await _game_pk(session, event.id) == 824101

    async def test_duplicate_score_is_carried_and_a_differing_one_reported(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_a", AFTERNOON),
                _event(
                    "op_live_b",
                    AFTERNOON + timedelta(minutes=20),
                    status=EventStatus.FINAL,
                    home_score=5,
                    away_score=3,
                ),
                _event(
                    "op_live_x", EVENING, home=YANKEES, away=RED_SOX, home_score=1, away_score=0
                ),
                _event(
                    "op_live_y",
                    EVENING + timedelta(minutes=20),
                    home=YANKEES,
                    away=RED_SOX,
                    home_score=2,
                    away_score=0,
                ),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher(
            [_game(824101, AFTERNOON), _game(824102, EVENING, home=YANKEES, away=RED_SOX)]
        )

        plan, outcomes = await _plan_and_apply(session, fetcher)

        assert [(o.kept_id, o.score, o.score_detail) for o in outcomes] == [
            ("op_live_a", ScoreCarry.CARRIED, ""),
            ("op_live_x", ScoreCarry.CONFLICT, "1-0_vs_2-0"),
        ]


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
def cli(pglite_async_session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> list[FakeFetcher]:
    """Points the events CLI at the test session; append the fetcher a command should use."""
    fetchers: list[FakeFetcher] = []
    monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(pglite_async_session))
    monkeypatch.setattr(events_cli, "MlbStatsFetcher", lambda: fetchers[-1])
    return fetchers


class TestAssignGamePksCommand:
    async def test_dry_run_reports_and_writes_nothing_then_apply_merges(
        self,
        pglite_async_session: AsyncSession,
        cli: list[FakeFetcher],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_a", AFTERNOON),
                _event("op_live_b", AFTERNOON + timedelta(minutes=20)),
                _event("op_live_ghost", AFTERNOON, home=YANKEES, away=RED_SOX),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        cli.append(FakeFetcher([_game(824101, AFTERNOON)]))

        await events_cli._assign_game_pks(None, apply=False, delete_ghosts_=True)

        out = capsys.readouterr().out
        assert "merge groups: 1 (read-only: 0; events merged: 1)" in out
        assert (
            "MERGE game_pk=824101 kept=op_live_a rule=single_game merged=op_live_b rows=none"
            " moved=no score=none settled=no same_time_pairs_identical=0"
            " same_time_pairs_conflicting=0"
        ) in out.splitlines()
        assert "GHOST event=op_live_ghost" in out
        assert await session.get(Event, "op_live_b") is not None
        assert await session.get(Event, "op_live_ghost") is not None

        await events_cli._assign_game_pks(None, apply=True, delete_ghosts_=True)

        assert await session.get(Event, "op_live_b") is None
        assert await session.get(Event, "op_live_ghost") is None
        assert await _game_pk(session, "op_live_a") == 824101


class TestApplyIsIdempotent:
    async def test_second_apply_changes_nothing(
        self,
        pglite_async_session: AsyncSession,
        cli: list[FakeFetcher],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        session = pglite_async_session
        played = AFTERNOON + timedelta(days=1)
        session.add_all(
            [
                _event("op_live_a", AFTERNOON),
                _event("op_live_b", AFTERNOON + timedelta(minutes=20)),
                _event(
                    "op_live_postponed",
                    EVENING,
                    home=YANKEES,
                    away=RED_SOX,
                    status=EventStatus.POSTPONED,
                ),
            ]
        )
        await session.flush()
        session.add(_snapshot("op_live_b", AFTERNOON - timedelta(hours=1), 1.8))
        await hold_game_pks(session)
        await session.commit()
        cli.append(
            FakeFetcher(
                [
                    _game(824101, AFTERNOON, score=(2, 1)),
                    _game(
                        824102, EVENING, home=YANKEES, away=RED_SOX, state=MlbGameState.POSTPONED
                    ),
                    _game(824102, played, home=YANKEES, away=RED_SOX, score=(0, 3)),
                ]
            )
        )

        async def state() -> list[tuple[Any, ...]]:
            rows = await session.execute(
                select(
                    Event.id,
                    EventExternalId.external_id,
                    Event.commence_time,
                    Event.status,
                    Event.home_score,
                    Event.away_score,
                )
                .outerjoin(
                    EventExternalId,
                    (EventExternalId.event_id == Event.surrogate_id)
                    & (EventExternalId.source == Source.MLB_STATS_API),
                )
                .order_by(Event.id)
            )
            snapshots = await session.execute(
                select(
                    OddsSnapshot.id,
                    OddsSnapshot.event_id,
                    OddsSnapshot.hours_until_commence,
                    OddsSnapshot.fetch_tier,
                ).order_by(OddsSnapshot.id)
            )
            return [*rows.all(), *snapshots.all()]

        await events_cli._assign_game_pks(None, apply=True, delete_ghosts_=False)
        after_first = await state()
        capsys.readouterr()

        await events_cli._assign_game_pks(None, apply=True, delete_ghosts_=False)

        assert await state() == after_first
        assert "MLB events without a gamePk: 0" in capsys.readouterr().out
        assert [row[:2] for row in after_first[:2]] == [
            ("op_live_a", "824101"),
            ("op_live_postponed", "824102"),
        ]


class TestSetGamePk:
    async def test_refuses_a_game_pk_another_event_holds_unless_merge(
        self, pglite_async_session: AsyncSession, cli: list[FakeFetcher]
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _event("op_live_holder", AFTERNOON + timedelta(days=1), 824101),
                _event("op_live_stray", EVENING),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        cli.append(
            FakeFetcher(
                [
                    _game(824101, EVENING, state=MlbGameState.POSTPONED),
                    _game(824101, AFTERNOON + timedelta(days=1)),
                ]
            )
        )

        with pytest.raises(typer.BadParameter, match="op_live_holder holds gamePk 824101"):
            await events_cli._set_game_pk("op_live_stray", 824101, merge=False, apply=True)
        assert await session.get(Event, "op_live_stray") is not None

        await events_cli._set_game_pk("op_live_stray", 824101, merge=True, apply=True)

        assert await session.get(Event, "op_live_stray") is None
        assert await _game_pk(session, "op_live_holder") == 824101

    async def test_sets_the_game_pk_and_scheduled_start(
        self, pglite_async_session: AsyncSession, cli: list[FakeFetcher]
    ) -> None:
        session = pglite_async_session
        session.add(_event("op_live_stray", EVENING))
        await hold_game_pks(session)
        await session.commit()
        cli.append(
            FakeFetcher(
                [
                    _game(824101, EVENING, state=MlbGameState.POSTPONED),
                    _game(824101, AFTERNOON + timedelta(days=1)),
                ]
            )
        )

        await events_cli._set_game_pk("op_live_stray", 824101, merge=False, apply=True)

        event = await session.get(Event, "op_live_stray")
        assert event.commence_time == AFTERNOON + timedelta(days=1)
        assert await _game_pk(session, event.id) == 824101

    async def test_refuses_a_game_of_other_teams(self, pglite_async_session: AsyncSession) -> None:
        session = pglite_async_session
        session.add(_event("op_live_stray", EVENING))
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, EVENING, home=YANKEES, away=RED_SOX)])

        with pytest.raises(GamePkRefused, match="Boston Red Sox @ New York Yankees"):
            await plan_hand_assignment(session, fetcher, "op_live_stray", 824101, merge=False)  # type: ignore[arg-type]

    async def test_refuses_a_game_of_another_series(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(_event("op_live_stray", EVENING))
        await hold_game_pks(session)
        await session.commit()
        fetcher = FakeFetcher([_game(824101, EVENING + timedelta(days=30))])

        with pytest.raises(GamePkRefused, match="has no slot on 2026-07-10"):
            await plan_hand_assignment(session, fetcher, "op_live_stray", 824101, merge=False)  # type: ignore[arg-type]


class TestMarkUnassignable:
    async def test_marked_event_leaves_the_warning_and_the_backfill(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        now = datetime.now(UTC)
        session.add_all(
            [
                _event(
                    "assigned", now - timedelta(days=4), 824101, created_at=now - timedelta(days=5)
                ),
                _event("orphan", now - timedelta(days=2), None, created_at=now - timedelta(days=3)),
            ]
        )
        await hold_game_pks(session)
        await session.commit()
        monitor = HealthMonitor(session, get_settings())
        assert await monitor.check_event_identity(now) == [
            "1 baseball_mlb events have no gamePk: orphan"
        ]

        await mark_no_game_pk(session, "orphan", "no MLB game matches")
        await hold_game_pks(session)
        await session.commit()

        assert await monitor.check_event_identity(now) == []
        plan = await plan_game_pks(session, FakeFetcher([]), None, TODAY)  # type: ignore[arg-type]
        assert (plan.pending, plan.marked) == (0, {"orphan": "no MLB game matches"})

    async def test_assigning_a_game_pk_clears_the_mark(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(_event("orphan", EVENING))
        await hold_game_pks(session)
        await session.commit()
        await mark_no_game_pk(session, "orphan", "unknown")
        fetcher = FakeFetcher([_game(824101, EVENING)])

        assignment = await plan_hand_assignment(session, fetcher, "orphan", 824101, merge=False)  # type: ignore[arg-type]
        await run_assignments(session, [assignment], commit=True)

        event = await session.get(Event, "orphan")
        assert await _game_pk(session, event.id) == 824101
        assert event.status is EventStatus.SCHEDULED
        assert await _acknowledged_reasons(session, event) == []

    async def test_a_marked_final_event_is_not_planned_or_listed(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        now = datetime.now(UTC)
        session.add_all(
            [
                _event(
                    "assigned", now - timedelta(days=4), 824101, created_at=now - timedelta(days=5)
                ),
                _event(
                    "allstar",
                    now - timedelta(days=2),
                    status=EventStatus.FINAL,
                    home_score=0,
                    away_score=4,
                    created_at=now - timedelta(days=3),
                ),
            ]
        )
        await hold_game_pks(session)
        await session.commit()

        await mark_no_game_pk(session, "allstar", "All-Star Game")
        await session.commit()

        event = await session.get(Event, "allstar")
        assert event.status is EventStatus.FINAL
        monitor = HealthMonitor(session, get_settings())
        assert await monitor.check_event_identity(now) == []
        plan = await plan_game_pks(session, FakeFetcher([]), None, TODAY)  # type: ignore[arg-type]
        assert (plan.pending, plan.marked) == (0, {"allstar": "All-Star Game"})

    async def test_marking_makes_the_event_unmatched_with_an_acknowledgement(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(_event("orphan", EVENING))
        await hold_game_pks(session)
        await session.commit()

        await mark_no_game_pk(session, "orphan", " no MLB game matches ")
        await hold_game_pks(session)
        await session.commit()

        event = await session.get(Event, "orphan")
        assert event.status is EventStatus.UNMATCHED
        assert await _acknowledged_reasons(session, event) == ["no MLB game matches"]

    async def test_refuses_an_event_holding_a_game_pk(
        self, pglite_async_session: AsyncSession
    ) -> None:
        session = pglite_async_session
        session.add(_event("assigned", EVENING, 824101))
        await hold_game_pks(session)
        await session.commit()

        with pytest.raises(GamePkRefused, match="already holds gamePk 824101"):
            await mark_no_game_pk(session, "assigned", "why")
