"""Integration tests for collapsing several events for one match into one."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_cli.commands import events as events_cli
from odds_core.match_brief_models import MatchBrief
from odds_core.models import (
    Event,
    EventStatus,
    HealthCheckAcknowledgement,
    Odds,
    OddsSnapshot,
    utc_now,
)
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.event_name_repair import repair_team_names
from odds_lambda.storage.event_unmatched import mark_unmatched
from sqlalchemy import select

from tests.test_helpers import (
    API_EVENT_ID,
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    make_epl_event,
)

CHILD_MODELS = (OddsSnapshot, Odds, PaperTrade, MatchBrief)


async def _event_ids(session, model) -> set[str]:
    return set((await session.execute(select(model.event_id))).scalars().all())


def _session_maker(session):
    class _Maker:
        def __call__(self):
            return self

        async def __aenter__(self):
            return session

        async def __aexit__(self, *args) -> bool:
            return False

    return _Maker()


async def _stored_event_ids(session) -> set[str]:
    return set((await session.execute(select(Event.id))).scalars().all())


def _decision(event_id: str, candidate: str) -> CandidateDecision:
    return CandidateDecision(
        candidate_name=candidate,
        candidate_version=1,
        event_id=event_id,
        minutes_before_kickoff=60,
        decision_time=TWIN_KICKOFF,
        kickoff_at_decision=TWIN_KICKOFF,
        known_at=TWIN_KICKOFF,
        bets_placed=1,
    )


def _trade(event_id: str, candidate: str, decision_id: int | None) -> PaperTrade:
    return PaperTrade(
        event_id=event_id,
        market="1x2",
        selection="home",
        bookmaker="bet365",
        price_decimal=2.0,
        stake=1.0,
        candidate_name=candidate,
        candidate_version=1,
        decision_id=decision_id,
    )


class TestMergeDecidedTwins:
    @pytest.mark.asyncio
    async def test_kept_decision_wins_and_other_decisions_move(self, pglite_async_session) -> None:
        session = pglite_async_session
        keep = make_epl_event(API_EVENT_ID)
        twin = make_epl_event(TWIN_EVENT_ID)
        session.add_all([keep, twin])
        await session.flush()
        decisions = {
            (event_id, name): _decision(event_id, name)
            for event_id in (API_EVENT_ID, TWIN_EVENT_ID)
            for name in ("shared", "only-twin")
            if not (event_id == API_EVENT_ID and name == "only-twin")
        }
        session.add_all(decisions.values())
        await session.flush()
        for (event_id, name), decision in decisions.items():
            session.add(_trade(event_id, name, decision.id))
        await session.commit()

        await EventMerger(session).merge(keep, [twin])
        await session.commit()
        session.expire_all()

        stored = (await session.execute(select(CandidateDecision))).scalars().all()
        assert {(d.event_id, d.candidate_name) for d in stored} == {
            (API_EVENT_ID, "shared"),
            (API_EVENT_ID, "only-twin"),
        }
        trades = (await session.execute(select(PaperTrade))).scalars().all()
        assert {(t.event_id, t.candidate_name) for t in trades} == {
            (API_EVENT_ID, "shared"),
            (API_EVENT_ID, "only-twin"),
        }
        assert len(trades) == 2
        assert await session.get(Event, TWIN_EVENT_ID) is None


class TestMerge:
    @pytest.mark.asyncio
    async def test_moves_every_row_and_deletes_twin(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        api_event = make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(minutes=30))
        session.add(api_event)
        await session.flush()

        await EventMerger(session).merge(api_event, [await session.get(Event, TWIN_EVENT_ID)])

        for model in CHILD_MODELS:
            assert await _event_ids(session, model) == {API_EVENT_ID}, model
        assert await session.get(Event, TWIN_EVENT_ID) is None

    @pytest.mark.asyncio
    async def test_reads_every_table_referencing_events_from_the_catalog(
        self, live_twin_with_rows
    ) -> None:
        columns = await EventMerger(live_twin_with_rows)._event_reference_columns()
        tables = {table for _, table, _, _ in columns}
        assert {m.__tablename__ for m in CHILD_MODELS} <= tables

    @pytest.mark.asyncio
    async def test_carries_result_when_api_event_has_none(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        twin = await session.get(Event, TWIN_EVENT_ID)
        twin.status = EventStatus.FINAL
        twin.home_score, twin.away_score = 2, 1
        twin.completed_at = utc_now()
        api_event = make_epl_event(API_EVENT_ID)
        session.add(api_event)
        await session.flush()

        await EventMerger(session).merge(api_event, [twin])

        stored = await session.get(Event, API_EVENT_ID)
        assert stored.status == EventStatus.FINAL
        assert (stored.home_score, stored.away_score) == (2, 1)
        assert stored.completed_at is not None

    @pytest.mark.asyncio
    async def test_keeps_api_result_when_present(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        twin = await session.get(Event, TWIN_EVENT_ID)
        twin.status = EventStatus.FINAL
        twin.home_score, twin.away_score = 2, 1
        api_event = make_epl_event(
            API_EVENT_ID, status=EventStatus.FINAL, home_score=3, away_score=0
        )
        session.add(api_event)
        await session.flush()

        await EventMerger(session).merge(api_event, [twin])

        stored = await session.get(Event, API_EVENT_ID)
        assert (stored.home_score, stored.away_score) == (3, 0)


FDUK_EVENT_ID = "fduk_2025-2026_ARS_CHE_2026-04-15"
HISTORICAL_EVENT_ID = "op_2025-2026_ARS_CHE_2026-04-15"


class TestMergeGroup:
    @pytest.mark.asyncio
    async def test_keeps_api_event_over_every_other_source(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(i)
                for i in (FDUK_EVENT_ID, TWIN_EVENT_ID, HISTORICAL_EVENT_ID, API_EVENT_ID)
            ]
        )
        await session.flush()

        groups = await EventMerger(session).find_twin_groups("soccer_epl")
        report = await EventMerger(session).merge_group(groups[0])
        await session.flush()

        assert report.kept_id == API_EVENT_ID
        assert set(report.merged_ids) == {FDUK_EVENT_ID, TWIN_EVENT_ID, HISTORICAL_EVENT_ID}
        assert await _stored_event_ids(session) == {API_EVENT_ID}

    @pytest.mark.asyncio
    async def test_without_api_event_keeps_historical_then_live_then_fduk(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(FDUK_EVENT_ID), make_epl_event(TWIN_EVENT_ID)])
        await session.flush()

        report = await EventMerger(session).merge_group(
            (await EventMerger(session).find_twin_groups())[0]
        )

        assert report.kept_id == TWIN_EVENT_ID

    @pytest.mark.asyncio
    async def test_moves_rows_from_every_duplicate(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        session.add_all([make_epl_event(API_EVENT_ID), make_epl_event(FDUK_EVENT_ID)])
        await session.flush()
        session.add(
            OddsSnapshot(
                event_id=FDUK_EVENT_ID,
                snapshot_time=TWIN_KICKOFF,
                raw_data={},
                bookmaker_count=0,
            )
        )
        await session.flush()

        await EventMerger(session).merge_all_twins("soccer_epl")
        await session.flush()

        assert await _event_ids(session, OddsSnapshot) == {API_EVENT_ID}
        snapshots = (await session.execute(select(OddsSnapshot))).scalars().all()
        assert len(snapshots) == 2

    @pytest.mark.asyncio
    async def test_distinct_matches_are_not_groups(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(API_EVENT_ID),
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF + timedelta(days=7)),
                make_epl_event(FDUK_EVENT_ID, away_team="Liverpool"),
            ]
        )
        await session.flush()

        assert await EventMerger(session).find_twin_groups() == []


class TestFindTwinGroups:
    @pytest.mark.asyncio
    async def test_chain_of_events_is_not_one_group(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event("a", TWIN_KICKOFF),
                make_epl_event("b", TWIN_KICKOFF + timedelta(hours=1.5)),
                make_epl_event("c", TWIN_KICKOFF + timedelta(hours=3)),
            ]
        )
        await session.flush()

        groups = await EventMerger(session).find_twin_groups()

        assert [[e.id for e in g] for g in groups] == [["a", "b"]]


class TestIdentityChecks:
    @pytest.mark.asyncio
    async def test_flags_twins_with_their_sport_and_clears_after_merge(
        self, pglite_async_session
    ) -> None:
        from odds_core.config import get_settings
        from odds_lambda.health_monitor import HealthMonitor

        session = pglite_async_session
        monitor = HealthMonitor(session, get_settings())
        assert await monitor.check_event_identity() == []

        upcoming = utc_now() + timedelta(days=3)
        session.add_all(
            [make_epl_event(TWIN_EVENT_ID, upcoming), make_epl_event(API_EVENT_ID, upcoming)]
        )
        await session.flush()
        assert await monitor.check_event_identity() == [
            "1 soccer_epl matches stored as more than one event"
        ]

        await EventMerger(session).merge_all_twins()
        await session.flush()
        assert await monitor.check_event_identity() == []

    @pytest.mark.asyncio
    async def test_flags_a_rescheduled_fixture_stored_twice_in_a_season(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF),
                make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(days=2)),
            ]
        )
        await session.flush()

        repeated = await EventMerger(session).find_repeated_fixtures("soccer_epl")

        assert [[e.id for e in g] for g in repeated] == [[TWIN_EVENT_ID, API_EVENT_ID]]
        assert await EventMerger(session).find_twin_groups() == []

    @pytest.mark.asyncio
    async def test_same_fixture_in_two_seasons_is_not_repeated(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF),
                make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(days=365)),
            ]
        )
        await session.flush()

        assert await EventMerger(session).find_repeated_fixtures("soccer_epl") == []


class TestMergeTwinsCommand:
    @pytest.mark.asyncio
    async def test_apply_merges_and_reports_counts(
        self, pglite_async_session, monkeypatch, capsys
    ) -> None:
        session = pglite_async_session
        session.add_all([make_epl_event(TWIN_EVENT_ID), make_epl_event(API_EVENT_ID)])
        await session.commit()

        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        await events_cli._twins("soccer_epl", apply=False)
        assert await session.get(Event, TWIN_EVENT_ID) is not None

        await events_cli._twins("soccer_epl", apply=True)
        assert await session.get(Event, TWIN_EVENT_ID) is None
        assert "Merged 1 events into 1; matches with twins now: 0" in capsys.readouterr().out


class TestMergeCommand:
    @pytest.mark.asyncio
    async def test_dry_run_then_apply_keeps_the_named_event(
        self, pglite_async_session, monkeypatch
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF, status=EventStatus.FINAL),
                make_epl_event(API_EVENT_ID, TWIN_KICKOFF + timedelta(days=2)),
            ]
        )
        await session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        await events_cli._merge(TWIN_EVENT_ID, API_EVENT_ID, apply=False)
        assert await session.get(Event, API_EVENT_ID) is not None

        await events_cli._merge(TWIN_EVENT_ID, API_EVENT_ID, apply=True)
        assert await session.get(Event, API_EVENT_ID) is None
        assert await session.get(Event, TWIN_EVENT_ID) is not None

    @pytest.mark.asyncio
    async def test_refuses_different_sports(self, pglite_async_session, monkeypatch) -> None:
        import typer

        session = pglite_async_session
        other = make_epl_event(API_EVENT_ID)
        other.sport_key = "baseball_mlb"
        session.add_all([make_epl_event(TWIN_EVENT_ID), other])
        await session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        with pytest.raises(typer.BadParameter):
            await events_cli._merge(TWIN_EVENT_ID, API_EVENT_ID, apply=True)


class TestMergeRebasesSnapshotTiming:
    @pytest.mark.asyncio
    async def test_moved_snapshots_follow_the_kept_events_kickoff(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        later = TWIN_KICKOFF + timedelta(days=2)
        session.add_all(
            [make_epl_event(TWIN_EVENT_ID, TWIN_KICKOFF), make_epl_event(API_EVENT_ID, later)]
        )
        await session.flush()
        snapshot = OddsSnapshot(
            event_id=TWIN_EVENT_ID,
            snapshot_time=later - timedelta(hours=1),
            raw_data={},
            bookmaker_count=0,
            hours_until_commence=-47.0,
            fetch_tier="closing",
        )
        session.add(snapshot)
        await session.flush()

        await EventMerger(session).merge(
            await session.get(Event, API_EVENT_ID), [await session.get(Event, TWIN_EVENT_ID)]
        )
        await session.flush()
        await session.refresh(snapshot)

        assert snapshot.event_id == API_EVENT_ID
        assert snapshot.hours_until_commence == pytest.approx(1.0)
        assert snapshot.fetch_tier == "closing"

    @pytest.mark.asyncio
    async def test_same_kickoff_leaves_snapshots_untouched(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        session.add(make_epl_event(API_EVENT_ID))
        await session.flush()
        before = (await session.execute(select(OddsSnapshot.hours_until_commence))).scalar_one()

        await EventMerger(session).merge_all_twins()
        await session.flush()

        after = (await session.execute(select(OddsSnapshot.hours_until_commence))).scalar_one()
        assert after == before

    @pytest.mark.asyncio
    async def test_merge_command_refuses_the_same_id(
        self, pglite_async_session, monkeypatch
    ) -> None:
        import typer

        session = pglite_async_session
        session.add(make_epl_event(API_EVENT_ID))
        await session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        with pytest.raises(typer.BadParameter):
            await events_cli._merge(API_EVENT_ID, API_EVENT_ID, apply=True)
        assert await session.get(Event, API_EVENT_ID) is not None


def _mlb_event(event_id: str, home: str, away: str, **fields) -> Event:
    fields.setdefault("status", EventStatus.SCHEDULED)
    return Event(
        id=event_id,
        sport_key="baseball_mlb",
        sport_title="MLB",
        commence_time=fields.pop("commence_time", TWIN_KICKOFF),
        home_team=home,
        away_team=away,
        **fields,
    )


class TestRepairAnnotatedNames:
    @pytest.mark.asyncio
    async def test_merges_into_the_clean_event_whose_id_sorts_later(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("op_live_CLE(S_TAMMAR", "Cleveland Guardians (S", "Tampa Bay Rays (N"),
                _mlb_event(
                    "op_live_CLEGUA_TAMRAY",
                    "Cleveland Guardians",
                    "Tampa Bay Rays",
                    commence_time=TWIN_KICKOFF + timedelta(minutes=8),
                    status=EventStatus.FINAL,
                    home_score=3,
                    away_score=1,
                ),
            ]
        )
        await session.commit()

        report = await repair_team_names(session, "baseball_mlb")
        await session.commit()

        assert report.renamed_ids == ("op_live_CLE(S_TAMMAR",)
        assert report.merged_count == 1
        assert await _stored_event_ids(session) == {"op_live_CLEGUA_TAMRAY"}
        kept = await session.get(Event, "op_live_CLEGUA_TAMRAY")
        assert kept.commence_time == TWIN_KICKOFF + timedelta(minutes=8)
        assert kept.status is EventStatus.FINAL

    @pytest.mark.asyncio
    async def test_event_without_a_clean_twin_is_renamed_in_place(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add(_mlb_event("op_live_A", "Chicago White Sox (TBD", "Cleveland Guardians"))
        await session.commit()

        report = await repair_team_names(session, "baseball_mlb")

        assert report.merged_count == 0
        assert report.renamed_in_place_ids == ("op_live_A",)
        assert report.renamed_in_place_statuses == {EventStatus.SCHEDULED: 1}
        assert (await session.get(Event, "op_live_A")).home_team == "Chicago White Sox"

    @pytest.mark.asyncio
    async def test_renamed_event_takes_the_canonical_team_name(self, pglite_async_session) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("op_live_ST.(A_X", "St. Louis Cardinals (A", "Milwaukee Brewers"),
                _mlb_event("op_live_ST.CAR_X", "St.Louis Cardinals", "Milwaukee Brewers"),
            ]
        )
        await session.commit()

        report = await repair_team_names(session, "baseball_mlb")

        assert report.merged_count == 1
        assert await _stored_event_ids(session) == {"op_live_ST.CAR_X"}

    @pytest.mark.asyncio
    async def test_annotated_twins_of_each_other_collapse_to_one(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("op_live_A", "New York Yankees (M Fri", "Tampa Bay Rays (N Marti"),
                _mlb_event(
                    "op_live_B",
                    "New York Yankees (M",
                    "Tampa Bay Rays",
                    commence_time=TWIN_KICKOFF + timedelta(minutes=8),
                ),
            ]
        )
        await session.commit()

        report = await repair_team_names(session, "baseball_mlb")

        assert report.merged_count == 1
        assert await _stored_event_ids(session) == {"op_live_A"}

    @pytest.mark.asyncio
    async def test_second_run_changes_nothing_and_other_sports_are_untouched(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("op_live_A", "Chicago White Sox (TBD", "Cleveland Guardians"),
                make_epl_event("op_live_E", home_team="Arsenal (W)"),
            ]
        )
        await session.commit()
        await repair_team_names(session, "baseball_mlb")

        again = await repair_team_names(session, "baseball_mlb")

        assert again.renamed_ids == ()
        assert again.merges == ()
        assert (await session.get(Event, "op_live_E")).home_team == "Arsenal (W)"

    @pytest.mark.asyncio
    async def test_command_dry_run_writes_nothing_and_apply_merges(
        self, pglite_async_session, monkeypatch, capsys
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("op_live_A", "Chicago White Sox (TBD", "Cleveland Guardians"),
                _mlb_event("op_live_B", "Chicago White Sox", "Cleveland Guardians"),
            ]
        )
        await session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(session))

        await events_cli._repair_names("baseball_mlb", apply=False)
        assert await _stored_event_ids(session) == {"op_live_A", "op_live_B"}
        assert (await session.get(Event, "op_live_A")).home_team == "Chicago White Sox (TBD"
        assert "merged into a real event: 1" in capsys.readouterr().out

        await events_cli._repair_names("baseball_mlb", apply=True)
        assert await _stored_event_ids(session) == {"op_live_B"}


class TestRepairGhostsFromRunnerNames:
    @pytest.mark.asyncio
    async def test_cut_off_name_merges_into_clean_event_and_moves_every_row(
        self, pglite_async_session
    ) -> None:
        from odds_core.match_brief_models import BriefDecision, MatchBrief

        session = pglite_async_session
        ghost = _mlb_event("op_live_ST.CAR_ARIDIA", "St. Louis Cardin", "Arizona Diamondbacks")
        session.add_all(
            [
                ghost,
                _mlb_event("op_live_ARIDIA_ST.CAR", "St.Louis Cardinals", "Arizona Diamondbacks"),
            ]
        )
        await session.flush()
        snapshot = OddsSnapshot(
            event_id=ghost.id,
            snapshot_time=TWIN_KICKOFF - timedelta(hours=1),
            raw_data={
                "source": "betfair_api",
                "bookmakers": [
                    {
                        "betfair_meta": {
                            "runners": [
                                {"runner_name": "Arizona Diamondbacks"},
                                {"runner_name": "St. Louis Cardinals"},
                            ]
                        }
                    }
                ],
            },
            bookmaker_count=1,
        )
        session.add(snapshot)
        await session.flush()
        session.add_all(
            [
                Odds(
                    event_id=ghost.id,
                    bookmaker_key="betfair_exchange",
                    bookmaker_title="Betfair Exchange",
                    market_key="h2h",
                    outcome_name="St.Louis Cardinals",
                    price_decimal=2.0,
                    odds_timestamp=snapshot.snapshot_time,
                    last_update=snapshot.snapshot_time,
                ),
                MatchBrief(
                    event_id=ghost.id, decision=BriefDecision.WATCHING, summary="s", brief_text="b"
                ),
            ]
        )
        await session.commit()

        report = await repair_team_names(session, "baseball_mlb")
        await session.commit()

        assert report.merged_count == 1
        assert await _stored_event_ids(session) == {"op_live_ARIDIA_ST.CAR"}
        for model in (OddsSnapshot, Odds, MatchBrief):
            assert await _event_ids(session, model) == {"op_live_ARIDIA_ST.CAR"}
        assert report.describe_merges() == [
            "op_live_ARIDIA_ST.CAR [scheduled] <- op_live_ST.CAR_ARIDIA [scheduled]"
        ]

    @pytest.mark.asyncio
    async def test_doubleheader_games_are_renamed_but_never_merged(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("op_live_PIT1", "Pittsburgh Pirates - Game 1", "Milwaukee Brewers"),
                _mlb_event(
                    "op_live_PIT2",
                    "Pittsburgh Pirates - Game 2",
                    "Milwaukee Brewers",
                    commence_time=TWIN_KICKOFF + timedelta(hours=4),
                ),
            ]
        )
        await session.commit()

        report = await repair_team_names(session, "baseball_mlb")

        assert report.merges == ()
        assert await _stored_event_ids(session) == {"op_live_PIT1", "op_live_PIT2"}
        assert (await session.get(Event, "op_live_PIT2")).home_team == "Pittsburgh Pirates"

    @pytest.mark.asyncio
    async def test_clean_doubleheader_is_not_conflated_by_a_repair(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add_all(
            [
                _mlb_event("op_live_G1", "Pittsburgh Pirates", "Milwaukee Brewers"),
                _mlb_event(
                    "op_live_G2",
                    "Pittsburgh Pirates",
                    "Milwaukee Brewers",
                    commence_time=TWIN_KICKOFF + timedelta(hours=3),
                ),
                _mlb_event("op_live_X", "Pittsburgh Pirates (A", "Milwaukee Brewers"),
            ]
        )
        await session.commit()

        await repair_team_names(session, "baseball_mlb")

        assert await _stored_event_ids(session) == {"op_live_G1", "op_live_G2"}


class TestMergeClearsUnmatched:
    @staticmethod
    def _mlb(event_id: str, game_pk: int | None = None) -> Event:
        return Event(
            id=event_id,
            sport_key="baseball_mlb",
            sport_title="MLB",
            commence_time=TWIN_KICKOFF,
            home_team="New York Yankees",
            away_team="Boston Red Sox",
            status=EventStatus.SCHEDULED,
            mlb_game_pk=game_pk,
        )

    @staticmethod
    async def _acknowledgements(session, event: Event) -> list[str]:
        rows = await session.execute(
            select(HealthCheckAcknowledgement.reason).where(
                HealthCheckAcknowledgement.event_id == event.surrogate_id
            )
        )
        return list(rows.scalars())

    @pytest.mark.asyncio
    async def test_an_unmatched_duplicate_merged_into_the_gamepk_holder_leaves_no_acknowledgement(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        keep, duplicate = self._mlb("holder", 777), self._mlb("orphan")
        session.add_all([keep, duplicate])
        await session.flush()
        await mark_unmatched(session, duplicate, "not in the schedule")

        await EventMerger(session).merge(keep, [duplicate])
        await session.flush()

        assert await self._acknowledgements(session, keep) == []
        assert keep.status is EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_an_unmatched_event_that_takes_a_duplicates_gamepk_is_scheduled_again(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        keep, duplicate = self._mlb("orphan"), self._mlb("holder", 777)
        session.add_all([keep, duplicate])
        await session.flush()
        await mark_unmatched(session, keep, "not in the schedule")

        await EventMerger(session).merge(keep, [duplicate])
        await session.flush()

        assert (keep.mlb_game_pk, keep.status) == (777, EventStatus.SCHEDULED)
        assert await self._acknowledgements(session, keep) == []
