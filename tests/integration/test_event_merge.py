"""Integration tests for collapsing several events for one match into one."""

from __future__ import annotations

from datetime import timedelta

import pytest
from odds_cli.commands import events as events_cli
from odds_core.match_brief_models import MatchBrief
from odds_core.models import (
    DataQualityLog,
    Event,
    EventExternalId,
    EventStatus,
    HealthCheckAcknowledgement,
    OddsSnapshot,
    SnapshotPayload,
    utc_now,
)
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.event_name_repair import _betfair_runner_names, repair_team_names
from odds_lambda.storage.event_unmatched import mark_unmatched
from sqlalchemy import select
from structlog.testing import capture_logs

from tests.test_helpers import (
    API_EVENT_ID,
    TWIN_EVENT_ID,
    TWIN_KICKOFF,
    game_pk_of,
    hold_game_pks,
    make_epl_event,
    with_game_pk,
)

CHILD_MODELS = (OddsSnapshot, PaperTrade, MatchBrief)


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
    async def test_keeps_the_first_stored_event(self, pglite_async_session) -> None:
        session = pglite_async_session
        for event_id in (FDUK_EVENT_ID, TWIN_EVENT_ID, HISTORICAL_EVENT_ID, API_EVENT_ID):
            session.add(make_epl_event(event_id))
            await session.flush()

        groups = await EventMerger(session).find_twin_groups("soccer_epl")
        report = await EventMerger(session).merge_group(groups[0])
        await session.flush()

        assert report.kept_id == FDUK_EVENT_ID
        assert set(report.merged_ids) == {API_EVENT_ID, TWIN_EVENT_ID, HISTORICAL_EVENT_ID}
        assert await _stored_event_ids(session) == {FDUK_EVENT_ID}

    @pytest.mark.asyncio
    async def test_moves_rows_from_every_duplicate(self, live_twin_with_rows) -> None:
        session = live_twin_with_rows
        session.add_all([make_epl_event(API_EVENT_ID), make_epl_event(FDUK_EVENT_ID)])
        await session.flush()
        session.add(
            OddsSnapshot(
                event_id=FDUK_EVENT_ID,
                snapshot_time=TWIN_KICKOFF,
                source=Source.FOOTBALL_DATA_UK,
                market=Market.ONE_X_TWO,
                time_basis=TimeBasis.CLOSE,
            )
        )
        await session.flush()

        merger = EventMerger(session)
        for group in await merger.find_twin_groups("soccer_epl"):
            await merger.merge_group(group)
        await session.flush()

        assert await _event_ids(session, OddsSnapshot) == {TWIN_EVENT_ID}
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

        merger = EventMerger(session)
        for group in await merger.find_twin_groups():
            await merger.merge_group(group)
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
        assert await session.get(Event, API_EVENT_ID) is None
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


class TestMergeCommandSameId:
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


class TestBetfairRunnerNames:
    @pytest.mark.asyncio
    async def test_totals_snapshot_at_the_same_time_does_not_hide_the_team_runners(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        session.add(make_epl_event("e1"))
        await session.flush()
        envelope = {
            "source": "betfair_api",
            "bookmakers": [
                {
                    "markets": [{"key": "1x2"}],
                    "betfair_meta": {
                        "runners": [{"runner_name": n} for n in ["Arsenal", "Chelsea", "The Draw"]]
                    },
                },
                {
                    "markets": [{"key": "totals"}],
                    "betfair_meta": {
                        "runners": [
                            {"runner_name": n} for n in ["Under 2.5 Goals", "Over 2.5 Goals"]
                        ]
                    },
                },
            ],
        }
        payload = SnapshotPayload(payload=envelope)
        session.add(payload)
        await session.flush()

        def snapshot(market: Market) -> OddsSnapshot:
            return OddsSnapshot(
                event_id="e1",
                snapshot_time=TWIN_KICKOFF,
                source=Source.BETFAIR,
                market=market,
                time_basis=TimeBasis.OBSERVED,
                payload_id=payload.id,
            )

        session.add_all([snapshot(Market.TOTALS), snapshot(Market.ONE_X_TWO)])
        await session.flush()

        assert await _betfair_runner_names(session, "soccer_epl") == {
            "e1": ["Arsenal", "Chelsea", "The Draw"]
        }


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
        assert ghost.surrogate_id is not None
        session.add(
            EventExternalId(source=Source.BETFAIR, external_id="35001", event_id=ghost.surrogate_id)
        )

        def envelope(betfair_event_id: str, runners: list[str]) -> dict:
            return {
                "source": "betfair_api",
                "betfair_event_id": betfair_event_id,
                "bookmakers": [
                    {
                        "markets": [{"key": "h2h"}],
                        "betfair_meta": {"runners": [{"runner_name": name} for name in runners]},
                    },
                    # The event's other markets ride in the same envelope.
                    {
                        "markets": [{"key": "totals"}],
                        "betfair_meta": {"runners": [{"runner_name": "Over 8.5"}]},
                    },
                ],
            }

        ghost_envelope = envelope("35001", ["Arizona Diamondbacks", "St. Louis Cardinals"])
        # The run's payload holds every event's envelope; the ghost's is found by its Betfair id.
        payload = SnapshotPayload(
            payload=[envelope("35002", ["New York Yankees", "Boston Red Sox"]), ghost_envelope]
        )
        session.add(payload)
        await session.flush()
        snapshot = OddsSnapshot(
            event_id=ghost.id,
            snapshot_time=TWIN_KICKOFF - timedelta(hours=1),
            source=Source.BETFAIR,
            market=Market.H2H,
            time_basis=TimeBasis.OBSERVED,
            payload_id=payload.id,
        )
        session.add(snapshot)
        await session.flush()
        session.add(
            MatchBrief(
                event_id=ghost.id, decision=BriefDecision.WATCHING, summary="s", brief_text="b"
            )
        )
        await session.commit()

        report = await repair_team_names(session, "baseball_mlb")
        await session.commit()

        assert report.merged_count == 1
        assert await _stored_event_ids(session) == {"op_live_ARIDIA_ST.CAR"}
        for model in (OddsSnapshot, MatchBrief):
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
        return with_game_pk(
            Event(
                id=event_id,
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=TWIN_KICKOFF,
                home_team="New York Yankees",
                away_team="Boston Red Sox",
                status=EventStatus.SCHEDULED,
            ),
            game_pk,
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
        await hold_game_pks(session)
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
        await hold_game_pks(session)
        await mark_unmatched(session, keep, "not in the schedule")

        await EventMerger(session).merge(keep, [duplicate])
        await session.flush()

        assert (await game_pk_of(session, keep), keep.status) == (777, EventStatus.SCHEDULED)
        assert await self._acknowledgements(session, keep) == []


class TestBetfairEnvelopeLookup:
    @staticmethod
    async def _event_with_run_payload(session, envelope_id: str) -> Event:
        event = _mlb_event("op_live_ST.CAR_ARIDIA", "St. Louis Cardin", "Arizona Diamondbacks")
        session.add(event)
        await session.flush()
        envelope = {
            "source": "betfair_api",
            "betfair_event_id": envelope_id,
            "bookmakers": [
                {
                    "markets": [{"key": "h2h"}],
                    "betfair_meta": {
                        "runners": [
                            {"runner_name": "Arizona Diamondbacks"},
                            {"runner_name": "St. Louis Cardinals"},
                        ]
                    },
                }
            ],
        }
        payload = SnapshotPayload(payload=[{"betfair_event_id": "99999"}, envelope])
        session.add(payload)
        await session.flush()
        session.add(
            OddsSnapshot(
                event_id=event.id,
                snapshot_time=TWIN_KICKOFF - timedelta(hours=1),
                source=Source.BETFAIR,
                market=Market.H2H,
                time_basis=TimeBasis.OBSERVED,
                payload_id=payload.id,
            )
        )
        await session.flush()
        return event

    @pytest.mark.asyncio
    async def test_an_id_the_event_was_refused_finds_its_envelope(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        event = await self._event_with_run_payload(session, "35001")
        session.add(
            DataQualityLog(
                event_id=event.id,
                severity="warning",
                issue_type="external_id_refused",
                description="betfair id 35001 refused",
                raw_data={"source": "betfair", "external_id": "35001"},
            )
        )
        await session.flush()

        names = await _betfair_runner_names(session, "baseball_mlb")

        assert names == {event.id: ["Arizona Diamondbacks", "St. Louis Cardinals"]}

    @pytest.mark.asyncio
    async def test_an_envelope_not_found_is_logged(self, pglite_async_session) -> None:
        session = pglite_async_session
        event = await self._event_with_run_payload(session, "35001")

        with capture_logs() as logs:
            names = await _betfair_runner_names(session, "baseball_mlb")

        assert names == {}
        assert [(log["event"], log["event_id"]) for log in logs] == [
            ("betfair_envelope_not_found", event.id)
        ]
