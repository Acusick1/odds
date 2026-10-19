"""Tests for the fixture-aware collection guards and the per-source stale-data check."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import get_settings
from odds_core.mlb_data_models import MlbProbablePitchers
from odds_core.models import AlertHistory, Event, EventStatus, OddsSnapshot
from odds_core.snapshot_sources import SnapshotSource
from odds_lambda import collection_calendar
from odds_lambda.collection_calendar import (
    EmptyCollectionError,
    fixture_calendar,
    latest_snapshot_time,
    require_rows_for_fixtures,
)
from odds_lambda.health_monitor import HealthMonitor
from odds_lambda.jobs import (
    fetch_betfair_exchange,
    fetch_odds,
    fetch_oddsportal,
    fetch_oddsportal_results,
)
from odds_lambda.scheduling.decision import ScheduleDecision
from odds_lambda.scheduling.jobs import JobContext

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
OP = SnapshotSource.ODDSPORTAL_LIVE
BETFAIR = SnapshotSource.BETFAIR_API
ODDS_API = None


class SessionMaker:
    def __init__(self, session: Any) -> None:
        self.session = session

    def __call__(self) -> SessionMaker:
        return self

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, *_: object) -> None:
        return None


@pytest.fixture
def seed(pglite_async_session: Any) -> Callable[..., Any]:
    counter = 0

    async def add(
        kickoff: datetime,
        *,
        sport_key: str = "soccer_epl",
        status: EventStatus = EventStatus.SCHEDULED,
        snapshots: tuple[tuple[SnapshotSource | None, datetime], ...] = (),
    ) -> None:
        nonlocal counter
        counter += 1
        pglite_async_session.add(
            Event(
                id=f"event_{counter}",
                sport_key=sport_key,
                sport_title=sport_key,
                commence_time=kickoff,
                home_team=f"Home {counter}",
                away_team=f"Away {counter}",
                status=status,
            )
        )
        await pglite_async_session.flush()
        for source, taken in snapshots:
            pglite_async_session.add(
                OddsSnapshot(
                    event_id=f"event_{counter}",
                    snapshot_time=taken,
                    raw_data={} if source is None else {"source": source.value},
                    bookmaker_count=1,
                )
            )
        await pglite_async_session.commit()

    return add


@pytest.fixture
def seed_mlb_game(pglite_async_session: Any) -> Callable[..., Any]:
    counter = 0

    async def add(kickoff: datetime) -> None:
        nonlocal counter
        counter += 1
        pglite_async_session.add(
            MlbProbablePitchers(
                game_pk=counter,
                commence_time=kickoff,
                fetched_at=NOW - timedelta(hours=2),
                home_team="A",
                away_team="B",
                game_type="R",
            )
        )
        await pglite_async_session.commit()

    return add


@pytest.fixture
def guard_session(monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> None:
    monkeypatch.setattr(
        collection_calendar, "async_session_maker", SessionMaker(pglite_async_session)
    )


class TestFixtureCalendar:
    async def test_counts_scheduled_fixtures_inside_the_horizon(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=5))
        await seed(NOW + timedelta(hours=30))
        await seed(NOW + timedelta(hours=100))
        await seed(NOW + timedelta(hours=2), status=EventStatus.FINAL)
        await seed(NOW + timedelta(hours=2), sport_key="baseball_mlb")

        calendar = await fixture_calendar(
            pglite_async_session, "soccer_epl", now=NOW, horizon=timedelta(hours=48)
        )

        assert calendar.kickoffs_in_horizon == 2
        assert calendar.next_kickoff == NOW + timedelta(hours=5)

    async def test_empty_calendar_has_no_next_kickoff(self, pglite_async_session: Any) -> None:
        calendar = await fixture_calendar(
            pglite_async_session, "soccer_epl", now=NOW, horizon=timedelta(hours=48)
        )

        assert calendar.kickoffs_in_horizon == 0
        assert calendar.next_kickoff is None


class TestLatestSnapshotTime:
    async def test_is_scoped_to_prefix_and_sport(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(
            NOW + timedelta(hours=5),
            snapshots=((OP, NOW - timedelta(hours=9)), (BETFAIR, NOW - timedelta(hours=1))),
        )
        await seed(
            NOW + timedelta(hours=5),
            sport_key="baseball_mlb",
            snapshots=((OP, NOW - timedelta(minutes=5)),),
        )

        assert await latest_snapshot_time(pglite_async_session, "soccer_epl", OP) == (
            NOW - timedelta(hours=9)
        )
        assert await latest_snapshot_time(pglite_async_session, "soccer_epl", ODDS_API) is None


@pytest.mark.usefixtures("guard_session")
class TestRequireRowsForFixtures:
    async def test_zero_rows_with_fixtures_in_the_horizon_raises(
        self, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=20))

        with pytest.raises(EmptyCollectionError, match="oddsportal wrote 0 rows for soccer_epl"):
            await require_rows_for_fixtures(fetch_oddsportal.SOURCE, {"soccer_epl": 0}, now=NOW)

    async def test_rows_written_passes(self, seed: Callable[..., Any]) -> None:
        await seed(NOW + timedelta(hours=20))

        await require_rows_for_fixtures(fetch_oddsportal.SOURCE, {"soccer_epl": 4}, now=NOW)

    async def test_no_fixtures_passes(self) -> None:
        await require_rows_for_fixtures(fetch_oddsportal.SOURCE, {"soccer_epl": 0}, now=NOW)

    async def test_fixtures_beyond_the_horizon_pass(self, seed: Callable[..., Any]) -> None:
        await seed(NOW + timedelta(days=6))

        await require_rows_for_fixtures(fetch_oddsportal.SOURCE, {"soccer_epl": 0}, now=NOW)

    async def test_one_empty_sport_raises_although_another_wrote_rows(
        self, seed_mlb_game: Callable[..., Any]
    ) -> None:
        await seed_mlb_game(NOW + timedelta(hours=5))

        with pytest.raises(EmptyCollectionError, match="baseball_mlb"):
            await require_rows_for_fixtures(
                fetch_oddsportal.SOURCE, {"soccer_epl": 6, "baseball_mlb": 0}, now=NOW
            )

    async def test_the_runs_errors_are_in_the_message(self, seed: Callable[..., Any]) -> None:
        await seed(NOW + timedelta(hours=5))

        with pytest.raises(EmptyCollectionError, match="errors: unusable_outcomes: A v B"):
            await require_rows_for_fixtures(
                fetch_betfair_exchange.SOURCE,
                {"soccer_epl": 0},
                errors_by_sport={"soccer_epl": ["unusable_outcomes: A v B"]},
                now=NOW,
            )

    async def test_betfair_mlb_horizon_ignores_fixtures_it_has_not_listed_yet(
        self, seed_mlb_game: Callable[..., Any]
    ) -> None:
        await seed_mlb_game(NOW + timedelta(hours=30))

        await require_rows_for_fixtures(fetch_betfair_exchange.SOURCE, {"baseball_mlb": 0}, now=NOW)

    async def test_betfair_mlb_game_inside_its_horizon_raises(
        self, seed_mlb_game: Callable[..., Any]
    ) -> None:
        await seed_mlb_game(NOW + timedelta(hours=4))

        with pytest.raises(EmptyCollectionError, match="baseball_mlb"):
            await require_rows_for_fixtures(
                fetch_betfair_exchange.SOURCE, {"baseball_mlb": 0}, now=NOW
            )

    async def test_betfair_epl_horizon_is_three_days(self, seed: Callable[..., Any]) -> None:
        await seed(NOW + timedelta(hours=70))

        with pytest.raises(EmptyCollectionError, match="soccer_epl"):
            await require_rows_for_fixtures(
                fetch_betfair_exchange.SOURCE, {"soccer_epl": 0}, now=NOW
            )

    async def test_a_sport_the_source_does_not_cover_is_ignored(
        self, seed_mlb_game: Callable[..., Any]
    ) -> None:
        await seed_mlb_game(NOW + timedelta(hours=1))

        await require_rows_for_fixtures(fetch_odds.SOURCE, {"baseball_mlb": 0}, now=NOW)


class TestCheckStaleSources:
    @staticmethod
    def _monitor(session: Any) -> HealthMonitor:
        return HealthMonitor(session, get_settings())

    async def test_flags_a_source_that_stopped_writing_before_a_fixture(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), snapshots=((OP, NOW - timedelta(hours=5)),))

        stale = await self._monitor(pglite_async_session).check_stale_sources(NOW)

        assert [(s.source, s.sport_key) for s in stale] == [("oddsportal", "soccer_epl")]
        assert stale[0].active_hours == pytest.approx(5.0)
        assert stale[0].threshold_hours == 3.0

    async def test_a_fresh_feed_does_not_mask_a_dead_one(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(
            NOW + timedelta(hours=3),
            snapshots=((OP, NOW - timedelta(hours=7)), (BETFAIR, NOW - timedelta(minutes=10))),
        )

        stale = await self._monitor(pglite_async_session).check_stale_sources(NOW)

        assert [s.source for s in stale] == ["oddsportal"]

    async def test_a_feed_inside_its_cadence_is_healthy(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), snapshots=((OP, NOW - timedelta(hours=2)),))

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_threshold_follows_the_cadence_of_the_next_kickoff(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=40), snapshots=((ODDS_API, NOW - timedelta(hours=20)),))

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_a_slow_cadence_wait_chosen_earlier_is_not_stale(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=10), snapshots=((ODDS_API, NOW - timedelta(hours=20)),))

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_a_slow_cadence_source_is_stale_past_the_interval_it_chose(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=2), snapshots=((ODDS_API, NOW - timedelta(hours=20)),))

        stale = await self._monitor(pglite_async_session).check_stale_sources(NOW)

        assert [(s.source, s.threshold_hours) for s in stale] == [("odds_api", 18.0)]

    async def test_betfair_epl_is_checked_72_hours_ahead(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=50), snapshots=((BETFAIR, NOW - timedelta(hours=30)),))

        stale = await self._monitor(pglite_async_session).check_stale_sources(NOW)

        assert [s.source for s in stale] == ["betfair_exchange"]

    async def test_betfair_mlb_is_not_checked_beyond_its_short_horizon(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(
            NOW + timedelta(hours=20),
            sport_key="baseball_mlb",
            snapshots=((BETFAIR, NOW - timedelta(hours=30)),),
        )

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_a_source_whose_job_is_not_bootstrapped_is_not_checked(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), snapshots=((OP, NOW - timedelta(hours=9)),))
        settings = get_settings().model_copy(deep=True)
        settings.scheduler.bootstrap_jobs = ["check-health"]

        assert await HealthMonitor(pglite_async_session, settings).check_stale_sources(NOW) == []

    async def test_a_failing_job_is_left_to_its_own_alert(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), snapshots=((OP, NOW - timedelta(hours=9)),))
        pglite_async_session.add(
            AlertHistory(
                alert_type="job_failure:fetch-oddsportal-epl",
                severity="critical",
                message="failed",
                sent_at=NOW - timedelta(hours=1),
            )
        )
        await pglite_async_session.commit()

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_a_heartbeat_after_the_failure_means_the_job_recovered(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), snapshots=((OP, NOW - timedelta(hours=9)),))
        for alert_type, ago in (("job_failure", 3), ("heartbeat", 1)):
            pglite_async_session.add(
                AlertHistory(
                    alert_type=f"{alert_type}:fetch-oddsportal-epl",
                    severity="info",
                    message="x",
                    sent_at=NOW - timedelta(hours=ago),
                )
            )
        await pglite_async_session.commit()

        stale = await self._monitor(pglite_async_session).check_stale_sources(NOW)

        assert [s.source for s in stale] == ["oddsportal"]

    async def test_mlb_fixtures_come_from_the_stats_api_schedule(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        evening = NOW + timedelta(hours=8)
        await seed(
            NOW - timedelta(days=30),
            sport_key="baseball_mlb",
            status=EventStatus.FINAL,
            snapshots=((OP, evening - timedelta(hours=9)),),
        )
        pglite_async_session.add(
            MlbProbablePitchers(
                game_pk=1,
                commence_time=evening + timedelta(hours=8),
                fetched_at=evening - timedelta(hours=2),
                home_team="A",
                away_team="B",
                game_type="R",
            )
        )
        await pglite_async_session.commit()

        stale = await self._monitor(pglite_async_session).check_stale_sources(evening)

        assert [(s.source, s.sport_key) for s in stale] == [("oddsportal", "baseball_mlb")]

    async def test_an_old_or_exhibition_mlb_schedule_is_no_calendar(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(
            NOW - timedelta(days=30),
            sport_key="baseball_mlb",
            status=EventStatus.FINAL,
            snapshots=((OP, NOW - timedelta(hours=9)),),
        )
        for game_pk, fetched, game_type in ((1, NOW - timedelta(days=3), "R"), (2, NOW, "S")):
            pglite_async_session.add(
                MlbProbablePitchers(
                    game_pk=game_pk,
                    commence_time=NOW + timedelta(hours=8),
                    fetched_at=fetched,
                    home_team="A",
                    away_team="B",
                    game_type=game_type,
                )
            )
        await pglite_async_session.commit()

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_a_stale_alert_repeats_only_after_the_reminder_window(
        self, pglite_async_session: Any
    ) -> None:
        monitor = self._monitor(pglite_async_session)
        monitor.settings.alerts.alert_enabled = True
        reminder = monitor.settings.alerts.stale_reminder_minutes
        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
            first = await monitor._send_alert(
                "stale_data:x:y", "critical", "m", rate_limit_minutes=reminder
            )
            again = await monitor._send_alert(
                "stale_data:x:y", "critical", "m", rate_limit_minutes=reminder
            )

        assert (first, again) == (True, False)
        assert reminder == 360

    async def test_overnight_pause_is_not_counted(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        morning = datetime(2026, 10, 19, 7, 0, tzinfo=UTC)
        await seed(
            morning + timedelta(hours=8),
            snapshots=((OP, datetime(2026, 10, 18, 21, 0, tzinfo=UTC)),),
        )

        assert await self._monitor(pglite_async_session).check_stale_sources(morning) == []

    async def test_odds_api_has_no_overnight_pause(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        morning = datetime(2026, 10, 19, 7, 0, tzinfo=UTC)
        await seed(
            morning + timedelta(hours=1),
            snapshots=((ODDS_API, datetime(2026, 10, 18, 9, 0, tzinfo=UTC)),),
        )

        stale = await self._monitor(pglite_async_session).check_stale_sources(morning)

        assert [s.source for s in stale] == ["odds_api"]

    async def test_international_break_raises_nothing(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW - timedelta(days=14), status=EventStatus.FINAL)
        await seed(NOW + timedelta(days=6), snapshots=((OP, NOW - timedelta(days=14)),))

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_gap_counts_from_the_fixture_entering_the_horizon(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=47), snapshots=((OP, NOW - timedelta(days=14)),))

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_a_source_that_never_wrote_is_not_checked(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3))

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_other_sports_calendars_do_not_trigger_a_sport(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), sport_key="baseball_mlb")
        await seed(NOW + timedelta(days=9), snapshots=((OP, NOW - timedelta(hours=30)),))

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []


class _AlertRecorder:
    def __init__(self) -> None:
        self.recorded: list[str] = []

    async def record(self, alert_type: str, severity: str, message: str) -> None:
        self.recorded.append(alert_type)

    @property
    def heartbeats(self) -> list[str]:
        return [t for t in self.recorded if t.startswith("heartbeat:")]


@pytest.fixture
def alerts(monkeypatch: pytest.MonkeyPatch) -> Iterator[_AlertRecorder]:
    recorder = _AlertRecorder()
    monkeypatch.setattr("odds_core.alerts.record_to_alert_history", recorder.record)
    monkeypatch.setattr("odds_core.alerts.alert_manager", SimpleNamespace(enabled=False))
    yield recorder


def _schedule_stubs(monkeypatch: pytest.MonkeyPatch, module: Any) -> None:
    decision = ScheduleDecision(
        should_execute=True, reason="test", next_execution=NOW + timedelta(hours=1), tier=None
    )
    monkeypatch.setattr(module, "decide_forward_resilient", AsyncMock(return_value=decision))
    monkeypatch.setattr(module, "self_schedule", AsyncMock())


@pytest.mark.usefixtures("guard_session")
class TestOddsPortalJob:
    @pytest.fixture(autouse=True)
    def stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _schedule_stubs(monkeypatch, fetch_oddsportal)
        monkeypatch.setattr("odds_lambda.model_loader.model_supports_sport", lambda sport: False)

    @staticmethod
    def _ingest(monkeypatch: pytest.MonkeyPatch, snapshots_stored: int) -> None:
        stats = fetch_oddsportal.IngestionStats(
            league="england-premier-league", snapshots_stored=snapshots_stored
        )
        monkeypatch.setattr(fetch_oddsportal, "ingest_league", AsyncMock(return_value=stats))

    async def test_zero_rows_with_fixtures_fails_without_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=20))
        self._ingest(monkeypatch, 0)

        with pytest.raises(EmptyCollectionError):
            await fetch_oddsportal.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == []

    async def test_zero_rows_without_fixtures_writes_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        self._ingest(monkeypatch, 0)

        await fetch_oddsportal.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-oddsportal-epl"]

    async def test_rows_written_writes_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=20))
        self._ingest(monkeypatch, 5)

        await fetch_oddsportal.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-oddsportal-epl"]

    async def test_a_failed_league_fails_the_job(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        monkeypatch.setattr(
            fetch_oddsportal, "ingest_league", AsyncMock(side_effect=RuntimeError("listing down"))
        )

        with pytest.raises(RuntimeError, match="listing down"):
            await fetch_oddsportal.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == []


@pytest.mark.usefixtures("guard_session")
class TestBetfairJob:
    @pytest.fixture(autouse=True)
    def stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _schedule_stubs(monkeypatch, fetch_betfair_exchange)
        settings = MagicMock()
        settings.betfair.enabled = True
        settings.betfair.sports = None
        monkeypatch.setattr(fetch_betfair_exchange, "get_settings", lambda: settings)
        monkeypatch.setattr(
            fetch_betfair_exchange, "resolve_cert_paths", lambda bf: ("cert", "key")
        )
        monkeypatch.setattr(fetch_betfair_exchange, "BetfairExchangeClient", MagicMock())

    @staticmethod
    def _ingest(monkeypatch: pytest.MonkeyPatch, snapshots_stored: int) -> None:
        stats = fetch_betfair_exchange.IngestionStats(
            sport="soccer_epl", events_seen=3, snapshots_stored=snapshots_stored
        )
        monkeypatch.setattr(fetch_betfair_exchange, "ingest_sport", AsyncMock(return_value=stats))

    async def test_zero_rows_with_fixtures_fails_without_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=5))
        self._ingest(monkeypatch, 0)

        with pytest.raises(EmptyCollectionError):
            await fetch_betfair_exchange.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == []

    async def test_zero_rows_without_fixtures_writes_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        self._ingest(monkeypatch, 0)

        await fetch_betfair_exchange.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-betfair-exchange-epl"]


@pytest.mark.usefixtures("guard_session")
class TestFetchOddsJob:
    @pytest.fixture(autouse=True)
    def stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _schedule_stubs(monkeypatch, fetch_odds)
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        monkeypatch.setattr(fetch_odds, "TheOddsAPIClient", lambda: client)
        sync = MagicMock()
        sync.sync_sports = AsyncMock(return_value=[])
        monkeypatch.setattr(fetch_odds, "build_event_sync_service", lambda c: sync)
        monkeypatch.setattr("odds_core.alerts.send_critical", AsyncMock())

    @staticmethod
    def _ingest(monkeypatch: pytest.MonkeyPatch, processed: int) -> None:
        sport_result = SimpleNamespace(
            sport_key="soccer_epl",
            processed_events=processed,
            total_events=processed,
            quota_remaining=None,
            failures=[],
        )
        results = SimpleNamespace(
            sport_results=[sport_result],
            total_processed=processed,
            total_events=processed,
            total_failures=0,
        )
        service = MagicMock()
        service.ingest_sports = AsyncMock(return_value=results)
        monkeypatch.setattr(fetch_odds, "build_ingestion_service", lambda c, s: service)

    async def test_zero_rows_with_fixtures_fails_and_alerts(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any]
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=10))
        self._ingest(monkeypatch, 0)

        with pytest.raises(EmptyCollectionError):
            await fetch_odds.main(JobContext(sport="soccer_epl"))

        import odds_core.alerts

        odds_core.alerts.send_critical.assert_awaited_once()  # type: ignore[attr-defined]

    async def test_zero_rows_without_fixtures_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._ingest(monkeypatch, 0)

        await fetch_odds.main(JobContext(sport="soccer_epl"))


class TestResultsJob:
    async def test_a_settlement_error_fails_the_job_without_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        stats = fetch_oddsportal_results.ResultsStats(errors=["Fulham vs Burnley: boom"])
        monkeypatch.setattr(
            fetch_oddsportal_results, "process_results", AsyncMock(return_value=stats)
        )
        monkeypatch.setattr(fetch_oddsportal_results, "self_schedule", AsyncMock())

        with pytest.raises(RuntimeError, match="failed to settle 1 event"):
            await fetch_oddsportal_results.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == []

    async def test_a_clean_run_writes_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        monkeypatch.setattr(
            fetch_oddsportal_results,
            "process_results",
            AsyncMock(return_value=fetch_oddsportal_results.ResultsStats()),
        )
        monkeypatch.setattr(fetch_oddsportal_results, "self_schedule", AsyncMock())

        await fetch_oddsportal_results.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-oddsportal-results-epl"]
