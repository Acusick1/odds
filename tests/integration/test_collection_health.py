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
from odds_core.paper_trade_models import PaperTrade
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda import collection_calendar, results_run
from odds_lambda.betfair.client import BetfairBook, BetfairRunner
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.collection_calendar import (
    EmptyCollectionError,
    EventKey,
    MlbGameKey,
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
from odds_lambda.scheduling import helpers as scheduling_helpers
from odds_lambda.scheduling.decision import ScheduleDecision
from odds_lambda.scheduling.jobs import JobContext

from tests.test_helpers import make_tagged_snapshot, store_tagged, three_way_bookmaker, totals_book

NOW = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
OP = Source.ODDSPORTAL
BETFAIR = Source.BETFAIR
ODDS_API = Source.ODDS_API


class SessionMaker:
    def __init__(self, session: Any) -> None:
        self.session = session

    def __call__(self) -> SessionMaker:
        return self

    async def __aenter__(self) -> Any:
        return self.session

    async def __aexit__(self, *_: object) -> None:
        return None


@pytest.fixture(autouse=True)
def deliverable_alerts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("odds_core.alerts.alert_manager.enabled", True)


@pytest.fixture
def seed(pglite_async_session: Any) -> Callable[..., Any]:
    counter = 0

    async def add(
        kickoff: datetime,
        *,
        sport_key: str = "soccer_epl",
        status: EventStatus = EventStatus.SCHEDULED,
        snapshots: tuple[tuple[Source, datetime], ...] = (),
        time_basis: TimeBasis = TimeBasis.OBSERVED,
    ) -> None:
        nonlocal counter
        counter += 1
        pglite_async_session.add(
            Event(
                id=counter,
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
                    event_id=counter,
                    snapshot_time=taken,
                    source=source,
                    time_basis=time_basis,
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

    async def test_leaves_out_backfilled_history(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(
            NOW + timedelta(hours=5),
            snapshots=((OP, NOW - timedelta(hours=1)),),
            time_basis=TimeBasis.SOURCE_STAMPED,
        )

        assert await latest_snapshot_time(pglite_async_session, "soccer_epl", OP) is None


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

    async def test_fixtures_all_accounted_for_pass(self, seed: Callable[..., Any]) -> None:
        await seed(NOW + timedelta(hours=2))

        await require_rows_for_fixtures(
            fetch_betfair_exchange.SOURCE,
            {"soccer_epl": 0},
            accounted_by_sport={"soccer_epl": {EventKey(1)}},
            now=NOW,
        )

    async def test_a_fixture_beyond_the_accounted_ones_raises(
        self, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=2))
        await seed(NOW + timedelta(hours=3))

        with pytest.raises(EmptyCollectionError, match="with 1 fixture"):
            await require_rows_for_fixtures(
                fetch_betfair_exchange.SOURCE,
                {"soccer_epl": 0},
                accounted_by_sport={"soccer_epl": {EventKey(1)}},
                now=NOW,
            )

    async def test_an_mlb_game_is_accounted_for_by_its_game_pk(
        self, seed_mlb_game: Callable[..., Any]
    ) -> None:
        await seed_mlb_game(NOW + timedelta(hours=2))
        await seed_mlb_game(NOW + timedelta(hours=3))

        with pytest.raises(EmptyCollectionError, match="with 1 fixture"):
            await require_rows_for_fixtures(
                fetch_betfair_exchange.SOURCE,
                {"baseball_mlb": 0},
                accounted_by_sport={"baseball_mlb": {MlbGameKey(1)}},
                now=NOW,
            )
        await require_rows_for_fixtures(
            fetch_betfair_exchange.SOURCE,
            {"baseball_mlb": 0},
            accounted_by_sport={"baseball_mlb": {MlbGameKey(1), MlbGameKey(2)}},
            now=NOW,
        )

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
        await seed(NOW + timedelta(hours=4), snapshots=((OP, NOW - timedelta(hours=5)),))

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

    async def test_a_failure_before_the_overnight_pause_still_suppresses_after_it(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), snapshots=((OP, NOW - timedelta(hours=11)),))
        pglite_async_session.add(
            AlertHistory(
                alert_type="job_failure:fetch-oddsportal-epl",
                severity="critical",
                message="failed",
                sent_at=NOW - timedelta(hours=10),
            )
        )
        await pglite_async_session.commit()

        assert await self._monitor(pglite_async_session).check_stale_sources(NOW) == []

    async def test_an_old_failure_without_a_heartbeat_no_longer_suppresses(
        self, pglite_async_session: Any, seed: Callable[..., Any]
    ) -> None:
        await seed(NOW + timedelta(hours=3), snapshots=((OP, NOW - timedelta(hours=9)),))
        pglite_async_session.add(
            AlertHistory(
                alert_type="job_failure:fetch-oddsportal-epl",
                severity="critical",
                message="failed",
                sent_at=NOW - timedelta(days=3),
            )
        )
        await pglite_async_session.commit()

        stale = await self._monitor(pglite_async_session).check_stale_sources(NOW)

        assert [s.source for s in stale] == ["oddsportal"]

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
            first = await monitor._send_alert("stale_data:x:y", "critical", "m")
            again = await monitor._send_alert("stale_data:x:y", "critical", "m")

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


class TestCheckMissingBetfairCloses:
    KICKOFF = NOW - timedelta(hours=3)

    @staticmethod
    def _settings() -> Any:
        settings = get_settings().model_copy(deep=True)
        settings.betfair.enabled = True
        return settings

    @classmethod
    def _monitor(cls, session: Any) -> HealthMonitor:
        return HealthMonitor(session, cls._settings())

    @staticmethod
    def _betfair(
        inplay: bool = False, outcomes: int = 3, prices: tuple[float, ...] = (2.0, 4.0, 4.0)
    ) -> dict[str, Any]:
        names = ("Home FC", "The Draw", "Away FC")[:outcomes]
        runners = [
            {"runner_name": name, "best_back": price}
            for name, price in zip(names, prices, strict=False)
        ]
        return {
            "source": "betfair_api",
            "bookmakers": [
                {
                    "key": "betfair_exchange",
                    "betfair_meta": {"inplay": inplay, "runners": runners},
                    "markets": [{"key": "1x2", "outcomes": []}],
                }
            ],
        }

    @pytest.fixture
    def seed_event(self, pglite_async_session: Any) -> Callable[..., Any]:
        counter = 0

        async def add(
            kickoff: datetime,
            snapshots: tuple[tuple[datetime, dict[str, Any]], ...],
            status: EventStatus = EventStatus.FINAL,
        ) -> int:
            nonlocal counter
            counter += 1
            event_id = counter
            pglite_async_session.add(
                Event(
                    id=event_id,
                    sport_key="soccer_epl",
                    sport_title="EPL",
                    commence_time=kickoff,
                    home_team="Home FC",
                    away_team="Away FC",
                    status=status,
                )
            )
            await pglite_async_session.flush()
            for taken, envelope in snapshots:
                await store_tagged(
                    pglite_async_session,
                    make_tagged_snapshot(
                        event_id, taken, envelope["bookmakers"], envelope["source"]
                    ),
                )
            await pglite_async_session.commit()
            return event_id

        return add

    @staticmethod
    def _earlier_betfair(kickoff: datetime) -> tuple[datetime, dict[str, Any]]:
        return kickoff - timedelta(hours=2), TestCheckMissingBetfairCloses._betfair()

    @staticmethod
    def _op(kickoff: datetime, minutes_before: int = 20) -> tuple[datetime, dict[str, Any]]:
        return kickoff - timedelta(minutes=minutes_before), {
            "source": "oddsportal_live",
            "bookmakers": [three_way_bookmaker("bet365", (2.0, 4.0, 4.0), "Home FC", "Away FC")],
        }

    async def test_event_without_a_snapshot_in_the_window_is_missing(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        event_id = await seed_event(
            self.KICKOFF, (self._earlier_betfair(self.KICKOFF), self._op(self.KICKOFF))
        )

        missing = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert [m.event_id for m in missing] == [event_id]

    async def test_snapshot_inside_the_window_is_a_close(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (self.KICKOFF - timedelta(minutes=29), self._betfair()),
            ),
        )

        assert await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW) == []

    async def test_snapshot_just_outside_the_window_is_not_a_close(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF,
            (
                (self.KICKOFF - timedelta(minutes=31), self._betfair()),
                self._op(self.KICKOFF),
            ),
        )

        missing = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert len(missing) == 1

    async def test_in_play_snapshot_is_not_a_close(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (self.KICKOFF - timedelta(minutes=5), self._betfair(inplay=True)),
            ),
        )

        missing = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert len(missing) == 1

    async def test_event_with_no_live_betfair_snapshot_in_72_hours_is_a_ghost(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF,
            (
                (self.KICKOFF - timedelta(hours=73), self._betfair()),
                self._op(self.KICKOFF),
            ),
        )

        assert await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW) == []

    async def test_twin_priced_only_through_the_other_event_is_not_reported(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(self.KICKOFF, (self._op(self.KICKOFF),))
        await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (self.KICKOFF - timedelta(minutes=10), self._betfair()),
            ),
        )

        assert await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW) == []

    async def test_snapshot_wider_than_the_overround_cap_is_not_a_close(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (self.KICKOFF - timedelta(minutes=10), self._betfair(prices=(2.5, 2.5, 2.5))),
            ),
        )

        missing = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert len(missing) == 1

    async def test_snapshot_without_every_outcome_is_not_a_close(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (self.KICKOFF - timedelta(minutes=10), self._betfair(outcomes=2)),
            ),
        )

        missing = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert len(missing) == 1

    @staticmethod
    def _totals_trade(event_id: int, line: float) -> PaperTrade:
        return PaperTrade(
            event_id=event_id,
            market="totals",
            selection="over",
            line=line,
            bookmaker="bet365",
            price_decimal=2.0,
            stake=1.0,
            bankroll_before=100.0,
        )

    @classmethod
    def _live_totals(cls, taken: datetime, lines: dict[float, tuple[float, float]]) -> Any:
        return (
            taken,
            {
                "source": "betfair_api",
                "bookmakers": [totals_book("betfair_exchange", lines, betfair_runners=True)],
            },
        )

    async def test_totals_trade_without_a_close_on_its_line_is_missing(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        close = self.KICKOFF - timedelta(minutes=10)
        event_id = await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (close, self._betfair()),
                self._live_totals(close, {2.5: (2.0, 2.0)}),
            ),
        )
        pglite_async_session.add(self._totals_trade(event_id, 2.5))
        pglite_async_session.add(self._totals_trade(event_id, 3.5))
        await pglite_async_session.commit()

        [missing] = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert (missing.event_id, missing.three_way, missing.totals_lines) == (
            event_id,
            False,
            (3.5,),
        )
        assert "totals 3.5" in missing.describe()

    async def test_totals_trade_with_a_close_on_its_line_is_not_missing(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        close = self.KICKOFF - timedelta(minutes=10)
        event_id = await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (close, self._betfair()),
                self._live_totals(close, {2.5: (2.0, 2.0)}),
            ),
        )
        pglite_async_session.add(self._totals_trade(event_id, 2.5))
        await pglite_async_session.commit()

        assert await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW) == []

    async def test_event_without_a_totals_trade_needs_no_totals_close(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF,
            (
                self._earlier_betfair(self.KICKOFF),
                (self.KICKOFF - timedelta(minutes=10), self._betfair()),
            ),
        )

        assert await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW) == []

    @pytest.mark.parametrize("status", [EventStatus.POSTPONED, EventStatus.CANCELLED])
    async def test_postponed_or_cancelled_event_is_skipped(
        self, pglite_async_session: Any, seed_event: Callable[..., Any], status: EventStatus
    ) -> None:
        await seed_event(self.KICKOFF, (self._op(self.KICKOFF),), status=status)
        await seed_event(self.KICKOFF, (self._earlier_betfair(self.KICKOFF),))

        missing = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert len(missing) == 1

    async def test_events_outside_the_lookback_or_not_yet_kicked_off_are_ignored(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        for kickoff in (NOW - timedelta(hours=25), NOW + timedelta(minutes=10)):
            await seed_event(kickoff, (self._earlier_betfair(kickoff), self._op(kickoff)))

        assert await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW) == []

    async def test_job_that_is_not_bootstrapped_is_not_checked(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF, (self._earlier_betfair(self.KICKOFF), self._op(self.KICKOFF))
        )
        settings = self._settings()
        settings.scheduler.bootstrap_jobs = ["check-health"]

        monitor = HealthMonitor(pglite_async_session, settings)

        assert await monitor.check_missing_betfair_closes(NOW) == []

    async def test_betfair_disabled_is_not_checked(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF, (self._earlier_betfair(self.KICKOFF), self._op(self.KICKOFF))
        )
        settings = self._settings()
        settings.betfair.enabled = False

        monitor = HealthMonitor(pglite_async_session, settings)

        assert await monitor.check_missing_betfair_closes(NOW) == []

    async def test_betfair_not_covering_epl_is_not_checked(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(
            self.KICKOFF, (self._earlier_betfair(self.KICKOFF), self._op(self.KICKOFF))
        )
        settings = self._settings()
        settings.data_collection.sports = ["baseball_mlb"]

        monitor = HealthMonitor(pglite_async_session, settings)

        assert await monitor.check_missing_betfair_closes(NOW) == []

    async def test_betfair_source_that_never_wrote_is_not_checked(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        await seed_event(self.KICKOFF, (self._op(self.KICKOFF),))

        assert await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW) == []

    async def test_events_are_aggregated_and_alerted_once(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        ids = [
            await seed_event(
                self.KICKOFF - timedelta(minutes=offset),
                (
                    self._earlier_betfair(self.KICKOFF - timedelta(minutes=offset)),
                    self._op(self.KICKOFF - timedelta(minutes=offset)),
                ),
            )
            for offset in (0, 60)
        ]
        monitor = self._monitor(pglite_async_session)
        settings = monitor.settings
        sent: list[str] = []

        async def alert(message: str, severity: str) -> None:
            sent.append(message)

        with (
            patch("odds_core.alerts.alert_manager.alert", side_effect=alert),
            patch.object(settings.alerts, "alert_enabled", True),
        ):
            first = await monitor.check_system_health(NOW)
            second = await monitor.check_system_health(NOW)

        close_alerts = [
            message for message in sent if "Betfair" in message and "snapshot" in message
        ]
        assert len(close_alerts) == 1
        assert all(str(event_id) in close_alerts[0] for event_id in ids)
        assert "missing_betfair_close" in first.alerts_sent
        assert "missing_betfair_close" not in second.alerts_sent

    async def test_rate_limited_alert_is_retried_on_the_next_run(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        event_id = await seed_event(
            self.KICKOFF, (self._earlier_betfair(self.KICKOFF), self._op(self.KICKOFF))
        )
        recent = AlertHistory(
            alert_type="missing_betfair_close",
            severity="critical",
            message="other event",
            context={"event_ids": [0]},
            sent_at=datetime.now(UTC) - timedelta(minutes=5),
        )
        pglite_async_session.add(recent)
        await pglite_async_session.commit()
        monitor = self._monitor(pglite_async_session)
        monitor.settings.alerts.alert_enabled = True

        with patch("odds_core.alerts.alert_manager.alert", new_callable=AsyncMock):
            limited = await monitor.check_system_health(NOW)
            await pglite_async_session.delete(recent)
            await pglite_async_session.commit()
            retried = await monitor.check_system_health(NOW)
            repeated = await monitor.check_system_health(NOW)

        listing = [i for i in limited.issues_detected if str(event_id) in i]
        assert len(listing) == 1
        assert "missing_betfair_close" not in limited.alerts_sent
        assert "missing_betfair_close" in retried.alerts_sent
        assert not any(str(event_id) in issue for issue in repeated.issues_detected)

    async def test_new_event_is_alerted_after_an_earlier_alert(
        self, pglite_async_session: Any, seed_event: Callable[..., Any]
    ) -> None:
        first = await seed_event(
            self.KICKOFF, (self._earlier_betfair(self.KICKOFF), self._op(self.KICKOFF))
        )
        pglite_async_session.add(
            AlertHistory(
                alert_type="missing_betfair_close",
                severity="critical",
                message="earlier",
                context={"event_ids": [first]},
                sent_at=NOW - timedelta(hours=1),
            )
        )
        later = NOW - timedelta(hours=1)
        second = await seed_event(later, (self._earlier_betfair(later), self._op(later)))

        missing = await self._monitor(pglite_async_session).check_missing_betfair_closes(NOW)

        assert [m.event_id for m in missing] == [second]


class TestMlbCalendar:
    @staticmethod
    def _game(
        game_pk: int, kickoff: datetime, fetched: datetime, **fields: Any
    ) -> MlbProbablePitchers:
        return MlbProbablePitchers(
            game_pk=game_pk,
            commence_time=kickoff,
            fetched_at=fetched,
            home_team="A",
            away_team="B",
            game_type=fields.pop("game_type", "R"),
            **fields,
        )

    @staticmethod
    async def _count(session: Any) -> int:
        calendar = await fixture_calendar(
            session, "baseball_mlb", now=NOW, horizon=timedelta(hours=48)
        )
        return calendar.kickoffs_in_horizon

    async def test_only_the_latest_fetch_is_the_calendar(self, pglite_async_session: Any) -> None:
        kickoff = NOW + timedelta(hours=5)
        pglite_async_session.add_all(
            [
                self._game(1, kickoff, NOW - timedelta(hours=20)),
                self._game(2, kickoff, NOW - timedelta(hours=20)),
                self._game(1, kickoff, NOW - timedelta(hours=2)),
            ]
        )
        await pglite_async_session.commit()

        assert await self._count(pglite_async_session) == 1

    @pytest.mark.parametrize(
        "fields",
        [
            {"start_time_tbd": True},
            {"game_status": "Postponed"},
            {"game_status": "Cancelled"},
            {"game_status": "If Necessary"},
        ],
    )
    async def test_unplayed_or_untimed_games_are_not_fixtures(
        self, pglite_async_session: Any, fields: dict[str, Any]
    ) -> None:
        pglite_async_session.add(
            self._game(1, NOW + timedelta(hours=5), NOW - timedelta(hours=2), **fields)
        )
        await pglite_async_session.commit()

        assert await self._count(pglite_async_session) == 0

    async def test_a_scheduled_game_is_a_fixture(self, pglite_async_session: Any) -> None:
        pglite_async_session.add(
            self._game(
                1, NOW + timedelta(hours=5), NOW - timedelta(hours=2), game_status="Scheduled"
            )
        )
        await pglite_async_session.commit()

        assert await self._count(pglite_async_session) == 1


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


def _schedule_stubs(
    monkeypatch: pytest.MonkeyPatch, module: Any, decider: str = "decide_forward_resilient"
) -> None:
    decision = ScheduleDecision(
        should_execute=True, reason="test", next_execution=NOW + timedelta(hours=1), tier=None
    )
    monkeypatch.setattr(module, decider, AsyncMock(return_value=decision))
    monkeypatch.setattr(scheduling_helpers, "self_schedule", AsyncMock())
    if hasattr(module, "self_schedule"):
        monkeypatch.setattr(module, "self_schedule", AsyncMock())


@pytest.mark.usefixtures("guard_session")
class TestOddsPortalJob:
    @pytest.fixture(autouse=True)
    def stubs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _schedule_stubs(monkeypatch, fetch_oddsportal, "decide_collection_run")

    @staticmethod
    def _ingest(
        monkeypatch: pytest.MonkeyPatch,
        snapshots_stored: int,
        events_listed: int = 0,
        upcoming_listed: int = 0,
    ) -> None:
        stats = fetch_oddsportal.IngestionStats(
            league="england-premier-league",
            snapshots_stored=snapshots_stored,
            events_listed=events_listed,
            upcoming_listed=upcoming_listed,
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

    async def test_listing_of_only_started_events_passes_with_fixtures(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=20))
        self._ingest(monkeypatch, 0, events_listed=1, upcoming_listed=0)

        await fetch_oddsportal.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-oddsportal-epl"]

    async def test_upcoming_events_with_zero_rows_still_fails(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=20))
        self._ingest(monkeypatch, 0, events_listed=2, upcoming_listed=1)

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
        _schedule_stubs(monkeypatch, fetch_betfair_exchange, "decide_collection_run")
        settings = MagicMock()
        settings.betfair.enabled = True
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

    @staticmethod
    def _ingest_accounted(monkeypatch: pytest.MonkeyPatch, *event_ids: int) -> None:
        stats = fetch_betfair_exchange.IngestionStats(
            sport="soccer_epl",
            events_seen=len(event_ids),
            books_fetched=len(event_ids),
            books_in_play=len(event_ids),
            accounted_fixtures={EventKey(i) for i in event_ids},
        )
        monkeypatch.setattr(fetch_betfair_exchange, "ingest_sport", AsyncMock(return_value=stats))

    async def test_a_run_whose_only_fixture_is_accounted_for_passes_with_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=1))
        self._ingest_accounted(monkeypatch, 1)

        await fetch_betfair_exchange.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-betfair-exchange-epl"]

    async def test_an_accounted_fixture_does_not_excuse_a_pre_play_one_with_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        now = datetime.now(UTC)
        await seed(now + timedelta(hours=1))
        await seed(now + timedelta(hours=4))
        self._ingest_accounted(monkeypatch, 1)

        with pytest.raises(EmptyCollectionError, match="with 1 fixture"):
            await fetch_betfair_exchange.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == []

    async def test_zero_rows_without_fixtures_writes_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        self._ingest(monkeypatch, 0)

        await fetch_betfair_exchange.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-betfair-exchange-epl"]


@pytest.mark.usefixtures("guard_session")
class TestBetfairEplAccounting:
    """An in-play EPL book accounts for its event through the real ingestion path (#634)."""

    @pytest.fixture(autouse=True)
    def session_maker(self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> None:
        monkeypatch.setattr(
            fetch_betfair_exchange, "async_session_maker", SessionMaker(pglite_async_session)
        )

    @staticmethod
    def _event(event_id: int, home: str, away: str, kickoff: datetime) -> Event:
        return Event(
            id=event_id,
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=kickoff,
            home_team=home,
            away_team=away,
            status=EventStatus.SCHEDULED,
        )

    @staticmethod
    async def _ingest(kickoff: datetime) -> fetch_betfair_exchange.IngestionStats:
        def runner(selection_id: int, name: str) -> BetfairRunner:
            return BetfairRunner(selection_id, name, None, None, None, None, None)

        book = BetfairBook(
            market_id="1.1",
            market_type="MATCH_ODDS",
            betfair_event_id="1",
            betfair_event_name="Arsenal v Chelsea",
            market_start_time=kickoff,
            market_status="OPEN",
            inplay=True,
            total_matched=1000.0,
            runners=[runner(1, "Arsenal"), runner(2, "Chelsea"), runner(3, "The Draw")],
        )
        client = MagicMock()
        client.list_events.return_value = ["1"]
        client.list_books.return_value = [book]
        return await fetch_betfair_exchange.ingest_sport(
            SPORT_CONFIG["soccer_epl"], client=client, lookahead_hours=72, snapshot_time=NOW
        )

    async def test_the_in_play_book_accounts_for_its_event_so_the_check_passes(
        self, pglite_async_session: Any
    ) -> None:
        kickoff = NOW + timedelta(hours=1)
        pglite_async_session.add(self._event(1, "Arsenal", "Chelsea", kickoff))
        await pglite_async_session.commit()

        stats = await self._ingest(kickoff)

        assert stats.accounted_fixtures == {EventKey(1)}
        await require_rows_for_fixtures(
            fetch_betfair_exchange.SOURCE,
            {"soccer_epl": stats.snapshots_stored},
            accounted_by_sport={"soccer_epl": stats.accounted_fixtures},
            now=NOW,
        )

    async def test_another_pre_play_fixture_with_no_rows_still_raises(
        self, pglite_async_session: Any
    ) -> None:
        kickoff = NOW + timedelta(hours=1)
        pglite_async_session.add_all(
            [
                self._event(1, "Arsenal", "Chelsea", kickoff),
                self._event(2, "Tottenham", "Everton", kickoff + timedelta(hours=2)),
            ]
        )
        await pglite_async_session.commit()

        stats = await self._ingest(kickoff)

        with pytest.raises(EmptyCollectionError, match="with 1 fixture"):
            await require_rows_for_fixtures(
                fetch_betfair_exchange.SOURCE,
                {"soccer_epl": stats.snapshots_stored},
                accounted_by_sport={"soccer_epl": stats.accounted_fixtures},
                now=NOW,
            )


@pytest.mark.usefixtures("guard_session")
class TestBetfairNbaAccounting:
    """An open book with no back prices accounts for its event, as a suspended one does."""

    @pytest.fixture(autouse=True)
    def session_maker(self, monkeypatch: pytest.MonkeyPatch, pglite_async_session: Any) -> None:
        monkeypatch.setattr(
            fetch_betfair_exchange, "async_session_maker", SessionMaker(pglite_async_session)
        )

    async def test_an_open_unpriced_book_accounts_for_its_event(
        self, pglite_async_session: Any
    ) -> None:
        tip = NOW + timedelta(hours=1)
        pglite_async_session.add(
            Event(
                id=1,
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=tip,
                home_team="Utah Jazz",
                away_team="Portland Trail Blazers",
                status=EventStatus.SCHEDULED,
            )
        )
        await pglite_async_session.commit()
        book = BetfairBook(
            market_id="1.1",
            market_type="MATCH_ODDS",
            betfair_event_id="1",
            betfair_event_name="Portland Trail Blazers @ Utah Jazz",
            market_start_time=tip,
            market_status="OPEN",
            inplay=False,
            total_matched=10.0,
            runners=[
                BetfairRunner(1, "Portland Trail Blazers", None, None, None, None, None),
                BetfairRunner(2, "Utah Jazz", None, None, None, None, None),
            ],
        )
        client = MagicMock()
        client.list_events.return_value = ["1"]
        client.list_books.return_value = [book]

        stats = await fetch_betfair_exchange.ingest_sport(
            SPORT_CONFIG["basketball_nba"], client=client, lookahead_hours=72, snapshot_time=NOW
        )

        assert (stats.snapshots_stored, stats.accounted_fixtures) == (0, {EventKey(1)})
        await require_rows_for_fixtures(
            fetch_betfair_exchange.SOURCE,
            {"basketball_nba": stats.snapshots_stored},
            accounted_by_sport={"basketball_nba": stats.accounted_fixtures},
            now=NOW,
        )


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
        monkeypatch.setattr("odds_core.alerts.send_job_failure", AsyncMock())

    @staticmethod
    def _ingest(monkeypatch: pytest.MonkeyPatch, processed: int) -> None:
        sport_result = SimpleNamespace(
            sport_key="soccer_epl",
            processed_events=processed,
            total_events=processed,
            quota_remaining=None,
            quota_used=None,
            failures=[],
            dropped_games=[],
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
        self, monkeypatch: pytest.MonkeyPatch, seed: Callable[..., Any], alerts: _AlertRecorder
    ) -> None:
        await seed(datetime.now(UTC) + timedelta(hours=10))
        self._ingest(monkeypatch, 0)

        with pytest.raises(EmptyCollectionError):
            await fetch_odds.main(JobContext(sport="soccer_epl"))

        import odds_core.alerts

        odds_core.alerts.send_job_failure.assert_awaited_once()  # type: ignore[attr-defined]
        assert odds_core.alerts.send_job_failure.await_args.args[0] == "fetch-odds-epl"  # type: ignore[attr-defined]
        assert alerts.heartbeats == []

    async def test_zero_rows_without_fixtures_passes(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        self._ingest(monkeypatch, 0)

        await fetch_odds.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == ["heartbeat:fetch-odds-epl"]


class TestResultsJob:
    async def test_a_settlement_error_fails_the_job_without_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        stats = results_run.ResultsStats(errors=["Fulham vs Burnley: boom"])
        monkeypatch.setattr(
            fetch_oddsportal_results, "process_results", AsyncMock(return_value=stats)
        )
        monkeypatch.setattr(results_run, "self_schedule", AsyncMock())
        monkeypatch.setattr(results_run, "record_run_outcome", AsyncMock())

        with pytest.raises(RuntimeError, match="failed to settle 1 event"):
            await fetch_oddsportal_results.main(JobContext(sport="soccer_epl"))

        assert alerts.heartbeats == []

    async def test_a_clean_run_writes_a_heartbeat(
        self, monkeypatch: pytest.MonkeyPatch, alerts: _AlertRecorder
    ) -> None:
        monkeypatch.setattr(
            fetch_oddsportal_results,
            "process_results",
            AsyncMock(return_value=results_run.ResultsStats()),
        )
        monkeypatch.setattr(results_run, "self_schedule", AsyncMock())
        record_outcome = AsyncMock()
        monkeypatch.setattr(results_run, "record_run_outcome", record_outcome)

        await fetch_oddsportal_results.main(JobContext(sport="soccer_epl"))

        record_outcome.assert_awaited_once()
        assert record_outcome.await_args.args[:2] == ("fetch-oddsportal-results-epl", "soccer_epl")
        assert alerts.heartbeats == ["heartbeat:fetch-oddsportal-results-epl"]
