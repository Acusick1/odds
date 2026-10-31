"""The NBA fetch-odds schedule: runs 3 h and 1 h before each slate's first tip-off, 1 h before
its main tip and in the morning, within a second key's 500 units a month."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from odds_lambda.jobs import fetch_odds
from odds_lambda.jobs.fetch_odds import PLANS
from odds_lambda.scheduling.decision import decide_forward_resilient
from odds_lambda.scheduling.helpers import main_tip, next_slate_main_tip, next_slate_start, slates
from odds_lambda.scheduling.jobs import JobContext

UNITS_PER_FETCH = 3
MONTHLY_QUOTA = 500
NBA = "basketball_nba"
PLAN = PLANS[NBA]
FIRST_DAY = datetime(2026, 11, 2, tzinfo=UTC)  # a Monday
# Tip-off hours after midnight UTC of the day a slate starts: a weeknight's games share 00:00,
# a weekend slate's share 20:00.
WEEKNIGHT = (23, 23, 23.5, 24, 24, 24, 24, 26)
WEEKEND = (17, 20, 20, 20, 22)


def _month_of_kickoffs(days: int = 30, off_days: tuple[int, ...] = ()) -> list[datetime]:
    kickoffs: list[datetime] = []
    for d in range(days):
        if d in off_days:
            continue
        day = FIRST_DAY + timedelta(days=d)
        hours = WEEKEND if day.weekday() >= 5 else WEEKNIGHT
        kickoffs += [day + timedelta(hours=h) for h in hours]
    return kickoffs


async def _run_times(kickoffs: list[datetime], start: datetime, end: datetime) -> list[datetime]:
    async def first_tip(sport_key: str, *, now: datetime) -> datetime | None:
        return next_slate_start(kickoffs, now=now)

    async def main_tip_of(sport_key: str, *, now: datetime) -> datetime | None:
        return next_slate_main_tip(kickoffs, now=now)

    leads = tuple(
        replace(lead, kickoff_lookup=main_tip_of if lead.kickoff_lookup else None)
        for lead in PLAN.leads
    )
    runs: list[datetime] = []
    now = start
    while now < end:
        decision = await decide_forward_resilient(
            [NBA],
            PLAN.cadence,
            overnight=PLAN.overnight,
            now=now,
            next_kickoff_lookup=first_tip,
            leads=leads,
            lead_kickoff_lookup=first_tip,
            run_started=now,
        )
        if decision.should_execute:
            runs.append(now)
        assert decision.next_execution is not None
        now = decision.next_execution
    return runs


def _first_tip(kickoffs: list[datetime], day: datetime) -> datetime:
    """The first tip-off of the slate that starts on ``day`` (the previous night's games run to 03:30)."""
    return min(k for k in kickoffs if day + timedelta(hours=6) <= k < day + timedelta(hours=30))


class TestNbaFetchSchedule:
    async def test_a_weeknight_slate_whose_main_tip_is_within_an_hour_of_its_first_has_three_runs(
        self,
    ) -> None:
        kickoffs = _month_of_kickoffs()
        day = FIRST_DAY + timedelta(days=1)  # Tuesday
        tip = _first_tip(kickoffs, day)
        previous_close_run = _first_tip(kickoffs, FIRST_DAY) - timedelta(hours=1)

        runs = await _run_times(kickoffs, previous_close_run, day + timedelta(days=1))

        assert runs == [
            previous_close_run,
            day + timedelta(hours=10),
            tip - timedelta(hours=3),
            tip - timedelta(hours=1),
        ]

    async def test_a_weekend_slate_adds_a_run_an_hour_before_its_main_tip(self) -> None:
        kickoffs = _month_of_kickoffs()
        day = FIRST_DAY + timedelta(days=5)  # Saturday, first tip 17:00, main tip 20:00
        tip = _first_tip(kickoffs, day)
        previous_close_run = _first_tip(kickoffs, day - timedelta(days=1)) - timedelta(hours=1)

        runs = await _run_times(kickoffs, previous_close_run, day + timedelta(days=1))

        assert runs[:5] == [
            previous_close_run,
            day + timedelta(hours=10),
            tip - timedelta(hours=3),
            tip - timedelta(hours=1),
            day + timedelta(hours=19),
        ]

    async def test_late_games_of_the_night_add_no_runs(self) -> None:
        kickoffs = _month_of_kickoffs()
        day = FIRST_DAY + timedelta(days=1)
        start = day + timedelta(hours=22, minutes=30)

        runs = await _run_times(kickoffs, start, day + timedelta(days=1, hours=9))

        assert runs == [start]

    async def test_a_month_fits_the_second_key_with_room_to_spare(self) -> None:
        kickoffs = _month_of_kickoffs(off_days=(3, 10, 17))
        end = FIRST_DAY + timedelta(days=30)

        runs = await _run_times(kickoffs, FIRST_DAY, end)

        assert len(runs) * UNITS_PER_FETCH <= MONTHLY_QUOTA * 0.7

    async def test_a_late_lead_run_is_taken_when_the_run_that_overran_it_started_before_it(
        self,
    ) -> None:
        tip = datetime(2026, 11, 3, 23, 0, tzinfo=UTC)
        started = tip - timedelta(hours=1, minutes=5)  # a run that began before the 1 h lead
        finished = tip - timedelta(minutes=55)  # and ended after it

        async def lookup(sport_key: str, *, now: datetime) -> datetime | None:
            return tip if now < tip else None

        decision = await decide_forward_resilient(
            [NBA],
            PLAN.cadence,
            overnight=PLAN.overnight,
            now=finished,
            next_kickoff_lookup=lookup,
            leads=PLAN.leads[:2],
            lead_kickoff_lookup=lookup,
            run_started=started,
        )

        assert decision.next_execution == finished

    async def test_an_off_season_gate_checks_in_daily(self) -> None:
        now = FIRST_DAY

        async def no_games(sport_key: str, *, now: datetime) -> datetime | None:
            return None

        decision = await decide_forward_resilient(
            [NBA],
            PLAN.cadence,
            overnight=PLAN.overnight,
            now=now,
            next_kickoff_lookup=no_games,
            leads=PLAN.leads,
            lead_kickoff_lookup=no_games,
        )

        assert decision.next_execution is not None
        assert decision.next_execution - now <= timedelta(hours=24)


class TestFetchOddsRunStarted:
    async def test_the_job_passes_its_start_so_a_late_lead_can_catch_up(self) -> None:
        decide = AsyncMock(
            return_value=type(
                "D",
                (),
                {"should_execute": False, "next_execution": None, "tier": None, "reason": ""},
            )()
        )
        before = datetime.now(UTC)
        with (
            patch.object(fetch_odds, "decide_forward_resilient", decide),
            patch.object(fetch_odds, "build_event_sync_service") as sync,
            patch.object(fetch_odds, "TheOddsAPIClient"),
            patch.object(fetch_odds, "job_alert_context"),
        ):
            sync.return_value.sync_sports = AsyncMock(return_value=[])
            await fetch_odds.main(JobContext(sport=NBA))

        assert {call.kwargs["run_started"] >= before for call in decide.await_args_list} == {True}


class TestSlates:
    def test_games_less_than_six_hours_apart_are_one_slate(self) -> None:
        tip = datetime(2026, 11, 3, 23, 0, tzinfo=UTC)
        kickoffs = [tip, tip + timedelta(hours=5, minutes=59), tip + timedelta(hours=26)]

        assert [len(slate) for slate in slates(kickoffs)] == [2, 1]

    def test_a_later_game_of_a_night_in_progress_does_not_start_a_slate(self) -> None:
        tip = datetime(2026, 11, 3, 23, 0, tzinfo=UTC)
        kickoffs = [tip, tip + timedelta(hours=2), tip + timedelta(hours=26)]

        assert next_slate_start(kickoffs, now=tip + timedelta(hours=1)) == tip + timedelta(hours=26)

    def test_games_six_hours_apart_are_two_slates(self) -> None:
        matinee = datetime(2026, 11, 26, 17, 0, tzinfo=UTC)
        kickoffs = [matinee, matinee + timedelta(hours=6)]

        assert next_slate_start(kickoffs, now=matinee + timedelta(hours=1)) == kickoffs[1]

    def test_no_kickoff_ahead_is_none(self) -> None:
        now = datetime(2026, 11, 3, tzinfo=UTC)

        assert next_slate_start([now - timedelta(hours=1)], now=now) is None

    def test_the_main_tip_is_the_one_most_games_share_and_the_latest_of_a_tie(self) -> None:
        first = datetime(2026, 11, 3, 19, 0, tzinfo=UTC)

        assert main_tip([first, first + timedelta(hours=3), first + timedelta(hours=3)]) == (
            first + timedelta(hours=3)
        )
        assert main_tip([first, first + timedelta(hours=1)]) == first + timedelta(hours=1)

    def test_a_main_tip_within_ninety_minutes_of_the_first_adds_no_lead(self) -> None:
        first = datetime(2026, 11, 3, 23, 0, tzinfo=UTC)
        kickoffs = [first, first + timedelta(hours=1), first + timedelta(hours=1)]

        assert next_slate_main_tip(kickoffs, now=first - timedelta(hours=5)) is None

    def test_a_slate_with_a_later_main_tip_leads_from_that_tip(self) -> None:
        first = datetime(2026, 11, 3, 19, 0, tzinfo=UTC)
        main = first + timedelta(hours=5)
        kickoffs = [first, main, main]

        assert next_slate_main_tip(kickoffs, now=first - timedelta(hours=5)) == main
        assert next_slate_main_tip(kickoffs, now=main) is None
