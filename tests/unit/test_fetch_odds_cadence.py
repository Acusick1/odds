from datetime import UTC, datetime, timedelta

from odds_lambda.jobs.fetch_odds import CADENCE
from odds_lambda.scheduling.decision import decide_forward

UNITS_PER_FETCH = 3
MONTHLY_QUOTA = 500
SCORES_RESERVE = 150
WEEKEND_EQUIVALENTS_PER_MONTH = 5
WEEKEND_BUDGET = (MONTHLY_QUOTA - SCORES_RESERVE) // WEEKEND_EQUIVALENTS_PER_MONTH
LONGEST_ALLOWED_GAP = timedelta(hours=48)
WEEK_START = datetime(2026, 11, 9, tzinfo=UTC)

WEEKEND_KICKOFFS = [
    WEEK_START + timedelta(weeks=week, days=days, hours=hours)
    for week in (0, 1)
    for days, hours in [(4, 20), (5, 12.5), (5, 15), (5, 17.5), (6, 14), (6, 16.5)]
]


async def _run_times(kickoffs: list[datetime], start: datetime, end: datetime) -> list[datetime]:
    runs: list[datetime] = []
    now = start
    while now < end:
        upcoming = [k for k in kickoffs if k > now]
        decision = await decide_forward(
            "soccer_epl", CADENCE, now=now, next_kickoff=min(upcoming) if upcoming else None
        )
        if decision.should_execute:
            runs.append(now)
        assert decision.next_execution is not None
        now = decision.next_execution
    return runs


class TestFetchOddsCadence:
    async def test_busy_weekend_fits_the_monthly_quota_budget(self) -> None:
        runs = await _run_times(WEEKEND_KICKOFFS, WEEK_START, WEEK_START + timedelta(days=7))

        assert len(runs) * UNITS_PER_FETCH <= WEEKEND_BUDGET

    async def test_runs_stay_within_the_heartbeat_gap(self) -> None:
        end = WEEK_START + timedelta(days=7)
        runs = await _run_times(WEEKEND_KICKOFFS, WEEK_START, end)

        gaps = [b - a for a, b in zip(runs, runs[1:], strict=False)]
        assert max(gaps) <= LONGEST_ALLOWED_GAP

    async def test_closing_tier_polls_every_two_hours(self) -> None:
        now = WEEK_START
        decision = await decide_forward(
            "soccer_epl", CADENCE, now=now, next_kickoff=now + timedelta(hours=1)
        )

        assert decision.next_execution == now + timedelta(hours=2)

    async def test_pregame_tier_polls_every_three_hours(self) -> None:
        now = WEEK_START
        decision = await decide_forward(
            "soccer_epl", CADENCE, now=now, next_kickoff=now + timedelta(hours=6)
        )

        assert decision.next_execution == now + timedelta(hours=3)
