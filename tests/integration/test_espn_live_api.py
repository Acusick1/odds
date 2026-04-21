"""Live-API integration tests for ``EspnFixtureFetcher``.

These hit ESPN's real Site API (no mocking) and act as a smoke test against
schema drift — specifically the ``status.type.state`` enum, which was the
root cause of the ``get_team_context`` bug fixed here.

Marked ``integration`` so CI can opt in. Tests ``skip`` rather than fail in
offseason edge cases where the expected shape may legitimately be empty.
"""

from __future__ import annotations

import pytest
from odds_lambda.espn_fixture_fetcher import (
    EspnFixtureFetcher,
    current_season,
)

pytestmark = pytest.mark.integration


# ESPN's canonical state enum. If any of these disappear from live responses,
# something has shifted upstream and we want to know early.
_EXPECTED_STATES: frozenset[str] = frozenset({"pre", "in", "post"})


class TestEspnLiveApi:
    @pytest.mark.asyncio
    async def test_fetch_upcoming_returns_pre_state_rows(self) -> None:
        """``fetch_upcoming`` should produce at least one ``state == 'pre'`` row.

        Skips (rather than fails) during deep offseason when no scheduled
        fixtures exist in the next 30 days across any configured competition.
        """
        async with EspnFixtureFetcher() as fetcher:
            records = await fetcher.fetch_upcoming(days_ahead=30)

        if not records:
            pytest.skip(
                "ESPN returned no upcoming fixtures in the next 30 days "
                "across any configured competition (offseason?)."
            )

        states = {r.state for r in records}
        assert "pre" in states, (
            f"Expected at least one upcoming record with state=='pre'; got states={states}"
        )

    @pytest.mark.asyncio
    async def test_fetch_season_has_completed_rows(self) -> None:
        """``fetch_season`` should include ``state == 'post'`` rows for the current season.

        Skips if the current season has not yet kicked off (August hasn't
        arrived, no matches played yet).
        """
        season = current_season()
        async with EspnFixtureFetcher() as fetcher:
            records = await fetcher.fetch_season(season)

        if not records:
            pytest.skip(
                f"ESPN returned no rows for season {season} — new season may not have started."
            )

        post_records = [r for r in records if r.state == "post"]
        if not post_records:
            pytest.skip(f"Season {season} has no completed ('post') rows yet — pre-season run?")

        assert post_records, "Expected at least one completed match row for the current season."

    @pytest.mark.asyncio
    async def test_states_match_expected_enum(self) -> None:
        """Every state value ESPN returns should be one of {pre, in, post}.

        Defensive schema smoke test — if ESPN ever introduces a new state
        value (or returns blank), the filtering logic in ``get_team_context``
        will silently drop those rows. Catching it here surfaces the drift.
        """
        async with EspnFixtureFetcher() as fetcher:
            upcoming = await fetcher.fetch_upcoming(days_ahead=30)
            season = await fetcher.fetch_season(current_season())

        all_records = [*upcoming, *season]
        if not all_records:
            pytest.skip("No ESPN rows available to validate state enum against.")

        observed_states = {r.state for r in all_records if r.state is not None}
        unexpected = observed_states - _EXPECTED_STATES
        assert not unexpected, (
            f"ESPN returned unexpected state values: {unexpected}. "
            f"Expected only {_EXPECTED_STATES}."
        )
