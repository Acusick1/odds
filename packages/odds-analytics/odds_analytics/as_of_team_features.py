"""EPL schedule and standings features as of a decision time before kick-off."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from odds_core.models import Event

from odds_analytics.epl_schedule_features import (
    EplScheduleFeatures,
    extract_epl_schedule_features,
)
from odds_analytics.standings_features import (
    StandingsCache,
    StandingsFeatures,
    extract_standings_features,
    get_prior_events_from_cache,
    load_season_events_cache,
)

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.ext.asyncio import AsyncSession

# A match counts at t only once its result is known; kick-off plus this covers full time (inferred).
RESULT_LAG = timedelta(hours=3)


def results_known_before(t: datetime) -> datetime:
    """Matches that kicked off strictly before this time have a result known at t."""
    return t - RESULT_LAG


@dataclass
class TeamHistory:
    """Completed EPL matches and all-competition fixtures, read as of any decision time."""

    season_results: StandingsCache
    fixtures: pd.DataFrame | None
    form_window: int
    _fixtures_by_event: dict[str, pd.DataFrame | None] = field(default_factory=dict, repr=False)

    def _event_fixtures(self, event: Event) -> pd.DataFrame | None:
        if event.id not in self._fixtures_by_event:
            df = self.fixtures
            if df is not None:
                teams = {event.home_team, event.away_team}
                df = df.loc[df["team"].isin(teams) | df["opponent"].isin(teams)]
            self._fixtures_by_event[event.id] = df
        return self._fixtures_by_event[event.id]

    def schedule_at(self, event: Event, t: datetime) -> EplScheduleFeatures:
        known = results_known_before(t)
        prior = get_prior_events_from_cache(self.season_results, event, before=known)
        return extract_epl_schedule_features(
            prior, event, fixtures_df=self._event_fixtures(event), known_at=known
        )

    def standings_at(self, event: Event, t: datetime) -> StandingsFeatures:
        prior = get_prior_events_from_cache(
            self.season_results, event, before=results_known_before(t)
        )
        return extract_standings_features(prior, event, form_window=self.form_window)


async def load_team_history(
    session: AsyncSession, sport_key: str, form_window: int, with_fixtures: bool
) -> TeamHistory:
    from odds_analytics.feature_groups import load_fixtures_df

    return TeamHistory(
        season_results=await load_season_events_cache(session, sport_key),
        fixtures=await load_fixtures_df(session) if with_fixtures else None,
        form_window=form_window,
    )
