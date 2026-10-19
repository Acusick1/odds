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
# ESPN statuses of a match that was played; rows left behind by a rescheduled match carry others.
FINISHED_FIXTURE_STATUSES: frozenset[str] = frozenset(
    {
        "Full Time",
        "Final Score - After Extra Time",
        "Final Score - After Penalties",
        "Final Score - After Golden Goal",
    }
)


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
                df = df.loc[
                    (df["team"].isin(teams) | df["opponent"].isin(teams))
                    & df["status"].isin(FINISHED_FIXTURE_STATUSES)
                ]
            self._fixtures_by_event[event.id] = df
        return self._fixtures_by_event[event.id]

    def schedule_at(self, event: Event, t: datetime) -> EplScheduleFeatures:
        """Schedule features from played matches known at t; NaN congestion for a team with none."""
        known = results_known_before(t)
        prior = get_prior_events_from_cache(self.season_results, event, before=known)
        fixtures = self._event_fixtures(event)
        if fixtures is None:
            return extract_epl_schedule_features(prior, event, known_at=known)
        fixtures = fixtures.loc[fixtures["date"] < known]
        features = extract_epl_schedule_features(prior, event, fixtures_df=fixtures, known_at=known)
        played = set(fixtures["team"]) | set(fixtures["opponent"])
        if event.home_team not in played:
            features.home_matches_last_14d = None
            features.home_european_last_7d = None
        if event.away_team not in played:
            features.away_matches_last_14d = None
            features.away_european_last_7d = None
        return features

    def standings_at(self, event: Event, t: datetime) -> StandingsFeatures:
        prior = get_prior_events_from_cache(
            self.season_results, event, before=results_known_before(t)
        )
        return extract_standings_features(prior, event, form_window=self.form_window)


async def load_team_history(
    session: AsyncSession,
    sport_key: str,
    form_window: int,
    with_fixtures: bool,
    until: datetime | None = None,
) -> TeamHistory:
    """Team history from matches up to ``until`` when it is given, else from every match."""
    from odds_analytics.feature_groups import load_fixtures_df

    return TeamHistory(
        season_results=await load_season_events_cache(session, sport_key, until),
        fixtures=await load_fixtures_df(session, until) if with_fixtures else None,
        form_window=form_window,
    )
