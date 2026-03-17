"""
5-Layer Feature Pipeline for ML Training.

Separates data collection from model-specific formatting via five layers:
  1. Collection   — EventDataBundle (loads all data once per event)
  2. Sampling     — SnapshotSampler protocol (TierSampler / TimeRangeSampler)
  3. Target       — pure functions from sequence_loader
  4. Adapters     — FeatureAdapter protocol (XGBoostAdapter, LSTMAdapter)
  5. Orchestrator — single prepare_training_data() entry point

Example:
    ```python
    from odds_analytics.feature_groups import prepare_training_data

    result = await prepare_training_data(events, session, config)
    X_train, y_train = result.X, result.y
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
import structlog
from odds_core.game_log_models import NbaTeamGameLog
from odds_core.injury_models import InjuryReport
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.player_stats_models import NbaPlayerSeasonStats
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import TabularFeatureExtractor
from odds_analytics.polymarket_features import (
    CrossSourceFeatureExtractor,
    CrossSourceFeatures,
    PolymarketFeatureExtractor,
    PolymarketTabularFeatures,
    resolve_home_outcome_index,
)
from odds_analytics.sequence_loader import (
    calculate_devigged_bookmaker_target,
    calculate_devigged_totals_target,
    calculate_regression_target,
    extract_devigged_h2h_probs,
    extract_devigged_totals_probs,
    extract_odds_from_snapshot,
    load_sequences_for_event,
)

if TYPE_CHECKING:
    from odds_analytics.epl_lineup_features import LineupCache
    from odds_analytics.fpl_availability_features import FplAvailabilityCache
    from odds_analytics.match_stats_features import MatchStatsCache
    from odds_analytics.training.config import FeatureConfig

logger = structlog.get_logger()

__all__ = [
    "PMEventContext",
    "EventDataBundle",
    "collect_event_data",
    "TierSampler",
    "TimeRangeSampler",
    "AdapterOutput",
    "XGBoostAdapter",
    "LSTMAdapter",
    "PreparedFeatureData",
    "prepare_training_data",
    "filter_completed_events",
    "resolve_outcome_name",
    "snapshot_has_bookmaker",
]


# =============================================================================
# Layer 1: Collection
# =============================================================================


@dataclass
class PMEventContext:
    """Polymarket context for a sportsbook event."""

    pm_event: PolymarketEvent
    sb_event: Event
    market: PolymarketMarket
    home_idx: int


@dataclass
class EventDataBundle:
    """All pre-loaded data for a single event."""

    event: Event
    snapshots: list[OddsSnapshot]
    closing_snapshot: OddsSnapshot | None
    pm_context: PMEventContext | None
    pm_prices: list[PolymarketPriceSnapshot] = field(default_factory=list)
    pm_orderbooks: list[PolymarketOrderBookSnapshot] = field(default_factory=list)
    injury_reports: list[InjuryReport] = field(default_factory=list)
    player_stats: dict[str, NbaPlayerSeasonStats] = field(default_factory=dict)
    game_logs: list[NbaTeamGameLog] = field(default_factory=list)
    prior_season_events: list[Event] = field(default_factory=list)
    prior_match_stats: dict[str, list[dict[str, int]]] = field(default_factory=dict)
    fixtures_df: pd.DataFrame | None = None
    lineup_cache: LineupCache | None = None
    fpl_availability_cache: FplAvailabilityCache | None = None
    sequences: list[list[Odds]] = field(default_factory=list)


def _snapshot_has_market(snapshot: OddsSnapshot, market: str) -> bool:
    """Check if a snapshot's raw_data contains data for the given market key."""
    raw = snapshot.raw_data
    if not raw or "bookmakers" not in raw:
        return False
    for bm in raw["bookmakers"]:
        for mkt in bm.get("markets", []):
            if mkt.get("key") == market:
                return True
    return False


def snapshot_has_bookmaker(snapshot: OddsSnapshot, bookmaker_key: str, market: str) -> bool:
    """Check if a snapshot's raw_data contains odds from a specific bookmaker for a market."""
    raw = snapshot.raw_data
    if not raw or "bookmakers" not in raw:
        return False
    for bm in raw["bookmakers"]:
        if bm.get("key") == bookmaker_key:
            for mkt in bm.get("markets", []):
                if mkt.get("key") == market:
                    return True
    return False


def _should_filter_missing_sharp(config: FeatureConfig) -> bool:
    """Whether to exclude events missing sharp bookmaker closing odds.

    Only applies when sharp_bookmakers differ from target_bookmaker — if they
    match, there are no cross-source features that would degrade from missing data.
    """
    return set(config.sharp_bookmakers) != {config.target_bookmaker}


async def collect_event_data(
    event: Event,
    session: AsyncSession,
    config: FeatureConfig,
    standings_cache: dict[str, list[Event]] | None = None,
    match_stats_cache: MatchStatsCache | None = None,
    fixtures_df: pd.DataFrame | None = None,
    lineup_cache: LineupCache | None = None,
    fpl_availability_cache: FplAvailabilityCache | None = None,
) -> EventDataBundle:
    """Load all data for an event in bulk (minimises per-snapshot DB queries).

    - Loads all OddsSnapshot records once
    - Filters snapshots to those containing the configured market
    - Finds closing snapshot (prefers target bookmaker when applicable)
    - Loads PM context (PM event, moneyline market, home_idx)
    - Bulk-loads all PM prices + orderbooks for the market (2 queries total)
    - Loads sequences via load_sequences_for_event when trajectory features requested
    """
    from odds_lambda.storage.polymarket_reader import PolymarketReader
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)
    all_snapshots_raw = await reader.get_snapshots_for_event(event.id)

    # Filter to snapshots containing the configured market so the sampler
    # doesn't pick snapshots with data for a different market (e.g. h2h
    # snapshot when we need totals).
    market = config.primary_market
    all_snapshots = [s for s in all_snapshots_raw if _snapshot_has_market(s, market)]

    # Derive closing snapshot from already-loaded snapshots (avoid extra DB query)
    closing_tier_value = config.closing_tier.value
    closing_candidates = [s for s in all_snapshots if s.fetch_tier == closing_tier_value]
    closing_snapshot = _select_closing_snapshot(closing_candidates, event, config)

    # PM context (optional)
    pm_context: PMEventContext | None = None
    pm_prices: list[PolymarketPriceSnapshot] = []
    pm_orderbooks: list[PolymarketOrderBookSnapshot] = []

    if "polymarket" in config.feature_groups:
        pm_result = await session.execute(
            select(PolymarketEvent).where(PolymarketEvent.event_id == event.id)
        )
        pm_event = pm_result.scalars().first()

        if pm_event is not None:
            pm_reader = PolymarketReader(session)
            market = await pm_reader.get_moneyline_market(pm_event.id)

            if market is not None:
                home_idx = resolve_home_outcome_index(market, event.home_team)
                if home_idx is not None:
                    pm_context = PMEventContext(
                        pm_event=pm_event,
                        sb_event=event,
                        market=market,
                        home_idx=home_idx,
                    )
                    # Bulk-load all PM data for this market (2 queries)
                    pm_prices = await pm_reader.get_prices_for_market(market.id)
                    pm_orderbooks = await pm_reader.get_orderbooks_for_market(market.id)
                else:
                    logger.debug(
                        "pm_home_outcome_unresolved",
                        event_id=event.id,
                        home_team=event.home_team,
                    )

    # Injury reports (bulk-load all, filter by time in adapter)
    injury_reports: list[InjuryReport] = []
    player_stats: dict[str, NbaPlayerSeasonStats] = {}
    if "injuries" in config.feature_groups:
        from odds_lambda.storage.injury_reader import InjuryReader

        injury_reader = InjuryReader(session)
        injury_reports = await injury_reader.get_injuries_for_event(event.id)

        # Load player stats for impact weighting
        if injury_reports:
            from odds_lambda.polymarket_matching import CANONICAL_TO_ABBREV
            from odds_lambda.storage.pbpstats_reader import PbpStatsReader

            from odds_analytics.injury_features import date_to_nba_season

            season = date_to_nba_season(event.commence_time)
            team_abbrevs = [
                a
                for a in [
                    CANONICAL_TO_ABBREV.get(event.home_team),
                    CANONICAL_TO_ABBREV.get(event.away_team),
                ]
                if a is not None
            ]
            if team_abbrevs:
                pbp_reader = PbpStatsReader(session)
                player_stats = await pbp_reader.get_players_for_teams(team_abbrevs, season)

    # Game logs for rest/schedule features
    game_logs: list[NbaTeamGameLog] = []
    if "rest" in config.feature_groups:
        from odds_lambda.storage.game_log_reader import GameLogReader

        gl_reader = GameLogReader(session)
        event_logs = await gl_reader.get_game_logs_for_event(event.id)
        all_logs = list(event_logs)
        for log in event_logs:
            prev = await gl_reader.get_team_previous_game(log.team_abbreviation, log.game_date)
            if prev:
                all_logs.append(prev)
        game_logs = all_logs

    # Prior season events for standings / epl_schedule features (from preloaded cache)
    prior_season_events: list[Event] = []
    if {"standings", "epl_schedule"} & set(config.feature_groups) and standings_cache is not None:
        from odds_analytics.standings_features import get_prior_events_from_cache

        prior_season_events = get_prior_events_from_cache(standings_cache, event)

    # Prior match stats for rolling match stats features (from preloaded cache)
    prior_match_stats: dict[str, list[dict[str, int]]] = {}
    if "match_stats" in config.feature_groups and match_stats_cache is not None:
        from odds_analytics.match_stats_features import get_prior_match_stats_from_cache

        prior_match_stats = get_prior_match_stats_from_cache(match_stats_cache, event)

    # Sequences for LSTM adapter
    sequences: list[list[Odds]] = []
    if config.adapter == "lstm":
        sequences = await load_sequences_for_event(event.id, session)

    return EventDataBundle(
        event=event,
        snapshots=all_snapshots,
        closing_snapshot=closing_snapshot,
        pm_context=pm_context,
        pm_prices=pm_prices,
        pm_orderbooks=pm_orderbooks,
        injury_reports=injury_reports,
        player_stats=player_stats,
        game_logs=game_logs,
        prior_season_events=prior_season_events,
        prior_match_stats=prior_match_stats,
        fixtures_df=fixtures_df,
        lineup_cache=lineup_cache,
        fpl_availability_cache=fpl_availability_cache,
        sequences=sequences,
    )


# =============================================================================
# Layer 2: Sampling
# =============================================================================


class SnapshotSampler(Protocol):
    """Protocol for snapshot sampling strategies."""

    def sample(self, bundle: EventDataBundle) -> list[OddsSnapshot]:
        """Return snapshots to use as decision points for this event."""
        ...


class TierSampler:
    """Returns the latest snapshot no closer to game than the decision tier (single row per event).

    Snapshots in the decision tier or any earlier tier (further from game) are
    candidates. When required_bookmakers is set, only snapshots containing all
    specified bookmakers for the given market are eligible. The most recent
    candidate by wall-clock time is returned.
    """

    def __init__(
        self,
        decision_tier: str,
        required_bookmakers: list[str] | None = None,
        market: str = "h2h",
    ) -> None:
        from odds_lambda.fetch_tier import FetchTier

        self._decision_tier = FetchTier(decision_tier)
        self._required_bookmakers = required_bookmakers or []
        self._market = market

    def sample(self, bundle: EventDataBundle) -> list[OddsSnapshot]:
        from odds_lambda.fetch_tier import FetchTier

        tier_order = FetchTier.get_priority_order()  # CLOSING first (closest)
        decision_idx = tier_order.index(self._decision_tier)

        candidates = []
        for s in bundle.snapshots:
            if s.fetch_tier:
                try:
                    tier = FetchTier(s.fetch_tier)
                    # IN_PLAY is placed last in priority order but is actually
                    # *closer* to (past) game time than any pre-game tier.
                    # Exclude it unless explicitly requested.
                    if tier == FetchTier.IN_PLAY and self._decision_tier != FetchTier.IN_PLAY:
                        continue
                    tier_idx = tier_order.index(tier)
                    if tier_idx >= decision_idx:
                        candidates.append(s)
                except ValueError:
                    pass

        # Filter to snapshots that contain at least one required bookmaker
        # (priority-ordered fallback: e.g. [pinnacle, betfair_exchange])
        if self._required_bookmakers:
            candidates = [
                s
                for s in candidates
                if any(
                    snapshot_has_bookmaker(s, bm, self._market) for bm in self._required_bookmakers
                )
            ]

        if not candidates:
            return []

        return [max(candidates, key=lambda s: s.snapshot_time)]


class TimeRangeSampler:
    """Stratified sampling across [min_hours, max_hours] before game."""

    def __init__(self, min_hours: float, max_hours: float, max_samples: int) -> None:
        self._min_hours = min_hours
        self._max_hours = max_hours
        self._max_samples = max_samples

    def sample(self, bundle: EventDataBundle) -> list[OddsSnapshot]:
        event = bundle.event
        commence = event.commence_time

        range_start = commence - timedelta(hours=self._max_hours)
        range_end = commence - timedelta(hours=self._min_hours)

        in_range = [s for s in bundle.snapshots if range_start <= s.snapshot_time <= range_end]

        if not in_range:
            return []

        if len(in_range) <= self._max_samples:
            return sorted(in_range, key=lambda s: s.snapshot_time)

        # Stratified: divide range into equal bins, pick nearest to each midpoint
        range_seconds = (range_end - range_start).total_seconds()
        bin_size = range_seconds / self._max_samples
        selected: list[OddsSnapshot] = []

        for i in range(self._max_samples):
            bin_mid = range_start + timedelta(seconds=(i + 0.5) * bin_size)
            best: OddsSnapshot | None = None
            best_diff = float("inf")

            for s in in_range:
                diff = abs((s.snapshot_time - bin_mid).total_seconds())
                if diff < best_diff:
                    best_diff = diff
                    best = s

            if best is not None and best not in selected:
                selected.append(best)

        selected.sort(key=lambda s: s.snapshot_time)
        return selected


def _make_sampler(config: FeatureConfig) -> SnapshotSampler:
    """Instantiate the sampler specified in config.sampling."""
    sc = config.sampling
    if sc.strategy == "tier":
        required_bms = config.sharp_bookmakers if _should_filter_missing_sharp(config) else None
        return TierSampler(sc.decision_tier.value, required_bms, config.primary_market)
    return TimeRangeSampler(sc.min_hours, sc.max_hours, sc.max_samples_per_event)


# =============================================================================
# Helpers
# =============================================================================


def filter_completed_events(events: list[Event]) -> list[Event]:
    """Filter to events with final scores."""
    return [
        e
        for e in events
        if e.status == EventStatus.FINAL and e.home_score is not None and e.away_score is not None
    ]


def make_backtest_event(event: Event) -> BacktestEvent:
    """Convert Event to BacktestEvent for feature extraction."""
    return BacktestEvent(
        id=event.id,
        commence_time=event.commence_time,
        home_team=event.home_team,
        away_team=event.away_team,
        home_score=event.home_score,
        away_score=event.away_score,
        status=event.status,
    )


def _find_nearest_pm_price(
    prices: list[PolymarketPriceSnapshot],
    target_time: datetime,
    tolerance_minutes: int,
) -> PolymarketPriceSnapshot | None:
    """Return the price snapshot closest to (and not after) target_time within tolerance."""
    time_lower = target_time - timedelta(minutes=tolerance_minutes)
    candidates = [p for p in prices if time_lower <= p.snapshot_time <= target_time]
    if not candidates:
        return None
    return min(candidates, key=lambda p: abs((p.snapshot_time - target_time).total_seconds()))


def _find_nearest_pm_orderbook(
    orderbooks: list[PolymarketOrderBookSnapshot],
    target_time: datetime,
    tolerance_minutes: int,
) -> PolymarketOrderBookSnapshot | None:
    """Return the orderbook snapshot closest to (and not after) target_time within tolerance."""
    time_lower = target_time - timedelta(minutes=tolerance_minutes)
    candidates = [o for o in orderbooks if time_lower <= o.snapshot_time <= target_time]
    if not candidates:
        return None
    return min(candidates, key=lambda o: abs((o.snapshot_time - target_time).total_seconds()))


def _find_velocity_prices(
    prices: list[PolymarketPriceSnapshot],
    target_time: datetime,
    velocity_window_hours: float,
) -> list[PolymarketPriceSnapshot]:
    """Return prices in the velocity window ending at target_time."""
    window_start = target_time - timedelta(hours=velocity_window_hours)
    return [p for p in prices if window_start <= p.snapshot_time <= target_time]


# =============================================================================
# Layer 4: Adapters
# =============================================================================


@dataclass
class AdapterOutput:
    """Structured output from a FeatureAdapter.transform() call."""

    features: np.ndarray  # 1D for XGBoost, 2D (timesteps, features) for LSTM
    mask: np.ndarray | None = None  # (timesteps,) boolean, LSTM only
    static_features: np.ndarray | None = None  # 1D static feature vector, LSTM only


class FeatureAdapter(Protocol):
    """Protocol for model-specific feature formatting."""

    def feature_names(self, config: FeatureConfig) -> list[str]:
        """Return ordered list of feature names."""
        ...

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: OddsSnapshot,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        """Extract features for one (event, snapshot) pair. Returns None to skip row."""
        ...


_STATIC_FEATURE_GROUPS = frozenset(
    {
        "tabular",
        "polymarket",
        "injuries",
        "rest",
        "standings",
        "match_stats",
        "epl_schedule",
        "epl_lineup",
        "fpl_availability",
    }
)


def _static_feature_group_names(config: FeatureConfig) -> list[str]:
    """Return ordered feature names for all configured static feature groups.

    Does NOT include ``hours_until_event`` (adapter-specific).
    """
    names: list[str] = []
    if "tabular" in config.feature_groups:
        from odds_analytics.feature_extraction import TabularFeatures

        names.extend(f"tab_{n}" for n in TabularFeatures.get_feature_names())
    if "polymarket" in config.feature_groups:
        names.extend(f"pm_{n}" for n in PolymarketTabularFeatures.get_feature_names())
        names.extend(f"xsrc_{n}" for n in CrossSourceFeatures.get_feature_names())
    if "injuries" in config.feature_groups:
        from odds_analytics.injury_features import InjuryFeatures

        names.extend(f"inj_{n}" for n in InjuryFeatures.get_feature_names())
    if "rest" in config.feature_groups:
        from odds_analytics.schedule_features import RestScheduleFeatures

        names.extend(f"rest_{n}" for n in RestScheduleFeatures.get_feature_names())
    if "standings" in config.feature_groups:
        from odds_analytics.standings_features import StandingsFeatures

        names.extend(f"stnd_{n}" for n in StandingsFeatures.get_feature_names())
    if "match_stats" in config.feature_groups:
        from odds_analytics.match_stats_features import MatchStatsFeatures

        names.extend(f"mstat_{n}" for n in MatchStatsFeatures.get_feature_names())
    if "epl_schedule" in config.feature_groups:
        from odds_analytics.epl_schedule_features import EplScheduleFeatures

        names.extend(f"eplsched_{n}" for n in EplScheduleFeatures.get_feature_names())
    if "epl_lineup" in config.feature_groups:
        from odds_analytics.epl_lineup_features import EplLineupFeatures

        names.extend(f"epllu_{n}" for n in EplLineupFeatures.get_feature_names())
    if "fpl_availability" in config.feature_groups:
        from odds_analytics.fpl_availability_features import FplAvailabilityFeatures

        names.extend(f"fplav_{n}" for n in FplAvailabilityFeatures.get_feature_names())
    return names


def resolve_outcome_name(config: FeatureConfig, event: Event) -> str:
    """Map config outcome to the outcome name stored in the database."""
    if config.outcome in ("over", "under"):
        return config.outcome.capitalize()
    return event.home_team if config.outcome == "home" else event.away_team


def _extract_static_feature_parts(
    bundle: EventDataBundle,
    snapshot: OddsSnapshot,
    config: FeatureConfig,
) -> list[np.ndarray] | None:
    """Extract static feature arrays for all configured groups.

    Returns ``None`` when tabular is configured but odds extraction fails
    (the row cannot be used). Other groups NaN-fill on failure.
    Does NOT include ``hours_until_event``.
    """
    event = bundle.event
    market = config.primary_market
    outcome = resolve_outcome_name(config, event)
    backtest_event = make_backtest_event(event)
    tab_extractor = TabularFeatureExtractor.from_config(config)

    parts: list[np.ndarray] = []

    # --- Tabular features ---
    if "tabular" in config.feature_groups:
        snap_odds = extract_odds_from_snapshot(snapshot, event.id, market=market)
        if not snap_odds:
            return None
        try:
            tab_feats = tab_extractor.extract_features(
                event=backtest_event,
                odds_data=snap_odds,
                outcome=outcome,
                market=market,
            )
            parts.append(tab_feats.to_array())
        except Exception:
            logger.debug("tabular_extraction_failed", event_id=event.id)
            return None

    # --- Polymarket features (NaN-fill when unavailable to keep row) ---
    if "polymarket" in config.feature_groups:
        n_pm = len(PolymarketTabularFeatures.get_feature_names())
        n_xsrc = len(CrossSourceFeatures.get_feature_names())
        nan_block = np.full(n_pm + n_xsrc, np.nan)

        if bundle.pm_context is None:
            parts.append(nan_block)
        else:
            ctx = bundle.pm_context
            target_time = snapshot.snapshot_time
            price_snap = _find_nearest_pm_price(
                bundle.pm_prices, target_time, config.pm_price_tolerance_minutes
            )

            if price_snap is None:
                parts.append(nan_block)
            else:
                ob_snap = _find_nearest_pm_orderbook(
                    bundle.pm_orderbooks, target_time, config.pm_price_tolerance_minutes
                )
                recent = _find_velocity_prices(
                    bundle.pm_prices, target_time, config.pm_velocity_window_hours
                )

                try:
                    pm_extractor = PolymarketFeatureExtractor(
                        velocity_window_hours=config.pm_velocity_window_hours
                    )
                    xsrc_extractor = CrossSourceFeatureExtractor()

                    pm_feats = pm_extractor.extract(
                        price_snapshot=price_snap,
                        orderbook_snapshot=ob_snap,
                        recent_prices=recent,
                        home_outcome_index=ctx.home_idx,
                    )

                    # Try to align SB odds for cross-source features
                    sb_feats = None
                    sb_odds_at_time = extract_odds_from_snapshot(snapshot, event.id, market=market)
                    if sb_odds_at_time:
                        try:
                            sb_feats = tab_extractor.extract_features(
                                event=backtest_event,
                                odds_data=sb_odds_at_time,
                                outcome=outcome,
                                market=market,
                            )
                        except Exception:
                            pass

                    xsrc_feats = xsrc_extractor.extract(
                        pm_features=pm_feats,
                        sb_features=sb_feats,
                    )
                    parts.append(np.concatenate([pm_feats.to_array(), xsrc_feats.to_array()]))
                except Exception:
                    logger.debug("pm_feature_extraction_failed", event_id=event.id)
                    parts.append(nan_block)

    # --- Injury features (NaN-fill when unavailable to keep row) ---
    if "injuries" in config.feature_groups:
        from odds_analytics.injury_features import InjuryFeatures, extract_injury_features

        n_inj = len(InjuryFeatures.get_feature_names())
        nan_block_inj = np.full(n_inj, np.nan)

        if not bundle.injury_reports:
            parts.append(nan_block_inj)
        else:
            try:
                inj_feats = extract_injury_features(
                    bundle.injury_reports,
                    event,
                    snapshot.snapshot_time,
                    player_stats=bundle.player_stats,
                )
                parts.append(inj_feats.to_array())
            except Exception:
                logger.debug("injury_feature_extraction_failed", event_id=event.id)
                parts.append(nan_block_inj)

    # --- Rest/schedule features (NaN-fill when unavailable to keep row) ---
    if "rest" in config.feature_groups:
        from odds_analytics.schedule_features import RestScheduleFeatures, extract_rest_features

        n_rest = len(RestScheduleFeatures.get_feature_names())
        nan_block_rest = np.full(n_rest, np.nan)

        if not bundle.game_logs:
            parts.append(nan_block_rest)
        else:
            try:
                rest_feats = extract_rest_features(bundle.game_logs, event)
                parts.append(rest_feats.to_array())
            except Exception:
                logger.debug("rest_feature_extraction_failed", event_id=event.id)
                parts.append(nan_block_rest)

    # --- Standings features (NaN-fill when unavailable to keep row) ---
    if "standings" in config.feature_groups:
        from odds_analytics.standings_features import StandingsFeatures, extract_standings_features

        n_stnd = len(StandingsFeatures.get_feature_names())
        nan_block_stnd = np.full(n_stnd, np.nan)

        if not bundle.prior_season_events:
            parts.append(nan_block_stnd)
        else:
            try:
                stnd_feats = extract_standings_features(
                    bundle.prior_season_events, event, form_window=config.form_window
                )
                parts.append(stnd_feats.to_array())
            except Exception:
                logger.debug("standings_feature_extraction_failed", event_id=event.id)
                parts.append(nan_block_stnd)

    # --- Match stats features (NaN-fill when unavailable to keep row) ---
    if "match_stats" in config.feature_groups:
        from odds_analytics.match_stats_features import (
            MatchStatsFeatures,
            extract_match_stats_features,
        )

        n_mstat = len(MatchStatsFeatures.get_feature_names())
        nan_block_mstat = np.full(n_mstat, np.nan)

        if not bundle.prior_match_stats:
            parts.append(nan_block_mstat)
        else:
            try:
                mstat_feats = extract_match_stats_features(
                    bundle.prior_match_stats, event, window=config.match_stats_window
                )
                parts.append(mstat_feats.to_array())
            except Exception:
                logger.debug("match_stats_feature_extraction_failed", event_id=event.id)
                parts.append(nan_block_mstat)

    # --- EPL schedule features (NaN-fill on failure to keep row) ---
    if "epl_schedule" in config.feature_groups:
        from odds_analytics.epl_schedule_features import (
            EplScheduleFeatures,
            extract_epl_schedule_features,
        )

        n_eplsched = len(EplScheduleFeatures.get_feature_names())
        nan_block_eplsched = np.full(n_eplsched, np.nan)

        try:
            eplsched_feats = extract_epl_schedule_features(
                bundle.prior_season_events, event, fixtures_df=bundle.fixtures_df
            )
            parts.append(eplsched_feats.to_array())
        except Exception:
            logger.debug("epl_schedule_feature_extraction_failed", event_id=event.id)
            parts.append(nan_block_eplsched)

    # --- EPL lineup features (NaN-fill on failure to keep row) ---
    if "epl_lineup" in config.feature_groups:
        from odds_analytics.epl_lineup_features import (
            EplLineupFeatures,
            extract_epl_lineup_features,
        )

        n_epllu = len(EplLineupFeatures.get_feature_names())
        nan_block_epllu = np.full(n_epllu, np.nan)

        try:
            epllu_feats = extract_epl_lineup_features(bundle.lineup_cache, event)
            parts.append(epllu_feats.to_array())
        except Exception:
            logger.debug("epl_lineup_feature_extraction_failed", event_id=event.id)
            parts.append(nan_block_epllu)

    # --- FPL availability features (NaN-fill on failure to keep row) ---
    if "fpl_availability" in config.feature_groups:
        from odds_analytics.fpl_availability_features import (
            FplAvailabilityFeatures,
            extract_fpl_availability_features,
        )

        n_fplav = len(FplAvailabilityFeatures.get_feature_names())
        nan_block_fplav = np.full(n_fplav, np.nan)

        try:
            fplav_feats = extract_fpl_availability_features(bundle.fpl_availability_cache, event)
            parts.append(fplav_feats.to_array())
        except Exception:
            logger.debug("fpl_availability_feature_extraction_failed", event_id=event.id)
            parts.append(nan_block_fplav)

    return parts


class XGBoostAdapter:
    """Tabular feature adapter for XGBoost (and other tree-based models)."""

    def feature_names(self, config: FeatureConfig) -> list[str]:
        return _static_feature_group_names(config) + ["hours_until_event"]

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: OddsSnapshot,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        parts = _extract_static_feature_parts(bundle, snapshot, config)
        if parts is None:
            return None

        hours_until = (bundle.event.commence_time - snapshot.snapshot_time).total_seconds() / 3600
        parts.append(np.array([hours_until]))

        return AdapterOutput(features=np.concatenate(parts))


class LSTMAdapter:
    """Sequence + optional static feature adapter for LSTM models.

    Produces a fixed 14-feature ``SequenceFeatures`` vector per timestep
    via ``SequenceFeatureExtractor``. When static feature groups (tabular,
    injuries, rest, etc.) are configured, also extracts a 1-D static feature
    vector returned via ``AdapterOutput.static_features``.
    """

    def _has_static_groups(self, config: FeatureConfig) -> bool:
        return bool(_STATIC_FEATURE_GROUPS & set(config.feature_groups))

    def feature_names(self, config: FeatureConfig) -> list[str]:
        from odds_analytics.feature_extraction import SequenceFeatures

        names = SequenceFeatures.get_feature_names()
        if self._has_static_groups(config):
            names = names + _static_feature_group_names(config)
        return names

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: OddsSnapshot,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        from odds_analytics.feature_extraction import SequenceFeatureExtractor

        event = bundle.event
        market = config.primary_market
        outcome = resolve_outcome_name(config, event)
        backtest_event = make_backtest_event(event)

        # Filter sequences to those whose first entry is at or before the snapshot time,
        # ensuring no look-ahead bias: snapshot_time is the look-ahead cutoff.
        seqs_up_to = [
            s for s in bundle.sequences if s and s[0].odds_timestamp <= snapshot.snapshot_time
        ]

        extractor = SequenceFeatureExtractor.from_config(config)
        try:
            result = extractor.extract_features(
                event=backtest_event,
                odds_data=seqs_up_to,
                outcome=outcome,
                market=market,
            )
        except Exception as e:
            logger.debug("sequence_extraction_failed", event_id=event.id, error=str(e))
            return None

        # Extract optional static features (injuries, rest, tabular, etc.)
        static_vec: np.ndarray | None = None
        if self._has_static_groups(config):
            parts = _extract_static_feature_parts(bundle, snapshot, config)
            if parts is None:
                return None
            if parts:
                static_vec = np.concatenate(parts)

        return AdapterOutput(
            features=result["sequence"],
            mask=result["mask"],
            static_features=static_vec,
        )


def _make_adapter(config: FeatureConfig) -> FeatureAdapter:
    if config.adapter == "xgboost":
        return XGBoostAdapter()
    if config.adapter == "lstm":
        return LSTMAdapter()
    raise NotImplementedError(f"Adapter '{config.adapter}' is not yet implemented")


# =============================================================================
# Layer 3: Target helpers
# =============================================================================


def _compute_target(
    snapshot: OddsSnapshot,
    closing_snapshot: OddsSnapshot,
    event: Event,
    config: FeatureConfig,
) -> float | None:
    """Compute regression target for one (snapshot, closing) pair."""
    market = config.primary_market

    closing_odds_all = extract_odds_from_snapshot(closing_snapshot, event.id, market=market)

    if config.target_type == "devigged_bookmaker":
        snapshot_odds_all = extract_odds_from_snapshot(snapshot, event.id, market=market)
        if market == "totals":
            outcome_name = resolve_outcome_name(config, event)
            return calculate_devigged_totals_target(
                snapshot_odds_all,
                closing_odds_all,
                bookmaker_key=config.target_bookmaker,
                outcome=outcome_name,
            )
        return calculate_devigged_bookmaker_target(
            snapshot_odds_all,
            closing_odds_all,
            event.home_team,
            event.away_team,
            bookmaker_key=config.target_bookmaker,
        )
    else:
        # "raw": avg implied prob delta (snapshot → closing)
        outcome = resolve_outcome_name(config, event)
        snap_odds = extract_odds_from_snapshot(snapshot, event.id, market=market, outcome=outcome)
        closing_odds = extract_odds_from_snapshot(
            closing_snapshot, event.id, market=market, outcome=outcome
        )
        return calculate_regression_target(snap_odds, closing_odds, market)


def _has_bookmaker_closing(
    closing_snapshot: OddsSnapshot, event: Event, bookmaker_key: str, market: str = "h2h"
) -> bool:
    """Check if closing snapshot has bookmaker data for the target market.

    Only h2h and totals support devigged bookmaker targets. For other markets
    (e.g. spreads), returns False since no devigging function exists.
    """
    closing_odds_all = extract_odds_from_snapshot(closing_snapshot, event.id, market=market)
    if market == "totals":
        return extract_devigged_totals_probs(closing_odds_all, bookmaker_key) is not None
    if market == "h2h":
        return (
            extract_devigged_h2h_probs(
                closing_odds_all, event.home_team, event.away_team, bookmaker_key
            )
            is not None
        )
    return False


def _select_closing_snapshot(
    candidates: list[OddsSnapshot], event: Event, config: FeatureConfig
) -> OddsSnapshot | None:
    """Pick the best closing snapshot from candidates ordered chronologically.

    Assumes candidates are sorted by snapshot_time ascending ([-1] = latest).
    When target_type is bookmaker-specific, prefer candidates with the target
    bookmaker's data. If closing_source_priority is set, prefer sources in that
    order among candidates with the target bookmaker. Falls back to last-by-time.
    """
    if not candidates:
        return None
    if config.target_type == "devigged_bookmaker":
        market = config.primary_market
        with_bookmaker = [
            c
            for c in candidates
            if _has_bookmaker_closing(c, event, config.target_bookmaker, market)
        ]
        if with_bookmaker:
            if config.closing_source_priority:
                for source in config.closing_source_priority:
                    source_matches = [c for c in with_bookmaker if c.api_request_id == source]
                    if source_matches:
                        return source_matches[-1]
            return with_bookmaker[-1]
    return candidates[-1]


# =============================================================================
# Layer 5: Orchestrator
# =============================================================================


async def _load_fixtures_df(session: AsyncSession) -> pd.DataFrame | None:
    """Load all ESPN fixtures from DB into a DataFrame.

    Returns None if no fixtures exist in the database.
    """
    from odds_lambda.storage.espn_fixture_reader import EspnFixtureReader

    reader = EspnFixtureReader(session)
    fixtures = await reader.get_all_fixtures()
    if not fixtures:
        logger.warning("espn_fixtures_not_found_in_db")
        return None

    rows = [f.model_dump(exclude={"id", "created_at"}) for f in fixtures]
    df = pd.DataFrame(rows).rename(columns={"match_round": "round"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    logger.info("espn_fixtures_loaded_from_db", rows=len(df))
    return df


async def _load_lineup_cache(session: AsyncSession) -> LineupCache | None:
    """Load all ESPN lineups from DB and build a per-team starting XI cache.

    Loads the full roster (starters + subs) with player_name so that downstream
    consumers like the FPL availability feature group can do fuzzy name matching.
    The build_lineup_cache function filters to starters internally.

    Returns None if no lineup data exists in the database.
    """
    from odds_lambda.storage.espn_lineup_reader import EspnLineupReader

    from odds_analytics.epl_lineup_features import build_lineup_cache

    reader = EspnLineupReader(session)
    lineups = await reader.get_all_lineups()
    if not lineups:
        logger.warning("espn_lineups_not_found_in_db")
        return None

    rows = [
        lu.model_dump(include={"team", "player_id", "player_name", "date", "starter"})
        for lu in lineups
    ]
    df = pd.DataFrame(rows).rename(columns={"date": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["match_date"] = df["datetime"].dt.date

    logger.info("espn_lineups_loaded_from_db", rows=len(df))
    return build_lineup_cache(df[df["starter"]].drop(columns=["starter"]))


class PreparedFeatureData:
    """Pre-split feature matrix with targets and metadata."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        masks: np.ndarray | None = None,
        event_ids: np.ndarray | None = None,
        static_features: np.ndarray | None = None,
        static_feature_names: list[str] | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.masks = masks
        self.event_ids = event_ids
        self.static_features = static_features
        self.static_feature_names = static_feature_names

    @property
    def num_samples(self) -> int:
        return len(self.X)

    @property
    def num_features(self) -> int:
        return len(self.feature_names)


async def prepare_training_data(
    events: list[Event],
    session: AsyncSession,
    config: FeatureConfig,
) -> PreparedFeatureData:
    """Unified training data preparation using the 5-layer adapter architecture.

    Sampling strategy and target type are both controlled via config:
      - config.sampling.strategy="tier"        → single row per event
      - config.sampling.strategy="time_range"  → multiple rows per event
      - config.target_type="raw"               → avg implied-prob delta
      - config.target_type="devigged_bookmaker" → devigged bookmaker delta

    For devigged_bookmaker, events without target bookmaker closing data are dropped.
    Polymarket features NaN-fill when PM data is unavailable, so events are
    kept rather than dropped when PM is missing.

    Args:
        events: List of Event objects (filters to FINAL status internally)
        session: Async database session
        config: FeatureConfig controlling sampling, features, and target

    Returns:
        PreparedFeatureData with X, y, feature_names, masks, and event_ids for CV
    """
    valid_events = filter_completed_events(events)
    if not valid_events:
        raise ValueError(f"No valid events found in {len(events)} total events")

    adapter = _make_adapter(config)
    sampler = _make_sampler(config)
    feature_names = adapter.feature_names(config)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    masks_list: list[np.ndarray] = []
    static_list: list[np.ndarray] = []
    event_id_list: list[str] = []
    skipped_events = 0
    sharp_filtered_count = 0
    total_rows = 0

    # Determine static feature names upfront (empty list if no static groups)
    lstm_adapter = isinstance(adapter, LSTMAdapter)
    static_names = _static_feature_group_names(config) if lstm_adapter else None

    # Resolve sport_key once for cache loaders
    sport_key = config.sport_key or (valid_events[0].sport_key if valid_events else None)

    # Preload standings cache to avoid N+1 queries (also used by epl_schedule)
    standings_cache: dict[str, list[Event]] | None = None
    if {"standings", "epl_schedule"} & set(config.feature_groups):
        from odds_analytics.standings_features import load_season_events_cache

        if sport_key:
            standings_cache = await load_season_events_cache(session, sport_key)

    # Preload match stats cache to avoid N+1 queries
    match_stats_cache: MatchStatsCache | None = None
    if "match_stats" in config.feature_groups:
        from odds_analytics.match_stats_features import load_match_stats_cache

        if sport_key:
            match_stats_cache = await load_match_stats_cache(session, sport_key)

    # Preload all-competition fixtures DataFrame for rest/congestion features
    fixtures_df: pd.DataFrame | None = None
    if "epl_schedule" in config.feature_groups:
        fixtures_df = await _load_fixtures_df(session)

    # Preload lineup cache for lineup-delta features
    lineup_cache: LineupCache | None = None
    if "epl_lineup" in config.feature_groups:
        lineup_cache = await _load_lineup_cache(session)

    # Preload FPL availability cache for expected-disruption features
    fpl_availability_cache: FplAvailabilityCache | None = None
    if "fpl_availability" in config.feature_groups:
        from odds_analytics.fpl_availability_features import load_fpl_availability_cache

        fpl_availability_cache = load_fpl_availability_cache()

    for event in valid_events:
        # Load all data for this event in bulk
        bundle = await collect_event_data(
            event,
            session,
            config,
            standings_cache=standings_cache,
            match_stats_cache=match_stats_cache,
            fixtures_df=fixtures_df,
            lineup_cache=lineup_cache,
            fpl_availability_cache=fpl_availability_cache,
        )

        # Closing snapshot is required
        if bundle.closing_snapshot is None:
            skipped_events += 1
            continue

        # For devigged_bookmaker, must have bookmaker closing data
        market = config.primary_market
        if config.target_type == "devigged_bookmaker" and not _has_bookmaker_closing(
            bundle.closing_snapshot, event, config.target_bookmaker, market
        ):
            skipped_events += 1
            continue

        # Filter events missing ALL sharp bookmaker closing odds (cross-source features).
        # Events are kept if at least one sharp bookmaker is present, enabling
        # priority-ordered fallback (e.g. [pinnacle, betfair_exchange]).
        if _should_filter_missing_sharp(config):
            has_any_sharp = any(
                snapshot_has_bookmaker(bundle.closing_snapshot, bm, market)
                for bm in config.sharp_bookmakers
            )
            if not has_any_sharp:
                sharp_filtered_count += 1
                continue

        # Get sampled decision snapshots
        decision_snapshots = sampler.sample(bundle)
        if not decision_snapshots:
            skipped_events += 1
            continue

        event_had_rows = False
        for snapshot in decision_snapshots:
            target = _compute_target(snapshot, bundle.closing_snapshot, event, config)
            if target is None:
                continue

            output = adapter.transform(bundle, snapshot, config)
            if output is None:
                continue

            X_list.append(output.features)
            if output.mask is not None:
                masks_list.append(output.mask)
            if output.static_features is not None:
                static_list.append(output.static_features)
            y_list.append(target)
            event_id_list.append(event.id)
            event_had_rows = True
            total_rows += 1

        if not event_had_rows:
            skipped_events += 1

    if not X_list:
        raise ValueError(
            f"No valid training data after processing {len(valid_events)} events "
            f"(skipped {skipped_events})"
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    event_ids = np.array(event_id_list)
    masks = np.array(masks_list, dtype=bool) if masks_list else None

    # Stack static features if collected
    static_features: np.ndarray | None = None
    if static_list:
        static_features = np.array(static_list, dtype=np.float32)

    from odds_analytics.training.feature_selection import apply_variance_filter

    X, feature_names, _ = apply_variance_filter(X, feature_names, config.variance_threshold)

    n_events = len(set(event_id_list))

    if sharp_filtered_count > 0:
        logger.info(
            "excluded_events_missing_sharp_closing",
            count=sharp_filtered_count,
            sharp_bookmakers=config.sharp_bookmakers,
        )

    logger.info(
        "prepared_training_data",
        num_samples=len(X),
        num_events=n_events,
        num_features=len(feature_names),
        num_static_features=static_features.shape[1] if static_features is not None else 0,
        avg_rows_per_event=total_rows / max(n_events, 1),
        skipped_events=skipped_events,
        sharp_filtered=sharp_filtered_count,
        sampling_strategy=config.sampling.strategy,
        target_type=config.target_type,
        target_mean=float(np.mean(y)),
        target_std=float(np.std(y)),
    )

    return PreparedFeatureData(
        X=X,
        y=y,
        feature_names=feature_names,
        masks=masks,
        event_ids=event_ids,
        static_features=static_features,
        static_feature_names=static_names if static_features is not None else None,
    )
