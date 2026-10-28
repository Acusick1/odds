"""
5-Layer Feature Pipeline for ML Training.

Separates data collection from model-specific formatting via five layers:
  1. Collection   — EventDataBundle (loads all data once per event)
  2. Sampling     — SnapshotSampler protocol (TierSampler / TimeRangeSampler)
  3. Target       — pure functions from targets
  4. Adapters     — FeatureAdapter protocol (XGBoostAdapter)
  5. Orchestrator — single prepare_training_data() entry point

Example:
    ```python
    from odds_analytics.feature_groups import prepare_training_data

    result = await prepare_training_data(events, session, config)
    X_train, y_train = result.X, result.y
    ```
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Protocol, get_args

import numpy as np
import pandas as pd
import structlog
from odds_core.as_of import by_snapshot
from odds_core.game_log_models import NbaTeamGameLog
from odds_core.injury_models import InjuryReport
from odds_core.models import Event, EventStatus
from odds_core.player_stats_models import NbaPlayerSeasonStats
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.tier_utils import snapshot_tiers
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.feature_extraction import FeatureEvent, TabularFeatureExtractor
from odds_analytics.polymarket_features import (
    CrossSourceFeatureExtractor,
    CrossSourceFeatures,
    PolymarketFeatureExtractor,
    PolymarketTabularFeatures,
    resolve_home_outcome_index,
)
from odds_analytics.targets import calculate_regression_target
from odds_analytics.training.config import FeatureGroupName

if TYPE_CHECKING:
    from odds_analytics.training.config import FeatureConfig

logger = structlog.get_logger()

__all__ = [
    "PMEventContext",
    "SnapshotQuotes",
    "EventDataBundle",
    "collect_event_data",
    "TierSampler",
    "TimeRangeSampler",
    "AdapterOutput",
    "XGBoostAdapter",
    "PreparedFeatureData",
    "prepare_training_data",
    "filter_completed_events",
    "raw_closes",
    "tiers_of",
    "load_fixtures_df",
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


@dataclass(frozen=True, eq=False)
class SnapshotQuotes:
    """The back quotes of one stored snapshot of one market; never empty."""

    quotes: tuple[Quote, ...]

    @property
    def snapshot_time(self) -> datetime:
        return self.quotes[0].snapshot_time

    @property
    def source(self) -> Source:
        return self.quotes[0].source

    @property
    def time_basis(self) -> TimeBasis:
        return self.quotes[0].time_basis

    @property
    def market(self) -> Market:
        return self.quotes[0].market

    def has_bookmaker(self, bookmaker: str) -> bool:
        return any(q.bookmaker == bookmaker for q in self.quotes)

    def outcome_quotes(self, outcome: PriceOutcome) -> list[Quote]:
        return [q for q in self.quotes if q.outcome is outcome]


@dataclass
class EventDataBundle:
    """All pre-loaded data for a single event."""

    event: Event
    snapshots: list[SnapshotQuotes]
    closes: dict[Source, SnapshotQuotes]
    pm_context: PMEventContext | None
    pm_prices: list[PolymarketPriceSnapshot] = field(default_factory=list)
    pm_orderbooks: list[PolymarketOrderBookSnapshot] = field(default_factory=list)
    injury_reports: list[InjuryReport] = field(default_factory=list)
    player_stats: dict[str, NbaPlayerSeasonStats] = field(default_factory=dict)
    game_logs: list[NbaTeamGameLog] = field(default_factory=list)


def tiers_of(snapshots: Sequence[SnapshotQuotes], kickoff: datetime) -> list[FetchTier]:
    """The tier each of an event's snapshots covers (``odds_lambda.tier_utils.snapshot_tiers``),
    from its time, source, time basis and market; never the stored ``fetch_tier``."""
    return snapshot_tiers(
        [(s.snapshot_time, s.source, s.time_basis, s.market) for s in snapshots], kickoff
    )


def raw_closes(
    snapshots: Sequence[SnapshotQuotes], kickoff: datetime
) -> dict[Source, SnapshotQuotes]:
    """The ``raw`` target's close of each source: its latest snapshot whose tier is CLOSING.

    A decision snapshot is compared with its own source's close, so the target never mixes
    one source's books with another's. This is a multi-book snapshot for the line-movement
    target, not the Betfair Exchange close ``odds_core.betfair_close`` defines.
    """
    closes: dict[Source, SnapshotQuotes] = {}
    for snapshot, tier in zip(snapshots, tiers_of(snapshots, kickoff), strict=True):
        if tier is FetchTier.CLOSING:
            closes[snapshot.source] = snapshot
    return closes


async def collect_event_data(
    event: Event,
    session: AsyncSession,
    config: FeatureConfig,
) -> EventDataBundle:
    """Load all data for an event in bulk (minimises per-snapshot DB queries).

    - Loads the back quotes of the configured market once, as one ``SnapshotQuotes`` per
      snapshot, leaving out Polymarket's, which carry no retail or target prices to build a
      decision row from
    - Takes each source's ``raw`` target close (``raw_closes``)
    - Loads PM context (PM event, moneyline market, home_idx)
    - Bulk-loads all PM prices + orderbooks for the market (2 queries total)
    """
    from odds_lambda.storage.polymarket_reader import PolymarketReader

    quoted = await quotes(session, [event.id], Market(config.primary_market), sides=[Side.BACK])
    all_snapshots = [
        SnapshotQuotes(tuple(group))
        for group in by_snapshot(quoted)
        if group[0].source is not Source.POLYMARKET
    ]
    closes = raw_closes(all_snapshots, event.commence_time)

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

    return EventDataBundle(
        event=event,
        snapshots=all_snapshots,
        closes=closes,
        pm_context=pm_context,
        pm_prices=pm_prices,
        pm_orderbooks=pm_orderbooks,
        injury_reports=injury_reports,
        player_stats=player_stats,
        game_logs=game_logs,
    )


# =============================================================================
# Layer 2: Sampling
# =============================================================================


class SnapshotSampler(Protocol):
    """Protocol for snapshot sampling strategies."""

    def sample(
        self, bundle: EventDataBundle, sources: Collection[Source] | None = None
    ) -> list[SnapshotQuotes]:
        """Return snapshots to use as decision points for this event, only of ``sources`` when
        given (the sources whose close a decision can be measured against)."""
        ...


class TierSampler:
    """Returns the latest snapshot no closer to game than the decision tier (single row per event).

    A snapshot's tier is the one ``tiers_of`` derives, never the stored ``fetch_tier``.
    Snapshots in the decision tier
    or any earlier tier (further from game) are candidates. When required_bookmakers is set,
    only snapshots quoting at least one of them are eligible, and with ``sources`` only
    snapshots of those sources. The most recent candidate by wall-clock time is returned.
    """

    def __init__(
        self,
        decision_tier: str,
        required_bookmakers: list[str] | None = None,
    ) -> None:
        self._decision_tier = FetchTier(decision_tier)
        self._required_bookmakers = required_bookmakers or []

    def sample(
        self, bundle: EventDataBundle, sources: Collection[Source] | None = None
    ) -> list[SnapshotQuotes]:
        tier_order = FetchTier.get_priority_order()  # CLOSING first (closest)
        decision_idx = tier_order.index(self._decision_tier)
        kickoff = bundle.event.commence_time

        candidates = []
        for s, tier in zip(bundle.snapshots, tiers_of(bundle.snapshots, kickoff), strict=True):
            if sources is not None and s.source not in sources:
                continue
            # IN_PLAY is placed last in priority order but is actually
            # *closer* to (past) game time than any pre-game tier.
            # Exclude it unless explicitly requested.
            if tier == FetchTier.IN_PLAY and self._decision_tier != FetchTier.IN_PLAY:
                continue
            if tier_order.index(tier) >= decision_idx:
                candidates.append(s)

        # Filter to snapshots that contain at least one required bookmaker
        # (priority-ordered fallback: e.g. [pinnacle, betfair_exchange])
        if self._required_bookmakers:
            candidates = [
                s
                for s in candidates
                if any(s.has_bookmaker(bm) for bm in self._required_bookmakers)
            ]

        if not candidates:
            return []

        return [max(candidates, key=lambda s: s.snapshot_time)]


class TimeRangeSampler:
    """Stratified sampling across [min_hours, max_hours] before game, of ``sources`` when given."""

    def __init__(self, min_hours: float, max_hours: float, max_samples: int) -> None:
        self._min_hours = min_hours
        self._max_hours = max_hours
        self._max_samples = max_samples

    def sample(
        self, bundle: EventDataBundle, sources: Collection[Source] | None = None
    ) -> list[SnapshotQuotes]:
        event = bundle.event
        commence = event.commence_time

        range_start = commence - timedelta(hours=self._max_hours)
        range_end = commence - timedelta(hours=self._min_hours)

        in_range = [
            s
            for s in bundle.snapshots
            if range_start <= s.snapshot_time <= range_end
            and (sources is None or s.source in sources)
        ]

        if not in_range:
            return []

        if len(in_range) <= self._max_samples:
            return sorted(in_range, key=lambda s: s.snapshot_time)

        # Stratified: divide range into equal bins, pick nearest to each midpoint
        range_seconds = (range_end - range_start).total_seconds()
        bin_size = range_seconds / self._max_samples
        selected: list[SnapshotQuotes] = []

        for i in range(self._max_samples):
            bin_mid = range_start + timedelta(seconds=(i + 0.5) * bin_size)
            best: SnapshotQuotes | None = None
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
        return TierSampler(sc.decision_tier.value, config.sharp_bookmakers)
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


def make_feature_event(event: Event) -> FeatureEvent:
    """Convert Event to FeatureEvent for feature extraction.

    Scores are passed through as-is; SCHEDULED events at inference time
    will have ``None`` scores. Feature extractors do not consume score
    fields, so this is safe.
    """
    return FeatureEvent(
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

    features: np.ndarray


class FeatureAdapter(Protocol):
    """Protocol for model-specific feature formatting."""

    def feature_names(self, config: FeatureConfig) -> list[str]:
        """Return ordered list of feature names."""
        ...

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: SnapshotQuotes,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        """Extract features for one (event, snapshot) pair. Returns None to skip row."""
        ...


_STATIC_FEATURE_GROUPS = frozenset(get_args(FeatureGroupName))


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
    return names


def _extract_static_feature_parts(
    bundle: EventDataBundle,
    snapshot: SnapshotQuotes,
    config: FeatureConfig,
) -> list[np.ndarray] | None:
    """Extract static feature arrays for all configured groups.

    Returns ``None`` when tabular is configured but feature extraction fails
    (the row cannot be used). Other groups NaN-fill on failure.
    Does NOT include ``hours_until_event``.
    """
    event = bundle.event
    outcome = PriceOutcome(config.outcome)
    feature_event = make_feature_event(event)
    tab_extractor = TabularFeatureExtractor.from_config(config)

    parts: list[np.ndarray] = []

    # --- Tabular features ---
    if "tabular" in config.feature_groups:
        try:
            tab_feats = tab_extractor.extract_features(
                event=feature_event,
                quoted=snapshot.quotes,
                outcome=outcome,
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
                    try:
                        sb_feats = tab_extractor.extract_features(
                            event=feature_event,
                            quoted=snapshot.quotes,
                            outcome=outcome,
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

    return parts


class XGBoostAdapter:
    """Tabular feature adapter for XGBoost (and other tree-based models)."""

    def feature_names(self, config: FeatureConfig) -> list[str]:
        return _static_feature_group_names(config) + ["hours_until_event"]

    def transform(
        self,
        bundle: EventDataBundle,
        snapshot: SnapshotQuotes,
        config: FeatureConfig,
    ) -> AdapterOutput | None:
        parts = _extract_static_feature_parts(bundle, snapshot, config)
        if parts is None:
            return None

        hours_until = (bundle.event.commence_time - snapshot.snapshot_time).total_seconds() / 3600
        parts.append(np.array([hours_until]))

        return AdapterOutput(features=np.concatenate(parts))


def _make_adapter(config: FeatureConfig) -> FeatureAdapter:
    if config.adapter == "xgboost":
        return XGBoostAdapter()
    raise NotImplementedError(f"Adapter '{config.adapter}' is not yet implemented")


# =============================================================================
# Layer 3: Target helpers
# =============================================================================


def _compute_target(
    snapshot: SnapshotQuotes,
    closing_snapshot: SnapshotQuotes,
    config: FeatureConfig,
) -> float | None:
    """Average implied-probability delta from one snapshot to the closing snapshot."""
    outcome = PriceOutcome(config.outcome)
    return calculate_regression_target(
        snapshot.outcome_quotes(outcome),
        closing_snapshot.outcome_quotes(outcome),
        config.primary_market,
    )


# =============================================================================
# Layer 5: Orchestrator
# =============================================================================


async def load_fixtures_df(
    session: AsyncSession, until: datetime | None = None
) -> pd.DataFrame | None:
    """Load all ESPN fixtures, up to ``until`` when it is given, from DB into a DataFrame.

    Returns None if no fixtures exist in the database.
    """
    from odds_lambda.storage.espn_fixture_reader import EspnFixtureReader

    reader = EspnFixtureReader(session)
    fixtures = await reader.get_all_fixtures(until)
    if not fixtures:
        logger.warning("espn_fixtures_not_found_in_db")
        return None

    rows = [f.model_dump(exclude={"id", "created_at"}) for f in fixtures]
    df = pd.DataFrame(rows).rename(columns={"match_round": "round"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    logger.info("espn_fixtures_loaded_from_db", rows=len(df))
    return df


class PreparedFeatureData:
    """Pre-split feature matrix with targets and metadata."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        event_ids: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        commence_times: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.event_ids = event_ids
        self.sample_weights = sample_weights
        self.commence_times = commence_times

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
    sampler: SnapshotSampler | None = None,
) -> PreparedFeatureData:
    """Unified training data preparation using the 5-layer adapter architecture.

    Sampling strategy and target type are both controlled via config:
      - config.sampling.strategy="tier"        → single row per event
      - config.sampling.strategy="time_range"  → multiple rows per event
      - config.target_type="raw"               → avg implied-prob delta
      - config.target_type="retail_ev"         → ADR-0012 rows (odds_analytics.retail_ev)

    Polymarket features NaN-fill when PM data is unavailable, so events are
    kept rather than dropped when PM is missing.

    Args:
        events: List of Event objects (filters to FINAL status internally)
        session: Async database session
        config: FeatureConfig controlling sampling, features, and target
        sampler: Decision-snapshot sampler replacing the one ``config.sampling`` specifies

    Returns:
        PreparedFeatureData with X, y, feature_names, and event_ids for CV
    """
    if config.target_type == "retail_ev":
        if sampler is not None:
            raise ValueError("retail_ev builds a row per retail snapshot and takes no sampler")
        from odds_analytics.retail_ev import prepare_retail_ev_training_data

        return await prepare_retail_ev_training_data(events, session, config)

    valid_events = filter_completed_events(events)
    if not valid_events:
        raise ValueError(f"No valid events found in {len(events)} total events")

    adapter = _make_adapter(config)
    if sampler is None:
        sampler = _make_sampler(config)
    feature_names = adapter.feature_names(config)

    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    event_id_list: list[str] = []
    skipped_events = 0
    sharp_filtered_count = 0
    total_rows = 0

    for event in valid_events:
        # Load all data for this event in bulk
        bundle = await collect_event_data(event, session, config)

        # A close is required
        if not bundle.closes:
            skipped_events += 1
            continue

        # Filter events whose closes all miss EVERY sharp bookmaker (cross-source features).
        # A close is kept if at least one sharp bookmaker is present, enabling
        # priority-ordered fallback (e.g. [pinnacle, betfair_exchange]).
        sharp_closes = {
            source: close
            for source, close in bundle.closes.items()
            if any(close.has_bookmaker(bm) for bm in config.sharp_bookmakers)
        }
        if not sharp_closes:
            sharp_filtered_count += 1
            continue

        # Decision snapshots only of sources whose close they can be measured against
        decision_snapshots = sampler.sample(bundle, sources=sharp_closes.keys())
        if not decision_snapshots:
            skipped_events += 1
            continue

        event_had_rows = False
        for snapshot in decision_snapshots:
            target = _compute_target(snapshot, sharp_closes[snapshot.source], config)
            if target is None:
                continue

            output = adapter.transform(bundle, snapshot, config)
            if output is None:
                continue

            X_list.append(output.features)
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
        event_ids=event_ids,
    )
