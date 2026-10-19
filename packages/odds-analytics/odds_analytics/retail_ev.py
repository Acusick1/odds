"""Decision rows for the retail-EV target of ADR-0012.

One row per event, outcome, UK retail book and retail snapshot time t. The target is
the bet's EV against the devigged Betfair Exchange close; the baseline treats the
exchange price at or before t as the close.
"""

from __future__ import annotations

import bisect
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast, get_args

import numpy as np
import structlog
from odds_core.betfair_close import EXCHANGE_BOOKMAKER, LIVE_CLOSE_WINDOW, is_in_play
from odds_core.bookmakers import UK_RETAIL_BOOKMAKERS, canonical_bookmaker_key
from odds_core.models import Event, OddsSnapshot
from odds_core.odds_math import devig_probabilities, outcome_price_decimal
from odds_core.snapshot_sources import SnapshotKind, snapshot_kind

from odds_analytics.epl_schedule_features import EplScheduleFeatures
from odds_analytics.standings_features import StandingsFeatures
from odds_analytics.training.config import RetailEvFeatureGroupName

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from odds_analytics.as_of_team_features import TeamHistory
    from odds_analytics.feature_groups import PreparedFeatureData
    from odds_analytics.training.config import FeatureConfig

logger = structlog.get_logger()

DECISION_WINDOW = timedelta(days=7)
DECISION_CUTOFF = timedelta(minutes=5)
MAX_EXCHANGE_AGE = timedelta(hours=2)
# A wider back book is a thin market whose best backs are not a price (ADR-0014).
MAX_EXCHANGE_OVERROUND = 0.05
POLYMARKET_BOOKMAKER = "polymarket"
# Twice the coarsest step of the Polymarket history grid, and the exchange's age cap.
MAX_POLYMARKET_AGE = timedelta(hours=2)
MOVE_WINDOWS: tuple[timedelta, ...] = (timedelta(hours=1), timedelta(hours=3), timedelta(hours=6))
MAX_LAGGED_PRICE_AGE = timedelta(hours=2)
# Fixed rather than fitted to the training rows, so min_child_weight and reg_lambda weigh
# the same number of events at every refit. The frozen 2024-25 trial was tuned with mean-one
# weights on 112,455 rows over 720 events, 156.19 per event; development events average
# 0.15413 in 1/odds², and 156.19 / 0.15413 = 1013 (#556).
LOSS_WEIGHT_SCALE = 1013.0
BASELINE_FEATURE = "baseline_ev"
DRAW_OUTCOME_NAME = "Draw"
THREE_WAY_MARKET = "1x2"
# Lower edges of the reporting horizon buckets, in hours to kick-off.
HORIZON_BUCKET_EDGES_HOURS: tuple[float, ...] = (0.0, 1.0, 6.0, 24.0, 72.0)
HORIZON_BUCKET_LABELS: tuple[str, ...] = ("0-1h", "1-6h", "6-24h", "1-3d", "3-7d")
SEASON_START_MONTH = 7
TWO_WAY_MARKET = "h2h"

RETAIL_KINDS: frozenset[SnapshotKind] = frozenset(
    {
        SnapshotKind.ODDS_API,
        SnapshotKind.ODDSPORTAL,
        SnapshotKind.ODDSPORTAL_LIVE,
        SnapshotKind.FDUK_OPENING,
    }
)
# Stamped at FDUK's latest documented collection time, an upper bound on when the price was taken.
UPPER_BOUND_TIME_KINDS: frozenset[SnapshotKind] = frozenset({SnapshotKind.FDUK_OPENING})
EXCHANGE_AT_T_KINDS: frozenset[SnapshotKind] = frozenset(
    {SnapshotKind.BETFAIR_LIVE, SnapshotKind.FDUK_OPENING}
)


class Outcome(StrEnum):
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"


def market_outcomes(market: str) -> tuple[Outcome, ...]:
    if market == THREE_WAY_MARKET:
        return (Outcome.HOME, Outcome.DRAW, Outcome.AWAY)
    if market == TWO_WAY_MARKET:
        return (Outcome.HOME, Outcome.AWAY)
    raise ValueError(f"retail_ev has no outcomes for market {market!r}")


@dataclass(frozen=True)
class ExchangePrice:
    """Devigged Betfair Exchange back prices from one snapshot."""

    time: datetime
    kind: SnapshotKind
    fair: dict[Outcome, float]
    overround: float


@dataclass(frozen=True)
class PolymarketPrice:
    """Devigged Polymarket Yes prices from one snapshot, with the live Yes spreads."""

    time: datetime
    fair: dict[Outcome, float]
    spread: dict[Outcome, float]


@dataclass(frozen=True)
class Dynamics:
    """Moves into t over each of ``MOVE_WINDOWS``, and the spread of UK prices at t."""

    exchange_moves: tuple[float | None, ...]
    consensus_moves: tuple[float | None, ...]
    retail_dispersion: float | None


@dataclass(frozen=True)
class _RetailQuote:
    time: datetime
    kind: SnapshotKind
    books: dict[str, dict[Outcome, float]]
    implied: dict[Outcome, list[float]]


@dataclass(frozen=True)
class RetailEvRow:
    event_id: str
    commence_time: datetime
    outcome: Outcome
    book: str
    retail_kind: SnapshotKind
    decision_time: datetime
    retail_decimal: float
    book_fair_prob: float | None
    book_overround: float | None
    consensus_prob: float
    num_bookmakers: int
    exchange: ExchangePrice
    close: ExchangePrice
    polymarket: PolymarketPrice | None
    dynamics: Dynamics

    @property
    def decision_time_is_exact(self) -> bool:
        """Whether the retail price was taken at ``decision_time``, not at some time before it."""
        return self.retail_kind not in UPPER_BOUND_TIME_KINDS

    @property
    def hours_to_kickoff(self) -> float:
        return (self.commence_time - self.decision_time).total_seconds() / 3600

    @property
    def horizon_bucket(self) -> str:
        hours = self.hours_to_kickoff
        if not 0 <= hours <= DECISION_WINDOW.total_seconds() / 3600:
            raise ValueError(f"hours to kick-off {hours} outside the decision window")
        idx = bisect.bisect_right(HORIZON_BUCKET_EDGES_HOURS, hours) - 1
        return HORIZON_BUCKET_LABELS[idx]

    @property
    def season(self) -> str:
        year = self.commence_time.year
        start = year if self.commence_time.month >= SEASON_START_MONTH else year - 1
        return f"{start}-{(start + 1) % 100:02d}"

    @property
    def exchange_age_hours(self) -> float:
        return (self.decision_time - self.exchange.time).total_seconds() / 3600

    @property
    def target(self) -> float:
        return self.close.fair[self.outcome] * self.retail_decimal - 1

    @property
    def baseline(self) -> float:
        return self.exchange.fair[self.outcome] * self.retail_decimal - 1


@dataclass
class DropCounts:
    events_without_close: int = 0
    rows_without_exchange: int = 0
    rows_stale_exchange: int = 0
    rows_without_own_exchange: int = 0
    wide_exchange_snapshots: int = 0
    prices_outside_roster: Counter[str] = field(default_factory=Counter)

    def add(self, other: DropCounts) -> None:
        self.events_without_close += other.events_without_close
        self.rows_without_exchange += other.rows_without_exchange
        self.rows_stale_exchange += other.rows_stale_exchange
        self.rows_without_own_exchange += other.rows_without_own_exchange
        self.wide_exchange_snapshots += other.wide_exchange_snapshots
        self.prices_outside_roster.update(other.prices_outside_roster)


@dataclass
class RetailEvDataset:
    rows: list[RetailEvRow] = field(default_factory=list)
    drops: DropCounts = field(default_factory=DropCounts)

    @property
    def event_ids(self) -> np.ndarray:
        return np.array([r.event_id for r in self.rows])

    @property
    def targets(self) -> np.ndarray:
        return np.array([r.target for r in self.rows], dtype=np.float64)

    @property
    def baselines(self) -> np.ndarray:
        return np.array([r.baseline for r in self.rows], dtype=np.float64)

    @property
    def commence_times(self) -> np.ndarray:
        return np.array(
            [r.commence_time.astimezone(UTC).replace(tzinfo=None) for r in self.rows],
            dtype="datetime64[us]",
        )

    @property
    def event_weights(self) -> np.ndarray:
        return event_weights(self.event_ids)

    @property
    def loss_weights(self) -> np.ndarray:
        odds = np.array([r.retail_decimal for r in self.rows], dtype=np.float64)
        return LOSS_WEIGHT_SCALE * self.event_weights / odds**2


def event_weights(event_ids: np.ndarray) -> np.ndarray:
    """``w_event = 1 / (rows in the event)`` for each row."""
    _, inverse, counts = np.unique(event_ids, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def _outcome_names(event: Event, market: str) -> dict[str, Outcome]:
    names = {event.home_team: Outcome.HOME, event.away_team: Outcome.AWAY}
    if Outcome.DRAW in market_outcomes(market):
        names[DRAW_OUTCOME_NAME] = Outcome.DRAW
    return names


def _book_prices(
    raw_data: dict[str, Any], kind: SnapshotKind, market: str, names: dict[str, Outcome]
) -> dict[str, dict[Outcome, float]]:
    """Valid decimal prices per canonical book key; the first entry wins for an aliased book."""
    books: dict[str, dict[Outcome, float]] = {}
    for bookmaker in raw_data.get("bookmakers", []):
        key = canonical_bookmaker_key(bookmaker.get("key", ""), kind)
        if key in books:
            continue
        for mkt in bookmaker.get("markets", []):
            if mkt.get("key") != market:
                continue
            prices: dict[Outcome, float] = {}
            for outcome in mkt.get("outcomes", []):
                side = names.get(outcome.get("name"))
                if side is None:
                    continue
                price = outcome_price_decimal(outcome)
                if price > 1.0:
                    prices[side] = price
            if prices:
                books[key] = prices
            break
    return books


def _devig(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> dict[Outcome, float]:
    fair = devig_probabilities(*(1 / prices[o] for o in outcomes))
    return dict(zip(outcomes, fair, strict=True))


def _overround(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> float:
    return sum(1 / prices[o] for o in outcomes) - 1


def _exchange_price(
    snapshot: OddsSnapshot,
    kind: SnapshotKind,
    market: str,
    names: dict[str, Outcome],
    outcomes: tuple[Outcome, ...],
) -> ExchangePrice | None:
    prices = _book_prices(snapshot.raw_data, kind, market, names).get(EXCHANGE_BOOKMAKER)
    if prices is None or set(prices) != set(outcomes):
        return None
    return ExchangePrice(
        time=snapshot.snapshot_time,
        kind=kind,
        fair=_devig(prices, outcomes),
        overround=_overround(prices, outcomes),
    )


def _polymarket_price(
    snapshot: OddsSnapshot, market: str, names: dict[str, Outcome], outcomes: tuple[Outcome, ...]
) -> PolymarketPrice | None:
    raw = snapshot.raw_data
    prices = _book_prices(raw, SnapshotKind.POLYMARKET, market, names).get(POLYMARKET_BOOKMAKER)
    if prices is None or set(prices) != set(outcomes):
        return None
    meta: dict[str, Any] = next(
        (
            bm.get("polymarket_meta") or {}
            for bm in raw.get("bookmakers", [])
            if bm.get("key") == POLYMARKET_BOOKMAKER
        ),
        {},
    )
    bids: dict[str, float | None] = meta.get("best_bid") or {}
    asks: dict[str, float | None] = meta.get("best_ask") or {}
    spread: dict[Outcome, float] = {}
    for outcome in outcomes:
        bid, ask = bids.get(outcome.value), asks.get(outcome.value)
        if bid is not None and ask is not None:
            spread[outcome] = ask - bid
    return PolymarketPrice(
        time=snapshot.snapshot_time, fair=_devig(prices, outcomes), spread=spread
    )


def _latest_at_or_before(times: list[datetime], at: datetime, max_age: timedelta) -> int | None:
    idx = bisect.bisect_right(times, at) - 1
    if idx < 0 or at - times[idx] > max_age:
        return None
    return idx


def _consensus_move(now: _RetailQuote, before: _RetailQuote, outcome: Outcome) -> float | None:
    moves = [
        1 / prices[outcome] - 1 / before.books[book][outcome]
        for book, prices in now.books.items()
        if outcome in prices and outcome in before.books.get(book, {})
    ]
    return float(np.mean(moves)) if moves else None


def _dynamics(
    outcome: Outcome,
    t: datetime,
    exchange: ExchangePrice,
    same_kind_exchange: list[ExchangePrice],
    same_kind_exchange_times: list[datetime],
    quote: _RetailQuote,
    same_kind: list[_RetailQuote],
    same_kind_times: list[datetime],
) -> Dynamics:
    """Moves compare prices from one source, so a change of source is never a move."""
    exchange_moves: list[float | None] = []
    consensus_moves: list[float | None] = []
    for window in MOVE_WINDOWS:
        idx = _latest_at_or_before(same_kind_exchange_times, t - window, MAX_LAGGED_PRICE_AGE)
        lagged = None if idx is None else same_kind_exchange[idx]
        exchange_moves.append(
            None
            if lagged is None or lagged.time == exchange.time
            else exchange.fair[outcome] - lagged.fair[outcome]
        )
        idx = _latest_at_or_before(same_kind_times, t - window, MAX_LAGGED_PRICE_AGE)
        consensus_moves.append(
            None if idx is None else _consensus_move(quote, same_kind[idx], outcome)
        )
    implied = quote.implied[outcome]
    return Dynamics(
        exchange_moves=tuple(exchange_moves),
        consensus_moves=tuple(consensus_moves),
        retail_dispersion=float(np.std(implied)) if len(implied) >= 2 else None,
    )


def build_event_rows(
    event: Event,
    snapshots: Sequence[OddsSnapshot],
    market: str,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> tuple[list[RetailEvRow], DropCounts]:
    """Build every decision row for one event from its snapshots, in time order.

    The close is the last live Betfair snapshot that is not in-play within
    ``LIVE_CLOSE_WINDOW`` before kick-off, else the FDUK closing price (stamped at
    kick-off). The exchange price at t comes only from live Betfair or FDUK opening
    snapshots at or before t, so nothing after t enters a row's inputs. Exchange
    prices whose back overround exceeds ``max_exchange_overround`` are not prices.
    The Polymarket price and the dynamics also come only from snapshots at or before t.
    """
    drops = DropCounts()
    outcomes = market_outcomes(market)
    names = _outcome_names(event, market)
    kickoff = event.commence_time
    ordered = sorted(snapshots, key=lambda s: s.snapshot_time)
    kinds = [snapshot_kind(s.raw_data) for s in ordered]

    exchange_at_t: list[ExchangePrice] = []
    live_closes: list[ExchangePrice] = []
    fduk_close: ExchangePrice | None = None
    polymarket: list[PolymarketPrice] = []
    for snapshot, kind in zip(ordered, kinds, strict=True):
        if kind is SnapshotKind.POLYMARKET:
            pm_price = _polymarket_price(snapshot, market, names, outcomes)
            if pm_price is not None and pm_price.time < kickoff:
                polymarket.append(pm_price)
            continue
        if kind not in EXCHANGE_AT_T_KINDS and kind is not SnapshotKind.FDUK_CLOSING:
            continue
        if kind is SnapshotKind.BETFAIR_LIVE and is_in_play(snapshot.raw_data):
            continue
        price = _exchange_price(snapshot, kind, market, names, outcomes)
        if price is None:
            continue
        if price.overround > max_exchange_overround:
            drops.wide_exchange_snapshots += 1
            continue
        if kind is SnapshotKind.FDUK_CLOSING:
            fduk_close = price
            continue
        if price.time < kickoff:
            exchange_at_t.append(price)
        if (
            kind is SnapshotKind.BETFAIR_LIVE
            and kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff
        ):
            live_closes.append(price)

    close = live_closes[-1] if live_closes else fduk_close
    if close is None:
        drops.events_without_close = 1
        return [], drops

    exchange_times = [p.time for p in exchange_at_t]
    exchange_by_kind: dict[SnapshotKind, list[ExchangePrice]] = {}
    for price in exchange_at_t:
        exchange_by_kind.setdefault(price.kind, []).append(price)
    exchange_times_by_kind = {
        kind: [p.time for p in prices] for kind, prices in exchange_by_kind.items()
    }
    polymarket_times = [p.time for p in polymarket]
    window_start = kickoff - DECISION_WINDOW
    window_end = kickoff - DECISION_CUTOFF
    quotes_by_kind: dict[SnapshotKind, list[_RetailQuote]] = {}
    in_window: list[_RetailQuote] = []
    for snapshot, kind in zip(ordered, kinds, strict=True):
        t = snapshot.snapshot_time
        if kind not in RETAIL_KINDS or t >= kickoff:
            continue
        decision = window_start <= t <= window_end
        retail: dict[str, dict[Outcome, float]] = {}
        for book, prices in _book_prices(snapshot.raw_data, kind, market, names).items():
            if book in UK_RETAIL_BOOKMAKERS:
                retail[book] = prices
            elif decision:
                drops.prices_outside_roster[book] += len(prices)
        if not retail:
            continue
        implied: dict[Outcome, list[float]] = {o: [] for o in outcomes}
        for prices in retail.values():
            for outcome, price in prices.items():
                implied[outcome].append(1 / price)
        quote = _RetailQuote(time=t, kind=kind, books=retail, implied=implied)
        quotes_by_kind.setdefault(kind, []).append(quote)
        if decision:
            in_window.append(quote)
    quote_times = {kind: [q.time for q in quotes] for kind, quotes in quotes_by_kind.items()}

    rows: list[RetailEvRow] = []
    for quote in in_window:
        t = quote.time
        n_candidates = sum(len(p) for p in quote.books.values())
        exact = quote.kind not in UPPER_BOUND_TIME_KINDS
        if exact:
            idx = bisect.bisect_right(exchange_times, t) - 1
            if idx < 0:
                drops.rows_without_exchange += n_candidates
                continue
            exchange = exchange_at_t[idx]
            if t - exchange.time > MAX_EXCHANGE_AGE:
                drops.rows_stale_exchange += n_candidates
                continue
        else:
            # t only bounds when the price was taken, so no other snapshot is known to precede it.
            own = [p for p in exchange_by_kind.get(quote.kind, []) if p.time == t]
            if not own:
                drops.rows_without_own_exchange += n_candidates
                continue
            exchange = own[-1]
        pm_idx = _latest_at_or_before(polymarket_times, t, MAX_POLYMARKET_AGE) if exact else None
        pm_price = None if pm_idx is None else polymarket[pm_idx]
        dynamics = {
            outcome: _dynamics(
                outcome,
                t,
                exchange,
                exchange_by_kind[exchange.kind] if exact else [],
                exchange_times_by_kind[exchange.kind] if exact else [],
                quote,
                quotes_by_kind[quote.kind],
                quote_times[quote.kind],
            )
            for outcome in outcomes
        }

        for book, prices in quote.books.items():
            complete = set(prices) == set(outcomes)
            book_fair = _devig(prices, outcomes) if complete else None
            book_overround = _overround(prices, outcomes) if complete else None
            for outcome, price in prices.items():
                implied_for_outcome = quote.implied[outcome]
                rows.append(
                    RetailEvRow(
                        event_id=event.id,
                        commence_time=kickoff,
                        outcome=outcome,
                        book=book,
                        retail_kind=quote.kind,
                        decision_time=t,
                        retail_decimal=price,
                        book_fair_prob=book_fair[outcome] if book_fair is not None else None,
                        book_overround=book_overround,
                        consensus_prob=float(np.mean(implied_for_outcome)),
                        num_bookmakers=len(implied_for_outcome),
                        exchange=exchange,
                        close=close,
                        polymarket=pm_price,
                        dynamics=dynamics[outcome],
                    )
                )
    return rows, drops


RETAIL_BOOK_COLUMNS: tuple[str, ...] = tuple(sorted(UK_RETAIL_BOOKMAKERS))

_PRICE_FEATURE_NAMES: tuple[str, ...] = (
    "hours_to_kickoff",
    "retail_implied_prob",
    "book_fair_prob",
    "book_overround",
    "consensus_prob",
    "retail_consensus_diff",
    "num_bookmakers",
    "sharp_prob",
    "retail_sharp_diff",
    "exchange_overround",
    "exchange_age_hours",
    BASELINE_FEATURE,
    "is_weekend",
    "day_of_week",
)
_POLYMARKET_FEATURE_NAMES: tuple[str, ...] = (
    "pm_prob",
    "pm_exchange_diff",
    "pm_consensus_diff",
    "pm_baseline_ev",
    "pm_spread",
    "pm_age_hours",
)


def _window_label(window: timedelta) -> str:
    return f"{int(window.total_seconds() // 3600)}h"


_DYNAMICS_FEATURE_NAMES: tuple[str, ...] = (
    *(f"exchange_move_{_window_label(w)}" for w in MOVE_WINDOWS),
    *(f"consensus_move_{_window_label(w)}" for w in MOVE_WINDOWS),
    "retail_dispersion",
)


def _nan_if_none(value: float | None) -> float:
    return np.nan if value is None else value


def _price_features(row: RetailEvRow, outcomes: tuple[Outcome, ...]) -> list[float]:
    implied = 1 / row.retail_decimal
    sharp = row.exchange.fair[row.outcome]
    weekday = row.commence_time.weekday()
    price_block = [
        row.hours_to_kickoff,
        implied,
        _nan_if_none(row.book_fair_prob),
        _nan_if_none(row.book_overround),
        row.consensus_prob,
        implied - row.consensus_prob,
        float(row.num_bookmakers),
        sharp,
        implied - sharp,
        row.exchange.overround,
        row.exchange_age_hours,
        row.baseline,
        float(weekday >= 5),
        float(weekday),
    ]
    outcome_block = [float(row.outcome is o) for o in outcomes]
    book_block = [float(row.book == b) for b in RETAIL_BOOK_COLUMNS]
    return price_block + outcome_block + book_block


def _polymarket_features(row: RetailEvRow) -> list[float]:
    pm = row.polymarket
    if pm is None:
        return [np.nan] * len(_POLYMARKET_FEATURE_NAMES)
    prob = pm.fair[row.outcome]
    return [
        prob,
        prob - row.exchange.fair[row.outcome],
        prob - row.consensus_prob,
        prob * row.retail_decimal - 1,
        _nan_if_none(pm.spread.get(row.outcome)),
        (row.decision_time - pm.time).total_seconds() / 3600,
    ]


def _dynamics_features(row: RetailEvRow) -> list[float]:
    d = row.dynamics
    return [
        *(_nan_if_none(m) for m in d.exchange_moves),
        *(_nan_if_none(m) for m in d.consensus_moves),
        _nan_if_none(d.retail_dispersion),
    ]


_SCHEDULE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"eplsched_{n}" for n in EplScheduleFeatures.get_feature_names()
)
_STANDINGS_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"stnd_{n}" for n in StandingsFeatures.get_feature_names()
)


@dataclass(frozen=True)
class _FeatureGroup:
    names: tuple[str, ...]
    values: Callable[[RetailEvFeatures, RetailEvRow], list[float]]
    needs_teams: bool = False


def _schedule_features(features: RetailEvFeatures, row: RetailEvRow) -> list[float]:
    return features.team_features(row, "schedule_as_of_t", lambda h, e, t: h.schedule_at(e, t))


def _standings_features(features: RetailEvFeatures, row: RetailEvRow) -> list[float]:
    return features.team_features(row, "standings_as_of_t", lambda h, e, t: h.standings_at(e, t))


_FEATURE_GROUPS: dict[RetailEvFeatureGroupName, _FeatureGroup] = {
    "polymarket_at_t": _FeatureGroup(
        _POLYMARKET_FEATURE_NAMES, lambda _, row: _polymarket_features(row)
    ),
    "dynamics": _FeatureGroup(_DYNAMICS_FEATURE_NAMES, lambda _, row: _dynamics_features(row)),
    "schedule_as_of_t": _FeatureGroup(
        _SCHEDULE_FEATURE_NAMES, _schedule_features, needs_teams=True
    ),
    "standings_as_of_t": _FeatureGroup(
        _STANDINGS_FEATURE_NAMES, _standings_features, needs_teams=True
    ),
}


class RetailEvFeatures:
    """The feature vector of a retail-EV row: price features plus the configured as-of-t groups.

    Every group reads only data at or before the row's decision time t. The schedule
    and standings groups need ``teams`` and the rows' events.
    """

    def __init__(
        self,
        outcomes: tuple[Outcome, ...],
        groups: Sequence[RetailEvFeatureGroupName] = (),
        teams: TeamHistory | None = None,
        events: Mapping[str, Event] | None = None,
    ) -> None:
        unknown = set(groups) - set(_FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"unknown retail_ev feature groups {sorted(unknown)}")
        self.outcomes = outcomes
        self.groups: tuple[RetailEvFeatureGroupName, ...] = tuple(
            g for g in get_args(RetailEvFeatureGroupName) if g in groups
        )
        if any(_FEATURE_GROUPS[g].needs_teams for g in self.groups) and (
            teams is None or events is None
        ):
            raise ValueError("schedule_as_of_t and standings_as_of_t need teams and events")
        self._teams = teams
        self._events = events or {}
        self._team_cache: dict[tuple[RetailEvFeatureGroupName, str, datetime], list[float]] = {}

    @staticmethod
    def needs_teams(groups: Sequence[RetailEvFeatureGroupName]) -> bool:
        return any(_FEATURE_GROUPS[g].needs_teams for g in groups)

    @property
    def names(self) -> list[str]:
        names = (
            list(_PRICE_FEATURE_NAMES)
            + [f"outcome_{o.value}" for o in self.outcomes]
            + [f"book_{b}" for b in RETAIL_BOOK_COLUMNS]
        )
        for group in self.groups:
            names += _FEATURE_GROUPS[group].names
        return names

    def team_features(
        self,
        row: RetailEvRow,
        group: RetailEvFeatureGroupName,
        extract: Callable[[TeamHistory, Event, datetime], EplScheduleFeatures | StandingsFeatures],
    ) -> list[float]:
        """One group's team features, computed once per event and decision time.

        NaN for a row whose decision time is only an upper bound: its price may predate
        results the team history holds at that time.
        """
        if not row.decision_time_is_exact:
            return [np.nan] * len(_FEATURE_GROUPS[group].names)
        key = (group, row.event_id, row.decision_time)
        if key not in self._team_cache:
            assert self._teams is not None
            event = self._events[row.event_id]
            self._team_cache[key] = (
                extract(self._teams, event, row.decision_time).to_array().tolist()
            )
        return self._team_cache[key]

    def row(self, row: RetailEvRow) -> np.ndarray:
        values = _price_features(row, self.outcomes)
        for group in self.groups:
            values += _FEATURE_GROUPS[group].values(self, row)
        return np.array(values, dtype=np.float64)

    def matrix(self, rows: Sequence[RetailEvRow]) -> np.ndarray:
        return np.array([self.row(r) for r in rows], dtype=np.float32).reshape(
            len(rows), len(self.names)
        )


async def load_retail_ev_dataset(
    events: Sequence[Event],
    session: AsyncSession,
    market: str,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
    now: datetime | None = None,
) -> RetailEvDataset:
    """Decision rows for every event that has kicked off by ``now``, in the events' order.

    An event that has not kicked off has no close yet, whatever snapshots it holds.
    ``now`` defaults to the current time.
    """
    from odds_lambda.storage.readers import OddsReader

    reader = OddsReader(session)
    dataset = RetailEvDataset()
    cutoff = now if now is not None else datetime.now(UTC)
    events = [event for event in events if event.commence_time <= cutoff]
    for event in events:
        snapshots = await reader.get_snapshots_for_event(event.id)
        rows, drops = build_event_rows(event, snapshots, market, max_exchange_overround)
        dataset.rows.extend(rows)
        dataset.drops.add(drops)
    logger.info(
        "retail_ev_rows_built",
        events=len(events),
        rows=len(dataset.rows),
        events_without_close=dataset.drops.events_without_close,
        rows_without_exchange=dataset.drops.rows_without_exchange,
        rows_stale_exchange=dataset.drops.rows_stale_exchange,
        rows_without_own_exchange=dataset.drops.rows_without_own_exchange,
        wide_exchange_snapshots=dataset.drops.wide_exchange_snapshots,
        prices_outside_roster=dict(dataset.drops.prices_outside_roster.most_common()),
    )
    return dataset


async def prepare_retail_ev_training_data(
    events: list[Event], session: AsyncSession, config: FeatureConfig
) -> PreparedFeatureData:
    """Feature matrix, targets, loss weights and kick-off times for ``retail_ev``."""
    from odds_analytics.feature_groups import PreparedFeatureData
    from odds_analytics.training.feature_selection import apply_variance_filter

    market = config.primary_market
    outcomes = market_outcomes(market)
    dataset = await load_retail_ev_dataset(events, session, market)
    if not dataset.rows:
        raise ValueError(f"No retail_ev rows after processing {len(events)} events")
    row_event_ids = set(dataset.event_ids.tolist())
    valid_events = [e for e in events if e.id in row_event_ids]

    groups = cast(tuple[RetailEvFeatureGroupName, ...], config.feature_groups)
    teams: TeamHistory | None = None
    if RetailEvFeatures.needs_teams(groups):
        from odds_analytics.as_of_team_features import load_team_history

        sport_keys = {e.sport_key for e in valid_events}
        if len(sport_keys) != 1:
            raise ValueError(f"retail_ev team features need one sport, got {sorted(sport_keys)}")
        teams = await load_team_history(
            session,
            sport_keys.pop(),
            config.form_window,
            with_fixtures="schedule_as_of_t" in groups,
        )
    features = RetailEvFeatures(
        outcomes, groups, teams=teams, events={e.id: e for e in valid_events}
    )
    X, feature_names, _ = apply_variance_filter(
        features.matrix(dataset.rows), features.names, config.variance_threshold
    )
    y = dataset.targets.astype(np.float32)

    logger.info(
        "prepared_retail_ev_training_data",
        num_samples=len(y),
        num_events=len(set(dataset.event_ids.tolist())),
        num_features=len(feature_names),
        target_mean=float(np.mean(y)),
        target_std=float(np.std(y)),
    )

    return PreparedFeatureData(
        X=X,
        y=y,
        feature_names=feature_names,
        event_ids=dataset.event_ids,
        sample_weights=dataset.loss_weights,
        commence_times=dataset.commence_times,
    )
