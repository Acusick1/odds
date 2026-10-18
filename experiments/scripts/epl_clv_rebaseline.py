"""EPL CLV re-baseline against the OLS ``retail_sharp_diff`` baseline over two seeds (#496).

Applies the keep test pre-registered on #496 to the tabular bet365 CLV model. Both seeds
walk forward over the evaluation range (events on or after 2024-08-01, ADR-0010) with
hyperparameters frozen from the tuning range:

- Seed A: the #452 expanding cell's best trial and its XGBoost ``random_state``.
- Seed B: the #452 expanding cell re-tuned on the tuning range with a different Optuna
  sampler seed and ``random_state``.

Decision rows are the day-before snapshot of each event (``DayBeforeSampler``), the row
type of every tuning-range row. The run fails if that sampler changes any tuning-range
row from the production sampler's.

Usage:
    uv run python experiments/scripts/epl_clv_rebaseline.py
    uv run python experiments/scripts/epl_clv_rebaseline.py --n-trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import optuna
import pandas as pd
from cv_protocol_common_range import (
    CELL_CONFIGS,
    EVALUATION_START,
    EXPERIMENTS_DIR,
    CellData,
    bootstrap_diff_ci,
    event_sq_errors,
    load_rows,
    refit_folds,
    season_of,
    tune,
)
from odds_analytics.feature_extraction import select_sharp_odds
from odds_analytics.feature_groups import (
    EventDataBundle,
    resolve_outcome_name,
    snapshot_has_bookmaker,
)
from odds_analytics.training.config import FeatureConfig, MLTrainingConfig
from odds_analytics.training.tuner import apply_trial_params
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.snapshot_utils import extract_odds_from_snapshot
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select

OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "epl_clv_rebaseline"
REBASELINE_CONFIG = EXPERIMENTS_DIR / "configs" / "xgboost_epl_clv_rebaseline.yaml"
TUNING_CELL_CONFIG = CELL_CONFIGS["expanding"]
PROTOCOL_SUMMARY = EXPERIMENTS_DIR / "results" / "cv_protocol_common_range" / "summary.json"
SEED_A = 42
SEED_B = 7
# ADR-0010's tuning range: 1,900 events, the last kicking off at 2024-05-19 15:00 UTC, so
# the first fold validates on the first evaluation-range event. Ten events share that
# kick-off, so which of them is 1,900th depends on the tie order.
TUNING_EVENTS = 1900
LAST_TUNING_COMMENCE = pd.Timestamp("2024-05-19T15:00:00Z")
DAY_BEFORE_HOURS = 24.0
# At the time of this run FDUK snapshots were stamped 24 h before the UK clock time read as
# UTC, so they sit 23 h before kick-off under British Summer Time and 24 h otherwise;
# tuning-range rows run from 23.0 to 24.25 h.
DAY_BEFORE_TOLERANCE_HOURS = 1.0
LIVE_TWIN_PREFIX = "op_live_"
EVAL_SEASONS = {"2024-25", "2025-26", "2026-27"}
# Fields the re-baseline config may change from the #452 expanding cell's config.
REBASELINE_DATA_FIELDS = {"end_date", "min_train_events"}


@dataclass(frozen=True)
class DecisionRow:
    event_id: str
    source: str
    fetch_tier: str | None
    hours_before: float
    sharp_bookmaker: str


def snapshot_source(snapshot: OddsSnapshot) -> str:
    raw = snapshot.raw_data or {}
    return str(raw.get("source") or snapshot.api_request_id or "odds_api")


class DayBeforeSampler:
    """The snapshot nearest 24 h before kick-off, within the tolerance, that can make a row.

    A candidate quotes the target bookmaker (the target needs its decision-time price)
    and a sharp bookmaker for the configured outcome. Ties go to the earlier snapshot.
    Each event's choice is recorded in ``selected``.
    """

    def __init__(self, config: FeatureConfig) -> None:
        assert config.target_bookmaker is not None
        self._config = config
        self._market = config.primary_market
        self.called: set[str] = set()
        self.selected: dict[str, DecisionRow] = {}

    @property
    def target_bookmaker(self) -> str:
        assert self._config.target_bookmaker is not None
        return self._config.target_bookmaker

    def _sharp_bookmaker(self, snapshot: OddsSnapshot, event: Event) -> str | None:
        odds = extract_odds_from_snapshot(snapshot, event.id, market=self._market)
        outcome = resolve_outcome_name(self._config, event)
        sharp = select_sharp_odds(odds, self._config.sharp_bookmakers, outcome)
        return sharp[0].bookmaker_key if sharp else None

    def sample(self, bundle: EventDataBundle) -> list[OddsSnapshot]:
        event = bundle.event
        self.called.add(event.id)
        candidates: list[tuple[float, float, OddsSnapshot, str]] = []
        for snapshot in bundle.snapshots:
            hours = (event.commence_time - snapshot.snapshot_time).total_seconds() / 3600
            if abs(hours - DAY_BEFORE_HOURS) > DAY_BEFORE_TOLERANCE_HOURS:
                continue
            if not snapshot_has_bookmaker(snapshot, self._config.target_bookmaker, self._market):
                continue
            sharp = self._sharp_bookmaker(snapshot, event)
            if sharp is None:
                continue
            candidates.append((abs(hours - DAY_BEFORE_HOURS), -hours, snapshot, sharp))
        if not candidates:
            return []
        _, neg_hours, pick, sharp = min(candidates, key=lambda c: (c[0], c[1]))
        self.selected[event.id] = DecisionRow(
            event.id, snapshot_source(pick), pick.fetch_tier, -neg_hours, sharp
        )
        return [pick]


def check_rebaseline_config(rebaseline: MLTrainingConfig, cell: MLTrainingConfig) -> None:
    """The re-baseline config is the #452 expanding cell with only the planned changes."""
    if rebaseline.training.data.model_dump(
        exclude=REBASELINE_DATA_FIELDS
    ) != cell.training.data.model_dump(exclude=REBASELINE_DATA_FIELDS):
        raise ValueError("re-baseline data section differs from the expanding cell's")
    if rebaseline.training.features != cell.training.features:
        raise ValueError("re-baseline features differ from the expanding cell's")
    if rebaseline.training.data.min_train_events != TUNING_EVENTS:
        raise ValueError(f"re-baseline min_train_events must be {TUNING_EVENTS}")
    protocol = json.loads(PROTOCOL_SUMMARY.read_text())
    frozen = protocol["chosen_best_params"]
    model = rebaseline.training.model
    if {k: getattr(model, k) for k in frozen} != frozen or model.random_state != SEED_A:
        raise ValueError("re-baseline model fields are not the #452 expanding best trial")
    if protocol["seed"] != SEED_A or cell.training.model.random_state != SEED_A:
        raise ValueError(f"#452 was not run with seed {SEED_A}")


async def load_events() -> pd.DataFrame:
    """FINAL EPL events from the start of the tuning range, with teams and scores."""
    async with async_session_maker() as session:
        result = await session.execute(
            select(
                Event.id,
                Event.commence_time,
                Event.home_team,
                Event.away_team,
                Event.home_score,
                Event.away_score,
            ).where(
                Event.sport_key == "soccer_epl",
                Event.status == EventStatus.FINAL,
                Event.commence_time >= datetime(2019, 8, 1, tzinfo=UTC),
            )
        )
        df = pd.DataFrame(result.all(), columns=list(result.keys())).set_index("id")
    df["commence_time"] = pd.to_datetime(df["commence_time"], utc=True)
    df["season"] = df["commence_time"].map(season_of)
    df["scored"] = df["home_score"].notna() & df["away_score"].notna()
    df["match"] = match_ids(df)
    return df


def match_ids(events: pd.DataFrame) -> pd.Series:
    """One ID per match: same teams, kick-offs within the twin-adoption window.

    Live-collected events can have a twin under another event ID for the same match;
    ``OddsWriter.EVENT_MATCH_WINDOW`` is the window adoption matches twins within.
    """
    window = pd.Timedelta(OddsWriter.EVENT_MATCH_WINDOW)
    ids = pd.Series(0, index=events.index)
    next_id = 0
    ordered = events.sort_values("commence_time")
    for _, group in ordered.groupby(["home_team", "away_team"], sort=False):
        start: pd.Timestamp | None = None
        for eid, kickoff in group["commence_time"].items():
            if start is None or kickoff - start > window:
                next_id += 1
                start = kickoff
            ids[eid] = next_id
    return ids


def check_tuning_rows(day_before: CellData, production: CellData, n_tuning_rows: int) -> None:
    """The day-before sampler must reproduce every production tuning-range row."""
    if not np.array_equal(
        day_before.event_ids[:n_tuning_rows], production.event_ids[:n_tuning_rows]
    ):
        raise ValueError("day-before sampler changes the tuning-range events")
    if not np.array_equal(
        day_before.X[:n_tuning_rows], production.X[:n_tuning_rows], equal_nan=True
    ):
        raise ValueError("day-before sampler changes tuning-range features")
    if not np.array_equal(day_before.y[:n_tuning_rows], production.y[:n_tuning_rows]):
        raise ValueError("day-before sampler changes tuning-range targets")


def dedupe_matches(data: CellData, events: pd.DataFrame) -> tuple[CellData, dict[str, str]]:
    """Keep one row per match; return the kept data and each dropped twin's kept ID.

    Of an ``op_live_`` twin and another ID, the other is kept, as
    ``OddsWriter.adopt_live_twins`` keeps the Odds API event. Any other duplicate raises.
    """
    row_events = list(data.event_ids)
    match = events.loc[row_events, "match"]
    kept: dict[int, str] = {}
    for eid in sorted(row_events, key=lambda e: (e.startswith(LIVE_TWIN_PREFIX), e)):
        kept.setdefault(match[eid], eid)
    dropped = {eid: kept[match[eid]] for eid in row_events if kept[match[eid]] != eid}
    if any(not eid.startswith(LIVE_TWIN_PREFIX) for eid in dropped):
        raise ValueError(f"duplicate match not explained by an op_live_ twin: {dropped}")
    keep = ~np.isin(data.event_ids, list(dropped))
    return (
        CellData(data.X[keep], data.y[keep], data.feature_names, data.event_ids[keep]),
        dropped,
    )


def collection(source: pd.Series) -> np.ndarray:
    """FDUK rows against rows from live-collected snapshots."""
    return np.where(source == "football_data_uk", "fduk", "live")


def slice_rows(data: CellData, n: int) -> CellData:
    return CellData(data.X[:n], data.y[:n], data.feature_names, data.event_ids[:n])


def with_params(config: MLTrainingConfig, params: dict[str, Any], seed: int) -> MLTrainingConfig:
    out = config.model_copy(deep=True)
    apply_trial_params(out, params)
    out.training.model.random_state = seed
    return out


def compare(
    oof: pd.DataFrame, events: list[str], n_resamples: int, seed: int
) -> dict[str, float | int]:
    """Pooled MSEs and paired bootstrap CIs over ``events``; every comparison resamples alike."""
    errs = {
        col: event_sq_errors(oof, col, events)
        for col in ("pred_seed_a", "pred_seed_b", "pred_ols", "pred_mean")
    }
    n_rows = int(errs["pred_ols"]["size"].sum())
    out: dict[str, float | int] = {"events": len(events), "rows": n_rows}
    for col, name in (
        ("pred_seed_a", "seed_a"),
        ("pred_seed_b", "seed_b"),
        ("pred_ols", "ols"),
        ("pred_mean", "mean_floor"),
    ):
        out[f"mse_{name}"] = float(errs[col]["sum"].sum() / n_rows)
    pairs = {
        "ols_minus_seed_a": ("pred_ols", "pred_seed_a"),
        "ols_minus_seed_b": ("pred_ols", "pred_seed_b"),
        "seed_a_minus_seed_b": ("pred_seed_a", "pred_seed_b"),
        "mean_minus_seed_a": ("pred_mean", "pred_seed_a"),
        "mean_minus_seed_b": ("pred_mean", "pred_seed_b"),
        "mean_minus_ols": ("pred_mean", "pred_ols"),
    }
    for name, (a, b) in pairs.items():
        point, lo, hi = bootstrap_diff_ci(
            errs[a], errs[b], n_resamples, np.random.default_rng(seed)
        )
        out[name], out[f"{name}_ci_low"], out[f"{name}_ci_high"] = point, lo, hi
    for name in ("seed_a", "seed_b", "ols"):
        out[f"r2_{name}_vs_mean"] = 1.0 - out[f"mse_{name}"] / out["mse_mean_floor"]
    return out


def keep_test(overall: dict[str, float | int]) -> dict[str, Any]:
    """The two conditions pre-registered on #496."""
    d_a, d_b = float(overall["ols_minus_seed_a"]), float(overall["ols_minus_seed_b"])
    mean_improvement = (d_a + d_b) / 2
    seed_spread = abs(float(overall["mse_seed_a"]) - float(overall["mse_seed_b"]))
    condition_1 = mean_improvement > seed_spread
    ci_pass = {
        seed: float(overall[f"ols_minus_{seed}_ci_low"]) > 0 for seed in ("seed_a", "seed_b")
    }
    condition_2 = all(ci_pass.values())
    return {
        "condition_1": {
            "rule": "mean over seeds of MSE(baseline) - MSE(model) > |MSE(seed A) - MSE(seed B)|",
            "mean_improvement": mean_improvement,
            "seed_spread": seed_spread,
            "pass": condition_1,
        },
        "condition_2": {
            "rule": "for each seed, the 95% paired bootstrap CI by event on "
            "MSE(baseline) - MSE(model) excludes zero, with the model better",
            "seed_a_ci": [overall["ols_minus_seed_a_ci_low"], overall["ols_minus_seed_a_ci_high"]],
            "seed_b_ci": [overall["ols_minus_seed_b_ci_low"], overall["ols_minus_seed_b_ci_high"]],
            "seed_a_pass": ci_pass["seed_a"],
            "seed_b_pass": ci_pass["seed_b"],
            "pass": condition_2,
        },
        "passes": condition_1 and condition_2,
    }


@dataclass(frozen=True)
class TargetRows:
    """One target's day-before rows, one per match, with the production sampler's for checks."""

    data: CellData
    sampler: DayBeforeSampler
    production: CellData
    twins: dict[str, str]
    n_tuning: int

    @property
    def eval_events(self) -> list[str]:
        return list(self.data.event_ids[self.n_tuning :])

    def production_hours(self) -> dict[str, float]:
        hours_col = self.production.feature_names.index("hours_until_event")
        return dict(zip(self.production.event_ids, self.production.X[:, hours_col], strict=True))


async def load_target_rows(config: MLTrainingConfig, events: pd.DataFrame) -> TargetRows:
    """Day-before rows for ``config``'s target, checked against ADR-0010's tuning range.

    Fails unless the rows are one per match in chronological order, the tuning range
    is ADR-0010's 1,900 events, and the day-before sampler reproduces every production
    tuning-range row.
    """
    sampler = DayBeforeSampler(config.training.features)
    data = await load_rows(config, sampler=sampler)
    production = await load_rows(config)

    if len(set(data.event_ids)) != len(data.event_ids):
        raise ValueError("expected one decision row per event")
    # A match with rows under two event IDs (live twins) is counted once. The tuning
    # range has no twins, so check_tuning_rows below compares undeduped production rows.
    data, twins = dedupe_matches(data, events)
    row_events = list(data.event_ids)
    row_times = events.loc[row_events, "commence_time"]
    if not row_times.is_monotonic_increasing:
        raise ValueError("rows are not in chronological order")
    if events.loc[row_events, "match"].duplicated().any():
        raise ValueError("a match still has rows under two event ids")
    eval_seasons = set(row_times[row_times >= EVALUATION_START].map(season_of))
    if not eval_seasons <= EVAL_SEASONS:
        raise ValueError(f"evaluation rows fall outside {sorted(EVAL_SEASONS)}: {eval_seasons}")

    n_tuning = int((row_times < EVALUATION_START).sum())
    if n_tuning != TUNING_EVENTS or row_times.iloc[n_tuning - 1] != LAST_TUNING_COMMENCE:
        raise ValueError(
            f"tuning range has {n_tuning} events ending {row_times.iloc[n_tuning - 1]}; "
            f"ADR-0010 expects {TUNING_EVENTS} ending {LAST_TUNING_COMMENCE}"
        )
    prod_times = events.loc[production.event_ids, "commence_time"].to_numpy()
    if int((prod_times < EVALUATION_START).sum()) != n_tuning:
        raise ValueError("the production sampler gives a different tuning-range row count")
    check_tuning_rows(data, production, n_tuning)
    return TargetRows(data, sampler, production, twins, n_tuning)


def decision_rows_frame(
    rows: TargetRows, events: pd.DataFrame, eval_events: list[str]
) -> pd.DataFrame:
    """Per-event decision rows over ``eval_events``, with the production pick's horizon."""
    prod_hours = rows.production_hours()
    eval_db = events[events["commence_time"] >= EVALUATION_START]
    rows_df = pd.DataFrame(
        [asdict(rows.sampler.selected[eid]) for eid in eval_events if eid in rows.sampler.selected]
    ).set_index("event_id")
    rows_df = rows_df.loc[eval_events]
    rows_df.insert(0, "commence_time", eval_db.loc[eval_events, "commence_time"])
    rows_df.insert(1, "season", eval_db.loc[eval_events, "season"])
    rows_df["production_hours_before"] = [prod_hours.get(eid, np.nan) for eid in eval_events]
    return rows_df


def lost_events_frame(rows: TargetRows, events: pd.DataFrame) -> pd.DataFrame:
    """Why each evaluation-range event in the database has no row for this target."""
    bookmaker = rows.sampler.target_bookmaker
    prod_hours = rows.production_hours()
    eval_db = events[events["commence_time"] >= EVALUATION_START]
    eval_events = set(rows.eval_events)

    def loss_reason(eid: str) -> str | None:
        if eid in eval_events:
            return None
        if eid in rows.twins:
            return f"twin of {rows.twins[eid]}, which has the match's row"
        if not eval_db.at[eid, "scored"]:
            return "no final score"
        if eid not in rows.sampler.called:
            return f"no closing snapshot with {bookmaker} and a sharp bookmaker"
        if eid not in rows.sampler.selected:
            return f"no day-before snapshot quoting {bookmaker} and a sharp bookmaker"
        return "target or features could not be computed"

    losses = pd.DataFrame(
        {
            "commence_time": eval_db["commence_time"],
            "season": eval_db["season"],
            "home_team": eval_db["home_team"],
            "away_team": eval_db["away_team"],
            "reason": [loss_reason(eid) for eid in eval_db.index],
            "production_hours_before": [prod_hours.get(eid, np.nan) for eid in eval_db.index],
        }
    ).dropna(subset=["reason"])
    losses.index.name = "event_id"
    return losses


def event_counts_frame(
    rows: TargetRows, events: pd.DataFrame, rows_df: pd.DataFrame, losses: pd.DataFrame
) -> pd.DataFrame:
    """Evaluation-range events in the database, with rows and losses, by season."""
    eval_db = events[events["commence_time"] >= EVALUATION_START]
    counts = (
        eval_db.groupby("season")
        .agg(
            events_in_db=("scored", "size"),
            matches_in_db=("match", "nunique"),
            events_scored=("scored", "sum"),
        )
        .join(rows_df.groupby("season").size().rename("events_with_day_before_row"))
        .join(
            pd.Series([season_of(t) for t in events.loc[rows.data.event_ids, "commence_time"]])
            .value_counts()
            .rename("rows")
        )
    )
    counts["rows_per_event"] = counts["rows"] / counts["events_with_day_before_row"]
    prod_eval = pd.Series(
        [season_of(events.at[eid, "commence_time"]) for eid in rows.production.event_ids]
    )
    counts["production_sampler_rows"] = prod_eval.value_counts().reindex(counts.index)
    by_reason = losses.groupby(["season", "reason"]).size().unstack(fill_value=0)
    counts = counts.join(by_reason.add_prefix("lost: "))
    count_cols = [c for c in counts.columns if c != "rows_per_event"]
    counts[count_cols] = counts[count_cols].fillna(0).astype(int)
    counts.index.name = "season"
    return counts


def tune_seed(
    name: str, cell: MLTrainingConfig, tuning: CellData, n_trials: int, seed: int
) -> tuple[optuna.Study, float]:
    """Tune ``cell`` on the tuning range with ``seed`` as Optuna sampler seed and ``random_state``.

    Fails unless a refit of the best trial reproduces its tuned objective.
    """
    study, tune_seconds = tune(name, with_params(cell, {}, seed), tuning, n_trials, seed)
    _, folds = refit_folds(name, with_params(cell, study.best_params, seed), tuning)
    if not np.isclose(folds["val_mse"].mean(), study.best_value, rtol=1e-7):
        raise RuntimeError(f"{name}: refit does not reproduce its tuned objective")
    return study, tune_seconds


def score_seeds(
    config: MLTrainingConfig,
    data: CellData,
    seeds: dict[str, tuple[dict[str, Any], int]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk ``config``'s folds over ``data`` once per seed, with its hyperparameters frozen.

    ``seeds`` maps ``seed_a`` and ``seed_b`` to (hyperparameters, ``random_state``). Returns
    the out-of-fold rows, with ``pred_seed_a`` and ``pred_seed_b``, and both seeds' folds.
    Fails unless both seeds score identical rows against identical baselines.
    """
    oof_by_seed: dict[str, pd.DataFrame] = {}
    folds_by_seed: list[pd.DataFrame] = []
    for name in ("seed_a", "seed_b"):
        params, seed = seeds[name]
        oof_by_seed[name], folds = refit_folds(name, with_params(config, params, seed), data)
        folds_by_seed.append(folds)
    oof_a, oof_b = oof_by_seed["seed_a"], oof_by_seed["seed_b"]
    for col in ("event_id", "y", "pred_ols", "pred_mean"):
        if not oof_a[col].equals(oof_b[col]):
            raise RuntimeError(f"seeds disagree on {col}; they must score identical rows")
    oof = oof_a.drop(columns=["cell", "pred_model"]).assign(
        pred_seed_a=oof_a["pred_model"], pred_seed_b=oof_b["pred_model"]
    )
    return oof, pd.concat(folds_by_seed, ignore_index=True)


async def main(n_trials: int, n_resamples: int, bootstrap_seed: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rebaseline = MLTrainingConfig.from_yaml(REBASELINE_CONFIG)
    cell = MLTrainingConfig.from_yaml(TUNING_CELL_CONFIG)
    check_rebaseline_config(rebaseline, cell)

    # Rows: the day-before sampler, checked against the production sampler.
    events = await load_events()
    rows = await load_target_rows(rebaseline, events)
    data, twins, n_tuning = rows.data, rows.twins, rows.n_tuning
    row_events = list(data.event_ids)
    tuning = slice_rows(data, n_tuning)
    eval_events = rows.eval_events

    prod_hours = rows.production_hours()
    eval_db = events[events["commence_time"] >= EVALUATION_START]
    rows_df = decision_rows_frame(rows, events, eval_events)
    losses = lost_events_frame(rows, events)
    counts = event_counts_frame(rows, events, rows_df, losses)
    print(counts.to_string())

    # Seed A: the frozen #452 trial, scored on the tuning-range folds for comparison with
    # seed B's tuned objective. #452 ordered kick-off ties by query plan, not event ID, so
    # this differs slightly from its recorded objective.
    frozen = json.loads(PROTOCOL_SUMMARY.read_text())
    seed_a_params = frozen["chosen_best_params"]
    _, seed_a_tuning_folds = refit_folds("seed_a", with_params(cell, seed_a_params, SEED_A), tuning)
    seed_a_check = float(seed_a_tuning_folds["val_mse"].mean())

    # Seed B: re-tune the #452 expanding cell on the tuning range with new seeds.
    study, tune_seconds = tune_seed("epl_clv_rebaseline_seed_b", cell, tuning, n_trials, SEED_B)
    seed_b_params = study.best_params

    # Evaluation range: walk forward with each seed's hyperparameters frozen.
    oof, folds = score_seeds(
        rebaseline,
        data,
        {"seed_a": (seed_a_params, SEED_A), "seed_b": (seed_b_params, SEED_B)},
    )
    if oof["event_id"].iloc[0] != eval_events[0]:
        raise RuntimeError("the first validation event is not the first evaluation-range event")
    oof = oof.join(rows_df, on="event_id")
    scored = list(dict.fromkeys(oof["event_id"]))
    unscored_tail = eval_events[len(scored) :]
    if eval_events[: len(scored)] != scored:
        raise RuntimeError("validation events are not a prefix of the evaluation range")

    overall = compare(oof, scored, n_resamples, bootstrap_seed)
    verdict = keep_test(overall)

    table = [{"slice": "all", "group": "all", **overall}]
    oof["collection"] = collection(oof["source"])
    for slice_col in ("season", "sharp_bookmaker", "source", "collection"):
        for group, sub in oof.groupby(slice_col):
            table.append(
                {
                    "slice": slice_col,
                    "group": group,
                    **compare(
                        sub, list(dict.fromkeys(sub["event_id"])), n_resamples, bootstrap_seed
                    ),
                }
            )

    # make_walk_forward_splits drops the partial last block, so it does not enter the
    # keep test. It is scored once, trained on every earlier event, for information.
    if unscored_tail:
        tail_layout = {
            "min_train_events": n_tuning + len(scored),
            "val_step_events": len(unscored_tail),
        }
        tail_oof = {}
        for name, params, seed in (
            ("seed_a", seed_a_params, SEED_A),
            ("seed_b", seed_b_params, SEED_B),
        ):
            cfg = with_params(rebaseline, params, seed)
            for field, value in tail_layout.items():
                setattr(cfg.training.data, field, value)
            tail_oof[name], _ = refit_folds(name, cfg, data)
        tail = (
            tail_oof["seed_a"]
            .drop(columns=["cell", "pred_model"])
            .assign(
                pred_seed_a=tail_oof["seed_a"]["pred_model"],
                pred_seed_b=tail_oof["seed_b"]["pred_model"],
            )
        )
        tail = tail.join(rows_df, on="event_id")
        table.append(
            {
                "slice": "unscored_tail",
                "group": "partial last block",
                **compare(tail, unscored_tail, n_resamples, bootstrap_seed),
            }
        )
        tail["collection"] = collection(tail["source"])
        for group, sub in tail.groupby("collection"):
            table.append(
                {
                    "slice": "unscored_tail_collection",
                    "group": group,
                    **compare(
                        sub, list(dict.fromkeys(sub["event_id"])), n_resamples, bootstrap_seed
                    ),
                }
            )
        tail.to_csv(OUTPUT_DIR / "tail_predictions.csv", index=False)
    results = pd.DataFrame(table)

    # Horizon of the production sampler's row against the day-before row, per event.
    prod_eval_hours = pd.Series(prod_hours).reindex(eval_db.index)
    within = (prod_eval_hours - DAY_BEFORE_HOURS).abs() <= DAY_BEFORE_TOLERANCE_HOURS
    horizon = pd.Series(
        np.select(
            [prod_eval_hours.isna(), within, prod_eval_hours < DAY_BEFORE_HOURS],
            ["no row", "23 to 25 h", "under 23 h"],
            "over 25 h",
        ),
        index=eval_db.index,
    )
    sampling = (
        pd.DataFrame(
            {
                "season": eval_db["season"],
                "production_row": horizon,
                "day_before_row": eval_db.index.isin(eval_events),
            }
        )
        .groupby(["season", "production_row", "day_before_row"])
        .size()
        .rename("events")
    )

    def at(eid: str) -> str:
        return events.at[eid, "commence_time"].isoformat()

    for col in ("first_train_event", "last_train_event", "first_val_event", "last_val_event"):
        folds[col.replace("event", "commence")] = folds[col].map(at)

    counts.to_csv(OUTPUT_DIR / "event_counts.csv")
    rows_df.to_csv(OUTPUT_DIR / "decision_rows.csv")
    losses.to_csv(OUTPUT_DIR / "lost_events.csv")
    sampling.to_csv(OUTPUT_DIR / "sampling_comparison.csv")
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    folds.to_csv(OUTPUT_DIR / "folds.csv", index=False)
    oof.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    trials = study.trials_dataframe()
    trials.to_csv(OUTPUT_DIR / "trials_seed_b.csv", index=False)

    summary = {
        "config": str(REBASELINE_CONFIG.relative_to(EXPERIMENTS_DIR.parent)),
        "tuning_cell_config": str(TUNING_CELL_CONFIG.relative_to(EXPERIMENTS_DIR.parent)),
        "evaluation_start": EVALUATION_START.isoformat(),
        "decision_row": {
            "rule": "snapshot nearest 24 h before kick-off within the tolerance, quoting "
            "bet365 and a sharp bookmaker for the home outcome; ties go to the earlier",
            "hours": DAY_BEFORE_HOURS,
            "tolerance_hours": DAY_BEFORE_TOLERANCE_HOURS,
        },
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": bootstrap_seed,
        "tuning_range": {
            "n_events": n_tuning,
            "last_event": row_events[n_tuning - 1],
            "first_evaluation_event": eval_events[0],
            "last_commence": at(row_events[n_tuning - 1]),
        },
        "evaluation_range": {
            "events_with_rows": len(eval_events),
            "scored_events": len(scored),
            "first_scored": scored[0],
            "first_scored_commence": at(scored[0]),
            "last_scored": scored[-1],
            "last_scored_commence": at(scored[-1]),
            "unscored_partial_block_events": len(unscored_tail),
            "twins_dropped": twins,
            "folds": int(folds["fold"].nunique()),
        },
        "seed_a": {
            "sampler_seed": SEED_A,
            "random_state": SEED_A,
            "params": seed_a_params,
            "tuning_range_cv_mse": seed_a_check,
            "tuning_range_cv_mse_in_452": frozen["cells"]["expanding"]["best_cv_mean_fold_mse"],
        },
        "seed_b": {
            "sampler_seed": SEED_B,
            "random_state": SEED_B,
            "n_trials": n_trials,
            "best_trial": study.best_trial.number,
            "params": seed_b_params,
            "tuning_range_cv_mse": study.best_value,
            "tune_seconds": round(tune_seconds, 1),
        },
        "overall": overall,
        "keep_test": verdict,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    pd.set_option("display.width", 250)
    print("\n" + results.to_string(index=False))
    print("\n" + json.dumps(verdict, indent=2, default=str))
    print(f"\nResults in {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    args = parser.parse_args()
    asyncio.run(main(args.n_trials, args.bootstrap, args.seed))
