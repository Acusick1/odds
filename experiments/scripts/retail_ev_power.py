"""Simulated power of ADR-0012's two keep-bar tests on a fresh season (#538).

Builds the retail-EV decision rows for finished EPL events, then for each collection
regime resamples whole development events into a synthetic fresh season of N events. A
synthetic model predicts each row, and the keep-bar tests of
``odds_analytics.retail_ev_metrics`` decide whether it passes. Power is the share of
simulated seasons whose 95% CI lies above zero.

Every model predicts the move of the exchange's fair probability from t to the close,
``dp = (target - baseline) / retail_decimal``, as a signal ``s``; its EV prediction is
``baseline + retail_decimal * s``. ``sigma`` is the weighted RMS of ``dp`` in the row's
horizon bucket and odds band, and ``z`` is unit noise centred across the three outcomes.

- calibrated(g): ``s = g * dp + sqrt(g * (1 - g)) * sigma * z`` on every row. Within
  each horizon and odds cell its expected weighted MSE is ``(1 - g)`` times the
  baseline's, and the slope of ``dp`` on ``s`` is 1.
- sparse(f, h): on a share ``f`` of events, drawn afresh per replicate, the calibrated
  signal with gain ``h``; elsewhere ``s = 0``, the baseline.
- no information(tau): ``s = tau * sigma * z``, noise with no edge, the size check.

Usage:
    uv run python experiments/scripts/retail_ev_power.py
    uv run python experiments/scripts/retail_ev_power.py --sims 50 --workers 8
    uv run python experiments/scripts/retail_ev_power.py --sweep-seeds 42 142 242 342 442
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog
from odds_analytics.retail_ev import (
    HORIZON_BUCKET_LABELS,
    RetailEvRow,
    event_weights,
    load_retail_ev_dataset,
)
from odds_analytics.retail_ev_metrics import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    mse_gain_of_event_totals,
    pick_ev_gain_of_event_totals,
    weighted_mse,
)
from odds_analytics.training.data_preparation import filter_events_by_date_range
from odds_core.database import async_session_maker
from odds_core.vocabulary import Source, TimeBasis
from retail_ev_target_report import MARKET, SPORT_KEY

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

RESULTS_DIR = Path("experiments/results/retail_ev_power")
SEASON_SIZES: tuple[int, ...] = (100, 190, 380, 760, 1140)
CALIBRATED_GAINS: tuple[float, ...] = (
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
)
SPARSE_SHARES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2)
SPARSE_GAINS: tuple[float, ...] = (0.5, 1.0)
NO_INFORMATION_SCALES: tuple[float, ...] = (0.05, 0.2, 0.5)
ODDS_BAND_EDGES: tuple[float, ...] = (2.0, 3.0, 4.5, 7.0)
TARGET_POWER = 0.8
SWEEP_SEASON_SIZE = 380
NOISE_REPLICATES = 256
REPLICATE_CHUNK = 32
SINGLE_THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


class Regime(StrEnum):
    FDUK_ONLY = "fduk_only"
    DENSE_LIVE = "dense_live"


class NoiseStructure(StrEnum):
    PER_OUTCOME = "per_outcome"
    PER_EXCHANGE_SNAPSHOT = "per_exchange_snapshot"


class Condition(StrEnum):
    MSE = "mse"
    PICK = "pick"
    BOTH = "both"


@dataclass(frozen=True)
class Pool:
    """Development rows of one regime, sorted by event."""

    regime: Regime
    event_starts: np.ndarray
    event_index: np.ndarray
    y: np.ndarray
    baseline: np.ndarray
    odds: np.ndarray
    w_event: np.ndarray
    dp: np.ndarray
    cell: np.ndarray
    sigma: np.ndarray
    outcome: np.ndarray
    unit_by_noise: dict[NoiseStructure, np.ndarray]

    @property
    def n_events(self) -> int:
        return len(self.event_starts)

    @property
    def w_loss(self) -> np.ndarray:
        return self.w_event / self.odds**2


@dataclass(frozen=True)
class CalibratedModel:
    gain: float

    def signal(self, pool: Pool, z: np.ndarray, informed: np.ndarray) -> np.ndarray:
        return calibrated_signal(pool, z, self.gain)

    def params(self) -> dict[str, Any]:
        return {"family": "calibrated", "gain": self.gain}


@dataclass(frozen=True)
class SparseModel:
    share: float
    informed_gain: float

    def signal(self, pool: Pool, z: np.ndarray, informed: np.ndarray) -> np.ndarray:
        rows_informed = informed[pool.event_index] < self.share
        return np.where(rows_informed, calibrated_signal(pool, z, self.informed_gain), 0.0)

    def params(self) -> dict[str, Any]:
        return {"family": "sparse", "share": self.share, "informed_gain": self.informed_gain}


@dataclass(frozen=True)
class NoInformationModel:
    noise_scale: float

    def signal(self, pool: Pool, z: np.ndarray, informed: np.ndarray) -> np.ndarray:
        return self.noise_scale * pool.sigma[:, None] * z

    def params(self) -> dict[str, Any]:
        return {"family": "no_information", "noise_scale": self.noise_scale}


Model = CalibratedModel | SparseModel | NoInformationModel


@dataclass(frozen=True)
class Cell:
    regime: Regime
    noise: NoiseStructure
    model: Model


@dataclass(frozen=True)
class EventTotals:
    """Per-event totals of the keep-bar statistics, one column per noise replicate."""

    se_base: np.ndarray
    w_loss: np.ndarray
    se_model: np.ndarray
    pick_diff: np.ndarray
    model_pick_ev: np.ndarray
    model_pick_w: np.ndarray
    base_pick_ev: np.ndarray
    base_pick_w: np.ndarray
    slope_numerator: np.ndarray
    slope_denominator: np.ndarray


def calibrated_signal(pool: Pool, z: np.ndarray, gain: float) -> np.ndarray:
    return gain * pool.dp[:, None] + np.sqrt(gain * (1 - gain)) * pool.sigma[:, None] * z


def frame_rows(dataset_rows: list[RetailEvRow]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [r.event_id for r in dataset_rows],
            "season": [r.season for r in dataset_rows],
            "horizon_bucket": [HORIZON_BUCKET_LABELS.index(r.horizon_bucket) for r in dataset_rows],
            "retail_source": [r.retail_source for r in dataset_rows],
            "retail_time_basis": [r.retail_time_basis for r in dataset_rows],
            "exchange_fduk_opening": [
                r.exchange.source is Source.FOOTBALL_DATA_UK
                and r.exchange.time_basis is TimeBasis.UPPER_BOUND
                for r in dataset_rows
            ],
            "exchange_time": [r.exchange.time for r in dataset_rows],
            "outcome": [r.outcome.value for r in dataset_rows],
            "odds": [r.retail_decimal for r in dataset_rows],
            "y": [r.target for r in dataset_rows],
            "baseline": [r.baseline for r in dataset_rows],
        }
    )


def regime_rows(df: pd.DataFrame, regime: Regime) -> pd.DataFrame:
    """FDUK-only keeps rows a season gets with no live collection; dense keeps live events."""
    if regime is Regime.FDUK_ONLY:
        keep = (df["retail_source"] == Source.FOOTBALL_DATA_UK) & df["exchange_fduk_opening"]
        return df[keep]
    live = (df["retail_source"] == Source.ODDSPORTAL) & (
        df["retail_time_basis"] == TimeBasis.OBSERVED
    )
    live_events = df.loc[live, "event_id"].unique()
    return df[df["event_id"].isin(live_events)]


def odds_band_index(odds: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(ODDS_BAND_EDGES), odds, side="right")


def odds_band_labels() -> list[str]:
    edges = ["1", *(f"{e:g}" for e in ODDS_BAND_EDGES), "inf"]
    return [f"{low}-{high}" for low, high in zip(edges[:-1], edges[1:], strict=True)]


def n_cells() -> int:
    return len(HORIZON_BUCKET_LABELS) * (len(ODDS_BAND_EDGES) + 1)


def cell_rms(dp: np.ndarray, w_event: np.ndarray, cell: np.ndarray) -> np.ndarray:
    num = np.bincount(cell, weights=w_event * dp**2, minlength=n_cells())
    den = np.bincount(cell, weights=w_event, minlength=n_cells())
    return np.sqrt(np.divide(num, den, out=np.full(n_cells(), np.nan), where=den > 0))


def build_pool(df: pd.DataFrame, regime: Regime) -> tuple[Pool, pd.DataFrame]:
    rows = regime_rows(df, regime).sort_values(["event_id", "exchange_time", "outcome"])
    event_ids = rows["event_id"].to_numpy()
    w_event = event_weights(event_ids)
    odds = rows["odds"].to_numpy()
    y = rows["y"].to_numpy()
    baseline = rows["baseline"].to_numpy()
    dp = (y - baseline) / odds
    bucket = rows["horizon_bucket"].to_numpy()
    band = odds_band_index(odds)
    cell = bucket * (len(ODDS_BAND_EDGES) + 1) + band
    outcome_codes, _ = pd.factorize(rows["outcome"], sort=True)
    per_outcome, _ = pd.factorize(rows["event_id"])
    per_snapshot, _ = pd.factorize(
        pd.MultiIndex.from_arrays([rows["event_id"], rows["exchange_time"]])
    )
    starts = np.flatnonzero(np.r_[True, event_ids[1:] != event_ids[:-1]])
    pool = Pool(
        regime=regime,
        event_starts=starts,
        event_index=per_outcome,
        y=y,
        baseline=baseline,
        odds=odds,
        w_event=w_event,
        dp=dp,
        cell=cell,
        sigma=cell_rms(dp, w_event, cell)[cell],
        outcome=outcome_codes,
        unit_by_noise={
            NoiseStructure.PER_OUTCOME: per_outcome,
            NoiseStructure.PER_EXCHANGE_SNAPSHOT: per_snapshot,
        },
    )
    described = rows.assign(w_event=w_event, dp=dp, bucket=bucket, band=band)
    return pool, described


def centred_noise(rng: np.random.Generator, units: int, outcomes: int, reps: int) -> np.ndarray:
    z = rng.standard_normal((units, outcomes, reps))
    z -= z.mean(axis=1, keepdims=True)
    return z * np.sqrt(outcomes / (outcomes - 1))


def event_totals(pool: Pool, noise: NoiseStructure, model: Model, seed: int) -> EventTotals:
    noise_rng = np.random.default_rng(seed)
    informed_rng = np.random.default_rng(seed + 2)
    units = pool.unit_by_noise[noise]
    n_units = int(units.max()) + 1
    n_outcomes = int(pool.outcome.max()) + 1
    starts = pool.event_starts
    w_loss = pool.w_loss
    base_picks = pool.baseline > 0

    def per_event(values: np.ndarray) -> np.ndarray:
        return np.add.reduceat(values, starts, axis=0)

    def per_cell(values: np.ndarray) -> np.ndarray:
        return np.bincount(pool.cell, weights=values.sum(axis=1), minlength=n_cells())

    se_model, model_pick_ev, model_pick_w = [], [], []
    slope_num = np.zeros(n_cells())
    slope_den = np.zeros(n_cells())
    for _ in range(NOISE_REPLICATES // REPLICATE_CHUNK):
        z = centred_noise(noise_rng, n_units, n_outcomes, REPLICATE_CHUNK)[units, pool.outcome]
        informed = informed_rng.random((pool.n_events, REPLICATE_CHUNK))
        signal = model.signal(pool, z, informed)
        pred = pool.baseline[:, None] + pool.odds[:, None] * signal
        model_picks = pred > 0
        se_model.append(per_event(w_loss[:, None] * (pool.y[:, None] - pred) ** 2))
        model_pick_ev.append(per_event(np.where(model_picks, (pool.w_event * pool.y)[:, None], 0)))
        model_pick_w.append(per_event(np.where(model_picks, pool.w_event[:, None], 0)))
        slope_num += per_cell(pool.w_event[:, None] * pool.dp[:, None] * signal)
        slope_den += per_cell(pool.w_event[:, None] * signal**2)

    base_pick_ev = per_event(np.where(base_picks, pool.w_event * pool.y, 0.0))
    model_ev = np.hstack(model_pick_ev)
    return EventTotals(
        se_base=per_event(w_loss * (pool.y - pool.baseline) ** 2),
        w_loss=per_event(w_loss),
        se_model=np.hstack(se_model),
        pick_diff=model_ev - base_pick_ev[:, None],
        model_pick_ev=model_ev,
        model_pick_w=np.hstack(model_pick_w),
        base_pick_ev=base_pick_ev,
        base_pick_w=per_event(np.where(base_picks, pool.w_event, 0.0)),
        slope_numerator=slope_num,
        slope_denominator=slope_den,
    )


def simulate_cell(
    pool: Pool, cell: Cell, sims: int, n_boot: int, seed: int, season_sizes: tuple[int, ...]
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    totals = event_totals(pool, cell.noise, cell.model, seed)
    labels = {"regime": cell.regime.value, "noise": cell.noise.value, **cell.model.params()}
    rng = np.random.default_rng(seed + 1)
    records = []
    for n_events in season_sizes:
        mse_pass = np.zeros(sims, dtype=bool)
        pick_pass = np.zeros(sims, dtype=bool)
        rel_gain = np.zeros(sims)
        for sim in range(sims):
            events = rng.integers(pool.n_events, size=n_events)
            reps = rng.integers(NOISE_REPLICATES, size=n_events)
            se_base = totals.se_base[events]
            se_model = totals.se_model[events, reps]
            boot_seed = int(rng.integers(2**31))
            mse = mse_gain_of_event_totals(
                se_model, se_base, totals.w_loss[events], n_boot, boot_seed
            )
            pick = pick_ev_gain_of_event_totals(totals.pick_diff[events, reps], n_boot, boot_seed)
            mse_pass[sim] = mse.ci_low > 0
            pick_pass[sim] = pick.ci_low > 0
            rel_gain[sim] = 1 - se_model.sum() / se_base.sum()
        records.append(
            {
                **labels,
                "n_events": n_events,
                "power_mse": float(mse_pass.mean()),
                "power_pick": float(pick_pass.mean()),
                "power_both": float((mse_pass & pick_pass).mean()),
                "mean_realised_gain": float(rel_gain.mean()),
            }
        )
    model_pick_w = totals.model_pick_w.sum()
    effect = {
        **labels,
        "realised_gain": float(
            1 - totals.se_model.sum() / (totals.se_base.sum() * NOISE_REPLICATES)
        ),
        "pick_gain_per_event": float(totals.pick_diff.mean()),
        "model_picks_per_event": float(totals.model_pick_w.mean()),
        "model_ev_per_pick": float(totals.model_pick_ev.sum() / model_pick_w)
        if model_pick_w > 0
        else None,
        "baseline_picks_per_event": float(totals.base_pick_w.mean()),
        "baseline_ev_per_pick": float(totals.base_pick_ev.sum() / totals.base_pick_w.sum()),
    }
    bands = odds_band_labels()
    calibration = [
        {
            **labels,
            "horizon": HORIZON_BUCKET_LABELS[index // len(bands)],
            "odds_band": bands[index % len(bands)],
            "slope_dp_on_signal": float(totals.slope_numerator[index] / den),
        }
        for index, den in enumerate(totals.slope_denominator)
        if den > 0
    ]
    return records, effect, calibration


def minimum_detectable(power: pd.DataFrame, effect_column: str, power_column: str) -> float | None:
    """Smallest effect whose interpolated power reaches the target, if the grid reaches it."""
    ordered = power.sort_values(effect_column)
    effects = ordered[effect_column].to_numpy()
    values = np.maximum.accumulate(ordered[power_column].to_numpy())
    above = np.flatnonzero(values >= TARGET_POWER)
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(effects[0])
    e0, e1, p0, p1 = effects[i - 1], effects[i], values[i - 1], values[i]
    return float(e0 + (TARGET_POWER - p0) * (e1 - e0) / (p1 - p0))


def baseline_wmse(rows: pd.DataFrame) -> float:
    w_loss = rows["w_event"] / rows["odds"] ** 2
    return weighted_mse(rows["y"].to_numpy(), rows["baseline"].to_numpy(), w_loss.to_numpy())


def rms_pp(rows: pd.DataFrame) -> float:
    return 100 * float(np.sqrt(np.average(rows["dp"] ** 2, weights=rows["w_event"])))


def describe_groups(rows: pd.DataFrame, column: str, labels: list[str]) -> list[dict[str, Any]]:
    described = []
    for index, label in enumerate(labels):
        group = rows[rows[column] == index]
        if group.empty:
            continue
        described.append(
            {
                column: label,
                "rows": len(group),
                "events": int(group["event_id"].nunique()),
                "sum_w_event": float(group["w_event"].sum()),
                "baseline_wmse": baseline_wmse(group),
                "dp_rms_pp": rms_pp(group),
            }
        )
    return described


def describe_pool(pool: Pool, rows: pd.DataFrame) -> dict[str, Any]:
    per_event = rows.groupby("event_id").agg(
        rows=("y", "size"), snapshots=("exchange_time", "nunique")
    )
    return {
        "regime": pool.regime.value,
        "events": pool.n_events,
        "rows": len(rows),
        "seasons": rows.groupby("season")["event_id"].nunique().to_dict(),
        "rows_per_event_median": float(per_event["rows"].median()),
        "exchange_snapshots_per_event_median": float(per_event["snapshots"].median()),
        "baseline_wmse": baseline_wmse(rows),
        "dp_rms_pp": rms_pp(rows),
        "baseline_pick_share": float(np.average(rows["baseline"] > 0, weights=rows["w_event"])),
        "by_horizon": describe_groups(rows, "bucket", list(HORIZON_BUCKET_LABELS)),
        "by_odds_band": describe_groups(rows, "band", odds_band_labels()),
    }


def cells() -> list[Cell]:
    calibrated = [
        Cell(regime, noise, CalibratedModel(gain))
        for regime, noise in (
            (Regime.FDUK_ONLY, NoiseStructure.PER_OUTCOME),
            (Regime.DENSE_LIVE, NoiseStructure.PER_OUTCOME),
            (Regime.DENSE_LIVE, NoiseStructure.PER_EXCHANGE_SNAPSHOT),
        )
        for gain in CALIBRATED_GAINS
    ]
    others: list[Cell] = []
    for regime in Regime:
        others += [
            Cell(regime, NoiseStructure.PER_OUTCOME, SparseModel(share, gain))
            for gain in SPARSE_GAINS
            for share in SPARSE_SHARES
        ]
        others += [
            Cell(regime, NoiseStructure.PER_OUTCOME, NoInformationModel(scale))
            for scale in NO_INFORMATION_SCALES
        ]
    return calibrated + others


async def load_rows(start: date, end: date) -> pd.DataFrame:
    async with async_session_maker() as session:
        events = await filter_events_by_date_range(
            session,
            start_date=datetime.combine(start, datetime.min.time(), tzinfo=UTC),
            end_date=datetime.combine(end, datetime.max.time(), tzinfo=UTC),
            status=None,
            sport_key=SPORT_KEY,
        )
        dataset = await load_retail_ev_dataset(events, session, MARKET)
    return frame_rows(dataset.rows)


def mde_tables(
    power: pd.DataFrame, baseline_wmses: dict[str, float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    calibrated = power[power["family"] == "calibrated"]
    for (regime, noise, n_events), group in calibrated.groupby(["regime", "noise", "n_events"]):
        record: dict[str, Any] = {
            "regime": regime,
            "noise": noise,
            "n_events": n_events,
        }
        for condition in Condition:
            record[f"mde_{condition.value}"] = minimum_detectable(
                group, "gain", f"power_{condition.value}"
            )
        mse = record["mde_mse"]
        record["mde_mse_absolute"] = None if mse is None else mse * baseline_wmses[regime]
        records.append(record)
    sparse_records: list[dict[str, Any]] = []
    sparse = power[power["family"] == "sparse"]
    for (regime, noise, informed_gain, n_events), group in sparse.groupby(
        ["regime", "noise", "informed_gain", "n_events"]
    ):
        record = {
            "regime": regime,
            "noise": noise,
            "informed_gain": informed_gain,
            "n_events": n_events,
        }
        for condition in Condition:
            record[f"mde_share_{condition.value}"] = minimum_detectable(
                group, "share", f"power_{condition.value}"
            )
        sparse_records.append(record)
    return pd.DataFrame(records), pd.DataFrame(sparse_records)


def markdown(table: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lines += [
        "| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)
    ]
    return "\n".join(lines)


SimulationOutput = tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]


def run_cells(
    pools: dict[Regime, Pool],
    jobs: list[Cell],
    seeds: tuple[int, ...],
    sims: int,
    n_boot: int,
    workers: int,
    season_sizes: tuple[int, ...],
) -> dict[int, list[SimulationOutput]]:
    streams = {key: i for i, key in enumerate(dict.fromkeys((c.regime, c.noise) for c in jobs))}
    # Spawned workers read these before importing numpy, so each runs single-threaded.
    for variable in SINGLE_THREAD_VARIABLES:
        os.environ[variable] = "1"
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=spawn) as executor:
        futures = {
            seed: [
                executor.submit(
                    simulate_cell,
                    pools[c.regime],
                    c,
                    sims,
                    n_boot,
                    seed + 1000 * streams[(c.regime, c.noise)],
                    season_sizes,
                )
                for c in jobs
            ]
            for seed in seeds
        }
        return {seed: [f.result() for f in fs] for seed, fs in futures.items()}


def seed_sweep(
    pools: dict[Regime, Pool],
    descriptions: list[dict[str, Any]],
    seeds: tuple[int, ...],
    sims: int,
    n_boot: int,
    workers: int,
) -> None:
    """One-season power and minimum detectable effects of the edge models under each seed."""
    jobs = [c for c in cells() if not isinstance(c.model, NoInformationModel)]
    outputs = run_cells(pools, jobs, seeds, sims, n_boot, workers, (SWEEP_SEASON_SIZE,))
    baseline_wmses = {d["regime"]: d["baseline_wmse"] for d in descriptions}
    power_frames, calibrated_frames, sparse_frames = [], [], []
    for seed, seed_outputs in outputs.items():
        power = pd.DataFrame([r for records, _, _ in seed_outputs for r in records])
        mde_calibrated, mde_sparse = mde_tables(power, baseline_wmses)
        power_frames.append(power.assign(seed=seed))
        calibrated_frames.append(mde_calibrated.assign(seed=seed))
        sparse_frames.append(mde_sparse.assign(seed=seed))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, frames in (
        ("seed_sweep_power.csv", power_frames),
        ("seed_sweep_mde_calibrated.csv", calibrated_frames),
        ("seed_sweep_mde_sparse.csv", sparse_frames),
    ):
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(RESULTS_DIR / name, index=False, float_format="%.6g")
        print(f"\n### {name}\n")
        print(markdown(table.round(4)))


def main(
    start: date,
    end: date,
    sims: int,
    n_boot: int,
    seed: int,
    workers: int,
    sweep_seeds: tuple[int, ...] | None,
) -> None:
    df = asyncio.run(load_rows(start, end))
    pools: dict[Regime, Pool] = {}
    descriptions = []
    for regime in Regime:
        pool, rows = build_pool(df, regime)
        pools[regime] = pool
        descriptions.append(describe_pool(pool, rows))

    if sweep_seeds is not None:
        seed_sweep(pools, descriptions, sweep_seeds, sims, n_boot, workers)
        return

    jobs = cells()
    outputs = run_cells(pools, jobs, (seed,), sims, n_boot, workers, SEASON_SIZES)[seed]
    power = pd.DataFrame([r for records, _, _ in outputs for r in records])
    effects = pd.DataFrame([effect for _, effect, _ in outputs])
    calibration = pd.DataFrame([row for _, _, rows in outputs for row in rows])
    mde_calibrated, mde_sparse = mde_tables(
        power, {d["regime"]: d["baseline_wmse"] for d in descriptions}
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    power.to_csv(RESULTS_DIR / "power.csv", index=False, float_format="%.6g")
    effects.to_csv(RESULTS_DIR / "effects.csv", index=False, float_format="%.6g")
    calibration.to_csv(RESULTS_DIR / "calibration.csv", index=False, float_format="%.4g")
    mde_calibrated.to_csv(RESULTS_DIR / "mde_calibrated.csv", index=False, float_format="%.6g")
    mde_sparse.to_csv(RESULTS_DIR / "mde_sparse.csv", index=False, float_format="%.6g")
    summary = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sims": sims,
        "n_boot": n_boot,
        "seed": seed,
        "noise_replicates": NOISE_REPLICATES,
        "target_power": TARGET_POWER,
        "season_sizes": list(SEASON_SIZES),
        "calibrated_gains": list(CALIBRATED_GAINS),
        "sparse_shares": list(SPARSE_SHARES),
        "sparse_informed_gains": list(SPARSE_GAINS),
        "no_information_scales": list(NO_INFORMATION_SCALES),
        "odds_band_edges": list(ODDS_BAND_EDGES),
        "pools": descriptions,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    print(json.dumps(descriptions, indent=2, default=str))
    print("\n### Power\n")
    print(markdown(power.round(3)))
    print("\n### Effects\n")
    print(markdown(effects.round(5)))
    print("\n### Minimum detectable gain at 80% power, calibrated model\n")
    print(markdown(mde_calibrated.round(4)))
    print("\n### Minimum detectable share of informed events at 80% power, sparse model\n")
    print(markdown(mde_sparse.round(4)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 8, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 7, 31))
    parser.add_argument("--sims", type=int, default=250)
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--sweep-seeds",
        type=int,
        nargs="+",
        help="instead of the main run, rerun the edge models at one season under these seeds",
    )
    args = parser.parse_args()
    sweep = tuple(args.sweep_seeds) if args.sweep_seeds else None
    main(args.start, args.end, args.sims, args.n_boot, args.seed, args.workers, sweep)
