"""Pinnacle as the EPL CLV target, against bet365 on a common event set (#482).

Applies the retarget test pre-registered on #482. The devigged Pinnacle target is tuned
on the tuning range (events before 2024-08-01, ADR-0010) with two seeds, then frozen and
walked forward over the evaluation range. It passes only if both conditions of #496 hold
against its own per-fold OLS ``retail_sharp_diff`` baseline.

Both targets are scored on the same events and folds: those with a day-before row for
each target, one per match (``epl_clv_rebaseline.load_target_rows``). The bet365 model is
re-scored there with its frozen #496 hyperparameters, for the comparison ADR-0002 needs.

Usage:
    uv run python experiments/scripts/epl_clv_pinnacle_retest.py
    uv run python experiments/scripts/epl_clv_pinnacle_retest.py --n-trials 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from cv_protocol_common_range import (
    EVALUATION_START,
    EXPERIMENTS_DIR,
    CellData,
    event_sq_errors,
)
from epl_clv_rebaseline import (
    REBASELINE_CONFIG,
    SEED_A,
    SEED_B,
    TUNING_CELL_CONFIG,
    TargetRows,
    check_rebaseline_config,
    compare,
    decision_rows_frame,
    event_counts_frame,
    keep_test,
    load_events,
    load_target_rows,
    lost_events_frame,
    score_seeds,
    slice_rows,
    tune_seed,
)
from odds_analytics.training.config import MLTrainingConfig

OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "epl_clv_pinnacle_retest"
REBASELINE_SUMMARY = EXPERIMENTS_DIR / "results" / "epl_clv_rebaseline" / "summary.json"
BET365 = "bet365"
PINNACLE = "pinnacle"
TARGETS = (BET365, PINNACLE)
SEEDS = {"seed_a": SEED_A, "seed_b": SEED_B}


def for_target(config: MLTrainingConfig, bookmaker: str) -> MLTrainingConfig:
    """``config`` with only its devigged target bookmaker changed, re-validated."""
    out = config.model_copy(deep=True)
    out.training.features.target_bookmaker = bookmaker
    return MLTrainingConfig.model_validate(out.model_dump())


def common_rows(
    rows: dict[str, TargetRows], events: pd.DataFrame
) -> tuple[dict[str, CellData], int]:
    """Each target's rows restricted to the events every target has a row for.

    Fails unless the tuning ranges are the same events, and every common event has the
    same decision row and features under each target, so only the targets differ.
    """
    n_tuning = {t: r.n_tuning for t, r in rows.items()}
    if len(set(n_tuning.values())) != 1:
        raise ValueError(f"tuning ranges differ in size: {n_tuning}")
    tuning_sets = {t: list(r.data.event_ids[: r.n_tuning]) for t, r in rows.items()}
    if tuning_sets[BET365] != tuning_sets[PINNACLE]:
        raise ValueError("the targets' tuning ranges are not the same events")

    shared = set.intersection(*(set(r.data.event_ids) for r in rows.values()))
    out: dict[str, CellData] = {}
    for target, r in rows.items():
        keep = np.isin(r.data.event_ids, list(shared))
        out[target] = CellData(
            r.data.X[keep], r.data.y[keep], r.data.feature_names, r.data.event_ids[keep]
        )
    b, p = out[BET365], out[PINNACLE]
    if not np.array_equal(b.event_ids, p.event_ids):
        raise ValueError("common events are not in the same order for both targets")
    if b.feature_names != p.feature_names or not np.array_equal(b.X, p.X, equal_nan=True):
        raise ValueError("common events have different features under the two targets")
    for eid in b.event_ids:
        if rows[BET365].sampler.selected[eid] != rows[PINNACLE].sampler.selected[eid]:
            raise ValueError(f"{eid}: the targets pick different decision rows")
    times = events.loc[b.event_ids, "commence_time"]
    if not times.is_monotonic_increasing:
        raise ValueError("common rows are not in chronological order")
    return out, n_tuning[BET365]


def extra_train_events(
    rows: TargetRows, common: CellData, folds: pd.DataFrame, events: pd.DataFrame
) -> list[int]:
    """Per fold, events this target has a row for, outside the common set, before validation.

    These are the training events a fold would add if each target trained on every earlier
    event with its own target, instead of only the common events.
    """
    extra = sorted(set(rows.data.event_ids) - set(common.event_ids))
    extra_times = events.loc[extra, "commence_time"] if extra else pd.Series(dtype=object)
    return [
        int((extra_times < events.at[first_val, "commence_time"]).sum())
        for first_val in folds["first_val_event"]
    ]


def skill_diffs(
    oof: dict[str, pd.DataFrame], events: list[str], n_resamples: int, seed: int
) -> dict[str, dict[str, float]]:
    """Pinnacle minus bet365 in R² against the floor and in relative gain over OLS.

    The relative gain is (MSE(OLS) - MSE(model)) / MSE(OLS). Both are ratios within one
    target, so they compare skill across targets whose variances differ. The paired
    bootstrap resamples events, the same resamples for both targets and every seed.
    """
    sse: dict[tuple[str, str], np.ndarray] = {}
    for target in TARGETS:
        for col in ("pred_seed_a", "pred_seed_b", "pred_ols", "pred_mean"):
            sse[target, col] = event_sq_errors(oof[target], col, events)["sum"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = np.vstack(
        [np.arange(len(events)), rng.integers(0, len(events), (n_resamples, len(events)))]
    )

    def sums(target: str, col: str) -> np.ndarray:
        return sse[target, col][idx].sum(axis=1)

    out: dict[str, dict[str, float]] = {}
    for name in SEEDS:
        col = f"pred_{name}"
        for metric in ("r2_vs_mean", "relative_gain_over_ols"):
            by_target = {}
            for target in TARGETS:
                model = sums(target, col)
                if metric == "r2_vs_mean":
                    by_target[target] = 1.0 - model / sums(target, "pred_mean")
                else:
                    ols = sums(target, "pred_ols")
                    by_target[target] = (ols - model) / ols
            diff = by_target[PINNACLE] - by_target[BET365]
            lo, hi = np.percentile(diff[1:], [2.5, 97.5])
            out[f"{metric}_{name}"] = {
                "bet365": float(by_target[BET365][0]),
                "pinnacle": float(by_target[PINNACLE][0]),
                "pinnacle_minus_bet365": float(diff[0]),
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
    return out


async def main(n_trials: int, n_resamples: int, bootstrap_seed: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rebaseline = MLTrainingConfig.from_yaml(REBASELINE_CONFIG)
    cell = MLTrainingConfig.from_yaml(TUNING_CELL_CONFIG)
    check_rebaseline_config(rebaseline, cell)
    if rebaseline.training.features.target_bookmaker != BET365:
        raise ValueError("the re-baseline config's target is not bet365")
    configs = {t: for_target(rebaseline, t) for t in TARGETS}

    # Rows per target, then the events both targets have.
    events = await load_events()
    rows = {t: await load_target_rows(configs[t], events) for t in TARGETS}
    common, n_tuning = common_rows(rows, events)
    common_ids = list(common[BET365].event_ids)
    eval_events = common_ids[n_tuning:]

    counts_parts, losses_parts = [], []
    for target, r in rows.items():
        rows_df = decision_rows_frame(r, events, r.eval_events)
        losses = lost_events_frame(r, events)
        counts = event_counts_frame(r, events, rows_df, losses)
        counts["common_events"] = (
            events.loc[eval_events, "season"].value_counts().reindex(counts.index).fillna(0)
        ).astype(int)
        counts.insert(0, "target", target)
        counts_parts.append(counts.reset_index())
        losses.insert(0, "target", target)
        losses_parts.append(losses.reset_index())
        # Events with this target's row that the other target lacks.
        only = [eid for eid in r.eval_events if eid not in set(eval_events)]
        other = PINNACLE if target == BET365 else BET365
        losses_parts.append(
            pd.DataFrame(
                {
                    "event_id": only,
                    "target": f"common ({target} only)",
                    "commence_time": events.loc[only, "commence_time"].to_numpy(),
                    "season": events.loc[only, "season"].to_numpy(),
                    "home_team": events.loc[only, "home_team"].to_numpy(),
                    "away_team": events.loc[only, "away_team"].to_numpy(),
                    "reason": f"no {other} day-before row",
                    "production_hours_before": np.nan,
                }
            )
        )
    counts = pd.concat(counts_parts, ignore_index=True)
    lost = pd.concat(losses_parts, ignore_index=True)
    print(counts.to_string(index=False))

    # Pinnacle: tune both seeds on the tuning range, then freeze.
    pinnacle_cell = for_target(cell, PINNACLE)
    tuning = slice_rows(common[PINNACLE], n_tuning)
    studies = {}
    for name, seed in SEEDS.items():
        studies[name] = tune_seed(
            f"epl_clv_pinnacle_retest_{name}", pinnacle_cell, tuning, n_trials, seed
        )
        study, seconds = studies[name]
        print(f"pinnacle {name}: best trial {study.best_trial.number}, {study.best_value:.7f}")

    # bet365: the frozen #496 hyperparameters of each seed.
    frozen = json.loads(REBASELINE_SUMMARY.read_text())
    seed_params = {
        BET365: {name: (frozen[name]["params"], seed) for name, seed in SEEDS.items()},
        PINNACLE: {name: (studies[name][0].best_params, seed) for name, seed in SEEDS.items()},
    }
    for name, seed in SEEDS.items():
        if frozen[name]["random_state"] != seed:
            raise ValueError(f"#496 {name} random_state is not {seed}")

    # Evaluation range: both targets walk the same folds over the common events.
    oof: dict[str, pd.DataFrame] = {}
    folds_parts = []
    for target in TARGETS:
        oof[target], folds = score_seeds(configs[target], common[target], seed_params[target])
        folds.insert(0, "target", target)
        first_fold = folds[folds["cell"] == "seed_a"]
        folds["extra_train_events"] = extra_train_events(
            rows[target], common[target], folds, events
        )
        if (first_fold["n_train_events"] != np.arange(len(first_fold)) * 50 + n_tuning).any():
            raise RuntimeError("folds do not expand from the tuning range in 50-event steps")
        folds_parts.append(folds)
    folds = pd.concat(folds_parts, ignore_index=True)
    layout_cols = ["fold", "n_train_events", "first_train_event", "last_train_event"]
    layout_cols += ["first_val_event", "last_val_event"]
    by_target = [folds[folds["target"] == t][layout_cols].reset_index(drop=True) for t in TARGETS]
    if not by_target[0].equals(by_target[1]):
        raise RuntimeError("the targets' folds differ")
    if not oof[BET365]["event_id"].equals(oof[PINNACLE]["event_id"]):
        raise RuntimeError("the targets score different rows")
    scored = list(dict.fromkeys(oof[BET365]["event_id"]))
    if eval_events[: len(scored)] != scored:
        raise RuntimeError("validation events are not a prefix of the common evaluation events")
    unscored_tail = eval_events[len(scored) :]

    table = []
    overall: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        overall[target] = compare(oof[target], scored, n_resamples, bootstrap_seed)
        y = oof[target]["y"]
        overall[target]["target_variance"] = float(np.var(y))
        overall[target]["target_mean"] = float(np.mean(y))
        for name in SEEDS:
            overall[target][f"relative_gain_{name}"] = (
                overall[target][f"ols_minus_{name}"] / overall[target]["mse_ols"]
            )
        table.append({"target": target, "slice": "all", "group": "all", **overall[target]})
        sub_oof = oof[target].assign(season=events.loc[oof[target]["event_id"], "season"].values)
        for group, sub in sub_oof.groupby("season"):
            table.append(
                {
                    "target": target,
                    "slice": "season",
                    "group": group,
                    **compare(
                        sub, list(dict.fromkeys(sub["event_id"])), n_resamples, bootstrap_seed
                    ),
                }
            )
    results = pd.DataFrame(table)
    verdict = keep_test(overall[PINNACLE])
    bet365_on_common = keep_test(overall[BET365])
    skill = skill_diffs(oof, scored, n_resamples, bootstrap_seed)
    target_corr = float(np.corrcoef(oof[BET365]["y"], oof[PINNACLE]["y"])[0, 1])

    def at(eid: str) -> str:
        return events.at[eid, "commence_time"].isoformat()

    for col in ("first_train_event", "last_train_event", "first_val_event", "last_val_event"):
        folds[col.replace("event", "commence")] = folds[col].map(at)

    preds = oof[BET365][["fold", "row", "event_id"]].copy()
    for target in TARGETS:
        for col in ("y", "pred_seed_a", "pred_seed_b", "pred_ols", "pred_mean"):
            preds[f"{col}_{target}"] = oof[target][col].to_numpy()
    decision = pd.DataFrame(
        [asdict(rows[BET365].sampler.selected[eid]) for eid in eval_events]
    ).set_index("event_id")
    decision.insert(0, "commence_time", events.loc[eval_events, "commence_time"])
    decision.insert(1, "season", events.loc[eval_events, "season"])
    decision["scored"] = decision.index.isin(scored)
    preds = preds.join(
        decision[["commence_time", "season", "source", "sharp_bookmaker"]], on="event_id"
    )

    counts.to_csv(OUTPUT_DIR / "event_counts.csv", index=False)
    lost.to_csv(OUTPUT_DIR / "lost_events.csv", index=False)
    decision.to_csv(OUTPUT_DIR / "decision_rows.csv")
    folds.to_csv(OUTPUT_DIR / "folds.csv", index=False)
    preds.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    results.to_csv(OUTPUT_DIR / "results.csv", index=False)
    for name, (study, _) in studies.items():
        study.trials_dataframe().to_csv(OUTPUT_DIR / f"trials_pinnacle_{name}.csv", index=False)

    eval_rows = {t: len(r.eval_events) for t, r in rows.items()}
    summary = {
        "config": str(REBASELINE_CONFIG.relative_to(EXPERIMENTS_DIR.parent)),
        "tuning_cell_config": str(TUNING_CELL_CONFIG.relative_to(EXPERIMENTS_DIR.parent)),
        "bet365_hyperparameters": str(REBASELINE_SUMMARY.relative_to(EXPERIMENTS_DIR.parent)),
        "evaluation_start": EVALUATION_START.isoformat(),
        "bootstrap_resamples": n_resamples,
        "bootstrap_seed": bootstrap_seed,
        "tuning_range": {
            "n_events": n_tuning,
            "same_events_for_both_targets": True,
            "last_event": common_ids[n_tuning - 1],
            "last_commence": at(common_ids[n_tuning - 1]),
        },
        "evaluation_range": {
            "events_with_rows": eval_rows,
            "twins_dropped": {t: r.twins for t, r in rows.items()},
            "common_events": len(eval_events),
            "first_common": eval_events[0],
            "last_common": eval_events[-1],
            "last_common_commence": at(eval_events[-1]),
            "scored_events": len(scored),
            "first_scored": scored[0],
            "first_scored_commence": at(scored[0]),
            "last_scored": scored[-1],
            "last_scored_commence": at(scored[-1]),
            "unscored_partial_block_events": len(unscored_tail),
            "folds": int(folds["fold"].nunique()),
        },
        "training_rule": {
            "rule": "every fold trains on the common events before its validation block, "
            "so both targets train on identical events and features",
            "extra_train_events_if_each_target_used_all_its_earlier_events": {
                t: int(folds.loc[folds["target"] == t, "extra_train_events"].max()) for t in TARGETS
            },
        },
        "pinnacle": {
            name: {
                "sampler_seed": seed,
                "random_state": seed,
                "n_trials": n_trials,
                "best_trial": studies[name][0].best_trial.number,
                "params": studies[name][0].best_params,
                "tuning_range_cv_mse": studies[name][0].best_value,
                "tune_seconds": round(studies[name][1], 1),
            }
            for name, seed in SEEDS.items()
        },
        "bet365": {
            name: {"random_state": seed, "params": frozen[name]["params"]}
            for name, seed in SEEDS.items()
        },
        "overall": overall,
        "keep_test_pinnacle": verdict,
        "keep_test_bet365_on_common_set": bet365_on_common,
        "skill_pinnacle_minus_bet365": skill,
        "target_correlation": target_corr,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    pd.set_option("display.width", 250)
    print("\n" + results.to_string(index=False))
    print("\n" + json.dumps({"pinnacle": verdict, "skill": skill}, indent=2, default=str))
    print(f"\nResults in {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42, help="bootstrap seed")
    args = parser.parse_args()
    asyncio.run(main(args.n_trials, args.bootstrap, args.seed))
