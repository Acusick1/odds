"""The #540 retail-EV protocol with each as-of-t feature group added alone, then all (#545).

Development data only, so nothing here is a keep-bar result. Each variant is the
price-only config of `retail_ev_price_only.py` with `feature_groups` set; tuning,
scoring folds, arms and seeds are that script's. Every variant scores the same rows,
so a variant's model is compared with the baseline and with the price-only model of
the same arm and seed, overall, per scoring fold and by event type.

Usage:
    uv run python experiments/scripts/retail_ev_as_of_t_groups.py
    uv run python experiments/scripts/retail_ev_as_of_t_groups.py --n-trials 3 --n-boot 200
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from odds_analytics.retail_ev_metrics import mse_gain, pick_ev_gain, weighted_mse
from odds_analytics.training.config import MLTrainingConfig, RetailEvFeatureGroupName
from retail_ev_price_only import (
    CONFIG_PATH,
    METRIC_SCALE,
    MODEL_LABELS,
    MODEL_RUNS,
    REPO_DIR,
    Arm,
    RunData,
    _est,
    data_fingerprint,
    git_dirty,
    git_sha,
    load_data,
    score_run,
    tuning_prefix,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = EXPERIMENTS_DIR / "results" / "retail_ev_as_of_t_groups"
REFERENCE = "price_only"
VARIANTS: dict[str, tuple[RetailEvFeatureGroupName, ...]] = {
    REFERENCE: (),
    "polymarket_at_t": ("polymarket_at_t",),
    "dynamics": ("dynamics",),
    "schedule_as_of_t": ("schedule_as_of_t",),
    "standings_as_of_t": ("standings_as_of_t",),
    "all_groups": ("polymarket_at_t", "dynamics", "schedule_as_of_t", "standings_as_of_t"),
}
MAX_WORKERS = 20


def variant_config(groups: tuple[RetailEvFeatureGroupName, ...]) -> MLTrainingConfig:
    raw = MLTrainingConfig.from_yaml(CONFIG_PATH).model_dump()
    raw["training"]["features"]["feature_groups"] = groups
    return MLTrainingConfig(**raw)


def group_coverage(data: RunData, scored: np.ndarray, reference: list[str]) -> dict[str, float]:
    """Share of scored rows with a value, for each column the variant adds."""
    added = [n for n in data.feature_names if n not in reference]
    X = data.X[scored]
    return {n: float(np.mean(~np.isnan(X[:, data.feature_names.index(n)]))) for n in added}


def comparison_metrics(oof: pd.DataFrame, variant: str, n_boot: int) -> dict[str, Any]:
    y = oof["y"].to_numpy()
    w = oof["w_loss"].to_numpy()
    we = oof["w_event"].to_numpy()
    ev = oof["event_id"].to_numpy()
    base = oof["baseline"].to_numpy()
    out: dict[str, Any] = {"rows": len(oof), "events": oof["event_id"].nunique()}
    out["wmse_baseline"] = weighted_mse(y, base, w) * METRIC_SCALE
    for label in MODEL_LABELS:
        pred = oof[f"{variant}:{label}"].to_numpy()
        out[f"wmse_{label}"] = weighted_mse(y, pred, w) * METRIC_SCALE
        out |= _est(f"mse_gain_{label}", mse_gain(y, pred, base, w, ev, n_boot), METRIC_SCALE)
        out |= _est(f"pick_gain_{label}", pick_ev_gain(y, pred, base, we, ev, n_boot), 1.0)
        if variant != REFERENCE:
            ref = oof[f"{REFERENCE}:{label}"].to_numpy()
            out |= _est(
                f"mse_gain_{label}_vs_{REFERENCE}",
                mse_gain(y, pred, ref, w, ev, n_boot),
                METRIC_SCALE,
            )
            out |= _est(
                f"pick_gain_{label}_vs_{REFERENCE}",
                pick_ev_gain(y, pred, ref, we, ev, n_boot),
                1.0,
            )
    return out


FAMILY_ALPHA = 0.05
RESIDUAL_LABELS: tuple[str, ...] = tuple(run.label for run in MODEL_RUNS if run.arm is Arm.RESIDUAL)


def bonferroni_table(oof: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    """Residual-arm gains over price-only with CIs at the family-wise level of each slice.

    The family is every non-reference variant and residual seed, so a CI excludes zero
    only at level 1 - FAMILY_ALPHA / (that many tests).
    """
    challengers = [v for v in VARIANTS if v != REFERENCE]
    n_tests = len(challengers) * len(RESIDUAL_LABELS)
    tail = FAMILY_ALPHA / n_tests / 2 * 100
    slices = [("all", "all", oof)] + [
        ("event_type", value, oof[oof["event_type"] == value])
        for value in sorted(oof["event_type"].unique())
    ]
    records: list[dict[str, Any]] = []
    for key, value, sub in slices:
        y, w = sub["y"].to_numpy(), sub["w_loss"].to_numpy()
        _, inverse = np.unique(sub["event_id"].to_numpy(), return_inverse=True)
        n_events = int(inverse.max()) + 1
        counts = np.random.default_rng(0).multinomial(
            n_events, np.full(n_events, 1.0 / n_events), size=n_boot
        )
        w_event = np.bincount(inverse, weights=w)
        for variant in challengers:
            for label in RESIDUAL_LABELS:
                se_ref = np.bincount(
                    inverse, weights=w * (y - sub[f"{REFERENCE}:{label}"].to_numpy()) ** 2
                )
                se_var = np.bincount(
                    inverse, weights=w * (y - sub[f"{variant}:{label}"].to_numpy()) ** 2
                )
                samples = (counts @ se_ref - counts @ se_var) / (counts @ w_event)
                low, high = np.percentile(samples, [tail, 100 - tail])
                records.append(
                    {
                        "slice": key,
                        "value": value,
                        "variant": variant,
                        "run": label,
                        "n_tests": n_tests,
                        "level": 1 - FAMILY_ALPHA / n_tests,
                        "gain_vs_price_only": (se_ref.sum() - se_var.sum())
                        / w_event.sum()
                        * METRIC_SCALE,
                        "ci_low": low * METRIC_SCALE,
                        "ci_high": high * METRIC_SCALE,
                        "excludes_zero": bool(low > 0 or high < 0),
                    }
                )
    return pd.DataFrame(records)


def results_table(oof: pd.DataFrame, n_boot: int) -> pd.DataFrame:
    slices: list[tuple[str, str, pd.DataFrame]] = [("all", "all", oof)]
    for event_type in sorted(oof["event_type"].unique()):
        slices.append(("event_type", event_type, oof[oof["event_type"] == event_type]))
    for fold in sorted(oof["fold"].unique()):
        slices.append(("fold", str(int(fold)), oof[oof["fold"] == fold]))
    records = [
        {
            "variant": variant,
            "slice": key,
            "value": value,
            **comparison_metrics(sub, variant, n_boot),
        }
        for variant in VARIANTS
        for key, value, sub in slices
    ]
    return pd.DataFrame(records)


async def main(n_trials: int, n_boot: int) -> None:
    sha, dirty = git_sha(), git_dirty()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = {name: variant_config(groups) for name, groups in VARIANTS.items()}
    data: dict[str, RunData] = {}
    for name, config in configs.items():
        data[name] = await load_data(config)
        print(f"{name}: {len(data[name].y):,} rows, {len(data[name].feature_names)} features")
    reference = data[REFERENCE]
    for name, d in data.items():
        if not np.array_equal(d.event_ids, reference.event_ids) or not np.array_equal(
            d.y, reference.y
        ):
            raise ValueError(f"{name} rows differ from the price-only rows")
    n_rows = tuning_prefix(reference, configs[REFERENCE])

    jobs = [(name, run) for name in VARIANTS for run in MODEL_RUNS]
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [
            pool.submit(
                score_run,
                configs[name],
                data[name],
                n_rows,
                n_trials,
                run,
                f"retail_ev_as_of_t_{name}",
            )
            for name, run in jobs
        ]
        results = {job: f.result() for job, f in zip(jobs, futures, strict=True)}

    oof = reference.rows.copy()
    scored = results[(REFERENCE, MODEL_RUNS[0])]["preds"]["row"].to_numpy()
    for (name, run), r in results.items():
        preds = r["preds"]
        if not np.array_equal(preds["row"].to_numpy(), scored):
            raise RuntimeError("runs scored different rows")
        oof.loc[preds["row"], f"{name}:{run.label}"] = preds["model"].to_numpy()
    oof.loc[scored, "fold"] = results[(REFERENCE, MODEL_RUNS[0])]["preds"]["fold"].to_numpy()
    oof = oof.loc[scored].reset_index(drop=True)

    table = results_table(oof, n_boot)
    table.to_csv(OUTPUT_DIR / "results.csv", index=False)
    bonferroni = bonferroni_table(oof, n_boot)
    bonferroni.to_csv(OUTPUT_DIR / "bonferroni.csv", index=False)
    folds = pd.concat(
        [r["folds"].assign(variant=name) for (name, _), r in results.items()], ignore_index=True
    )
    folds.to_csv(OUTPUT_DIR / "folds.csv", index=False)
    importance = pd.concat(
        [r["importance"].assign(variant=name) for (name, _), r in results.items()],
        ignore_index=True,
    )
    importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    summary = {
        "git_sha": sha,
        "git_dirty": dirty,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
        "config": str(CONFIG_PATH.relative_to(REPO_DIR)),
        "data_sha256": data_fingerprint(reference),
        "n_trials": n_trials,
        "n_boot": n_boot,
        "rows": len(reference.y),
        "scored_rows": len(oof),
        "scored_events": int(oof["event_id"].nunique()),
        "variants": {
            name: {
                "groups": list(groups),
                "features": data[name].feature_names,
                "coverage_scored": group_coverage(data[name], scored, reference.feature_names),
                "coverage_tuning": group_coverage(
                    data[name], np.arange(n_rows), reference.feature_names
                ),
                "runs": [
                    {
                        k: results[(name, run)][k]
                        for k in ("run", "best_trial", "best_tuning_cv_wmse", "best_params")
                    }
                    for run in MODEL_RUNS
                ],
            }
            for name, groups in VARIANTS.items()
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    for label in MODEL_LABELS:
        cols = [
            f"mse_gain_{label}",
            f"mse_gain_{label}_ci_low",
            f"mse_gain_{label}_ci_high",
            f"mse_gain_{label}_vs_{REFERENCE}",
            f"mse_gain_{label}_vs_{REFERENCE}_ci_low",
            f"mse_gain_{label}_vs_{REFERENCE}_ci_high",
        ]
        print(label)
        print(table[["variant", "slice", "value", *cols]].round(1).to_string(index=False))
    print(bonferroni.round(1).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    asyncio.run(main(args.n_trials, args.n_boot))
