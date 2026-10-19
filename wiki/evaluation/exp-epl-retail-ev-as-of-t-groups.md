---
type: explanation
title: As-of-t feature groups on the retail-EV target
description: The October 2026 exploratory rerun of the price-only retail-EV protocol with the Polymarket, dynamics, schedule and standings groups added one at a time and together, against the baseline and the price-only model, overall, by event type and per fold.
sources:
  - experiments/scripts/retail_ev_as_of_t_groups.py
  - experiments/scripts/retail_ev_price_only.py
  - experiments/configs/xgboost_epl_retail_ev_price_only.yaml
  - experiments/results/retail_ev_as_of_t_groups/results.csv
  - experiments/results/retail_ev_as_of_t_groups/summary.json
  - experiments/results/retail_ev_as_of_t_groups/folds.csv
  - experiments/results/retail_ev_as_of_t_groups/feature_importance.csv
  - odds_analytics.retail_ev.RetailEvFeatures
  - odds_analytics.training.config.RetailEvFeatureGroupName
  - odds_analytics.retail_ev_metrics.mse_gain
  - odds_analytics.retail_ev_metrics.pick_ev_gain
status: stable
---

# As-of-t feature groups on the retail-EV target

This run repeats the protocol of [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md) (#540) with the as-of-t feature groups of #545 ([retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)). Each group is added alone, and then all four together. Each variant's models are compared with the martingale baseline and with the price-only model of the same arm and seed, on the same rows.

**This is development data, and the results are exploratory.** No keep-bar claim is made: ADR-0012 judges a model only on a pre-registered fresh season. The run reads events kicking off up to 2026-07-31 and leaves 2026-27 untouched.

**Findings.**

- **Polymarket and all groups gain over the baseline.** In the residual arm, `polymarket_at_t` and all four groups together beat the baseline by 100 to 127 × 10⁻⁶ weighted MSE, about 3 to 4% of the baseline's MSE. The price-only model gained 35 to 49.
  - With all groups, both seeds' CIs exclude zero, against the baseline and against the price-only model.
  - With `polymarket_at_t` alone, seed 42's do.
- **Dynamics adds little; schedule and standings alone add nothing.** `dynamics` is +8 and +28 over price-only, and seed 7's CI excludes zero. `schedule_as_of_t` and `standings_as_of_t` alone are level with price-only.
- **The gain comes mostly from the FDUK-only events.** On the 69 live-collected events every CI includes zero, although the points are positive for `polymarket_at_t`, `dynamics` and all groups.
- **Picks don't improve.** No variant's pick gain over the baseline or over price-only has a CI that excludes zero.
- **The direct arm stays at or below the baseline** in every variant. With all groups it reaches about the baseline.

## Setup

- **Script.** `experiments/scripts/retail_ev_as_of_t_groups.py` builds each variant from `experiments/configs/xgboost_epl_retail_ev_price_only.yaml` with `feature_groups` set. It runs the functions of `experiments/scripts/retail_ev_price_only.py`: the same tuning range, scoring folds, arms and seeds, with each variant's Optuna studies under their own name. It writes `experiments/results/retail_ev_as_of_t_groups/`.
- **Rows.** All variants share the 112,650 price-only rows, and the script checks this. 108,789 rows over 340 events are scored, in five 68-event folds from 2025-08-15 to 2026-05-24.
  - Fold 4 (2026-02-28 onward) holds 102,186 rows and the 44 OddsPortal live events.
  - `summary.json` records the git SHA, `git_dirty`, the script and config hashes, and `data_sha256`.
  - The price-only variant reproduces the #540 numbers exactly.
- **Tuning sees few of the new features.** Hyperparameters are tuned on 2024-25 only. Those are FDUK opening rows, with no Polymarket price (it starts in 2025-08) and no exchange or consensus move (an FDUK opening row has none). Only `retail_dispersion`, the schedule and the standings columns are populated there (`summary.json`, `coverage_tuning`).
  - The `polymarket_at_t` residual seed-42 run picked the same hyperparameters as price-only.
  - The new columns are learnt only when the frozen hyperparameters are refitted on each scoring fold's training rows.
- **Coverage on scored rows** (`coverage_scored`):
  - Polymarket columns: 100%. `pm_spread` is dropped as all NaN, because the backfill stores no bid or ask.
  - Exchange moves: 86 to 88%. Consensus moves at 1, 3 and 6 h: 79%, 64% and 46%.
  - `retail_dispersion` and schedule: 100%. Standings: 99.8%.
- **Metrics.** These are the price-only page's metrics: weighted-MSE gain and pick-EV gain, each with event-clustered bootstrap 95% CIs (2,000 resamples), in units of 10⁻⁶ and event units. "Over price-only" compares a variant's model with the price-only model of the same arm and seed.

## Results

Weighted-MSE gain over 340 events, with the baseline at 3419.6 (`results.csv`, slice `all`):

| Variant | Residual 42 vs baseline | Residual 7 vs baseline | Residual 42 vs price-only | Residual 7 vs price-only |
|---|---|---|---|---|
| price-only | 48.8 [−13.8, 105.6] | 34.7 [−52.3, 115.9] | | |
| `polymarket_at_t` | 108.7 [29.0, 188.8] | 100.5 [−18.1, 223.7] | 60.0 [18.4, 103.8] | 65.8 [−6.3, 144.4] |
| `dynamics` | 56.5 [−1.2, 109.9] | 62.7 [−11.2, 130.4] | 7.8 [−3.6, 19.4] | 28.0 [3.3, 53.4] |
| `schedule_as_of_t` | 40.5 [−19.8, 97.7] | 32.5 [−30.9, 93.2] | −8.3 [−34.6, 18.0] | −2.2 [−42.0, 39.0] |
| `standings_as_of_t` | 40.9 [−4.0, 86.9] | 33.0 [−7.5, 73.4] | −7.8 [−47.5, 34.6] | −1.7 [−58.9, 58.3] |
| all four | 126.9 [34.8, 219.0] | 114.9 [33.5, 195.3] | 78.2 [15.7, 144.7] | 80.2 [14.5, 150.7] |

Direct arm against the baseline, seeds 42 and 7:

- price-only: −76.2 and −76.5;
- `polymarket_at_t`: −35.5 and 14.9;
- `dynamics`: −91.5 and −53.0;
- `schedule_as_of_t`: −87.6 and −83.2;
- `standings_as_of_t`: −93.5 and −65.2;
- all four: 1.1 and 7.1.

Every one of these CIs includes zero.

Residual-arm pick gain over the baseline, in event units, seeds 42 and 7:

- price-only: 0.029 and 0.007;
- all four: 0.027 [−0.039, 0.094] and 0.030 [−0.023, 0.086].

No variant's pick gain over the baseline or over price-only excludes zero.

### By event type

Residual-arm gain over price-only, seeds 42 and 7:

| Variant | FDUK-only (271 events, 5,325 rows) | Live-collected (69 events, 103,464 rows) |
|---|---|---|
| `polymarket_at_t` | 63.4 [15.9, 111.9]; 69.3 [−13.7, 155.9] | 46.8 [−27.8, 128.3]; 52.5 [−75.6, 187.4] |
| `dynamics` | 8.3 [−3.6, 21.0]; 22.9 [−2.8, 52.5] | 6.0 [−16.4, 28.3]; 47.3 [−0.8, 98.0] |
| `schedule_as_of_t` | 2.7 [−25.8, 32.5]; −0.4 [−42.5, 40.3] | −50.3 [−118.7, 9.1]; −9.3 [−121.6, 95.7] |
| `standings_as_of_t` | −3.3 [−46.3, 41.8]; −9.6 [−72.2, 53.8] | −25.1 [−119.5, 68.2]; 28.2 [−109.8, 164.0] |
| all four | 88.8 [17.2, 167.5]; 81.3 [8.6, 158.5] | 37.6 [−89.2, 164.2]; 76.1 [−63.0, 224.2] |

FDUK-only events are where the Polymarket price at t is most different from the exchange price at t. An FDUK opening row's exchange price is the FDUK opening Betfair price, taken 1 to 4 days out. The Polymarket price is the backfill's grid point at most 2 hours old (inferred).

### By fold

Residual-arm gain over price-only, seed 42 then seed 7, each 68 events (`results.csv`, slice `fold`):

| Fold | `polymarket_at_t` | `dynamics` | `schedule_as_of_t` | `standings_as_of_t` | all four |
|---|---|---|---|---|---|
| 0 | −2.8; −13.4 | 12.5; 19.1 | 12.1; 24.4 | 59.5; 22.4 | 117.7; 73.2 |
| 1 | 15.0; −14.3 | −13.7; 11.9 | 1.3; 15.6 | −33.0; −8.4 | −81.5; −19.7 |
| 2 | 103.5; 157.5 | −4.7; 21.5 | −9.9; 26.1 | 58.0; 81.1 | 176.1; 200.4 |
| 3 | 62.5; 59.5 | 24.3; 35.8 | −7.0; −49.6 | −97.1; −97.4 | 42.7; 15.3 |
| 4 | 119.1; 136.2 | 20.1; 50.9 | −37.2; −26.6 | −25.7; −6.2 | 132.9; 129.2 |

Only a few fold CIs exclude zero: `polymarket_at_t` seed 42 in folds 2 and 4, `dynamics` seed 7 in fold 4, `standings_as_of_t` seed 42 in fold 3 (a loss), and all four in fold 2 for both seeds. Fold 4, with the dense live rows, has the largest point gain for `polymarket_at_t` and for all four, but its CIs for all four still include zero.

## Feature importance

The mean total-gain share over the residual runs (`feature_importance.csv`):

- **`polymarket_at_t`:** `pm_exchange_diff` 0.081, `pm_consensus_diff` 0.066 and `pm_prob` 0.050, behind `sharp_prob` 0.200 and `consensus_prob` 0.175.
- **`dynamics`:** `retail_dispersion` is third at 0.111. No move column is in the top eight.
- **All four:** the standings goal rate and goal difference and the home rest days rank beside the Polymarket columns. Standings and schedule alone gain nothing, so these splits may fit noise (inferred).

## What it means

- **Polymarket is the one group with a consistent gain.** It adds to the exchange price at t on development data. It is the only group whose gain over price-only holds across both seeds' point estimates, and on both event types.
- **The live-collected events decide too little.** With 69 events, mostly in one fold, they can't separate any variant from price-only. The fresh season's live collection is what can test it.
- **The tuning range can't see the new groups.** A follow-up could tune on folds that include 2025-26 data, giving up some of the scoring range, so the hyperparameters see Polymarket and the moves (not run).

## Related

- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md)
- [Target functions: retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)
- [Polymarket match prices as odds snapshots](../collection/polymarket-odds-snapshots.md)
