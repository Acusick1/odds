---
type: explanation
title: Power of the retail-EV keep bar
description: The October 2026 simulation of how often ADR-0012's two keep-bar tests pass a synthetic model on a fresh season of 100 to 1,140 events, under FDUK-only rows and dense live collection, for a diffuse calibrated model and a sparse information-edge model, with the minimum detectable effects, the size check and the assumptions behind them.
sources:
  - experiments/scripts/retail_ev_power.py
  - experiments/results/retail_ev_power/power.csv
  - experiments/results/retail_ev_power/effects.csv
  - experiments/results/retail_ev_power/calibration.csv
  - experiments/results/retail_ev_power/mde_calibrated.csv
  - experiments/results/retail_ev_power/mde_sparse.csv
  - experiments/results/retail_ev_power/summary.json
  - experiments/results/retail_ev_power/seed_sweep_power.csv
  - experiments/results/retail_ev_power/seed_sweep_mde_calibrated.csv
  - experiments/results/retail_ev_power/seed_sweep_mde_sparse.csv
  - odds_analytics.retail_ev_metrics.mse_gain_of_event_totals
  - odds_analytics.retail_ev_metrics.pick_ev_gain_of_event_totals
status: stable
---

# Power of the retail-EV keep bar

This experiment estimates, for #538 on the map #506, how many fresh-season events [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md)'s keep bar needs before a model with a real edge passes it. The target, baseline, weights and the two tests are on [CLV target](../modelling/clv-target.md#the-retail-ev-target). It resamples development events into synthetic seasons, gives each a synthetic model of known shape and gain, and runs the two tests.

- **The weighted-MSE condition is reachable in one season at a gain of about 4.5% to 6%** of the baseline's weighted MSE, with FDUK-only rows or with dense live rows when the model's errors persist across horizons. Only the optimistic bound, where errors renew at every exchange snapshot, brings that below 1%.
- **The pick condition's power depends on the model's shape more than on its gain.** A diffuse model, with the same share of the move known on every row, needs a gain of about 16% to 21% to reach 80% power in one season. A concentrated edge needs far less: a model that knows the move exactly on 2% to 2.6% of events, and predicts the baseline elsewhere, has a gain of about that size and reaches 80% on FDUK-only rows.
- **Passing both conditions is decided by the pick condition** for the diffuse model, whose MSE condition reaches 80% power at a much smaller gain. For the concentrated edge that knows the move exactly, the two need about the same share of events.

The minimum detectable effects below are therefore conditional on the model shape assumed. The one-season ranges are over six noise seeds ([seed spread](#seed-spread)); the other figures are from one seed.

## Setup

- **Script:** `experiments/scripts/retail_ev_power.py`, with outputs in `experiments/results/retail_ev_power/`. It builds the retail-EV decision rows for EPL events with a close (any status) from 2024-08-01 to 2026-07-31 on the local development database, with `odds_analytics.retail_ev.load_retail_ev_dataset` (at the current time) as it stood for the run (its row rules are on [CLV and the training target](../modelling/clv-target.md#the-retail-ev-target)), and trains no model. Its FDUK-only regime keeps the rows whose retail price and exchange price both come from FDUK opening snapshots. The end date leaves out 2026-27, the fresh season of [ADR-0016](../../docs/adr/0016-pre-register-retail-ev-evaluation.md). The main run uses 250 simulated seasons per cell, 2,000 bootstrap resamples per test and seed 42, and takes about 15 minutes on 18 worker processes. `--sweep-seeds` instead reruns the calibrated and sparse models at 380 events under several seeds, in about 9 minutes for six.
- **The tests:** each simulated season is scored by `odds_analytics.retail_ev_metrics.mse_gain_of_event_totals` and `odds_analytics.retail_ev_metrics.pick_ev_gain_of_event_totals`, the event-total forms of the keep-bar functions, with the default resample count. A condition passes when its 95% CI lies above zero. Power is the share of seasons that pass.
- **Seasons:** each simulated season draws N events with replacement from the regime's development events, keeping each event's rows, weights and realised targets whole. N is 100, 190, 380 (one EPL season), 760 and 1,140. `w_event` is recomputed within each regime.

### Collection regimes

| Regime | Rows | Events | Seasons | Rows per event (median) | Exchange snapshots per event (median) | Baseline weighted MSE | RMS of the exchange's move to the close |
|---|---|---|---|---|---|---|---|
| FDUK-only | 10,833 | 739 | 2024-25, 2025-26 | 12 | 1 | 0.00333 | 2.26 pp |
| Dense live | 102,003 | 49 | 2025-26 | 2,145 | 43 | 0.00555 | 2.96 pp |

- **FDUK-only** is what a season yields with no live collection: rows whose retail price and exchange price at t both come from the FDUK opening snapshot. Every such event has one exchange price, so its rows differ only by book and outcome. By event weight, 52% of these rows are 1 to 3 days out and 41% 6 to 24 hours out (`summary.json` `pools`).
- **Dense live** is what full live collection yields: every row of the 49 events with OddsPortal live rows, kicking off from 2026-03-14 to 2026-05-24, including their FDUK rows. 41 of them also have live Betfair job prices. By event weight, 50% of their rows are 3 to 7 days out, 32% 1 to 3 days out, and 1.5% within the last hour, because the scrape takes more snapshots further out.

### The synthetic models

The weighted squared error of a row is a squared probability error, `w_event × dp²`, where `dp = (target − baseline) / retail_decimal` is the move of the devigged exchange price from t to the close ([CLV target](../modelling/clv-target.md#the-retail-ev-target)). `dp` is the same for every book quoted against one exchange snapshot, and sums to zero across the three outcomes. Its scale depends on the price: on FDUK-only rows its RMS is about 3.0 pp at odds under 3 and 1.4 to 1.7 pp above (`summary.json` `by_odds_band`).

Each model predicts a signal `s` for `dp`, and its EV prediction is `baseline + retail_decimal × s`. It bets a row when that prediction is positive. `σ` is the weighted RMS of `dp` in the row's cell of horizon bucket and odds band (under 2, 2–3, 3–4.5, 4.5–7, over 7), and `z` is unit normal noise, centred across the three outcomes.

- **Calibrated (diffuse), gain g:** `s = g × dp + sqrt(g × (1 − g)) × σ × z` on every row. Within each cell its expected weighted MSE is `(1 − g)` times the baseline's, and the expected slope of `dp` on `s` is 1. The measured slopes (`calibration.csv`) have a median of 0.99 to 1.00 and lie within 0.95 to 1.03 for nine cells in ten at gains of 5% or more; the outliers are cells with few events, such as FDUK-only rows 1 to 6 hours out (20 events).
- **Sparse (information edge), share f and informed gain h:** on a share f of events, drawn afresh for each replicate, the calibrated signal with gain h; on the rest `s = 0`, so the model is the baseline. Its overall gain is about f × h (`effects.csv` `realised_gain`). h = 1 knows the move exactly; h = 0.5 knows it at a correlation of 0.71.
- **No information, scale τ:** `s = τ × σ × z`, noise with no edge. It is the size check: with no edge, a test should pass in at most 2.5% of seasons.

The noise `z` has two structures. **Per outcome** draws once per event and outcome, so the model's error persists across horizons; this is the case for pre-registration. **Per exchange snapshot** draws once per event, exchange snapshot and outcome, so the error renews at each snapshot and dense events average it out; this is an optimistic upper bound, not a case to plan on. In FDUK-only rows the two are the same, because each event has one exchange snapshot. The sparse and no-information models use per-outcome noise. Each event has 256 noise replicates per model, and the replicates and season draws are shared across the models in a regime.

## Results

### Weighted-MSE condition, calibrated model

Power at 380 events and, in brackets, at 760 and 1,140 (`power.csv`):

| Gain g | FDUK-only | Dense, errors persist | Dense, errors renew (upper bound) |
|---|---|---|---|
| 1% | 0.24 (0.43, 0.50) | 0.22 (0.44, 0.60) | 0.82 (0.99, 1.00) |
| 2% | 0.38 (0.71, 0.83) | 0.42 (0.73, 0.90) | 0.98 (1.00, 1.00) |
| 5% | 0.72 (0.98, 1.00) | 0.78 (0.98, 1.00) | 1.00 |
| 10% | 0.97 (1.00, 1.00) | 0.98 (1.00, 1.00) | 1.00 |

### Pick condition, calibrated model

| Gain g (correlation) | FDUK-only | Dense, errors persist | Dense, errors renew (upper bound) |
|---|---|---|---|
| 2% (0.14) | 0.16 (0.30, 0.38) | 0.09 (0.14, 0.20) | 0.31 (0.55, 0.69) |
| 5% (0.22) | 0.24 (0.45, 0.60) | 0.30 (0.50, 0.64) | 0.72 (0.93, 0.98) |
| 10% (0.32) | 0.37 (0.72, 0.86) | 0.44 (0.72, 0.87) | 0.85 (0.99, 1.00) |
| 20% (0.45) | 0.82 (0.97, 1.00) | 0.85 (0.98, 1.00) | 0.99 (1.00, 1.00) |

At a 10% gain the model picks 4.2% to 4.5% of the event weight, against the baseline's 1.0% to 1.3%. Its picks' `w_event`-weighted realised EV is 1.1% (FDUK-only) and 2.0% to 2.1% (dense) a pick, and its total gain over the baseline is 0.0005 to 0.0007 per event (`effects.csv`).

### Pick condition, sparse model

Power at 380 events, with the overall gain f × h in brackets:

| Share of informed events f | FDUK-only, h = 1 | Dense, h = 1 | FDUK-only, h = 0.5 | Dense, h = 0.5 |
|---|---|---|---|---|
| 1% | 0.27 (1%) | 0.53 (1%) | 0.04 (0.5%) | 0.08 (0.5%) |
| 2% | 0.75 (2%) | 0.96 (2%) | 0.13 (1%) | 0.13 (1%) |
| 5% | 1.00 (5%) | 1.00 (5%) | 0.37 (2.5%) | 0.24 (2.5%) |
| 10% | 1.00 | 1.00 | 0.60 (5%) | 0.48 (5%) |
| 20% | 1.00 | 1.00 | 0.91 (10%) | 0.77 (10%) |

At a gain of about 2%, the calibrated model's pick power on FDUK-only rows is 0.16, the sparse model's with h = 1 is 0.75, and with h = 0.5 (f = 5%, gain 2.5%) 0.37. The pick total rewards a few large, correct edges more than many small ones, because each picked row's realised EV carries the whole move to the close (inferred).

### Size check

With no information (`power.csv`, family `no_information`):

- **The MSE condition** passes in at most 1.2% of seasons at every scale and size.
- **The pick condition** never passes on the dense events, and passes in at most 1.2% of FDUK-only seasons at τ = 0.2 and 0.5. At τ = 0.05 it passes in about 10% of FDUK-only seasons from 190 events upwards, four times the nominal rate.

The excess at small noise comes from resampling a fixed pool. Small noise changes only the picks of rows priced within a hair of zero EV, and those rows' realised outcomes are fixed in the 739 development events: on them, the added picks happened to earn more than the baseline's (0.28% against 0.11% a pick, `effects.csv`). Every simulated season reuses those outcomes, so the test treats the pool's luck as an edge. The same effect sets a floor of about 0.1 under the calibrated model's FDUK-only pick power at small gains (0.10 at g = 0.5% and 380 events). So on FDUK-only rows a pick power below about 0.15 is not evidence of an edge.

### Seed spread

`--sweep-seeds 42 142 242 342 442 542` reruns every calibrated and sparse cell at 380 events under six seeds. A seed fixes the model's noise replicates, the informed events and the season draws. One-season pick power, minimum to maximum over the six (`seed_sweep_power.csv`):

| Model | FDUK-only | Dense, errors persist | Dense, errors renew (upper bound) |
|---|---|---|---|
| Calibrated, g = 10% | 0.36–0.44 | 0.48–0.61 | 0.82–0.92 |
| Calibrated, g = 20% | 0.79–0.85 | 0.84–0.94 | 0.98–1.00 |
| Sparse h = 1, f = 2% | 0.75–0.83 | 0.90–0.96 | |
| Sparse h = 0.5, f = 10% | 0.51–0.61 | 0.43–0.65 | |

The spread is wider on the dense pool, up to about 0.2 in a cell against about 0.1 on FDUK-only rows. With 49 events, each seed's 256 noise replicates per event fix how much the model realises on each event, and a season of 380 draws every event about eight times, so that realised gain is not averaged away. The spread exceeds the 2.5-point binomial error of a single cell, so differences between regimes smaller than these ranges are within noise.

### Minimum detectable effect at 80% power

Calibrated model, as a share of the baseline's weighted MSE, interpolated between grid points. The 380-event rows give the range over six noise seeds (`seed_sweep_mde_calibrated.csv`); the other rows are seed 42 (`mde_calibrated.csv`):

| Events | Condition | FDUK-only | Dense, errors persist | Dense, errors renew (upper bound) |
|---|---|---|---|---|
| 190 | MSE | 9.8% | 9.5% | 2.1% |
| 190 | Pick | 30% | 28% | 19% |
| 380 | MSE | 4.9–6.2% | 4.4–5.6% | 0.75–0.97% |
| 380 | Pick | 19–21% | 16–19% | 6.3–9.4% |
| 760 | MSE | 2.7% | 2.5% | 0.5% or less |
| 760 | Pick | 12% | 12% | 3.0% |
| 1,140 | MSE | 1.9% | 1.7% | 0.5% or less |
| 1,140 | Pick | 8.9% | 8.5% | 2.5% |

Under seed 42, the 380-event MSE minimum detectable gains are 2.1, 3.0 and 0.52 × 10⁻⁴ in weighted-MSE units. "0.5% or less" means power already exceeds 80% at the smallest gain on the grid. At 100 events the MSE condition needs 20%, 18% and 4.4%, and the pick condition reaches 80% only under the upper bound (35%).

Sparse model, as the share of informed events f; the 380-event rows are six-seed ranges (`seed_sweep_mde_sparse.csv`) and the 760-event row seed 42 (`mde_sparse.csv`):

| Events | Condition | FDUK-only, h = 1 | Dense, h = 1 | FDUK-only, h = 0.5 | Dense, h = 0.5 |
|---|---|---|---|---|---|
| 380 | MSE | 1.6–1.7% | 1.6–1.7% | 8.3–9.6% | 8.6–11% |
| 380 | Pick | 1.9–2.6% | 1.6–1.8% | 18–20% | 16–19% in 2 seeds; not reached by 20% in 4 |
| 760 | Pick | 1% or less | 1% or less | 8.7% | 13% |

## What it means for pre-registration

- **One season is enough for the MSE condition at a gain of about 4.5% to 6%** on either collection regime under persistent errors. Which error structure a real model has is not known; a model that mostly learns event-level information, such as team news, would have persistent errors (inferred).
- **One season is enough for the pick condition only for a large diffuse gain (about 16% to 21%) or a concentrated edge.** A model whose edge is information on a few events, such as late team news, is the shape the pick condition can detect in one season.
- **Dense collection raises pick power for the sparse model with h = 1 and under the renewing-error bound.** For the calibrated model with persistent errors it gives a minimum detectable gain of 16–19% against 19–21% on FDUK-only rows, and for the sparse model with h = 0.5 the two regimes' ranges overlap; both differences are within the seed spread.

## Assumptions and limitations

- **The synthetic models' structure is assumed.** Both families are calibrated within each horizon and odds cell, with normal errors independent of the move. A real model's gain is likely uneven across horizons, books and outcomes, and its picks could be better or worse than these (inferred).
- **The development seasons stand in for the fresh one.** Resampling assumes the fresh season's moves, books, horizons and odds look like 2024-08 to 2026-07. Realised outcomes are reused across seasons, which the size check shows inflates FDUK-only pick power at small effects.
- **The dense pool is 49 events from 2026-03-14 to 2026-05-24.** The spread between events, and any difference between the end of a season and the rest of it, are under-represented. Drawing up to 1,140 events from it repeats each event about 23 times, so its multi-season results show the effect of row density, not of more distinct events.
- **Collection is assumed to keep its development shape.** The dense events put 1.5% of their weight in the last hour and half of it 3 to 7 days out. A fresh season collected on another schedule would weight horizons differently.
- **Power has simulation error.** With 250 seasons a cell, the binomial standard error of a power near 80% is about 2.5 percentage points. On top of that, the noise seed moves one-season pick power by up to about 0.2 on the dense pool and 0.1 on FDUK-only rows ([seed spread](#seed-spread)); the tables other than the six-seed ranges are from one seed. The calibrated gain grid steps from 20% to 30% to 40%, so minimum detectable gains above 20% carry interpolation error of a few points.

## Related

- [CLV target](../modelling/clv-target.md), [Target functions](../modelling/clv-target-functions.md#keep-bar-metrics)
