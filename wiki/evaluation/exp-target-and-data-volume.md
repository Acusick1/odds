---
type: explanation
title: Target bookmaker and data volume experiments
description: The NBA bet365-versus-Pinnacle runs and the Exp 6 learning curve behind ADR-0002.
sources:
  - experiments/results/exp6_learning_curve/FINDINGS.md
  - experiments/configs/xgboost_bet365_tuning.yaml
  - experiments/configs/xgboost_bet365_baseline_tuning.yaml
  - experiments/configs/xgboost_pinnacle_tuning.yaml
  - experiments/scripts/exp6_learning_curve.py
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Target bookmaker and data volume experiments

These runs are the evidence for [ADR-0002](../../docs/adr/0002-bet365-clv-target.md): the target bookmaker is configurable, Pinnacle was the first target (February 2026, NBA), and bet365 became the target for NBA at scale and for EPL. The evidence is the MODELING.md experiment log (commits 2840086, b950dc5 and d7c6621), PR #168 and `experiments/results/exp6_learning_curve/`; the production config is PR #362.

## What the runs show

- Pinnacle CLV was unpredictable with public features: XGBoost CV R² -0.017 ± 0.015 (about 800 NBA events, tuner converged to maximum regularisation; commit d7c6621) and LSTM -0.075 ([early NBA signal and LSTM](exp-early-nba-signal-and-lstm.md#lstm-sequence-models), [line-movement models](../modelling/line-movement-models.md)).
- Pinnacle lines move less: its MSE (0.000431) is 6.7x lower than bet365's (0.002883), leaving little variance to explain (commit d7c6621; bet365's MSE first recorded in b950dc5). No results artefact holds the MSE figures.
- bet365 reached CV R² 0.036 on about 5K NBA events and 0.058 on EPL, the deployed model (PR #361 and PR #362; [search space and ablation](exp-epl-feature-ablation-and-search-space.md), [CLV model findings](clv-model-findings.md)).

## NBA target and data-volume runs

**bet365 at scale (2026-02-27).** About 5K OddsPortal NBA events, 11-fold walk-forward CV, 100 Optuna trials, devigged bet365 target. Tabular-only CV R² = 0.036 ± 0.028; adding injuries gave 0.036 ± 0.033 ([NBA timing and injuries](exp-nba-timing-and-injuries.md)). Configs (expanding walk-forward, 2,000 minimum training events, step 200): `experiments/configs/xgboost_bet365_tuning.yaml`, `experiments/configs/xgboost_bet365_baseline_tuning.yaml`. First recorded in commit 2840086 (PR #164); no results directory holds the 11-fold numbers.

**Pinnacle as target (2026-03-01).** About 800 Odds API NBA events, 4-fold walk-forward, 100 trials: CV R² = -0.017 ± 0.015. Config: `experiments/configs/xgboost_pinnacle_tuning.yaml` (expanding walk-forward, 300 minimum training events, step 80). Recorded in commit d7c6621.

**Learning curve, Exp 6 (2026-03-01).** Tuned tabular-only bet365 model on 500-4,524 OddsPortal events, expanding walk-forward CV (`experiments/scripts/exp6_learning_curve.py`, PR #168, results in `experiments/results/exp6_learning_curve/FINDINGS.md`):

| Events | 500 | 1,000 | 1,500 | 2,000 | 3,000 | 4,000 | 4,524 |
|---|---|---|---|---|---|---|---|
| R² mean | -0.066 | -0.040 | +0.036 | +0.031 | +0.014 | -0.006 | +0.018 |

R² plateaus at about 1.5K events and oscillates between about 0 and +0.035 afterwards (`FINDINGS.md`: -0.006 to +0.036). A log fit gives R² = 0.0301 ln(N) - 0.2302, marginal dR²/dN = 6.66e-06 at N = 4,524, and extrapolates 0.047 at 10K and 0.068 at 20K events; the recorded decision was that more data alone will not help. EPL has about 1.8K events, so a similar plateau is likely but not confirmed.

## Caveats

- The extrapolation to 20K events (+0.07) sits uneasily with "more data will not help"; the reading rests on the small marginal slope.
- All Exp 6 subsets sample the sharp tier (mean about 19 h before the game) and OddsPortal has no 0-3 h snapshots. The pregame subset was diluted, because about 93% of its events fell back to the sharp snapshot ([look-ahead bias](../modelling/look-ahead-bias.md#where-the-rule-breaks)).
- Tuned CV R² is optimistically biased (best of N trials); comparisons between equally tuned runs are fair ([CV protocol grid](exp-cv-protocol-grid.md)).
- The Pinnacle and bet365 runs differ in source (Odds API against OddsPortal), size (about 800 against about 5K events) and folds (4 against 11); no run compares the two targets on the same source and event count (#482).
- The production YAML (`experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`) also lists bet365 first in `retail_bookmakers`, so the target bookmaker's snapshot price is an input to `retail_sharp_diff` ([sharp benchmark](../modelling/sharp-benchmark.md)).
- The `odds train` bet365 and Pinnacle runs set `early_stopping_rounds` on the validation fold, so their fold metrics are optimistically biased ([caveat](clv-model-findings.md#caveats)). The Exp 6 learning curve was script-based with no `eval_set` and is not affected.

## Related

- [CLV model findings](clv-model-findings.md), [CLV target](../modelling/clv-target.md), [Training config](../modelling/howto-train-and-publish-model.md#choose-data)
