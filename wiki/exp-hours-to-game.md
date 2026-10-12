---
type: experiment
title: Hours-to-game effect (Exp 4)
summary: CLV delta grows with hours to the game, and sharp-retail divergence peaks 3-6 h before NBA games.
sources:
  - docs/MODELING.md#hours-to-game-effect
  - docs/MODELING.md#experiment-log
  - experiments/results/exp4_hours_to_game/FINDINGS.md
  - experiments/scripts/exp4_hours_to_game.py
verified_at: 7887039
---

# Hours-to-game effect (Exp 4)

Question: when before the game should a decision be made? Exp 4 (2026-02-28, git `afab058`) used 1,593 samples from 477 Odds API NBA events, 1-48 h before the game, Pinnacle target, 13 features (tabular 6 + injury 6 + `hours_until_event`). Script: `experiments/scripts/exp4_hours_to_game.py`.

## Results

| Hours before game | 0-3 | 3-6 | 6-9 | 9-12 | 12-18 | 18-24 | 24-36 |
|---|---|---|---|---|---|---|---|
| Samples | 165 | 454 | 419 | 275 | 40 | 200 | 39 |
| Target std | 0.029 | 0.022 | 0.035 | 0.040 | 0.027 | 0.035 | 0.038 |
| Mean abs CLV delta | 0.014 | 0.015 | 0.023 | 0.026 | 0.021 | 0.027 | 0.030 |

- Mean absolute CLV delta grows from 1.4 pp at 0-3 h to 3.0 pp at 24-36 h, so there is more room the earlier the decision; the 3-6 h bin has the lowest variance.
- `tab_retail_sharp_diff` peaks at 3-6 h (r = +0.097) and is weakest at 0-3 h (r = +0.030).
- `inj_impact_gtd_away` is the strongest single correlation in the project at 0-3 h (r = +0.28, p = 0.0003), flipping negative at 18-24 h; it did not survive on bet365 ([exp-nba-injury-features](exp-nba-injury-features.md)).
- Probability level correlates negatively with CLV close to the game (reversion) and positively far from it.
- Per-bin XGBoost models were all R² < 0 (27-317 training samples per bin); the pooled model is -0.009, with 12-18 h and 18-24 h the only bins with MSE ratio below 1.

## Conclusion

The pregame window (3-12 h) stays the default decision window; no evidence to change it ([decision-time](decision-time.md)). The dataset is too small to name an optimal hour, and the 5K-event OddsPortal data (opening and closing snapshots only) cannot answer timing questions. Timing on EPL is untested.

## Hazards

- Snapshot coverage is bimodal (peaks at 3-4 h and 8-10 h, secondary cluster 22-24 h), so the 12-18 h and 24-36 h bins have only 40 and 39 samples.
- Features were 13, including the injury group, which later runs found uninformative ([feature-groups](feature-groups.md)).

## Related

- [Learning curve](exp-nba-target-and-data-volume.md), [CLV model findings](clv-model-findings.md)
