---
type: experiment
title: LSTM sequence models
summary: Three LSTM variants on the Pinnacle target, all worse than predicting the mean, so LSTM was ruled out.
sources:
  - docs/MODELING.md#lstm-sequence-models
  - docs/MODELING.md#experiment-log
  - experiments/configs/lstm_pinnacle_tuning.yaml
  - experiments/configs/lstm_line_movement_tuning.yaml
  - packages/odds-analytics/odds_analytics/lstm_line_movement.py::LSTMLineMovementStrategy
verified_at: 7887039
---

# LSTM sequence models

Conclusion: sequence models add nothing; no LSTM variant beat predicting the mean, and LSTM was ruled out. XGBoost remains the model ([line-movement-models](line-movement-models.md), [clv-model-findings](clv-model-findings.md)).

| Date | Variant | Data | CV R² |
|---|---|---|---|
| 2026-02-21 | Sequence only (15 features x 24 timesteps) | 800 events, Pinnacle | -0.010 ± 0.039 |
| 2026-02-21 | Sequence + 17 static features | 800 events, Pinnacle | -0.122 ± 0.115 |
| 2026-02-28 | Sequence only, packed sequences, 15 x 8 timesteps, 50 trials, 5-fold walk-forward | about 1K Odds API events, Pinnacle | -0.075 ± 0.113 |

Config for the Pinnacle runs: `experiments/configs/lstm_pinnacle_tuning.yaml` (also `experiments/configs/lstm_line_movement_tuning.yaml`). Best of the first tuning: hidden 48, 2 layers; the static branch increased overfitting.

- **2026-02-27 mask fix (#162).** The log row says packed sequences were introduced so the mask is applied correctly (code: `_compact_sequences` in `packages/odds-analytics/odds_analytics/lstm_line_movement.py`). The experiment log files it as a bug fix with no result.
- **Exp 2 cross-check** (230 events, see [exp-early-nba-signal-search](exp-early-nba-signal-search.md)): LSTM at both decision times is R² about 0 once the `IN_PLAY` leak is fixed; the earlier LSTM +0.02 was contaminated.
- The 2026-02-28 result is worse than a constant predictor.

## Hazards

- Sequence features and the LSTM adapter still exist in code and are selectable through `strategy_type: lstm_line_movement`; nothing prevents using them ([feature-pipeline](feature-pipeline.md)).
- All runs used the Pinnacle target on NBA; LSTM was not retried with the bet365 target or on EPL (`docs/MODELING.md` records no such run).

## Related

- [Pinnacle as target](exp-nba-target-and-data-volume.md)
