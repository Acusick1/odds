---
status: accepted
---

# Evaluate EPL CLV with expanding walk-forward CV, tuned before 2024-08-01 and evaluated after

The EPL CLV model is evaluated with expanding-window walk-forward CV in 50-event steps, and there is no held-out test set. Hyperparameters are tuned only on events before 2024-08-01 (the tuning range). Events on or after that date (the evaluation range) are scored only by folds that walk forward with those hyperparameters frozen. This supersedes [ADR-0004](0004-walk-forward-cv-protocol.md). The evidence is on [CV protocol on a common validation range](https://github.com/Acusick1/odds/blob/1ca09fe/wiki/evaluation/exp-cv-protocol-common-range.md).

## Context

ADR-0004 chose sliding-380:50 from a grid that scored each cell on its own validation range: expanding cells from 700 events, sliding cells from their window size (#452). Its numbers also predate the training and CV validity fixes (#474, #434, #477, #437, #475, #481), and the map #494 voids every pre-fix result. The keep, retarget or retire decision on #494 needs a re-baseline (#496) on a fixed validation range that no tuning has seen, against the baseline defined on #495.

In the tuning range, rows exist from 2019-20, the first season whose FDUK closing snapshot carries bet365. That makes 1,900 events with one row each. Every row's decision snapshot is the FDUK day-before snapshot, because no OddsPortal snapshot in the range quotes a sharp bookmaker (`experiments/results/cv_protocol_common_range/event_counts.csv`, `snapshot_coverage.csv`).

## Decision

Adopted 2026-10-18 (#452, under the map #494):

- **Protocol:** `cv_method: walk_forward`, `window_type: expanding`, `val_step_events: 50`, `test_split: 0.0`, on tabular features, from `start_date: 2019-08-01`. The config is `experiments/configs/xgboost_epl_cv_protocol_expanding.yaml`.
- **Range split:**
  - The tuning range is events before 2024-08-01, so the 2019-20 to 2023-24 seasons. Protocol and hyperparameter choices are made on it.
  - The evaluation range is events on or after 2024-08-01. It is never used to make a protocol or hyperparameter choice: no tuning, and no validation during tuning.
- **Re-baseline layout:** `min_train_events` equals the tuning-range event count (1,900), so the first validation block is the first 50 evaluation-range events. Each later fold trains on every earlier event. The hyperparameters are frozen from the tuning range: the expanding cell's best trial, recorded in `experiments/results/cv_protocol_common_range/summary.json`.

## Considered options

The three cells were scored on the same 1,100 validation events, 2021-08-13 to 2024-04-27, each tuned with 100 Optuna trials on its own folds (`experiments/results/cv_protocol_common_range/results.csv`):

- **Sliding-760 (two seasons):** lowest pooled MSE, 0.000659 against expanding's 0.000662. The paired bootstrap 95% CI on the difference is [-11.2, 6.2] × 10⁻⁶ and includes zero.
- **Sliding-380 (one season, ADR-0004's choice):** pooled MSE 0.000667, CI against expanding [-4.9, 16.6] × 10⁻⁶.
- **Expanding:** chosen. The rule, fixed in the ticket's design before the run (#503), was to take the lowest pooled MSE, or expanding when that cell's CI against expanding includes zero. The design's reason for the fallback is that expanding is simpler and has more data.
- **Expanding from 2015:** not a separate option. The 2015-16 to 2018-19 seasons yield no rows, because their FDUK close quotes only Pinnacle, so expanding already starts at the first season with rows.
- **A fixed chronological test split:** rejected, as in ADR-0004. The range split serves its purpose: the evaluation range is never tuned on, and its folds still follow the production retrain cycle.

## Consequences

- The deployed model (`experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`, sliding-380:50) predates this decision and is unchanged. Whether it is kept, retargeted or retired is decided by #494 on the #496 re-baseline.
- The protocol was chosen on the tabular group with one seed. A feature-group experiment that adopts it inherits that choice untested.
- The cells' differences are smaller than their bootstrap CIs. Any window in this range is defensible on this evidence; expanding wins on the rule, not on measured skill.
- The pooled MSEs are optimistic, because each cell is scored on rows its tuner saw. The bias is largest for sliding-760, whose tuning folds are exactly the common range, so it tilts the comparison towards the cell the rule did not choose. The keep test is run only on the evaluation range.
- #496 must check that the tuning-range event count is still 1,900 before using `min_train_events: 1900`. A backfill or a sampler change would shift the first fold into the tuning range or past the evaluation start.
- The code defaults stay `cv_method: timeseries` and `test_split: 0.2`, so the protocol exists only where a YAML sets it.

## Evidence

- Issues #452, #494, #495, #496.
- `experiments/scripts/cv_protocol_common_range.py`; `experiments/results/cv_protocol_common_range/` (`results.csv`, `event_counts.csv`, `snapshot_coverage.csv`, `folds.csv`, `summary.json`).
- `experiments/configs/xgboost_epl_cv_protocol_base.yaml` and its cell configs.
- Wiki: [CV protocol on a common validation range](https://github.com/Acusick1/odds/blob/1ca09fe/wiki/evaluation/exp-cv-protocol-common-range.md), [Cross-validation](../../wiki/modelling/cross-validation.md).
