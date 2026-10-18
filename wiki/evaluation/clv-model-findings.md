---
type: explanation
title: What the CLV model can and cannot do
description: Hub for the CLV model, with the deployed model, current conclusions linked to ADRs and experiments, the data used, and a dated index.
sources:
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - odds_cli.commands.model.publish_model
  - odds_mcp.server.find_retail_edges
  - experiments/results/exp6_learning_curve/FINDINGS.md
status: stable
---

# What the CLV model can and cannot do

The model predicts line movement: the devigged bet365 CLV delta between a snapshot and the close ([CLV target](../modelling/clv-target.md)). The signal is weak but real: best EPL CV R² about 0.058, NBA plateau about 0.036. It cannot be bet alone: flat betting on NBA loses at every useful threshold ([betting simulation](exp-betting-simulation.md)); that simulation is NBA only and EPL has not been simulated. The agent treats the model as a supplementary sanity check (`packages/odds-mcp/agents/common.md`); its strongest feature is the sharp-retail spread the agent already sees through `odds_mcp.server.find_retail_edges` ([sharp benchmark](../modelling/sharp-benchmark.md), [ADR-0005](../../docs/adr/0005-information-edge.md)).

Several results below used feature sets the code does not define, for example the 7-feature tabular group and the `trajectory` and `cross_source` groups of experiments before 2026-03-15; their numbers are as recorded and are not reproducible ([feature groups](../modelling/feature-groups.md)).

## Deployed model

`epl-clv-home` version `v308058c`, published 2026-04-23: XGBoost on tabular + match_stats, sliding-380:50 walk-forward CV (`test_split: 0.0`, so nothing is held out beyond the CV folds), principled search space, hybrid sharp reference; CV val_R² +0.058, val_MSE 0.000716, replacing a tabular-only model (+0.035) (#361). Source config (#362): `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`; `v308058c` was trained from that YAML as at commit `2bf3712`, which set `early_stopping_rounds: 50` on the validation fold (see [Caveats](#caveats)). The live S3 version is the authority and cannot be verified from the repo; check with `aws s3 ls s3://<model-bucket>/epl-clv-home/` ([model store](../modelling/model-store.md)). The production scorer handles models trained on several feature groups, including the standings, match_stats and `epl_schedule` cache preload (#359). Publishing uses `odds_cli.commands.model.publish_model`. Predictions exist only where inline scoring runs, which is a local scheduler, one row per snapshot ([prediction scoring](../modelling/prediction-scoring.md)).

## Current conclusions

Every result below that predates the training and CV validity fixes (#474, #434, #477, #437, #475, #481) is void, including the deployed model's CV metrics, the EPL feature ablation, the CV protocol grid and ADR-0002's rejection of Pinnacle as a target (#494). They are kept as the record, not as evidence.

- The current evidence on the EPL model's role is [EPL CLV re-baseline against the simple baseline](exp-epl-clv-rebaseline.md). Out of sample on the ADR-0010 evaluation range, the tabular bet365 model beats a one-feature OLS fit on `tab_retail_sharp_diff` by about 6 × 10⁻⁶ MSE over two seeds, but neither seed's bootstrap CI excludes zero. So the model fails the keep test pre-registered on #496.
- bet365, not Pinnacle, is the target: [ADR-0002](../../docs/adr/0002-bet365-clv-target.md), [target and data volume](exp-target-and-data-volume.md).
- Sharp reference is Pinnacle with Betfair fallback: [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md), [EPL sharp-reference experiments](exp-epl-sharp-reference.md).
- Sharp-vs-retail divergence is the core feature: [sharp benchmark](../modelling/sharp-benchmark.md).
- On EPL, match_stats helps and the rest do not: [EPL ablation](exp-epl-feature-ablation-and-search-space.md). Injuries, rest, FPL and Polymarket add nothing ([NBA timing and injuries](exp-nba-timing-and-injuries.md), [early NBA signal search](exp-early-nba-signal-and-lstm.md)).
- Expanding walk-forward in 50-event steps is the protocol, tuned before 2024-08-01 and evaluated after: [ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md), [CV protocol on a common validation range](exp-cv-protocol-common-range.md). On a common validation range, expanding, sliding-380 and sliding-760 are within bootstrap noise of each other. Sliding windows need the principled search space ([search-space experiment](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test)).
- NBA injury features add nothing: [NBA timing and injuries](exp-nba-timing-and-injuries.md#nba-injury-features-exp-6-6b) (untested on EPL).
- LSTM adds nothing: [early NBA signal and LSTM](exp-early-nba-signal-and-lstm.md#lstm-sequence-models).
- More NBA data will not help (plateau about 1.5K events; the log-fit marginal return at 4,524 events is dR²/dN = 6.7e-06, `experiments/results/exp6_learning_curve/FINDINGS.md`); EPL has about 1.8K events and a plateau is likely but unconfirmed.
- Pregame is the default window ([sampling defaults](../modelling/training-config.md)): [NBA timing](exp-nba-timing-and-injuries.md#hours-to-game-exp-4), [decision time](../modelling/look-ahead-bias.md). Timing on EPL is untested.
- Polymarket features are untested at scale (230 events, no effect): [Polymarket market research](../collection/polymarket-market-research.md).
- NBA bookmakers release lines at different times: [line release timing](exp-nba-timing-and-injuries.md#bookmaker-line-release-timing).
- Flat betting does not survive the vig, and line shopping does not close the gap: [betting simulation](exp-betting-simulation.md).

## Data used

| | OddsPortal | Odds API | football-data.co.uk |
|---|---|---|---|
| Events | about 5K NBA (5 seasons) + 1.8K EPL (5 seasons) | about 1K NBA (Mar-Apr 2025, Oct 2025-Feb 2026) | about 4K EPL (11 seasons, 2015-2026) |
| Snapshots per event | 2 (open, close) | 15+ | 2 |

1,810 EPL events match between OddsPortal and FDUK (#217); 141 NBA events match between the Odds API and OddsPortal. Only the 1,810 and the 81 events lacking Pinnacle closing trace to #217; the other counts are unverified, so re-check with a count query before quoting. FDUK is compiled by hand twice weekly, its Pinnacle odds became unreliable after July 2025 (81 events in 2025-26 lack Pinnacle closing), and Betfair Exchange closing odds exist from 2024-25 ([historical ingest scripts](../collection/historical-ingest-scripts.md), [data sources](../collection/data-sources.md)).

## Dated index

- 2026-02-14 to 02-18 (#134, #136): [early NBA signal search](exp-early-nba-signal-and-lstm.md).
- 2026-02-20 (#157, #160) and 03-01 (6b, #169): [NBA injury features](exp-nba-timing-and-injuries.md#nba-injury-features-exp-6-6b).
- 2026-02-21 to 02-28 (#160, #163, #166): [LSTM](exp-early-nba-signal-and-lstm.md#lstm-sequence-models).
- 2026-02-27, 03-01: [bet365 target and data volume](exp-target-and-data-volume.md).
- 2026-02-28: [hours to game](exp-nba-timing-and-injuries.md#hours-to-game-exp-4).
- 2026-03-01: [betting simulation](exp-betting-simulation.md).
- 2026-03-10 to 03-19 (#217, #234, #253): [EPL sharp reference](exp-epl-sharp-reference.md).
- 2026-03-16, 03-19: [CV protocol grid](exp-cv-protocol-grid.md).
- 2026-03-20 (#255), 04-23 (#361): [EPL feature ablation](exp-epl-feature-ablation-and-search-space.md).
- 2026-04-23: [principled search space](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test).
- 2026-10-18 (#452): [CV protocol on a common validation range](exp-cv-protocol-common-range.md).
- 2026-10-18 (#496): [EPL CLV re-baseline against the simple baseline](exp-epl-clv-rebaseline.md).

## Caveats

- XGBoost runs through `odds train`, including the deployed model, set `early_stopping_rounds`, which XGBoost evaluated on the validation fold, so their fold metrics are optimistically biased (#474). Script-based experiments that fit without an `eval_set` are not affected.

## Related

- [Cross-validation](../modelling/cross-validation.md), [Training pipeline](../modelling/training-pipeline.md), [Run an experiment](howto-run-experiment.md)
