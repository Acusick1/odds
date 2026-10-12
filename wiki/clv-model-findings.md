---
type: concept
title: CLV model findings
summary: Hub - what the CLV model can and cannot do, the deployed model, current conclusions and data used, with links to every experiment and decision.
sources:
  - docs/MODELING.md#what-the-model-predicts
  - docs/MODELING.md#deployed-model
  - docs/MODELING.md#current-conclusions
  - docs/MODELING.md#data-used
  - docs/MODELING.md#experiment-log
  - docs/BETTING_AGENT.md#role-of-the-xgboost-model
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
verified_at: 7887039
---

# CLV model findings

The model predicts line movement: the devigged bet365 CLV delta between a snapshot and the close ([clv](clv.md)). The signal is weak but real: best EPL CV R² about 0.058, NBA plateau about 0.036. It cannot be bet alone: flat betting on NBA loses at every useful threshold ([exp-betting-simulation](exp-betting-simulation.md)). The agent treats it as a supplementary sanity check; its strongest feature is the sharp-retail spread the agent already sees ([retail-sharp-divergence](retail-sharp-divergence.md), [decision-pivot-to-information-edge](decision-pivot-to-information-edge.md)).

## Deployed model

`epl-clv-home` version `v308058c`, published 2026-04-23: XGBoost on tabular + match_stats, sliding-380:50 walk-forward CV, principled search space, hybrid sharp reference; CV val_R² +0.058, val_MSE 0.000716, replacing a tabular-only model (+0.035). Source config: `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`. The live S3 version is the authority and cannot be verified from the repo; check with `aws s3 ls s3://<model-bucket>/epl-clv-home/` ([s3-model-store](s3-model-store.md), [howto-train-and-publish-model](howto-train-and-publish-model.md)).

## Current conclusions

- bet365, not Pinnacle, is the target: [decision-bet365-target](decision-bet365-target.md), [exp-nba-target-and-data-volume](exp-nba-target-and-data-volume.md).
- Sharp reference is Pinnacle with Betfair fallback: [decision-hybrid-sharp-reference](decision-hybrid-sharp-reference.md), [exp-epl-sharp-reference](exp-epl-sharp-reference.md).
- Sharp-vs-retail divergence is the core feature: [sharp-benchmark](sharp-benchmark.md).
- On EPL, match_stats helps and the rest do not: [exp-epl-feature-group-ablation](exp-epl-feature-group-ablation.md).
- Sliding-380:50 is the protocol: [decision-cv-protocol](decision-cv-protocol.md), [exp-cv-protocol-grid](exp-cv-protocol-grid.md), [exp-principled-search-space](exp-principled-search-space.md).
- NBA injury features add nothing: [exp-nba-injury-features](exp-nba-injury-features.md) (untested on EPL).
- LSTM adds nothing: [exp-lstm-sequence-models](exp-lstm-sequence-models.md).
- More NBA data will not help (plateau about 1.5K events); EPL has about 1.8K events and a plateau is likely but unconfirmed.
- Pregame (3-12 h) is the default window: [exp-hours-to-game](exp-hours-to-game.md), [decision-time](decision-time.md). Timing on EPL is untested.
- Polymarket features are untested at scale (230 events, no effect): [polymarket-pipeline](polymarket-pipeline.md).
- NBA bookmakers release lines at different times: [exp-bookmaker-line-release](exp-bookmaker-line-release.md).

## Data used

| | OddsPortal | Odds API | football-data.co.uk |
|---|---|---|---|
| Events | about 5K NBA + 1.8K EPL | about 1K NBA | about 4K EPL (2015-2026) |
| Snapshots per event | 2 (open, close) | 15+ | 2 |

1,810 EPL events match between OddsPortal and FDUK; 141 NBA events match between the Odds API and OddsPortal. FDUK is compiled by hand twice weekly, its Pinnacle odds became unreliable after July 2025 (81 events in 2025-26 lack Pinnacle closing), and Betfair Exchange closing odds exist from 2024-25 ([historical-odds-ingestion](historical-odds-ingestion.md), [data-sources](data-sources.md)).

## Dated index

2026-02-14 to 02-18: [exp-early-nba-signal-search](exp-early-nba-signal-search.md). 02-20 and 03-01 (6b): [exp-nba-injury-features](exp-nba-injury-features.md). 02-21 to 02-28: [exp-lstm-sequence-models](exp-lstm-sequence-models.md). 02-27, 03-01: [exp-nba-target-and-data-volume](exp-nba-target-and-data-volume.md). 02-28: [exp-hours-to-game](exp-hours-to-game.md). 03-01: [exp-betting-simulation](exp-betting-simulation.md). 03-10 to 03-19: [exp-epl-sharp-reference](exp-epl-sharp-reference.md). 03-16, 03-19: [exp-cv-protocol-grid](exp-cv-protocol-grid.md). 03-20, 04-23: [exp-epl-feature-group-ablation](exp-epl-feature-group-ablation.md). 04-23: [exp-principled-search-space](exp-principled-search-space.md).

## Hazards

- Results are as recorded at the time; several used feature sets that no longer exist ([feature-groups](feature-groups.md)).

## Related

- [Walk-forward CV](walk-forward-cv.md), [Training pipeline](training-pipeline.md)
