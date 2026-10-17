---
type: explanation
title: CLV and the training target
description: What CLV means here, why the target is a close-minus-snapshot delta of one devigged bookmaker, and how sampling changes the target.
sources:
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - odds_analytics.training.config.FeatureConfig
status: stable
---

# CLV and the training target

CLV (closing line value) is how far a bookmaker's line moves between a decision snapshot and the close. The CLV model regresses that movement: `target = devigged_fair_close - devigged_fair_at_snapshot` for one bookmaker (`target_bookmaker`). A positive value means the price at the snapshot was too low and the line moved up (`odds_analytics.sequence_loader.calculate_devigged_bookmaker_target` returns closing probability minus snapshot probability). Why a delta rather than the closing price is `(inferred)`: the decision needs the size of the move, and a delta is stationary and roughly mean-zero.

The functions, dispatch and config fields that compute it are on [Target functions](clv-target-functions.md).

## Sampling changes the target

The delta is measured from the sampled snapshot to the close ([decision time](look-ahead-bias.md)), so the same event can yield different targets under different samplers. A target measured from a closing-tier snapshot has std 0.0016 against 0.054 at the sharp tier, so any R² near 0.6 on it is an artefact, not skill ([NBA timing and injury experiments](../evaluation/exp-nba-timing-and-injuries.md#nba-injury-features-exp-6-6b)).

## Why bet365 and not Pinnacle

The production target is the devigged bet365 delta. Pinnacle's lines move too little to predict: CV R² -0.017 on NBA, against 0.036 for bet365 on NBA and 0.058 on EPL. The choice is recorded in [ADR-0002](../../docs/adr/0002-bet365-clv-target.md) and the runs behind it are on [Target bookmaker and data volume experiments](../evaluation/exp-target-and-data-volume.md). The target bookmaker's snapshot price is also an input to the model's strongest feature, `retail_sharp_diff` ([sharp benchmark](sharp-benchmark.md)).

## Related

- [Training config](training-config.md), [training pipeline](training-pipeline.md) and [cross-validation](cross-validation.md): how the target is trained and evaluated
- [CLV model findings](../evaluation/clv-model-findings.md): what the model can predict
- [Look-ahead bias](look-ahead-bias.md): the snapshot must precede the close
