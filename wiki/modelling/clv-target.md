---
type: explanation
title: CLV and the training target
description: What CLV means here, why the target is a close-minus-snapshot delta of one devigged bookmaker, and how sampling changes the target.
sources:
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
status: stable
---

# CLV and the training target

CLV (closing line value) is how far a bookmaker's line moves between a decision snapshot and the close. The CLV model regresses that movement: `target = devigged_fair_close - devigged_fair_at_snapshot` for one bookmaker (`target_bookmaker`). A positive value means the price at the snapshot was too low and the line moved up (`odds_analytics.sequence_loader.calculate_devigged_bookmaker_target` returns closing probability minus snapshot probability for the configured `outcome`, a required argument). Why a delta rather than the closing price is `(inferred)`: the decision needs the size of the move, and a delta is stationary and roughly mean-zero.

The functions, dispatch and config fields that compute it are on [Target functions](clv-target-functions.md).

## Sampling changes the target

The delta is measured from the sampled snapshot to the close ([decision time](look-ahead-bias.md)), so the same event can yield different targets under different samplers. A target measured from a closing-tier snapshot has std 0.0016 against 0.054 at the sharp tier, so any R² near 0.6 on it is an artefact, not skill ([NBA timing and injury experiments](../evaluation/exp-nba-timing-and-injuries.md#nba-injury-features-exp-6-6b)).

## Why the suspended model uses bet365

The suspended model's target is the devigged bet365 delta, recorded in [ADR-0002](../../docs/adr/0002-bet365-clv-target.md) on NBA runs the map #494 voids. On EPL, scored on the same events, folds and features, a Pinnacle-target model has no out-of-sample skill (R² about 0 against the training-fold mean), against about 0.023 for bet365. The difference is within bootstrap noise ([Pinnacle against bet365](../evaluation/exp-epl-clv-pinnacle-retest.md)). Pinnacle's EPL lines move almost as much as bet365's, with a target variance 91% of bet365's, so the NBA reason that Pinnacle moves too little does not hold on EPL. [ADR-0011](../../docs/adr/0011-bet365-clv-target-on-like-for-like-evidence.md) records this evidence. The earlier NBA runs are on [Target bookmaker and data volume experiments](../evaluation/exp-target-and-data-volume.md). The target bookmaker's snapshot price is also an input to the model's strongest feature, `retail_sharp_diff` ([sharp benchmark](sharp-benchmark.md)).

## The next target

[ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) supersedes both ADRs. The next EPL model predicts a retail bet's expected value against the Betfair Exchange close, with a row for each outcome, retail book and decision time. The training pipeline does not compute that target yet.

## Related

- [Training config](training-config.md), [training pipeline](training-pipeline.md) and [cross-validation](cross-validation.md): how the target is trained and evaluated
- [CLV model findings](../evaluation/clv-model-findings.md): what the model can predict
- [Look-ahead bias](look-ahead-bias.md): the snapshot must precede the close
