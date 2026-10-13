---
status: proposed
---

# Use Pinnacle with Betfair Exchange fallback as the sharp reference

`sharp_bookmakers` is `[pinnacle, betfair_exchange]` for EPL, and the first bookmaker with odds in the market wins. The runs behind it are on [EPL sharp-reference experiments](../../wiki/evaluation/exp-epl-sharp-reference.md).

## Context

Pinnacle's public API stopped covering EPL in January 2026, so 71 of about 1.8K events had no Pinnacle closing odds. FDUK's Pinnacle odds are unreliable after July 2025 (#217). Betfair Exchange closing odds exist in FDUK from 2024-25 (#253) and as live snapshots from the Betfair job. With bet365 as its own sharp reference, the combined EPL data gave no signal (CV R² -0.005 ± 0.008; #217). A sampler bug let OddsPortal snapshots without Pinnacle beat FDUK snapshots with it (#231, fixed 2026-03-15).

## Decision

Decided 2026-03-19 (#253 merged). `sharp_bookmakers: [pinnacle, betfair_exchange]`, first bookmaker with odds in the market wins; `required_bookmakers` is any-of. EPL production uses `data_source: null` and `closing_source_priority: [football_data_uk]`. The agent's MCP screener and brief stamp use the same list; the model code default stays Pinnacle-only, and the two lists differ on purpose (maintainer decision recorded in #421).

## Considered options

- **Pinnacle only.** Loses the 71 events without Pinnacle closing odds.
- **bet365 as its own sharp reference.** No signal on the combined data (CV R² -0.005 ± 0.008; #217).

## Consequences

- On 589 overlapping events Betfair `sharp_prob` correlates 0.9994 with Pinnacle with about a 1.1 pp level shift; XGBoost absorbing the shift through split points is `(inferred)`.
- The hybrid recovers all 71 events (1,800 vs 1,729); folds 0-19 match Pinnacle-only, the two new folds score R² +7.4% and +2.7%; fixed-parameter expanding:50 R² 0.060 ± 0.080.
- Semantics changed from all-of to any-of in `TierSampler.required_bookmakers`, the `prepare_training_data` event filter and `TabularFeatureExtractor`.
- A training set mixes Pinnacle-based and Betfair-based `sharp_prob`; the model cannot tell them apart except through the level shift.
- A YAML that omits `betfair_exchange` silently loses the fallback.
- The result tables are as recorded in the PRs; no results artefact holds them.

## Evidence

PRs #217, #231, #234, #253; prior wiki text `wiki/modelling/decision-hybrid-sharp-reference.md` at f2a0862, which labels the XGBoost split-point explanation `(inferred)`; `experiments/configs/xgboost_epl_hybrid_sharp.yaml`; `experiments/scripts/hybrid_sharp_validation.py`; #421. Wiki: [Sharp benchmark](../../wiki/modelling/sharp-benchmark.md), [Betfair Exchange](../../wiki/collection/betfair-exchange.md), [Data sources](../../wiki/collection/data-sources.md).
