---
status: accepted
---

# Use Pinnacle with Betfair Exchange fallback as the sharp reference

`sharp_bookmakers` is `[pinnacle, betfair_exchange]` for EPL, and the first bookmaker with odds in the market wins. The runs behind it are on [EPL sharp-reference experiments](../../wiki/evaluation/exp-epl-sharp-reference.md).

## Context

FDUK sources its Pinnacle odds from Pinnacle's public API. PR #217 recorded that the API became unreliable after July 2025, leaving 81 events in the 2025-26 season without Pinnacle closing odds. PR #253 recorded a Pinnacle API shutdown in January 2026 that excluded 71 EPL events (January to March 2026) from training. Betfair Exchange closing odds exist in FDUK from the 2024-25 season (`BFE*`/`BFEC*` columns, PR #253), and the Betfair job writes live snapshots (`odds_lambda.jobs.fetch_betfair_exchange`). On the combined OddsPortal and FDUK EPL data, bet365 as its own sharp reference gave CV R² −0.005 ± 0.008 (PR #217). A sampler bug let OddsPortal snapshots without Pinnacle beat FDUK snapshots with it (#231, fixed 2026-03-15).

## Decision

Decided 2026-03-19 (PR #253 merged). `sharp_bookmakers: [pinnacle, betfair_exchange]`, first bookmaker with odds in the market wins; `required_bookmakers` is any-of. EPL production uses `data_source: null` and `closing_source_priority: [football_data_uk]`. The agent's MCP screener and brief stamp use the same list. The model code default stays Pinnacle-only, and the two lists differ on purpose (maintainer decision recorded in #421).

## Considered options

- **Pinnacle only.** Loses the 71 events without Pinnacle closing odds.
- **bet365 as its own sharp reference.** CV R² −0.005 ± 0.008 on the combined data (PR #217).
- **Betfair Exchange only.** PR #253 added `experiments/configs/xgboost_epl_betfair_sharp.yaml`; no result for it is recorded.

## Consequences

- On 589 overlapping events Betfair `sharp_prob` correlates 0.9994 with Pinnacle, with a level shift of about 1.1 pp. PR #253's `docs/MODELING.md` entry states that XGBoost handles the shift through split-point adjustment; no run measures that.
- The hybrid recovers all 71 events (1,800 against 1,729). Folds 0-19 match Pinnacle-only, and the two new folds score R² +7.4% and +2.7%. A fixed-parameter expanding:50 run with 32 features scored R² 0.060 ± 0.080.
- Semantics changed from all-of to any-of in `TierSampler.required_bookmakers`, the `prepare_training_data` event filter and `TabularFeatureExtractor`.
- A training set mixes Pinnacle-based and Betfair-based `sharp_prob`. The model cannot tell them apart except through the level shift.
- `TabularFeatureExtractor` falls back only when the first sharp bookmaker has no rows in the market. When that bookmaker lacks the selected outcome, the sharp features stay NaN (#477).
- A YAML that omits `betfair_exchange` silently loses the fallback.
- PR #253's `docs/MODELING.md` entry points to `experiments/results/hybrid_sharp_validation/`, which was never committed. The hybrid figures exist only in PR #253's text.
- The bet365-as-sharp control ran on 2026-03-10, before the #231 fix. `docs/MODELING.md` marks the 2026-03-10 Pinnacle run invalid, does not mark the control, and records no re-run of the control after the fix.
- The comment above `odds_mcp.server._DEFAULT_SHARP_BOOKMAKERS` still says to keep it in sync with the model default; #421 replaces it with one saying the two differ on purpose.

## Evidence

- PRs #217, #231, #253; issues #421, #477.
- `docs/MODELING.md` commits 7346c2c (PR #217) and 1d1381f (PR #253); text at `git show 7887039:docs/MODELING.md`.
- `experiments/configs/xgboost_epl_hybrid_sharp.yaml`, `experiments/configs/xgboost_epl_betfair_sharp.yaml`, `experiments/scripts/hybrid_sharp_validation.py`, `experiments/scripts/sharp_feature_correlation.py`.
- Wiki: [Sharp benchmark](../../wiki/modelling/sharp-benchmark.md), [Betfair Exchange](../../wiki/collection/betfair-exchange.md), [Data sources](../../wiki/collection/data-sources.md).
