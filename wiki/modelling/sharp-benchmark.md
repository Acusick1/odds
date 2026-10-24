---
type: explanation
title: Sharp benchmark and retail-sharp divergence
description: What the sharp reference is and why, its priority semantics, the divergence feature and screener, and how the three code copies differ.
sources:
  - odds_analytics.feature_extraction.TabularFeatureExtractor
  - odds_analytics.feature_extraction.calculate_sharp_retail_metrics
  - odds_analytics.feature_extraction.select_sharp_odds
  - odds_analytics.feature_extraction.SequenceFeatureExtractor
  - odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS
  - odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.training.config.FeatureConfig
  - odds_mcp.server.find_retail_edges
  - packages/odds-mcp/agents/common.md
  - odds_core.models.Odds
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
status: stable
---

# Sharp benchmark and retail-sharp divergence

The sharp benchmark is the price treated as closest to fair: Pinnacle, with Betfair Exchange as the fallback after the Pinnacle API shutdown in January 2026 (71 EPL events lost, PR #253; Betfair `sharp_prob` correlates 0.9994 with Pinnacle, about a 1.1 pp level shift). The decision and its evidence are [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md) and [EPL sharp-reference experiments](../evaluation/exp-epl-sharp-reference.md). FDUK's Pinnacle odds are also unreliable after July 2025 ([data sources](../collection/data-sources.md)).

`polymarket` is a further reference for EPL matches from 2025-08: its midpoints sit within 0.35-0.5 pp of Betfair Exchange from three days out ([#518](https://github.com/Acusick1/odds/issues/518)) and are stored as snapshots holding no other bookmaker ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md#snapshot-shape)). Whether it adds information beyond Betfair is untested.

The [retail-EV target](clv-target.md#the-retail-ev-target) does not use this priority list for its close: ADR-0012 fixes the close and the baseline to Betfair Exchange, and its retail rows come from the UK roster `odds_core.bookmakers.UK_RETAIL_BOOKMAKERS` rather than from every non-sharp book ([target functions](clv-target-functions.md#retail-ev-rows)).

The sharp reference anchors the delta model's strongest feature, the retail-sharp divergence, and the agent's retail screener ([MCP market tools](../agent/mcp-market-tools.md#retail-edges)). Because the agent sees the same quantity directly through `odds_mcp.server.find_retail_edges`, the model adds little for it ([betting agent](../agent/betting-agent.md)).

## Priority semantics

`sharp_bookmakers` is an ordered list and the first bookmaker that has odds for the requested outcome wins (`odds_analytics.feature_extraction.select_sharp_odds`):

- Features: `odds_analytics.feature_extraction.TabularFeatureExtractor` and `odds_analytics.feature_extraction.calculate_sharp_retail_metrics` both pick the sharp bookmaker with `odds_analytics.feature_extraction.select_sharp_odds`, which filters to the requested outcome itself. `odds_analytics.feature_extraction.SequenceFeatureExtractor` called without an outcome leaves its sharp features unset. None of the three reads the target fields.
- Sampling: `odds_analytics.feature_groups.TierSampler` keeps only snapshots containing any (not all) of `required_bookmakers`, and `odds_analytics.feature_groups.prepare_training_data` drops events lacking every sharp bookmaker at close (a check that does not depend on the sampler), after `odds_analytics.feature_groups.collect_event_data` has already loaded the event's group data. That function leaves out snapshots whose only bookmaker is a prediction market (`odds_core.bookmakers.is_reference_only_snapshot`), so no sampler can pick a `polymarket` snapshot as a decision row; the retail-EV builder reads its own snapshots and takes `polymarket` ones only for its `polymarket_at_t` features, never as a decision row, baseline or close ([retail-EV features](clv-target-functions.md#retail-ev-features)). This filter always applies to a raw target, and to a devigged target only when `set(sharp_bookmakers) != {target_bookmaker}` ([feature pipeline](feature-pipeline.md)).
- MCP: `odds_lambda.storage.readers.OddsReader.get_sharp_prices` scans snapshots newest first within a 2 h lookback and, per outcome, takes the first sharp book in priority order that has a price ([odds reader](../storage/odds-reader.md)); `odds_mcp.server.find_retail_edges` defaults to `[pinnacle, betfair_exchange]` and never lists `odds_core.bookmakers.NON_BETTABLE_BOOKMAKERS` (Pinnacle, `polymarket`) as retail.

## Why a sharp reference

- The model's signal needs it: combined EPL data with bet365 as its own sharp reference gave no signal (CV R² -0.005; PR #217) ([EPL sharp-reference experiments](../evaluation/exp-epl-sharp-reference.md)).
- Pinnacle first posts NBA lines 12-24 h before the game `(inferred)`; the measurement behind it is an Odds API snapshot analysis whose date and query were not recorded, so re-check against `odds_snapshots`. Opening-tier snapshots (72+ h) hold only soft-book lines ([NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md#bookmaker-line-release-timing), [fetch tier](../storage/fetch-tier.md)).
- The agent prompt treats sharp prices as strong but imperfect, and the model as supplementary sanity checks (`packages/odds-mcp/agents/common.md`); why the prompt says so is `(inferred)`.

## Retail-sharp divergence

**Definition.** `divergence = retail_implied_prob - sharp_implied_prob` for one outcome. Each implied probability is below 1 because `price_decimal` is constrained above 1.0. Negative means the retail book prices that outcome longer than the sharp reference. Implied probabilities come from `odds_core.models.Odds.implied_probability` (1 over the stored decimal price) and are not devigged, so a retail book with a larger margin shows a more positive divergence on every outcome; compare with `market_hold` in the screener output to separate margin from divergence ([devigging](../foundations/devigging.md)). The target is devigged ([CLV target](clv-target.md)).

It is the strongest raw correlate of CLV from the first experiment onward (r = 0.12 in Exp 1, `experiments/results/exp1_feature_correlations/FINDINGS.md`; r = 0.356 at the sharp tier in Exp 6b, `experiments/results/exp6b_injury_closing_tier/FINDINGS.md`) and the core model feature `retail_sharp_diff` ([early NBA signal](../evaluation/exp-early-nba-signal-and-lstm.md), [NBA timing and injuries](../evaluation/exp-nba-timing-and-injuries.md)). The production YAML also lists bet365, the target bookmaker, in `retail_bookmakers`, so the target bookmaker's snapshot price is an input to `retail_sharp_diff` (PR #362, [training config](training-config.md)).

## Code copies

| | Feature, tabular path | Feature, helper path | MCP screener |
|---|---|---|---|
| Where | `odds_analytics.feature_extraction.TabularFeatureExtractor` | `odds_analytics.feature_extraction.calculate_sharp_retail_metrics`, called by `odds_analytics.feature_extraction.SequenceFeatureExtractor` | `odds_mcp.server.find_retail_edges`, sharp from `odds_lambda.storage.readers.OddsReader.get_sharp_prices` |
| Retail side | Mean of per-book implied probabilities for the outcome, over configured `retail_bookmakers` | Mean of per-book implied probabilities (`odds_analytics.feature_extraction.calculate_avg_probability`) | One row per book, every non-sharp book in the window |
| Sharp side | First bookmaker in the list with odds for the requested outcome (`odds_analytics.feature_extraction.select_sharp_odds`) | Same | First bookmaker per outcome, so outcomes can resolve to different books, and sharp and retail prices can come from different snapshots |
| Defaults | `["pinnacle"]`; retail `fanduel`, `draftkings`, `betmgm` (`odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS`) | Same | `["pinnacle", "betfair_exchange"]`; no retail list |
| Time | Snapshot at decision time | Same | Newest price per book within a lookback window (default 2 h) |

Both feature paths average per-book implied probabilities, so a feature of the same name agrees between them.

The sharp roster is defined in several places in the code. `odds_analytics.training.config.FeatureConfig.sharp_bookmakers` and `odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS` default to `["pinnacle"]`, the strategies default to `pinnacle`, and the default sharp list in `odds_mcp.server` is `["pinnacle", "betfair_exchange"]` with a comment saying to keep it in sync. The two lists differ on purpose, a maintainer decision recorded in #421 and [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md). `odds_mcp.server.save_match_brief` stamps the hybrid list literal ([match briefs](../agent/match-briefs.md)). A training YAML that omits `betfair_exchange` silently loses the fallback, and a single training set can mix Pinnacle-based and Betfair-based `sharp_prob` values that the model cannot tell apart except through the level shift. A `retail_ev` config rejects any `sharp_bookmakers` other than the default, because its baseline and close are fixed to Betfair Exchange ([training config](training-config.md)). Hybrid production configs set `sharp_bookmakers` explicitly ([training config](training-config.md)). The MCP-side point-bucket and bookmaker-mixing behaviour is on [MCP market tools](../agent/mcp-market-tools.md#retail-edges).

## Related

- [Feature groups](feature-groups.md): the tabular group that consumes it
- [CLV model findings](../evaluation/clv-model-findings.md)
- [Decision time and look-ahead bias](look-ahead-bias.md): the feature uses decision-time snapshots
