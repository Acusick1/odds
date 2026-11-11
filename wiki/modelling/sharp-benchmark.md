---
type: explanation
title: Sharp references and retail-sharp divergence
description: Which sharp price each use reads (the scoring close, the retail-EV model, the mechanical rules, the tabular features and the agent tools), the priority semantics of the Pinnacle-first lists, the divergence feature and screener, and how the code copies differ.
sources:
  - odds_analytics.retail_ev.build_event_rows
  - odds_analytics.candidate_model.state_inputs
  - odds_analytics.candidate_model.training_rows
  - odds_analytics.feature_extraction.TabularFeatureExtractor
  - odds_analytics.feature_extraction.select_sharp_quotes
  - odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS
  - odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.retail_ev.EXCHANGE_SOURCES
  - odds_lambda.candidate_rules.inputs.RETAIL_SOURCES
  - odds_mcp.server.find_retail_edges
  - odds_mcp.prices.resolve_sharp_prices
  - odds_mcp.prices.MarketWindow
  - packages/odds-mcp/agents/common.md
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
  - odds_core.bookmakers.SHARP_BOOKMAKERS
mentions:
  - odds_core.close.event_close
status: stable
---

# Sharp references and retail-sharp divergence

A sharp reference is the price treated as closest to fair. The project has no single one: each use reads its own, and only the scoring close decides a candidate's verdict ([goal and strategy](../foundations/goal-and-strategy.md#how-a-candidate-is-judged)).

## Which reference is used for what

| Use | Sharp reference | Code |
|---|---|---|
| Scoring: a candidate's EV | The Betfair Exchange close, from the direct Betfair feed; a football league falls back to FDUK's Betfair Exchange close and NBA `h2h` to Polymarket, and MLB has none. Defined on [CLV and the training target](clv-target.md#the-retail-ev-target) | `odds_core.close.event_close` |
| The `retail_ev` training target's close | The same Betfair close and FDUK fallback, without Polymarket: the builder leaves Polymarket out of its close inputs, so an NBA match closed only by Polymarket gives no rows | `odds_analytics.retail_ev.build_event_rows` |
| Retail-EV training pipeline: baseline and sharp features | The Betfair Exchange price at t, from the direct Betfair feed or an FDUK opening snapshot, never from the Odds API's Betfair rows; the optional `polymarket_at_t` group adds Polymarket ([retail-EV features](clv-target-functions.md#retail-ev-features)) | `odds_analytics.retail_ev.EXCHANGE_SOURCES` |
| The `odds_lambda.candidate_rules.retail_ev_model.retail_ev_model` candidate | The Betfair Exchange price in its own co-timed retail snapshot, from the Odds API or OddsPortal, as inputs; trained only on matches with a live Betfair close | `odds_analytics.candidate_model.state_inputs`, `odds_analytics.candidate_model.training_rows` |
| Mechanical rules | `odds_lambda.candidate_rules.gap.exchange_gap`: the Betfair Exchange price in the rule's own retail snapshot, from the Odds API or OddsPortal; `odds_lambda.candidate_rules.gap.polymarket_gap`: Polymarket ([gap rules](../evaluation/candidate-rules.md#the-gap-rules)) | `odds_lambda.candidate_rules.inputs.RETAIL_SOURCES` |
| Tabular features of a `raw` config | `sharp_bookmakers`, Pinnacle alone by default; a config may list `[pinnacle, betfair_exchange]` | `odds_analytics.training.config.FeatureConfig.sharp_bookmakers` |
| Agent tools: the retail screener, match-brief stamps and paper-trade sharp fields | Pinnacle, then Betfair Exchange | `odds_core.bookmakers.SHARP_BOOKMAKERS` |

Pinnacle appears only in the last two rows. Neither decides a verdict: the agent tools give screening and placement context, and the tabular path is the older modelling path behind the `raw` target. A `retail_ev` config ignores `sharp_bookmakers` and rejects any value other than its default, because its baseline and close are fixed to Betfair Exchange ([training config](training-config.md)). [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) allows other sharp sources, Pinnacle included, as retail-EV features; Polymarket is the only one the builder reads. Its retail rows come from the UK roster `odds_core.bookmakers.UK_RETAIL_BOOKMAKERS` rather than from every non-sharp book ([target functions](clv-target-functions.md#retail-ev-rows)). Pinnacle's coverage by source is on [data sources](../collection/data-sources.md#pinnacle-coverage).

`polymarket` prices EPL matches from 2025-08: its midpoints sit within 0.35-0.5 pp of Betfair Exchange from three days out ([#518](https://github.com/Acusick1/odds/issues/518)), and it is stored as snapshots holding no other bookmaker ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md#snapshot-shape)).

## The Pinnacle-first lists

The tabular features and the agent tools both put Pinnacle first. The tabular path's hybrid list is [ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md): FDUK's Pinnacle prices stop in January 2026 (71 EPL events were lost from training when PR #253 added the fallback), and Betfair `sharp_prob` correlates 0.9994 with Pinnacle's, about a 1.1 pp level shift.

The model and agent lists differ on purpose, a maintainer decision recorded in #421 and ADR-0003, and the comment on `odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS` says so. `odds_analytics.training.config.FeatureConfig.sharp_bookmakers` and `odds_analytics.feature_extraction.DEFAULT_SHARP_BOOKMAKERS` default to `["pinnacle"]`. The agent's list is `odds_core.bookmakers.SHARP_BOOKMAKERS`, `("pinnacle", "betfair_exchange")`, which the MCP server imports. `odds_mcp.server.find_retail_edges` defaults to it, and `odds_mcp.server.save_match_brief` (for an event given by integer id) and `odds_mcp.server.paper_bet` always use it ([match briefs](../agent/match-briefs.md), [paper trading](../agent/paper-trading.md#sharp-fields)). A training YAML that omits `betfair_exchange` silently loses the fallback, and a single training set can mix Pinnacle-based and Betfair-based `sharp_prob` values that the model cannot tell apart except through the level shift. A `raw` config sets the hybrid list explicitly.

### Priority semantics

`sharp_bookmakers` is an ordered list, but the two paths apply its order differently:

- Features: `odds_analytics.feature_extraction.TabularFeatureExtractor` takes, from one snapshot's quotes, the first bookmaker in the list that prices the requested outcome (`odds_analytics.feature_extraction.select_sharp_quotes`, which filters to that outcome itself). It does not read the target fields, and the extractor's `odds_analytics.feature_extraction.FeatureEvent` (integer `id`) carries no prices.
- Sampling: `odds_analytics.feature_groups.TierSampler` keeps only snapshots quoting any (not all) of `required_bookmakers`, and `odds_analytics.feature_groups.prepare_training_data` drops events whose closes all lack every sharp bookmaker and samples only sources whose close has one, after `odds_analytics.feature_groups.collect_event_data` has already loaded the event's group data and picked each source's raw-target close, a snapshot rather than the exchange price of `odds_core.close.event_close` (`odds_analytics.feature_groups.raw_closes`). That function leaves out Polymarket's snapshots (source `polymarket`), so no sampler can pick a `polymarket` snapshot as a decision row; the retail-EV builder reads its own snapshots and takes `polymarket` ones only for its `polymarket_at_t` features, never as a decision row, baseline or close ([retail-EV features](clv-target-functions.md#retail-ev-features)). The sampler filter and the closing check both apply to every `raw` target ([feature pipeline](feature-pipeline.md)).
- MCP: `odds_mcp.prices.MarketWindow.sharp` walks the loaded snapshots newest first within the lookback (2 h for the screener). Each outcome and line takes its price from the newest snapshot that has a sharp book pricing it, and only within that snapshot from the first book in priority order. Recency comes first, so a newer Betfair-only snapshot beats an older one holding Pinnacle ([MCP market tools](../agent/mcp-market-tools.md#retail-edges)). `odds_mcp.server.find_retail_edges` compares each retail row with the sharp price at its own line and never lists `odds_core.bookmakers.NON_BETTABLE_BOOKMAKERS` (Pinnacle and Polymarket, `odds_core.bookmakers.POLYMARKET_BOOKMAKER`) as retail.

### Why the agent tools read a sharp price

- Pinnacle first posts NBA lines 12-24 h before the game `(inferred)`; the measurement behind it is an Odds API snapshot analysis whose date and query were not recorded, so re-check against `odds_snapshots`. Snapshots of the opening tier (72+ h before kick-off) hold only soft-book lines ([fetch tier](../storage/fetch-tier.md)).
- The agent prompt treats sharp prices as strong but imperfect (`packages/odds-mcp/agents/common.md`); why the prompt says so is `(inferred)`.

## Retail-sharp divergence

**Definition.** `divergence = retail_implied_prob - sharp_implied_prob` for one outcome. Each implied probability is below 1 because `price_decimal` is constrained above 1.0. Negative means the retail book prices that outcome longer than the sharp reference. Implied probabilities are 1 over the stored decimal price and are not devigged, so a retail book with a larger margin shows a more positive divergence on every outcome; compare with `market_hold` in the screener output to separate margin from divergence ([devigging](../foundations/devigging.md)). The `retail_ev` target is devigged ([CLV target](clv-target.md#the-retail-ev-target)).

It is the core tabular feature `retail_sharp_diff` ([feature groups](feature-groups.md)) and the quantity the agent's retail screener ranks by ([MCP market tools](../agent/mcp-market-tools.md#retail-edges)). The agent reads it, vig-inclusive on both sides, through `odds_mcp.server.find_retail_edges` (an event by integer id; [betting agent](../agent/betting-agent.md)).

## Code copies

| | Feature (tabular group) | MCP screener |
|---|---|---|
| Where | `odds_analytics.feature_extraction.TabularFeatureExtractor` | `odds_mcp.server.find_retail_edges`, sharp from `odds_mcp.prices.MarketWindow.sharp` |
| Retail side | Mean of per-book implied probabilities for the outcome, over configured `retail_bookmakers` | One row per book, every non-sharp book in the window |
| Sharp side | First bookmaker in the list with a price for the requested outcome (`odds_analytics.feature_extraction.select_sharp_quotes`) | Per outcome and line, the newest snapshot with a sharp price, then the first book in priority order within it, so outcomes can resolve to different books, and sharp and retail prices can come from different snapshots |
| Defaults | `["pinnacle"]`; retail `fanduel`, `draftkings`, `betmgm_us` (`odds_analytics.feature_extraction.DEFAULT_RETAIL_BOOKMAKERS`; canonical keys, so US BetMGM is `betmgm_us`) | `["pinnacle", "betfair_exchange"]`; no retail list |
| Time | Snapshot at decision time | Newest price per book within a lookback window (default 2 h) |

The MCP-side point-bucket and bookmaker-mixing behaviour is on [MCP market tools](../agent/mcp-market-tools.md#retail-edges).

## Related

- [Feature groups](feature-groups.md): the tabular group that consumes it
- [Decision time and look-ahead bias](look-ahead-bias.md): the feature uses decision-time snapshots
