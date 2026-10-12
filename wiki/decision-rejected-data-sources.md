---
type: decision
title: Rejected agent data sources and lineup sourcing
summary: "Why each evaluated source was rejected, and why lineups come from web search rather than an API."
sources:
  - docs/AGENT_DATA_SOURCES.md#evaluated-and-rejected
  - docs/AGENT_DATA_SOURCES.md#lineup-data-2026-04-13
  - packages/odds-lambda/odds_lambda/espn_fixture_fetcher.py::EspnFixtureFetcher
  - scripts/ingest_espn_lineups.py
verified_at: 7887039
---

# Rejected agent data sources and lineup sourcing

**Status:** in effect. No ADR exists; the reasons below are the ones recorded in [AGENT_DATA_SOURCES.md](../docs/AGENT_DATA_SOURCES.md#evaluated-and-rejected), with dates where the doc gives them. Context: [Agent data sources](agent-data-sources.md).

## Rejected sources

| Source | Reason recorded |
|---|---|
| Betfair Exchange volume | Needs a £499 Live App Key and active betting; the free tier gives prices only |
| Twitter/X API | $100-5,000 per month under 2023 pricing, economically unviable |
| Action Network, BetQL | US sportsbook data; the "public" they track does not move EPL lines |
| Accumulator popularity | No source publishes most-backed acca legs with volume; tipster output is not crowd data |
| Tipster consensus (Predictz, Forebet) | Model-driven, lags sharp money, no signal for anticipating line movement |
| FPL API ownership and transfers | As XGBoost features (2026-03-20): R² -0.017 under sliding-380 and halved R² under expanding; transfer spikes lag the news sources the agent searches directly. See [EPL feature group ablation](exp-epl-feature-group-ablation.md) |
| API-Football | Free tier blocks current-season fixture lookup (2022-2024 only) and the lineups endpoint needs the gated fixture ids; Pro plan about $20 per month |
| football-data.org | A different site from football-data.co.uk; lineups cost €29 per month and timing is undocumented |
| ESPN API (for lineups) | Lineups appear around KO-60 but non-monotonically (present at KO-60, gone by KO-45 in a 2026-04-14 Champions League test; nothing at KO-59 in a 2026-04-13 EPL test); 403 from cloud and datacenter IPs |
| Polymarket | EPL volume thin ($10-100K per match) and AMM-driven, so the book reflects maker parameters, not public sentiment; not accessible from the UK. See [Polymarket pipeline](polymarket-pipeline.md) |

## Lineup sourcing (2026-04-13)

Every structured lineup API investigated is paywalled, blocks current-season data on its free tier, or has unconfirmed pre-kickoff timing. The agent's web search and Playwright are the lineup source: BBC Sport publishes confirmed lineups within minutes of the official team sheet (KO-60), and the agent reads them directly ([Matchday wake-up](matchday-wakeup.md)).

## Hazards

- ESPN is still used in the pipeline for fixtures (`EspnFixtureFetcher`) and a manual lineup ingestion script exists (`scripts/ingest_espn_lineups.py`); the rejection applies to lineup timing for the agent, not to those feeds ([EPL context data](epl-context-data.md)).
- Prices and tier names are as of the doc's dates and are external facts the repo cannot verify.

## Related

- [Pivot to information edge](decision-pivot-to-information-edge.md)
- [Data sources](data-sources.md)
