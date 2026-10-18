---
type: explanation
title: Data sources
description: Every odds feed and what it supplies, bookmaker roles, the Pinnacle coverage gap, and where each feed runs.
sources:
  - odds_core.config.DataCollectionConfig
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.storage.writers.OddsWriter.LIVE_EVENT_ID_PREFIX
  - scripts/ingest_football_data_uk.py
status: stable
---

# Data sources

Four feeds write `raw_data` snapshots in The Odds API shape ([odds tables](../storage/odds-tables.md)). Each is tagged by `odds_core.models.Event.id` prefix or `odds_core.models.OddsSnapshot.api_request_id` ([events](../storage/events.md)). The keys and defaults each feed uses are in [sport-and-market-keys](sport-and-market-keys.md).

| Source | Supplies | Collected by | Where it runs | Event ids |
|---|---|---|---|---|
| The Odds API | US books incl. Pinnacle, dense snapshots | `fetch-odds` job ([fetch-odds-job](fetch-odds-job.md), [client](odds-api-client.md)) | mini-PC, EPL only | API hex ids |
| OddsPortal | UK books, upcoming and closing odds | `fetch-oddsportal` job ([fetch-oddsportal-job](fetch-oddsportal-job.md)) | mini-PC | `op_live_*` (`odds_lambda.storage.writers.OddsWriter.LIVE_EVENT_ID_PREFIX`) |
| Betfair Exchange | Best back price per outcome | `fetch-betfair-exchange` job ([betfair-exchange](betfair-exchange.md)) | mini-PC | joins existing events |
| football-data.co.uk (FDUK) | Historical EPL opening and closing odds from Pinnacle, the exchange and UK retail books, with snapshot times stamped from FDUK's documented collection times | manual script ([historical-ingest-scripts](historical-ingest-scripts.md)) | manual | `fduk_*` / `op_<season>_*` |

ESPN, MLB and Polymarket are non-odds feeds: [epl-context-data](epl-context-data.md), [mlb-probable-pitchers](mlb-probable-pitchers.md), [polymarket](polymarket.md). Polymarket was built for NBA and is deprioritised ([ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md), [market research](polymarket-market-research.md)). The ESPN and MLB collectors run as a daily cron. Every scheduled collector runs on the mini-PC ([job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs)). The agent-facing view of the same feeds, including sources it rejected, is [agent-data-sources](../agent/agent-data-sources.md).

## Bookmaker roles

(Inferred.) Pinnacle and Circa Sports are sharp, low-margin books (Circa is Vegas-based). DraftKings, FanDuel, BetMGM and Caesars (`williamhill_us`) are major US retail books, BetRivers is regional retail and Bovada is offshore. UK books come from OddsPortal: bet365 is the suspended model's target ([ADR-0002](../../docs/adr/0002-bet365-clv-target.md)); Betway, Betfred and bwin are retail. The default Odds API bookmaker list is `odds_core.config.DataCollectionConfig.bookmakers`.

## Pinnacle coverage gap

Pinnacle's Odds API coverage of EPL stopped in January 2026 (71 events lost), and FDUK's Pinnacle odds are unreliable after July 2025. Betfair Exchange is the sharp fallback ([ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md); the experiments are in [exp-epl-sharp-reference](../evaluation/exp-epl-sharp-reference.md)). Event counts and season coverage are data facts the repo cannot confirm; query the database.

## Deployed state

The deployed schedule of each collector is not in the repo. `odds scheduler list-jobs --backend aws` lists what runs on AWS, and [job-deployment-matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs) gives the design placement of every job.

## Related

- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Odds API quota accounting](odds-api-quota.md)
- [CLV model findings](../evaluation/clv-model-findings.md), for which source each experiment used
