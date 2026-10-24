---
title: Storage
description: The events, odds_snapshots and odds tables, the OddsWriter/OddsReader storage layer, odds validation and coverage checks, and fetch tiers.
---

# Storage

Everything the pipeline persists about games and prices lives behind `packages/odds-lambda/odds_lambda/storage/` and the models in `odds_core.models`. Start with [how odds are stored](odds-storage-model.md) for how a fetch becomes rows, and [fetch-tier](fetch-tier.md) for the tier vocabulary used throughout.

## Table map

Every table is a SQLModel class in `packages/odds-core/odds_core/`; migrations are in [migrations](../foundations/howto-database-migration.md).

| Table | Model | Owning page |
|---|---|---|
| `events` | `odds_core.models.Event` | [Events](events.md) |
| `odds_snapshots` | `odds_core.models.OddsSnapshot` | [Odds tables](odds-tables.md) |
| `odds` | `odds_core.models.Odds` | [Odds tables](odds-tables.md) |
| `data_quality_logs` | `odds_core.models.DataQualityLog` | [Odds validation](odds-validator.md) |
| `fetch_logs` | `odds_core.models.FetchLog` | [Odds API ingestion](../collection/odds-ingestion-service.md) |
| `alert_history` | `odds_core.models.AlertHistory` | [Alerts and health](../operations/alerts.md) |
| `match_briefs` | `odds_core.match_brief_models.MatchBrief` | [Match briefs](../agent/match-briefs.md) |
| `paper_trades` | `odds_core.paper_trade_models.PaperTrade` | [Paper trading](../agent/paper-trading.md) |
| `agent_wakeups` | `odds_core.agent_wakeup_models.AgentWakeup` | [Agent runtime and the agent-run job](../agent/agent-run-job.md) |
| `espn_fixtures` | `odds_core.epl_data_models.EspnFixture` | [EPL context data](../collection/epl-context-data.md) |
| `espn_lineups` | `odds_core.epl_data_models.EspnLineup` | [EPL context data](../collection/epl-context-data.md) |
| `fpl_availability` | `odds_core.epl_data_models.FplAvailability` | [EPL context data](../collection/epl-context-data.md) |
| `mlb_probable_pitchers` | `odds_core.mlb_data_models.MlbProbablePitchers` | [MLB probable pitchers](../collection/mlb-probable-pitchers.md) |
| `nba_injury_reports` | `odds_core.injury_models.InjuryReport` | [NBA injuries and stats](../collection/nba-injuries-and-stats.md) |
| `nba_player_season_stats` | `odds_core.player_stats_models.NbaPlayerSeasonStats` | [NBA injuries and stats](../collection/nba-injuries-and-stats.md) |
| `nba_team_game_logs` | `odds_core.game_log_models.NbaTeamGameLog` | [NBA injuries and stats](../collection/nba-injuries-and-stats.md) |
| `polymarket_events` | `odds_core.polymarket_models.PolymarketEvent` | [Polymarket](../collection/polymarket.md) |
| `polymarket_markets` | `odds_core.polymarket_models.PolymarketMarket` | [Polymarket](../collection/polymarket.md) |
| `polymarket_price_snapshots` | `odds_core.polymarket_models.PolymarketPriceSnapshot` | [Polymarket](../collection/polymarket.md) |
| `polymarket_orderbook_snapshots` | `odds_core.polymarket_models.PolymarketOrderBookSnapshot` | [Polymarket](../collection/polymarket.md) |
| `polymarket_fetch_logs` | `odds_core.polymarket_models.PolymarketFetchLog` | [Polymarket](../collection/polymarket.md) |

<!-- pages:start -->
### Reference

- [Coverage and completeness checks](coverage-checks.md) — TierCoverageValidator, odds validate, odds quality, QualityMetrics, the daily-validation workflow and the read-only scripts/check_*.py.
- [Event matching across sources](event-matching.md) — The matcher each ingest path uses to find its events row (window, team comparison, ambiguity rule) and the duplicates and NULL event ids that follow.
- [Event merge](event-merge.md) — How events for one match are merged into one canonical event, which event is kept, who merges, how twins and rescheduled duplicates are detected, and the odds events merge-twins and repair-names commands.
- [Events table](events.md) — The events columns and statuses, id prefixes by source, team-name normalizers, and the upsert verbs that write events.
- [Fetch tier](fetch-tier.md) — The FetchTier enum and bounds, where tiers are written and read, and the boundary encodings.
- [Odds snapshot and odds tables](odds-tables.md) — The odds_snapshots and odds columns and indexes, the raw_data format, and the provenance tags.
- [Odds validation](odds-validator.md) — OddsValidator checks, the data_quality_logs rows it produces, and DataQualityConfig.
- [OddsReader](odds-reader.md) — Every OddsReader query, which read normalized rows and which re-parse raw_data, and the time-window semantics.
- [OddsWriter](odds-writer.md) — store_odds_snapshot, event upserts, find_or_create_event, logging helpers, and their edge cases.
- [Storage layer](storage-layer.md) — The odds_lambda.storage module map, the commit and flush contract, and the upsert/insert/store verbs.

### Explanation

- [How odds are stored](odds-storage-model.md) — Why odds are stored as raw snapshots plus normalized rows, which sources lack the normalized half, and which readers use which.
<!-- pages:end -->
