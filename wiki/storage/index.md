---
title: Storage
description: The events, odds_snapshots, odds and prices tables, event external ids and the event resolver, the OddsWriter/OddsReader storage layer, odds validation and coverage checks, and fetch tiers.
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
| `prices` | `odds_core.models.Price` | [Prices and snapshot payloads tables](price-tables.md) |
| `snapshot_payloads` | `odds_core.models.SnapshotPayload` | [Prices and snapshot payloads tables](price-tables.md) |
| `event_external_ids` | `odds_core.models.EventExternalId` | [Events](events.md#external-ids) |
| `health_check_acknowledgements` | `odds_core.models.HealthCheckAcknowledgement` | [Events](events.md#health-check-acknowledgements) |
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

- [Coverage and completeness checks](coverage-checks.md) — TierCoverageValidator, odds validate, odds quality, QualityMetrics, the daily-validation workflow and the read-only scripts/check_games_in_database.py.
- [Event matching across sources](event-matching.md) — The matcher each ingest path uses to find its events row (window, team comparison, ambiguity rule) and the duplicates and NULL event ids that follow.
- [Event merge](event-merge.md) — How events for one match are merged into one canonical event, which event is kept, who merges, how twins and rescheduled duplicates are detected, how stored MLB events get their gamePk, and the odds events merge-twins, repair-names, assign-game-pks, set-game-pk and mark-unassignable commands.
- [Event resolver](event-resolver.md) — How resolve_event finds or creates the one event of a listed game (lock, lookup order, ±2 h match, creation) and records the source's id, and which ids it refuses.
- [Events table](events.md) — The events columns and statuses, id prefixes by source, team-name normalizers, and the writers that create and update events.
- [Fetch tier](fetch-tier.md) — The FetchTier enum and bounds, the one hours-to-tier rule, the tier a snapshot covers by its time basis, and where tiers are written and read.
- [Kick-off authority](kickoff-authority.md) — Which source may move an event's commence_time for each sport, the move_kickoff function every move goes through, and what a move does to the event and its snapshots.
- [Odds snapshot and odds tables](odds-tables.md) — The odds_snapshots and odds columns and indexes, how an odds row links to its snapshot, the raw_data format, the provenance tags, and same-time snapshot pairs.
- [Odds validation](odds-validator.md) — OddsValidator checks, the data_quality_logs rows it produces, and DataQualityConfig.
- [OddsReader](odds-reader.md) — Every OddsReader query and its time-window semantics.
- [OddsWriter](odds-writer.md) — store_odds_snapshot (envelope, odds rows and price rows), event status updates, logging helpers, and their edge cases.
- [Prices and snapshot payloads tables](price-tables.md) — The prices and snapshot_payloads tables, the closed sets their text columns take, how writers fill them from each source's envelope, and the policy store_snapshot applies when a snapshot's key is already stored.
- [Shape backfill and verify](shape-backfill.md) — What odds data backfill-shape writes for stored history (twin merges, UNMATCHED events, external ids, snapshot source, market, time basis, payloads, price rows and Odds API splits, legacy Polymarket prices), how it tells sources apart, the read-only plan and the gates that stop it before any write, its duplicate rule, and what odds data verify-shape checks.
- [Storage layer](storage-layer.md) — The odds_lambda.storage module map, the commit and flush contract, and the upsert/insert/store verbs.

### Explanation

- [How odds are stored](odds-storage-model.md) — Why odds are stored as raw snapshots plus normalized rows, which sources lack the normalized half, and which readers use which.

### How-to guides

- [Assign MLB events their gamePk](howto-assign-mlb-game-pks.md) — Steps to give stored MLB events without a gamePk their MLB Stats API gamePk, merge the events of one game, and settle the events the backfill cannot resolve.
- [Backfill and verify the new shape](howto-backfill-the-new-shape.md) — Steps to move production's stored history into the new snapshot, price and event-identity shape with odds data backfill-shape, check it with odds data verify-shape, and recover when either fails.
- [Delete same-time snapshot pairs](howto-delete-same-time-pairs.md) — Steps to delete the MLB snapshot pairs that hold two games' prices and the duplicate snapshots of one fetch, with their odds rows, and to check that every odds row then names its snapshot.
<!-- pages:end -->
