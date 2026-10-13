---
title: Storage
description: The events, odds_snapshots and odds tables, the OddsWriter/OddsReader storage layer, odds validation and coverage checks, and fetch tiers.
---

# Storage

Everything the pipeline persists about games and prices lives behind `packages/odds-lambda/odds_lambda/storage/` and the models in `packages/odds-core/odds_core/models.py`. Start with [odds-storage](odds-storage.md) for how a fetch becomes rows, and [fetch-tier](fetch-tier.md) for the tier vocabulary used throughout.

## Table map

Every table is a SQLModel class in `packages/odds-core/odds_core/`; migrations are in [migrations](../foundations/howto-database-migration.md).

| Table | Model | Owning page |
|---|---|---|
| `events` | `packages/odds-core/odds_core/models.py::Event` | [Events](events.md) |
| `odds_snapshots` | `packages/odds-core/odds_core/models.py::OddsSnapshot` | [Odds storage](odds-storage.md) |
| `odds` | `packages/odds-core/odds_core/models.py::Odds` | [Odds storage](odds-storage.md) |
| `data_quality_logs` | `packages/odds-core/odds_core/models.py::DataQualityLog` | [Validation](data-validation.md) |
| `fetch_logs` | `packages/odds-core/odds_core/models.py::FetchLog` | [Odds API ingestion](../collection/odds-api-ingestion.md) |
| `alert_history` | `packages/odds-core/odds_core/models.py::AlertHistory` | [Alerts and health](../operations/alerts-and-health.md) |
| `predictions` | `packages/odds-core/odds_core/prediction_models.py::Prediction` | [Model publishing and scoring](../modelling/model-publishing-and-scoring.md) |
| `match_briefs` | `packages/odds-core/odds_core/match_brief_models.py::MatchBrief` | [Match briefs and paper trades](../agent/match-briefs-and-paper-trades.md) |
| `paper_trades` | `packages/odds-core/odds_core/paper_trade_models.py::PaperTrade` | [Match briefs and paper trades](../agent/match-briefs-and-paper-trades.md) |
| `agent_wakeups` | `packages/odds-core/odds_core/agent_wakeup_models.py::AgentWakeup` | [Betting agent](../agent/betting-agent.md) |
| `espn_fixtures` | `packages/odds-core/odds_core/epl_data_models.py::EspnFixture` | [EPL context data](../collection/epl-context-data.md) |
| `espn_lineups` | `packages/odds-core/odds_core/epl_data_models.py::EspnLineup` | [EPL context data](../collection/epl-context-data.md) |
| `fpl_availability` | `packages/odds-core/odds_core/epl_data_models.py::FplAvailability` | [EPL context data](../collection/epl-context-data.md) |
| `mlb_probable_pitchers` | `packages/odds-core/odds_core/mlb_data_models.py::MlbProbablePitchers` | [MLB probable pitchers](../collection/mlb-probable-pitchers.md) |
| `nba_injury_reports` | `packages/odds-core/odds_core/injury_models.py::InjuryReport` | [NBA injuries and stats](../collection/nba-injuries-and-stats.md) |
| `nba_player_season_stats` | `packages/odds-core/odds_core/player_stats_models.py::NbaPlayerSeasonStats` | [NBA injuries and stats](../collection/nba-injuries-and-stats.md) |
| `nba_team_game_logs` | `packages/odds-core/odds_core/game_log_models.py::NbaTeamGameLog` | [NBA injuries and stats](../collection/nba-injuries-and-stats.md) |
| `polymarket_events` | `packages/odds-core/odds_core/polymarket_models.py::PolymarketEvent` | [Polymarket](../collection/polymarket.md) |
| `polymarket_markets` | `packages/odds-core/odds_core/polymarket_models.py::PolymarketMarket` | [Polymarket](../collection/polymarket.md) |
| `polymarket_price_snapshots` | `packages/odds-core/odds_core/polymarket_models.py::PolymarketPriceSnapshot` | [Polymarket](../collection/polymarket.md) |
| `polymarket_orderbook_snapshots` | `packages/odds-core/odds_core/polymarket_models.py::PolymarketOrderBookSnapshot` | [Polymarket](../collection/polymarket.md) |
| `polymarket_fetch_logs` | `packages/odds-core/odds_core/polymarket_models.py::PolymarketFetchLog` | [Polymarket](../collection/polymarket.md) |

<!-- pages:start -->
### Concepts

- [Fetch tier](fetch-tier.md) — The opening/early/sharp/pregame/closing/in-play tiers by hours to kickoff, where tiers are written and read, the four boundary encodings that disagree, and how tiers drive cadence, validation and sampling.

### Components

- [Events table and event matching](events.md) — The events row and its id-prefix provenance, the five source-specific matchers and two team-name normalizers that decide which event a row attaches to, and why duplicates or NULL event ids appear.
- [Odds snapshots, odds rows and OddsWriter](odds-storage.md) — The hybrid raw-plus-normalized odds storage, the raw_data format and provenance tags, how OddsWriter stores snapshots, assigns tiers and finds or creates events, and which sources lack normalized rows.
- [Odds validation and coverage checks](data-validation.md) — The two unrelated "quality" systems, per-snapshot OddsValidator checks into data_quality_logs (log-only) and completeness checks (odds validate, odds quality, TierCoverageValidator, the daily-validation workflow, scripts/check_*), with their NBA-only parts.
- [OddsReader and the storage layer](odds-reader-and-storage-layer.md) — Map of odds_lambda/storage/, the commit/flush contract and upsert verbs, and which OddsReader queries read normalized rows versus re-parsed JSON, including the symmetric get_odds_at_time window.
<!-- pages:end -->
