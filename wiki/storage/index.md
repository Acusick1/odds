---
title: Storage
description: The events, odds_snapshots and odds tables, the OddsWriter/OddsReader storage layer, odds validation and coverage checks, and fetch tiers.
---

# Storage

Everything the pipeline persists about games and prices lives behind `packages/odds-lambda/odds_lambda/storage/` and the models in `packages/odds-core/odds_core/models.py`. Start with [odds-storage](odds-storage.md) for how a fetch becomes rows, and [fetch-tier](fetch-tier.md) for the tier vocabulary used throughout.

<!-- pages:start -->
### Concepts

- [Fetch tier](fetch-tier.md) — The opening/early/sharp/pregame/closing/in-play tiers by hours to kickoff, where tiers are written and read, the four boundary encodings that disagree, and how tiers drive cadence, validation and sampling.

### Components

- [Events table and event matching](events.md) — The events row and its id-prefix provenance, the five source-specific matchers and two team-name normalizers that decide which event a row attaches to, and why duplicates or NULL event ids appear.
- [Odds snapshots, odds rows and OddsWriter](odds-storage.md) — The hybrid raw-plus-normalized odds storage, the raw_data format and provenance tags, how OddsWriter stores snapshots, assigns tiers and finds or creates events, and which sources lack normalized rows.
- [Odds validation and coverage checks](data-validation.md) — The two unrelated "quality" systems, per-snapshot OddsValidator checks into data_quality_logs (log-only) and completeness checks (odds validate, odds quality, TierCoverageValidator, the daily-validation workflow, scripts/check_*), with their NBA-only parts.
- [OddsReader and the storage layer](odds-reader-and-storage-layer.md) — Map of odds_lambda/storage/, the commit/flush contract and upsert verbs, and which OddsReader queries read normalized rows versus re-parsed JSON, including the symmetric get_odds_at_time window.
<!-- pages:end -->
