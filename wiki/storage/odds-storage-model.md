---
type: explanation
title: How odds are stored
description: Why odds are stored as per-market snapshots with one row per price and the source's envelope archived apart, and how every writer and reader shares that one shape.
sources:
  - odds_core.models.OddsSnapshot
  - odds_core.models.Price
  - odds_core.models.SnapshotPayload
  - odds_lambda.price_rows
  - odds_lambda.storage.shaped_snapshots.store_shaped_snapshot
  - odds_core.quotes.quotes
mentions:
  - odds_lambda.storage.readers.OddsReader
status: stable
---

# How odds are stored

A fetch of one event from one source becomes one `odds_core.models.OddsSnapshot` per market it quotes, each holding the event's integer id, each with one `odds_core.models.Price` row per bookmaker, outcome, line and side. The envelope the writer built from the source's response is archived as an `odds_core.models.SnapshotPayload` that the snapshots point at. Columns are on [Odds snapshots table](odds-tables.md) and [Prices and snapshot payloads tables](price-tables.md).

## One shape for every source

Every writer, live or historical, builds an envelope in one shape ([envelope format](odds-tables.md#envelope-format)) and hands it to `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot`, which reads it with `odds_lambda.price_rows.expand_envelope` (a market element's own `last_update` and `betfair_matched` take precedence over its bookmaker entry's, and a price that is not a finite decimal above 1.0 is dropped) ([Writing price rows](price-tables.md#writing-price-rows)). The source-specific parsing (team names to home and away, bookmaker aliases, Betfair runners, Polymarket probabilities, lay and traded sides) happens there, once, at write time, so no reader parses JSON (#663). What a snapshot's time means is a column, `time_basis`, not something a reader infers from the source (#662).

## Why prices are rows

A price row carries its side (back, lay, mid or traded), size and matched volume next to the price, so exchange depth and Polymarket bid and ask are queryable like any bookmaker quote (#663). Readers filter in SQL by source, time basis, bookmaker, side and time through one query, `odds_core.quotes.quotes` ([Reading quotes](price-tables.md#reading-quotes)), and select as of a moment with the snapshot's `created_at`. Every source fills the same rows, historical loads included, so a reader that works for one source works for all.

## Why the envelope is kept

The payload is a cold archive: it is read again only to re-parse snapshots after an adapter bug, since the price rows are what the writer understood of it (#662). Live jobs archive one payload per run, holding every event's envelope; the readers that need a field no row holds, such as the Betfair runner names that team-name repair compares against, find the event's envelope in it.

## Snapshots are per market

One snapshot per market lets a source's 1x2 and totals prices at one time be keyed, deduplicated and closed independently: (event, source, market, time) is unique ([storing a snapshot under its key](price-tables.md#storing-a-snapshot-under-its-key)). An Odds API fetch quoting 1x2, spreads and totals stores three snapshots at one time, so readers that count fetches count distinct snapshot times (`odds_lambda.storage.readers.OddsReader`, [OddsReader](odds-reader.md)).

## Related

- [Odds snapshots table](odds-tables.md)
- [Prices and snapshot payloads tables](price-tables.md)
- [OddsWriter](odds-writer.md)
- [Fetch tier](fetch-tier.md)
