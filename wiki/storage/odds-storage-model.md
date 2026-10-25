---
type: explanation
title: How odds are stored
description: Why odds are stored as raw snapshots plus normalized rows, which sources lack the normalized half, and which readers use which.
sources:
  - odds_core.models.OddsSnapshot
  - odds_core.models.Odds
  - odds_core.snapshot_utils.extract_odds_from_snapshot
  - odds_lambda.storage.writers.OddsWriter._parse_and_store_odds
  - odds_lambda.storage.readers.OddsReader
  - scripts/ingest_oddsportal.py
  - scripts/ingest_football_data_uk.py
mentions:
  - odds_core.models.Event
status: stable
---

# How odds are stored

Odds are stored twice (the normalized rows keep `price_decimal`, constrained above 1.0). Each fetch becomes one `odds_core.models.OddsSnapshot` row holding the complete payload as JSON in `raw_data`, plus one `odds_core.models.Odds` row per bookmaker, market and outcome, which names its snapshot in `snapshot_id`. `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` writes both: raw for debugging, auditing and exact preservation, normalized for SQL filtering and time series `(inferred)`. Columns and indexes are on [Odds snapshot and odds tables](odds-tables.md). A third shape, one `prices` row per price with the payload archived apart, is written alongside by the OddsPortal live and Betfair jobs ([Writing price rows](price-tables.md#writing-price-rows)).

## The normalized half is not universal

`raw_data` is the one complete record and its football three-way market key is `1x2` for every source ([sport-and-market-keys](../collection/sport-and-market-keys.md#market-key-dialects)). The historical ingest scripts (`scripts/ingest_oddsportal.py`, `scripts/ingest_football_data_uk.py`) write `odds_core.models.Event` and `odds_core.models.OddsSnapshot` ([columns and indexes](odds-tables.md#odds_snapshots)) rows, and no `odds` rows (FDUK snapshot times are approximations tagged with an `odds_core.snapshot_sources.FdukTimeBasis`; both scripts attach to an existing event of the same sport when one matches, and the FDUK one gives it the full-time score when it has none) (#435; [Historical ingest scripts](../collection/historical-ingest-scripts.md)). Anything that queries the normalized table therefore returns nothing for those events: `odds_lambda.storage.readers.OddsReader.get_odds_at_time`. The [replay backtest](../evaluation/candidate-backtest.md) reads `raw_data` instead. Live fetches write both halves, so recent events work either way.

## Which readers use which

`odds_lambda.storage.readers.OddsReader` divides its methods by path ([OddsReader](odds-reader.md#which-path-each-method-uses)): three price lookups read normalized rows, while most price lookups rebuild `odds_core.models.Odds` objects from `raw_data` with `odds_core.snapshot_utils.extract_odds_from_snapshot` and never persist them. Those raw-based readers work for every source, including historical ones, which is why the sharp-price and latest-book queries behind the agent's tools do not depend on the normalized rows `(inferred)`.

## Two parsers, one price rule

Two code paths turn `raw_data` into `odds_core.models.Odds`, the writer's `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` and `odds_core.snapshot_utils.extract_odds_from_snapshot`. Both read each outcome's price with `odds_core.odds_math.outcome_price_decimal`, which raises `KeyError` when an outcome has neither `price_decimal` nor `price` ([Odds arithmetic helpers](../foundations/odds-math.md#price-encodings)). The same snapshot therefore yields the same prices from the normalized table and from the raw-based readers, and both kinds of row carry the snapshot's id, so a normalized row can be checked against the `raw_data` it came from. Without it the only join is event, time and key, which two snapshots at one time can share ([same-time pairs](odds-tables.md#same-time-pairs), #535). Provenance tags and indexes are on [Odds snapshot and odds tables](odds-tables.md).

## Related

- [Odds snapshot and odds tables](odds-tables.md)
- [OddsWriter](odds-writer.md)
- [OddsReader](odds-reader.md)
