---
type: reference
title: Live-close alternatives comparison
description: What the live-close alternatives comparison script compares (OddsPortal's and the Odds API's live Betfair Exchange prices before kick-off against the Betfair job's live close, per market), how each price is chosen, and the report it prints and writes.
sources:
  - scripts.compare_live_close_alternatives.PAIRINGS
  - scripts.compare_live_close_alternatives.alternative_price
  - scripts.compare_live_close_alternatives.compare_event
  - scripts.compare_live_close_alternatives.load_comparisons
  - scripts.compare_live_close_alternatives.build_section
  - scripts.compare_live_close_alternatives.main
mentions:
  - odds_core.close.event_close
  - odds_core.close.live_close_quotes
  - odds_core.exchange_price.MAX_EXCHANGE_OVERROUND
  - odds_core.quotes.quotes
  - odds_analytics.holdout.require_clear
status: stable
---

# Live-close alternatives comparison

`scripts.compare_live_close_alternatives` measures whether the Betfair Exchange prices collected outside the Betfair job agree with the job's live close, so that they could back it up ([#735](https://github.com/Acusick1/odds/issues/735)). Its result is the evidence of [ADR-0027](../../docs/adr/0027-no-live-close-fallback-from-other-betfair-feeds.md). It reads only: its transaction is `SET TRANSACTION READ ONLY`, and it calls no API.

## Usage

```
uv run python scripts/compare_live_close_alternatives.py [--from 2026-05-01] [--to DATE] [--max-age-minutes N] [--json report.json]
```

The league is EPL. `--from` and `--to` are inclusive UTC days of kick-off. `--from` defaults to the first day with a live close and `--to` to today. A range that touches a [sealed holdout](candidate-backtest.md#sealed-holdout) is refused before anything is read (`odds_analytics.holdout.require_clear`). The script prints a section per pairing and, with `--json`, writes the same figures.

## What is compared

`scripts.compare_live_close_alternatives.PAIRINGS` names each alternative source with a market and line: OddsPortal on `1x2` and on totals at `scripts.compare_live_close_alternatives.TOTALS_LINE`, and the Odds API on `1x2`. The Odds API has no Betfair Exchange totals price (#707).

`scripts.compare_live_close_alternatives.load_comparisons` takes each `FINAL` EPL event in the range. For each pairing it reads the event's prices through `odds_core.quotes.quotes`:

- **Live close:** the Betfair job's Betfair Exchange backs within the live close window (`odds_core.close.live_close_quotes`), picked by `odds_core.close.event_close` on the pairing's line. An event without one has nothing to compare.
- **Alternative price:** `betfair_exchange` back prices from the pairing's source with time basis `observed`, the live collectors' snapshots. With `--max-age-minutes`, only snapshots observed at most that long before kick-off are read.

`scripts.compare_live_close_alternatives.alternative_price` takes the latest of those snapshots at or before kick-off that is complete on the line, not in-play, and has a back vig of at most `odds_core.exchange_price.MAX_EXCHANGE_OVERROUND`. A later snapshot that fails these checks is skipped, as the close skips one ([ADR-0014](../../docs/adr/0014-ignore-wide-exchange-prices.md)). `scripts.compare_live_close_alternatives.compare_event` pairs it with the live close. The rows, thresholds and verdicts are the [close agreement test](close-agreement-test.md), with the live close as the reference.

## Report

Each pairing's section (`scripts.compare_live_close_alternatives.build_section`) has:

- The overall verdict, then n, mean and sd in pp with a verdict for each outcome and each band of the live close probability.
- Events: final events in the range, those with a live close, those compared and their share of the events with a live close (the coverage), and the missing counts. An event can have no live close, no alternative snapshot in the window, or only alternative snapshots that are incomplete, in-play or wide.
- The [freshness shares and price ages](close-agreement-test.md#freshness-and-age) over the events with a live close.

## Tests

`tests/unit/test_compare_live_close_alternatives.py` tests the price choice and the report. `tests/integration/test_compare_live_close_alternatives.py` runs `scripts.compare_live_close_alternatives.load_comparisons` over snapshots on the test database.
