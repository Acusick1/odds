---
type: reference
title: Substitute-close comparison
description: What the substitute-close comparison script compares (the Odds API's Betfair Exchange price near kick-off against FDUK's Betfair Exchange close), how its prices are chosen, and the report it prints and writes.
sources:
  - scripts.compare_substitute_close.compare_event
  - scripts.compare_substitute_close.load_comparisons
  - scripts.compare_substitute_close.build_report
  - scripts.compare_substitute_close.main
mentions:
  - odds_core.close.event_close
  - odds_core.close.substitute_close
  - odds_core.exchange_price.MAX_EXCHANGE_OVERROUND
  - odds_core.quotes.quotes
  - odds_analytics.holdout.require_clear
status: stable
---

# Substitute-close comparison

`scripts.compare_substitute_close` measures whether the Odds API's Betfair Exchange price near kick-off can stand in for a backtest's close where no Betfair close exists ([#730](https://github.com/Acusick1/odds/issues/730)). Its EPL result is the evidence of [ADR-0023](../../docs/adr/0023-odds-api-betfair-substitute-close-for-epl-backtests.md), which makes that price the EPL backtest's substitute close ([replay close](candidate-backtest.md#replay)), chosen by the same function within 10 minutes of kick-off. It reads only: its transaction is `SET TRANSACTION READ ONLY`, and it calls no API. The close it tests against is FDUK's Betfair Exchange close, which [scoring](../modelling/clv-target.md#the-retail-ev-target) uses for the football leagues in `odds_core.close.FDUK_CLOSED_LEAGUES`.

## Usage

```
uv run python scripts/compare_substitute_close.py [--sport soccer_epl] [--from 2024-08-01] [--to 2026-07-31] [--max-age-minutes N] [--json report.json]
```

`--sport` is one of `odds_core.close.FDUK_CLOSED_LEAGUES` and the market is `1x2`. `--from` and `--to` are inclusive UTC days of kick-off; a range that touches the sport's [sealed holdout](candidate-backtest.md#sealed-holdout) is refused before anything is read (`odds_analytics.holdout.require_clear`). It prints the tables below and, with `--json`, writes the same figures.

## What is compared

`scripts.compare_substitute_close.load_comparisons` takes each `FINAL` event of the sport in the range, then reads its prices through `odds_core.quotes.quotes`:

- **FDUK close:** Betfair Exchange back prices of the source and time basis in `odds_core.close.FDUK_CLOSE`, picked by `odds_core.close.event_close`.
- **Odds API price:** `betfair_exchange` back prices of source `odds_api` and time basis `source_stamped`, the historical backfill's ([stored shape](../collection/odds-api-backfill.md#executing-calls)). A live fetch (`observed`) is not read. With `--max-age-minutes`, only snapshots stamped at most that long before kick-off are read.

`scripts.compare_substitute_close.compare_event` takes, for each event, the price `odds_core.close.substitute_close` picks with the script's `--max-age-minutes` (no age cap without it): the latest Odds API historical snapshot stamped at or before kick-off that is complete and whose back vig is at most `odds_core.exchange_price.MAX_EXCHANGE_OVERROUND`; a later, wider snapshot is skipped, as the close skips one ([ADR-0014](../../docs/adr/0014-ignore-wide-exchange-prices.md)). The FDUK close has the same vig cap. The rows, thresholds and verdicts are the [close agreement test](close-agreement-test.md), with FDUK's close as the reference.

## Report

- The overall verdict, then n, mean and sd in pp with a verdict for each outcome and each band of the FDUK probability.
- Events: final events in the range, those with a usable FDUK close, those compared, and the missing counts: no FDUK close stored, an FDUK close over the vig cap, no Odds API snapshot in the window, and Odds API snapshots that are all incomplete or wide.
- The [freshness shares and price ages](close-agreement-test.md#freshness-and-age) over the events with a usable FDUK close. Without `--max-age-minutes` the latest snapshot may be days old.

## Tests

`tests/unit/test_compare_substitute_close.py` tests the price choice, the report and the refusal of sealed ranges; `tests/integration/test_compare_substitute_close.py` runs `scripts.compare_substitute_close.load_comparisons` over snapshots on the test database.
