---
type: reference
title: Substitute-close comparison
description: What the substitute-close comparison script compares (the Odds API's Betfair Exchange price near kick-off against FDUK's Betfair Exchange close), the pass thresholds, the report it prints and writes, and how its prices are chosen.
sources:
  - scripts.compare_substitute_close.compare_event
  - scripts.compare_substitute_close.load_comparisons
  - scripts.compare_substitute_close.build_report
  - scripts.compare_substitute_close.stat
  - scripts.compare_substitute_close.outcome_verdict
  - scripts.compare_substitute_close.band_verdict
  - scripts.compare_substitute_close.overall_verdict
  - scripts.compare_substitute_close.band_of
  - scripts.compare_substitute_close.main
mentions:
  - odds_core.close.event_close
  - odds_core.exchange_price.exchange_price
  - odds_core.exchange_price.MAX_EXCHANGE_OVERROUND
  - odds_core.as_of.latest_snapshot
  - odds_core.quotes.quotes
status: stable
---

# Substitute-close comparison

`scripts.compare_substitute_close` measures whether the Odds API's Betfair Exchange price near kick-off can stand in for a backtest's close where no Betfair close exists ([#730](https://github.com/Acusick1/odds/issues/730)). It reads only: its transaction is `SET TRANSACTION READ ONLY`, and it calls no API. The close it tests against is FDUK's Betfair Exchange closing price, which [scoring](../modelling/clv-target.md#the-retail-ev-target) uses for the football leagues in `odds_core.close.FDUK_CLOSED_LEAGUES`.

## Usage

```
uv run python scripts/compare_substitute_close.py [--sport soccer_epl] [--from 2024-08-01] [--to 2026-07-31] [--max-age-minutes N] [--json report.json]
```

`--sport` is one of `odds_core.close.FDUK_CLOSED_LEAGUES` and the market is `1x2`. `--from` and `--to` are inclusive UTC days of kick-off. It prints the tables below and, with `--json`, writes the same figures.

## What is compared

`scripts.compare_substitute_close.load_comparisons` takes each `FINAL` event of the sport in the range, then reads its quotes through `odds_core.quotes.quotes`:

- **FDUK close:** Betfair Exchange back prices of the source and time basis in `odds_core.close.FDUK_CLOSE`, picked by `odds_core.close.event_close`.
- **Odds API price:** `betfair_exchange` back prices of source `odds_api` and time basis `source_stamped`, the historical backfill's ([stored shape](../collection/odds-api-backfill.md#executing-calls)). A live fetch (`observed`) is not read. With `--max-age-minutes`, only snapshots stamped at most that long before kick-off are read.

`scripts.compare_substitute_close.compare_event` takes, for each event, the latest Odds API snapshot stamped at or before kick-off (`odds_core.as_of.latest_snapshot`) that is complete and whose back overround is at most `odds_core.exchange_price.MAX_EXCHANGE_OVERROUND`; a later, wider snapshot is skipped, as the close skips one ([ADR-0014](../../docs/adr/0014-ignore-wide-exchange-prices.md)). The FDUK close has the same overround cap. Both prices are devigged proportionally ([devigging](../foundations/devigging.md)), and `dp = p_oddsapi - p_fduk` in percentage points is one row per outcome and compared match.

## Pass thresholds

Pre-set on #730, as constants in the script:

- **Per outcome** (`scripts.compare_substitute_close.outcome_verdict`): pass when `|mean dp|` is at most 0.3 pp and the sample sd at most 1.0 pp. An outcome with under two rows is inconclusive.
- **Per band of the FDUK probability** (`scripts.compare_substitute_close.band_verdict`, bands from `scripts.compare_substitute_close.band_of`: below 0.25, 0.25 up to 0.5, and 0.5 and above): pass when `|mean dp|` is at most 0.5 pp. A band under 50 rows is inconclusive, not a pass.
- **Overall** (`scripts.compare_substitute_close.overall_verdict`): fail if any slice fails, pass only if every slice passes, otherwise inconclusive.

## Report

- The overall verdict, then n, mean and sd in pp with a verdict for each outcome and each band.
- Matches: final events in the range, those with a usable FDUK close, those compared, and the missing counts: no FDUK close stored, an FDUK close over the overround cap, no Odds API snapshot in the window, and Odds API snapshots that are all incomplete or wide.
- The share of matches with a usable FDUK close whose Odds API price is within 5, 10, 30 and 60 minutes before kick-off, with the counts. The share is over every such match, so a match with no usable price counts against it.
- The age of the prices used (minutes before kick-off): n, minimum, 25th, 50th, 75th and 90th percentiles, maximum and mean.

Without `--max-age-minutes` the latest snapshot may be days old, so read the age distribution with the verdict; freshness alone does not make a valid substitute ([#730](https://github.com/Acusick1/odds/issues/730)).

## Tests

`tests/unit/test_compare_substitute_close.py` tests the pure functions; `tests/integration/test_compare_substitute_close.py` runs `scripts.compare_substitute_close.load_comparisons` over snapshots on the test database.
