---
status: accepted
---

# Score EPL backtests against the Odds API's Betfair Exchange price where FDUK has no close

## Context

A backtest scores each bet against the Betfair Exchange close of [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md): the last live Betfair job snapshot within 30 minutes before kick-off, else the FDUK Betfair Exchange closing price. FDUK has that price only from 2024-25, so EPL `1x2` bets before then have no close and can only generate ideas. The FDUK Pinnacle close was tested as a substitute and rejected: its ≥ 0.5 probability band missed the threshold by −0.64 pp (#730).

[ADR-0021](0021-sealed-holdout-and-one-shot-gate.md) seals EPL 2022-23 and 2023-24 as holdout on the condition that this substitute close is accepted, behind the switch `odds_analytics.holdout.EPL_2022_24_SEALED`, which defaults to sealed and goes false only if #730 rejects it.

#730 pre-set the test. Measure `Δp = p_oddsapi_bfe − p_fduk_bfe_close`, with both prices devigged proportionally, on EPL matches that have an FDUK Betfair close. The substitute passes when every outcome has `|mean Δp| ≤ 0.3 pp` and `sd ≤ 1.0 pp`, and every probability band (< 0.25, 0.25–0.5, ≥ 0.5) with at least 50 rows has `|mean Δp| ≤ 0.5 pp`. A band under 50 rows is inconclusive. Freshness alone does not make a valid substitute. The outcome #730 set for a pass is an ADR adding the substitute to ADR-0012's close order for backtests, with the live forward-test score unchanged.

## Decision

- **Close order for EPL `1x2` in a backtest:**
  1. The live Betfair close (ADR-0012).
  2. Else the FDUK Betfair Exchange closing price (ADR-0012).
  3. Else the Odds API's `betfair_exchange` back price from historical (`source_stamped`) snapshots.

  The Odds API price is the latest complete snapshot stamped at or before kick-off and at most 10 minutes before it, whose back overround is within `odds_core.exchange_price.MAX_EXCHANGE_OVERROUND` ([ADR-0014](0014-ignore-wide-exchange-prices.md)). It is devigged proportionally, like the other closes.
- **Backtests only.** The substitute scores development replays and holdout runs. The live forward test's score, the candidate report on paper trades, and the retail-EV target are unchanged.
- **EPL `1x2` only.** The test ran on EPL. Other leagues keep the ADR-0012 close order. Totals get no substitute, because the Odds API has no Betfair Exchange totals price (#859).
- **The substitute scores EPL bets wherever FDUK has no close,** including EPL 2022-23 and 2023-24, which ADR-0024 makes development data in place of ADR-0021's EPL seal. EPL development screens can therefore score EPL from 2023-04-01, when the Odds API's Betfair EPL h2h begins.

## Considered options

- **No age cap.** The test passed without one, but on 738 matches it took one price 148 minutes before kick-off. #730 bought only the KO−10 and KO−5 calls as the near-kick-off price.
- **A 30-minute cap, the live close's window (`odds_core.close.LIVE_CLOSE_WINDOW`).**
  - It would reuse the existing windowed-fallback check.
  - In the comparison and in the sealed seasons' audit, no match has its price between 10 and 30 minutes before kick-off, so the cap would accept exactly the same matches. Ages from 10 to 30 minutes are untested.
- **A 5-minute cap.** It also passes, but drops 32 of the 738 matches.
- **The substitute before FDUK.** FDUK's close is the reference the substitute was measured against, so it stays first where it exists.
- **All six FDUK leagues.**
  - A check on 2026-11-12, with the same script and the same thresholds on each league's 2024-25 development season, passed in the Championship, La Liga, Serie A, the Bundesliga and Ligue 1.
  - That check was not part of #730's pre-set scope, which names EPL and NBA, and [#820](https://github.com/Acusick1/odds/issues/820) ties the sealed EPL seasons to "#730 accepts the substitute close for EPL".
  - Those leagues' 2024-25 seasons lack an FDUK close for only 3 matches (Ligue 1). Extending the substitute to them is a separate decision.

## Consequences

- **Where EPL gains a close.** The EPL Odds API run starts 2023-04-01, when its Betfair EPL h2h begins (#859). So:
  - The 110 FINAL fixtures of EPL 2022-23 from 2023-04-01 and all 380 of 2023-24 have a price within 10 minutes of kick-off, so their bets are scoreable.
  - EPL 2022-23 before 2023-04-01 still has no close, and neither does EPL 2020-22; bets there can only generate ideas.
- **The 22 EPL 2024-26 FINAL matches without an FDUK close gain a close.** All are from April and May 2026, and each has an Odds API price 4.4 minutes before kick-off.
- **The pass is EPL evidence only.** NBA has its own sample under #730, and MLB inherits the NBA result (#787).

## Evidence

- Issues: #730 (pre-set thresholds and outcome), #820 (sealed EPL seasons conditional on #730), #859 (Odds API month plan), #527 (purchase), #731 (audit).
- Comparison: `uv run python scripts/compare_substitute_close.py --sport soccer_epl --from 2024-08-01 --to 2026-06-30`, read-only, on production data on 2026-11-12 ([substitute-close comparison](../../wiki/evaluation/substitute-close-comparison.md)). The window starts 2024-08-01 because EPL 2022-08-01 to 2024-07-31 is sealed.

  | Slice | n | mean Δp (pp) | sd Δp (pp) | result |
  |---|---|---|---|---|
  | home | 738 | −0.01 | 0.32 | pass |
  | draw | 738 | +0.01 | 0.21 | pass |
  | away | 738 | −0.00 | 0.30 | pass |
  | p < 0.25 | 790 | +0.02 | 0.25 | pass |
  | 0.25–0.5 | 1023 | +0.00 | 0.30 | pass |
  | p ≥ 0.5 | 401 | −0.03 | 0.30 | pass |

  - 738 of the 760 FINAL matches were compared. The other 22 have no FDUK close.
  - Price age: median 4.4 minutes before kick-off. 95.7% of prices were within 5 minutes and 99.9% within 10 minutes. The maximum was 148.4 minutes.
  - With `--max-age-minutes 10`, every slice passes on 737 matches, with sd at most 0.30 pp.
  - With `--max-age-minutes 5`, every slice passes on 706 matches.
- Sealed-season coverage: [EPL backfill audit](../../backfills/odds-api-epl-2022-26.audit.md#summary), which reports counts and ages only.
- Code: `odds_core.close`, `odds_analytics.holdout.EPL_2022_24_SEALED`, `scripts/compare_substitute_close.py`.
