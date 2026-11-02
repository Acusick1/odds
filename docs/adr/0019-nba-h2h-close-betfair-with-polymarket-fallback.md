---
status: accepted
---

# Close NBA h2h on Betfair, falling back to Polymarket

NBA h2h candidates are scored against the live Betfair Exchange close that [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md) defines for EPL, and fall back to the last Polymarket mid before tip-off where Betfair has no price in the window. This adds a sport to ADR-0012 and does not supersede it.

## Context

ADR-0012's close is the last live Betfair job snapshot that is not in-play and falls within 30 minutes before kick-off, within the [ADR-0014](0014-ignore-wide-exchange-prices.md) overround cap, else the FDUK closing price. FDUK covers only the EPL, so the candidate report and backtest had no close for NBA, and `candidate_report.CLOSE_MARKETS` was keyed by market alone: MLB and NBA both use `h2h`, and MLB has no close.

NBA h2h is collected live from Betfair (back and lay prices, `in_play`, `total_matched`) and Polymarket (`mid`, `back` and `lay` sides; backfilled history has `mid` only), and from The Odds API and OddsPortal. The last h2h price before tip-off, for the 11 games every source collected live on 2026-10-31 and 2026-11-01 (production, decided 2026-11-02, [#724](https://github.com/Acusick1/odds/issues/724)):

| Source | Games priced 10 min or less before tip | Median min before tip | Overround | Liquidity |
|---|---|---|---|---|
| Betfair | 10/11 (11/11 within 30 min) | 5 | 0.3-1.2% | £9k-108k matched |
| Polymarket | 87/88 (backfilled from 2026-10-21) | 0 | 1.0% (mid) | depth not stored |
| Pinnacle (Odds API) | 1/11 | 60 | 2-4% | none |
| Pinnacle (OddsPortal) | none: OddsPortal's UK feed lists no Pinnacle | none | none | none |

On every game, Betfair's and Polymarket's devigged home probabilities are within about 0.02 of each other.

## Decision

- **NBA h2h has the same live Betfair close as EPL.** The window, the in-play exclusion and the overround cap are ADR-0012's and ADR-0014's.
- **Its fallback is the last Polymarket snapshot observed before tip-off within the same 30-minute window.** It is priced at its `mid` side and devigged proportionally across home and away, under the same overround cap. EPL's fallback stays the FDUK closing price. An NBA event with neither has no close.
- **The close is per sport.** `odds_core.betfair_close.CLOSE_FALLBACKS` maps (sport, market) to the fallback source; a pair without an entry has no close: MLB `h2h`, and NBA spreads and totals, which Betfair does not collect.
- **Each snapshot is read by its source.** `odds_core.betfair_close.event_close` reads Betfair and FDUK snapshots as Betfair Exchange backs and Polymarket snapshots as mids, so no call site special-cases a source.

## Considered options

- **Betfair only.** It has no close for the one game in 11 whose last Betfair price is more than 30 minutes out, and for any game the job missed. Polymarket prices 87 of 88 games.
- **Polymarket only.** Its liquidity is not stored, so a thin book cannot be told from a deep one; Betfair's matched volume is stored. Polymarket is the fallback because its devigged prices match Betfair's to about 0.02 on every game compared.
- **Pinnacle.** Only 1 of 11 games has a Pinnacle price from The Odds API within 10 minutes of tip-off (median 60 minutes out, 2-4% overround), and OddsPortal's UK feed lists none.
- **The Polymarket back or lay side.** The backfilled history has only `mid`, so a back or lay close would exist only for live-collected games.

## Consequences

- NBA h2h candidates are scored by EV against `p_close` in the report and backtest, as EPL's are.
- The evidence is 11 games on two days. The Betfair-to-Polymarket agreement is measured on the same games, so a fallback close is not shown to match Betfair's across a season (inferred).
- The retail-EV model builds its rows from Betfair closes only, so an NBA event closed by Polymarket gives it no rows.
- The missing-close alert is unchanged: it checks EPL for a live Betfair close.
