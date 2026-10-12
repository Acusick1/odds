---
type: experiment
title: Bookmaker line release timing
summary: When NBA bookmakers first post lines (soft books days out, Pinnacle 12-24 h before), and what that means for tiers and opening lines.
sources:
  - docs/BOOKMAKER_LINE_RELEASE.md
  - packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier
verified_at: 7887039
---

# Bookmaker line release timing

Measured from Odds API NBA snapshots: when each bookmaker first appears relative to game time. The analysis date and the query behind it are not recorded in `docs/BOOKMAKER_LINE_RELEASE.md`, so the numbers cannot be regenerated from the repo.

**Key finding.** The sharp book (Pinnacle) releases lines 12-24 hours before game time; soft books (DraftKings, FanDuel) release them 2-3+ days earlier.

## First capture, hours before game

| Bookmaker | Average | Median | Max |
|---|---|---|---|
| DraftKings | 64.4 | 24.0 | 2293 |
| Caesars | 41.2 | 23.1 | 1624 |
| BetMGM | 37.6 | 19.6 | 1696 |
| BetRivers | 35.1 | 13.0 | 1696 |
| Bovada | 35.0 | 12.0 | 1696 |
| FanDuel | 25.3 | 23.8 | 1696 |
| Pinnacle | 16.1 | 12.0 | 134 |

## Bookmakers available by window

| Hours before game | Average books | Meaning |
|---|---|---|
| 48+ | 2.7 | only soft books (DK, FD) |
| 24-48 | 3.5 | more books coming online |
| 12-24 | 4.3 | Pinnacle typically opens |
| under 12 | 6.9 | all books available |

## Implications

1. "Opening line" varies by bookmaker: DraftKings opens days out, Pinnacle about 12-24 h out.
2. The sharp opening line is Pinnacle's first line.
3. Most line movement occurs 12-24 h out, when sharp money enters via Pinnacle `(inferred)`.
4. Early soft-book lines are less predictive because they are adjusted when the sharp book opens `(inferred)`.
5. Opening-tier snapshots (72+ h) capture only soft-book lines; the sharp tier (12-24 h) is where Pinnacle first appears. These are the code's tier bounds: `FetchTier.max_hours` gives OPENING above 72 h, EARLY up to 72 h, SHARP up to 24 h, PREGAME up to 12 h, CLOSING up to 3 h ([fetch-tier](fetch-tier.md), [sharp-benchmark](sharp-benchmark.md)).

## Hazards

- NBA and Odds API only; EPL release timing is unmeasured here ([decision-time](decision-time.md)).
- Tier boundaries are encoded more than once and disagree at exactly 3, 12, 24 and 72 h: `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier` uses `<=` while `packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier` uses `<` ([fetch-tier](fetch-tier.md)).

## Related

- [Hours-to-game](exp-hours-to-game.md), [CLV model findings](clv-model-findings.md)
