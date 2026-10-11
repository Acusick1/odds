# Bookmaker Line Release Timing

When NBA bookmakers release betting lines relative to game time, measured from Odds API
snapshots. The date of the analysis and the query behind it are not recorded.

## Key Finding

**The sharp book (Pinnacle) releases lines 12–24 hours before game time**, while soft books
(DraftKings, FanDuel) release them 2–3+ days earlier.

## First Capture Timing by Bookmaker

| Bookmaker | Avg First Seen (hrs) | Median (hrs) | Max (hrs) |
|-----------|---------------------|--------------|-----------|
| DraftKings | 64.4 | 24.0 | 2293 |
| Caesars | 41.2 | 23.1 | 1624 |
| BetMGM | 37.6 | 19.6 | 1696 |
| BetRivers | 35.1 | 13.0 | 1696 |
| Bovada | 35.0 | 12.0 | 1696 |
| FanDuel | 25.3 | 23.8 | 1696 |
| Pinnacle | 16.1 | 12.0 | 134 |

## Bookmaker Availability by Time Window

| Time Before Game | Avg Bookmakers | Implication |
|------------------|----------------|-------------|
| 48+ hours | 2.7 | Only soft books (DK, FD) |
| 24-48 hours | 3.5 | More books coming online |
| 12-24 hours | 4.3 | Pinnacle typically opens |
| <12 hours | 6.9 | All books available |

## Implications

1. **"Opening line" varies by bookmaker.** DraftKings opens days out; Pinnacle opens
   ~12–24 hours out.
2. **The sharp opening line is Pinnacle's first line.**
3. **Most line movement occurs 12–24 hours out**, when sharp money enters via Pinnacle
   (inferred).
4. **Early soft-book lines are less predictive**: they are subject to significant
   adjustment when the sharp book opens (inferred).
5. **Opening-tier snapshots (72+ hours) capture only soft-book lines.** The sharp tier
   (12–24 hours) is where Pinnacle first appears (see
   [Fetch Tier System](ARCHITECTURE.md#fetch-tier-system)).
