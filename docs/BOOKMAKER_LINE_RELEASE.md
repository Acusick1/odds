# Bookmaker Line Release Timing

Analysis of when NBA bookmakers release betting lines relative to game time.

## Key Finding

**Sharp books (Pinnacle) release lines 12-24 hours before game time**, while soft books (DraftKings, FanDuel) release 2-3+ days earlier.

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

## Implications for Line Movement Analysis

1. **"Opening line" varies by bookmaker** - DraftKings opens days out, Pinnacle opens ~12-24 hours out
2. **Sharp opening = Pinnacle's first line** - This is the benchmark for EV calculations
3. **Most line movement occurs 12-24 hours out** - When sharp money enters via Pinnacle
4. **Early soft book lines are less predictive** - Subject to significant adjustment when sharps open

## Data Collection Strategy

Focus sampling frequency on the 12-24 hour window (SHARP tier) where Pinnacle opens and professional bettors act. Early tier data (72+ hours) captures only soft book lines of limited predictive value.
