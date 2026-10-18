# FDUK coverage for 2025-26 and 2026-27 (ticket #511)

Fetched 2026-10-18 from `https://football-data.co.uk/mmz4281/{2425,2526,2627}/E0.csv` (the `www.` host answers 302 to the bare host; `urlopen` follows it). Samples are in `research/fduk-coverage-2026/` (`E0_2526.csv`, `E0_2627.csv`, and `notes.txt`, the site's column dictionary).

## Verdict

PROVEN: FDUK has the full 2025-26 season and 2026-27 to date, with bet365 and Betfair Exchange opening and closing prices. Pinnacle is gone: the last Pinnacle price is Arsenal v Liverpool, 2026-01-08. The repo loader parses all three files with zero errors but has no 2026-27 season key.

## Coverage (non-empty counts in the downloaded CSVs)

| Season | Rows | Kick-offs | Pinnacle open/close | bet365 open/close | Betfair Exchange open/close |
|---|---|---|---|---|---|
| 2024-25 | 380 | 2024-08-16 to 2025-05-25 | 380/380 | 380/380 | 380/380 |
| 2025-26 | 380 (all with results) | 2025-08-15 to 2026-05-24 | 210/210 | 380/380 | 360/358 |
| 2026-27 | 60 (all with results) | 2026-08-21 to 2026-10-12 | columns absent | 60/60 | 60/60 |

- Pinnacle: complete through December 2025, 24 of 49 rows in January 2026 (last 2026-01-08), none from 2026-01-17. The 2026-27 file has no PS columns.
- Betfair Exchange gaps in 2025-26: 22 matches between 2026-04-20 and 2026-05-09 have no closing price (20 have no opening price either). 2026-27 BFE is complete; BFDC (Betfred closing) is 57/60.
- 2026-27 is current: 60 matches is six matchweeks, to 2026-10-12.
- Cost: three plain HTTP downloads; the host advertises `x-ws-ratelimit-limit: 1000`.

## Column dictionary (notes.txt)

- Opening columns are `<book><H|D|A>`; closing adds `C` after the book (`B365CH`) (line 48).
- `BFE*` is Betfair Exchange (lines 129-131), `PS*`/`P*` Pinnacle (89-91). `BFD*` is Betfred (59-61), not Betfair; `BF*` (56-58) is the old Betfair sportsbook.
- Timing: "Betting odds for weekend games are collected Friday afternoons, and on Tuesday afternoons for midweek games" (line 198). The opening column is a Friday or Tuesday snapshot, roughly 1-4 days before kick-off, not a fixed horizon. The notes give no closing timestamp.
- `Time` is kick-off time (line 9) with no stated timezone; values such as `20:00`, `12:30` look like UK local time (inferred).

## Does the repo loader handle the current format?

I imported `scripts/ingest_football_data_uk.py` and ran its parse path (`read_csv`, `normalize_team`, `_parse_date`, `_build_snapshot_raw_data`) over each file, with no DB writes.

| File | Rows | Opening snapshots | Closing snapshots | Errors |
|---|---|---|---|---|
| 2425 | 380 | 380 | 380 | 0 |
| 2526 | 380 | 380 | 380 | 0 |
| 2627 | 60 | 60 | 60 | 0 |

Missing Pinnacle columns are handled: `_safe_float` returns None and the book is dropped from the snapshot. Issues for a backfill:

1. **No 2026-27 season.** `SEASONS` (`scripts/ingest_football_data_uk.py:62`) ends at `2025-2026`, so `--all` skips 2026-27 and `--seasons 2026-2027` is rejected. Fix: add `"2026-2027": "2627"`.
2. **Stale local CSV.** `data/football_data_uk/E0_2526.csv` has 291 rows against 380 now. The loader skips events that already have a `football_data_uk` snapshot and only merges match_stats into them, so a re-download only adds snapshots for events with none.
3. **Horizon stamps are invented.** Opening is stamped kick-off minus 24 h and closing kick-off minus 1 h. Per notes.txt line 198 the real opening horizon is roughly 1-4 days and varies by match. This matters for the horizon experiments.
4. **Kick-off is treated as UTC.** `_parse_date` stamps UK local time as UTC, so summer matches are 1 h out; the docstring accepts this. It matters for any hour-level horizon work.
5. **Retail books are dropped.** `BOOKMAKER_PREFIXES` reads pinnacle, bet365, bwin, BFE, BF, IW, WH, VC, LB, 1XB, plus Max and Avg. It does not read `BV` (BetVictor; the mapped `VC` no longer appears), `BFD` (Betfred), `PP` (Paddy Power), `SKB` (Sky Bet), `BMGM` or `CL`. The 2026-27 file carries `B365`, `BFD`, `BV`, `BW`, `PP`, `SKB`; the loader would keep only bet365 and bwin of these.
6. **Pinnacle is unavailable after 2026-01-08.** The sharp reference for the remaining 2025-26 matches and all of 2026-27 has to be Betfair Exchange, with the gaps above.

## What the map can build on

- A backfill of 2025-26 from 2026-03-06 to the season end is 170 matches with bet365 and BFE opening and closing (BFE closing 148 of 170), plus 60 matches of 2026-27, growing as the file updates.
- Pinnacle as target or sharp reference ends at 210 matches of 2025-26.
- Code changes needed: the season key, the extra retail prefixes (`BV`, `BFD`, `PP`, `SKB`, and `BMGM`, `CL` for 2025-26), and a decision on how to represent the real Friday/Tuesday opening horizon.

## Cost and limits

No paid calls, no Odds API quota, no DB writes. The local dev DB (localhost:5433) holds no 2025-26 `fduk_` events, so it cannot confirm the "stops at 2026-03-05" observation.
