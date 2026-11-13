# Odds API extra purchases, 2026-11: costs and recommendations

Costed options for step 10 of the Odds API month ([#859](https://github.com/Acusick1/odds/issues/859)), and the cost of step 9's NBA sample ([#730](https://github.com/Acusick1/odds/issues/730)). The acquisition key's balance is 3,899,110 units, with no run active (#859, 2026-11-13). NBA 2024-26 (`odds-api-nba-2024-26.yaml`, 161,170 units) waits on step 9 and is not part of these options. No units were spent to produce these figures. The manifests named below are in `backfills/proposed/`, outside the tests that every bought manifest in `backfills/` passes (`tests/unit/test_backfill_plan.py`, `tests/integration/test_backfill_replay.py`).

## Summary

Every call is `h2h` on `regions=uk` at 10 units, apart from item 3 and item 5. Runtime is at 20 to 40 calls a minute per process; the month's runs took about 1.3 to 2.2 s a call (#859). Manifests of different sports run in parallel ([limits](../wiki/collection/odds-api-backfill.md#limits)).

| Item | Calls | Units | Runtime | Probe units | What it adds | What it decides | Recommendation |
|---|---|---|---|---|---|---|---|
| NBA sample, step 9 (`odds-api-nba-close-sample-2026-27.yaml`) | 114 now; ≈224 at the run | 1,140 now; ≈2,240 at the run | 6 to 11 min | 10, shared with item 6 | The exchange price about 9.4 and 4.4 min before tip-off for 2026-27 games with a live Betfair close | Whether the Odds API exchange price can be the NBA close, and through #787 the MLB close | **Buy** when the `<0.25` band reaches 50 rows, projected 2026-11-25 ([below](#nba-sample-readiness)) |
| 1. Older football: La Liga, Serie A, Bundesliga, Ligue 1 and the Championship, 2022-23 from 2023-04-01 and 2023-24 (`odds-api-<league>-2022-24.yaml`) | 36,767 | 367,670 | 4.0 to 8.0 h in 5 processes | 100 | A second development season for the five leagues, 2,430 fixtures | Nothing under the ADRs in force: FDUK has no Betfair Exchange close before 2024-25, and ADR-0023's substitute close is EPL only. An ADR extending ADR-0023 to these leagues would give it a close. Round 1 still could not use it, since its development blocks (ADR-0024, `odds_analytics.cell_selection.DEVELOPMENT_BLOCKS`) leave it out; later rounds could | **Buy after a probe**, and only with that ADR; otherwise skip |
| 2. A KO−5 close band where no close exists | 0 | 0 | — | 0 | Nothing: by row counts, no bought fixture lacks both a close and an exchange quote within 10 min ([method](#method)) | — | **Skip**: the target set is empty |
| 3. Narrow football totals, KO−1h to KO, 2024-26, six leagues (`odds-api-<league>-totals-2024-26.yaml`) | 20,451 | 204,510 | 2.0 to 4.0 h in 6 processes | 60 | Pinnacle and William Hill totals, every 10 min in the last hour | Nothing scheduled: step 7 judges totals ideas on live data only, and no totals rule exists in code | **Skip** |
| 4. NBA 2023-24 (`odds-api-nba-2023-24.yaml`) | 6,153 on stored games; ≈7,928 for the full season | 61,530; ≈79,280 | 3.3 to 6.6 h | 20 | A third NBA season | Nothing until step 9 passes: 2023-24 has no live Betfair or Polymarket close | **Buy after step 9 passes**, with NBA 2024-26; skip if it fails |
| 5. NBA spreads on the NBA 2024-26 calls | 16,117 (0 extra if requested with `h2h`) | 161,170 | none extra with `h2h`; 6.7 to 13.4 h alone | 20, after a code change | Retail spreads from William Hill, Ladbrokes, Coral, 888sport, LeoVegas and Matchbook | Nothing: no exchange spreads price, no spreads close in `odds_core.close.CLOSES`, and no spreads candidate | **Skip** |
| 6. The current season to 2026-12-01: EPL, La Liga, Serie A, Bundesliga, Ligue 1, the Championship and NBA (`odds-api-<league>-2026-27.yaml`, `odds-api-nba-2026-27.yaml`) | 11,313 to date; ≈13,849 to 2026-12-01, net of the NBA sample | 113,130 to date; ≈138,490 to 2026-12-01 | 1.2 to 2.5 h in 7 processes | 70 | The only pre-kick-off prices for the five leagues' 2026-27 games before 2026-11-20, their first stored 2026-27 event, and UK retail with the exchange in one snapshot for the 69 EPL games where live Odds API collection has none | Nothing in round 1, whose development and holdout blocks are fixed (ADR-0024). The data can be bought only this month, and FDUK publishes 2026-27 Betfair Exchange closes, so it is scoreable for later rounds | **Buy after a probe**, run last, with the window ending on the run's start day |
| 7. MLB KO−5 (`odds-api-mlb-ko5-2024-26.yaml`) | 2,240 | 22,400 | 56 to 112 min | 0 | An exchange price at most 4.4 min before first pitch on the 3,307 games without one | Only matters if step 9 sets a 5-minute cap on the substitute close: 99.3 to 99.4% of MLB games already have one within 10 min | **Skip** unless step 9's close needs the 5-minute cap; then buy with no probe |

Recommended: the NBA sample and item 6, about 140,800 units with their 70 probe units. Item 1 adds 367,770 with its probes, if the substitute close is extended. Item 4 adds about 79,300 if step 9 passes. That comes to about 587,870 units, against 3,737,940 left after NBA 2024-26.

## NBA sample readiness

The live Betfair close is the latest Betfair Exchange back price, not in-play, within 30 min before tip-off, whose overround is within the cap (`odds_core.close.is_live_close`, `odds_core.close.live_close_quotes`, ADR-0012). It is stored as `betfair` `observed` snapshots of `h2h`.

On 2026-11-13, 93 of 171 FINAL 2026-27 NBA games have one: every game from 2026-11-01 and 3 of the 8 on 2026-10-31 (UTC kick-off dates). The closes are 2.9 to 5.0 min old. With two outcomes per game, the rows per band of the devigged close are:

| Band | Rows | Rows a day, 2026-11-06 to 2026-11-12 | Reaches 50 |
|---|---|---|---|
| `<0.25` | 26 | 2.0 | about 2026-11-25 (2026-11-23 to 2026-11-27 at 1.7 to 2.4 a day) |
| 0.25–0.5 | 67 | 5.0 | reached |
| `≥0.5` | 93 | 7.0 | reached |

Games arrive at about 7.0 a day over the same week, and 28% of games have a `<0.25` row. Rows whose Odds API price is incomplete or over the overround cap drop out of the comparison, so the band may need a day or two more.

The sample's manifest plans the two grid snapshots before each tip-off (band 1 to 6 min, max age 5 min), so games tipping off in the same 5 minutes share calls. The planner costs only stored FINAL games before the season's end, set to 2026-11-26.

## Method

- **Stored fixtures.** `odds backfill plan --cost` gives items 3, 4, 7, EPL and NBA in item 6, and the NBA sample. `odds backfill execute --dry-run` gives item 7's calls net of stored ones.
- **Fixtures not stored.** No FINAL events are stored for the five leagues before 2024-25. Item 1 is costed by running `odds_analytics.backfill_plan.plan` on football-data.co.uk's fixture lists, with kick-offs read as UK time. On La Liga and the Championship 2024-26 and EPL 2026-27, this gives exactly the calls `plan --cost` gives on stored events. The five leagues' 2026-27 FINAL events to 2026-11-13 are stored (580 fixtures), and `plan --cost` on them reproduces item 6's figures.
- **Projections.** Item 6 to 2026-12-01 and the sample to 2026-11-26 add the stored SCHEDULED events up to those dates. NBA 2023-24's full-season figure is NBA 2024-25's (1,321 games, 7,928 calls).
- **Item 2.** Among the 5,098 FINAL fixtures of the six bought football manifests, every fixture without an FDUK or live Betfair close (626) has an Odds API exchange quote within 10 min of kick-off. That counts rows, with no overround check, because the five leagues' 2025-26 is sealed. Each of the 203 fixtures without an exchange quote within 5 min has a close or a quote within 10 min.
- **Item 7.** The executor makes 2,240 calls: 724 in 2024, 777 in 2025 and 739 in 2026. The audit's 2,231 (718, 777, 736) counts snapshot times not yet stored. The other 9 calls serve only doubleheader games whose listing the matcher could not attach, and the All-Star Game, so they would probably store nothing again `(inferred)`.

## Before buying

- **Load fixtures.** The planner and executor work only on stored events ([cost and execute a manifest](../wiki/collection/howto-backfill-odds-api-history.md#cost-and-execute-a-manifest)).
  - Item 1 needs the five leagues' FINAL events before 2024-25, loaded with `scripts/ingest_football_data_uk.py`. That is a production write, so back up first. Item 6 needs the 2026-27 fixtures played since the last load, loaded the same way.
  - Item 4 has 1,036 of about 1,320 2023-24 games stored, and December to April are thin. Load the rest with `odds discover games --sport basketball_nba`, which makes one historical events call per day on the NBA live key.
  - Rerun `plan --cost` once the fixtures are loaded.
- **Item 6's window.** Set each 2026-27 season's end to the run's start day.
- **Item 5.** `odds_analytics.backfill_plan.Manifest` accepts only `h2h` and `totals`, so a spreads manifest needs a code change before it can be probed or run. Bought with `h2h`, spreads cost 10 more units on each NBA call. Bought alone later, they cost the same units in a second set of calls.

## Probes

[#729](https://github.com/Acusick1/odds/issues/729) requires a passed probe before each full run. Each probe call costs 10 units.

| Item | Snapshots | Units | What it checks beyond units, grid and lag |
|---|---|---|---|
| 1 | 2023-04-01T09:55:38Z and 2023-09-16T09:55:38Z, per league | 100 | That each league has the exchange on `h2h` from 2023-04-01 as EPL does; legacy book keys; the fixture horizon (174 h, or 102 h for the Championship) |
| 3 | 2025-09-13T09:55:38Z, per league | 60 | That Pinnacle and William Hill price totals when named as the only books |
| 4 | 2023-12-16T15:55:38Z and 2024-03-16T15:55:38Z | 20 | That the exchange prices NBA in 2023-24 |
| 5 | One `h2h,spreads` call at 2025-01-18T15:55:38Z | 20 | Which `uk` books price NBA spreads historically, and whether the exchange does |
| 6 | 2026-10-24T09:55:38Z per football league; 2026-11-07T00:55:38Z for NBA | 70 | The 2026-27 books and horizon |
| NBA sample | 2026-11-07T00:55:38Z, the same call as item 6's NBA probe | 0 extra | — |
| 7 | none: the audited MLB run already shows its content, grid and exchange presence; the manifest's probe snapshot is not run | 0 | — |

- **Item 6's probe passed** (logs on the production host in `logs/backfill/probes-2026-27/`). EPL, La Liga, the Bundesliga, Ligue 1 and NBA pass every check. Two checks fail on behaviour the 2024-26 audits already record, so neither changes the manifests:
  - **Serie A, fixture horizon.** The farthest fixture is 130 h ahead, against 174 h, because the next round is midweek; the [Serie A audit](odds-api-serie-a-2024-26.audit.md) records rounds first listed after the previous one is played.
  - **Championship, exchange presence.** The exchange prices 11 of 12 events, all within 244 min of kick-off, and not the farthest; the [Championship audit](odds-api-championship-2024-26.audit.md) records the exchange in 77.7 to 84.7% of snapshots and on every fixture within 10 min of kick-off.
- **Probe calls that are reused.** The NBA probe snapshot is a planned call of both the sample and item 6's NBA manifest, and the football one is a planned call of the EPL, La Liga and Championship 2026-27 manifests, so a run after the probe skips those calls. A probe stores only games with a stored event, so load the fixtures first.
- **Totals.** All probes come to 270 units. The recommended package's 70 are spent; item 1 needs 100 more and item 4 20 more.
