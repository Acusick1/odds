# Football league coverage for the Odds API purchase

Research for [#775](https://github.com/Acusick1/odds/issues/775), on map [#774](https://github.com/Acusick1/odds/issues/774). Gathered 2026-11-05 at `d85fe8c1`. No paid Odds API call was made, no Betfair login was made, and production Postgres was read with `SELECT` only.

## Answer

| Rank | League | Betfair match-odds traded volume 1 min before KO (avg, £) | Est. O/U 2.5 (£) | FDUK usable rows 2024-26 | Planner units 2024-26 | Verdict |
|---|---|---|---|---|---|---|
| 1 | La Liga | 358,739 | ~27K | 729 of 760 | 291,060 | Include |
| 2 | Serie A | 289,941 | ~22K | 739 of 760 | 275,880 | Include |
| 3 | Bundesliga | 204,890 | ~15K | 594 of 612 | 186,560 | Include; cheapest of the top five |
| 4 | Ligue 1 | 148,999 | ~11K | 593 of 612 | 209,880 | Include |
| 5 | Championship | 42,010 | ~3K | 1,068 of 1,104 | 197,400 | Include if a large sample matters more than a deep close |
| 6 | Primeira Liga | 69,348 | ~5K | 591 of 612 | 241,640 | Marginal: more liquid than the Championship but the dearest per fixture |
| 7 | Eredivisie | 19,557 | ~1.5K | 593 of 612 | 206,760 | Skip |
| 8 | Scottish Premiership | 17,277 (row looks inconsistent) | ~1.3K | 443 of 456 | 149,960 | Skip |
| ref | EPL | 591,539 | 59K (measured, n=10) | 738 of 760 | 214,540 | Already in scope |

- The top four plus the Championship cost 1,160,780 units for 2024-26, and all eight candidates 1,759,140.
- Every cell marked `~` is an estimate: the match-odds volume scaled by 7.5%, the EPL's own O/U 2.5 to match-odds ratio (median of 10 matches).
- The one thing not verifiable without a paid call is whether the 10 books list each league. Spend about 160 units (one planner call per league, 20 units each) on the day-1 probe to settle it ([#729](https://github.com/Acusick1/odds/issues/729)).

## Betfair liquidity

### Match odds, by league

Source: [Betaminic, "Which Football Leagues Have the Most Liquidity on Betfair?"](https://www.betaminic.com/betting-research/which-football-leagues-have-the-most-liquidity-on-betfair/) (secondary, third-party). It records Betfair match-odds traded volume per match, 26 July to 25 September 2025, at 60 min and 1 min before kick-off and at full time.

| League (Betfair name) | 60 min avg | 1 min avg | 1 min low–high | Full time avg | 1 min avg vs EPL |
|---|---|---|---|---|---|
| English Premier League | 381,467 | 591,539 | 127,750–1,359,161 | 1,928,091 | 1.00 |
| Spanish La Liga | 172,648 | 358,739 | 67,772–1,260,027 | 1,477,574 | 0.61 |
| Italian Serie A | 136,500 | 289,941 | 77,010–961,364 | 970,824 | 0.49 |
| German Bundesliga | 80,606 | 204,890 | 28,254–393,259 | 1,198,239 | 0.35 |
| French Ligue 1 | 68,547 | 148,999 | 37,999–688,352 | 403,745 | 0.25 |
| Portuguese Primeira Liga | 26,013 | 69,348 | 5,113–254,462 | 458,961 | 0.12 |
| English Sky Bet Championship | 23,670 | 42,010 | 7,807–136,442 | 161,813 | 0.07 |
| Dutch Eredivisie | 11,063 | 19,557 | 6,537–62,631 | 192,208 | 0.033 |
| Scottish Premiership | 19,897 | 17,277 | 3,814–41,216 | 137,425 | 0.029 |

- **Eredivisie row:** the table labels it "Netherlands Eerste Divisie" under the Betfair name "Dutch Eredivisie", tier 1, and a separate row "Netherlands Eredivisie" under "Dutch Eerste Divisie", tier 2. I took the tier-1 row; the labels look swapped.
- **Scottish row:** its 60 min average is above its 1 min average and its full-time high (82,775) is below its average (137,425). Treat it as unreliable.
- **Calibration against our own data:** the EPL's last pre-play `betfair` match-odds snapshot (median 4 min before kick-off) has `total_matched` p10 £191K, median £620K, mean £796K, p90 £1.67M over 128 FINAL events (production `odds_snapshots`, `source='betfair'`, since 2026-04-25). That is the same order as Betaminic's EPL £592K, so its relative ranking is usable. It is not an independent check of the other leagues.
- Trading-blog figures of "£4–12M per EPL match" (not linked, conflicting and with an unstated metric) are about 10x either measure and are not used.

### Over/under 2.5

No source gives O/U 2.5 volume for any league but the EPL. We measured the EPL's:

| Metric | Value |
|---|---|
| Events with a close snapshot (collection began 2026-10-31) | 10 |
| O/U 2.5 `total_matched`, p10 / median / mean / p90 | £14,352 / £58,719 / £66,961 / £114,841 |
| Ratio to the same match's match-odds volume | 1.7%–34.5%, median 7.5%, aggregate 7.8% |

The `~` column in the answer table applies 7.5% to each league's 1 min match-odds average. The ratio is from 10 matches in one league and ranges from 1.7% to 34.5%, so a smaller league's O/U could be well under or over the estimate. A trader rule of thumb (unsourced, from trading blogs; not measured) says Betfair prices stop being a sharp signal below about £5K matched. On that rule the estimated Championship, Eredivisie and Scottish Premiership O/U 2.5 closes are unlikely to be usable.

### Competition ids

Not found in a primary source. The Betfair API returns them from `listCompetitions` with `eventTypeIds=["1"]` (free, but needs a login), which I did not run.

| League | Betfair competition name (from Betaminic) | Id | Support |
|---|---|---|---|
| La Liga | Spanish La Liga | 117 | Two unofficial sources: [abettor R docs](https://rdrr.io/github/phillc73/abettor/man/listCompetitions.html) and a [third-party GitHub PR](https://github.com/rianp7221-ship-it/rs-bet/pull/19). Unverified. |
| Serie A | Italian Serie A | 81 | The same PR only. Unverified. |
| Bundesliga | German Bundesliga | 59 | The same PR only. Unverified. |
| Ligue 1 | French Ligue 1 | not found | |
| Championship | English Sky Bet Championship | not found | |
| Primeira Liga | Portuguese Primeira Liga | not found | |
| Eredivisie | Dutch Eredivisie | not found | |
| Scottish Premiership | Scottish Premiership | not found | |

The EPL id in `odds_lambda.betfair.constants.EPL_COMPETITION_ID` (`10932509`) is the id the [Betfair data-scientists tutorial](https://betfair-datascientists.github.io/api/apiPythontutorial) prints for "English Premier League". The abettor docs that give La Liga as 117 give the EPL as 31, which is not the current id, so the 3-digit ids above may be stale. Pin ids only after a `listCompetitions` run.

`odds_lambda.betfair.constants.SportBetfairConfig.competition_id` is optional and `competition_name_hints` is the discovery fallback, so a missing id does not block a league. Use the full names above as hints; the bare "Championship" would also match the Scottish Championship.

## Odds API

### Sport keys

From the [Odds API sports page](https://the-odds-api.com/sports-odds-data/sports-apis.html) (all with the scores ✔):

| League | Sport key |
|---|---|
| Championship | `soccer_efl_champ` |
| La Liga | `soccer_spain_la_liga` |
| Serie A | `soccer_italy_serie_a` |
| Bundesliga | `soccer_germany_bundesliga` |
| Ligue 1 | `soccer_france_ligue_one` |
| Eredivisie | `soccer_netherlands_eredivisie` |
| Primeira Liga | `soccer_portugal_primeira_liga` |
| Scottish Premiership | `soccer_spl` |

### The 10 books

All ten keys exist on the [bookmaker page](https://the-odds-api.com/sports-odds-data/bookmaker-apis.html): `betfair_ex_uk`, `williamhill`, `paddypower`, `skybet`, `ladbrokes_uk`, `coral`, `betfred_uk`, `betvictor`, `sport888` in the `uk` region, and `pinnacle` in the `eu` region. Pinnacle is named through `bookmakers`, which the API honours whatever the region ([sport-and-market-keys](../../wiki/collection/sport-and-market-keys.md#odds-api-scope)).

Not verifiable without a paid call:

- **Which of the 10 list each league, for h2h and totals.** The docs have no per-league matrix and say "Some bookmakers may not list for less popular sports". Discovering it needs an odds call, a historical call, or a free API key's `/odds` response; I made none.
- **Since when a league is listed.** The [v4 guide](https://the-odds-api.com/liveapi/guides/v4/) gives one start for all sports: 2020-06-06 at 10-minute snapshots, 5-minute from September 2022. 2024-26 is inside it. Per-league start dates are not documented, and I infer, without evidence, that the five biggest leagues have been listed throughout.

Indirect evidence only: OddsPortal (below) prices all eight leagues at 18 books each, including 888sport, BetVictor, Betfred, Paddy Power, Sky Bet and William Hill, and the FDUK 2025-26 files carry closing prices for BetVictor and Betfred on about 97% of rows and for Coral and Ladbrokes on about 70% in every league. That shows the books price these leagues, not that the Odds API lists them. OddsPortal does not serve Pinnacle, Coral or Ladbrokes to a GB caller.

## FDUK

All eight leagues use the main layout (`https://football-data.co.uk/mmz4281/<season>/<code>.csv`, one file per season, 119–132 columns, `Time` in UK time: Spain's modal 20:00 slot is a 21:00 local kick-off). Codes: Championship `E1`, La Liga `SP1`, Serie A `I1`, Bundesliga `D1`, Ligue 1 `F1`, Eredivisie `N1`, Primeira `P1`, Scottish Premiership `SC0`, EPL `E0`. Every league has the same header within a season, and `E1` has the same as `E0`. None needs the "extra" layout.

"Usable" = `BFECH`, `BFECD`, `BFECA`, `BFEC>2.5` and `BFEC<2.5` all populated. `BFECH` and `BFEC>2.5` counts were equal in every file except EPL 2025-26.

| League | 2024-25 rows | `BFECH` | `BFEC>2.5` | 2025-26 rows | `BFECH` | `BFEC>2.5` | Usable total |
|---|---|---|---|---|---|---|---|
| EPL | 380 | 380 | 380 | 380 | 358 | 358 | 738 |
| Championship | 552 | 552 | 552 | 552 | 516 | 516 | 1,068 |
| La Liga | 380 | 380 | 380 | 380 | 349 | 349 | 729 |
| Serie A | 380 | 380 | 380 | 380 | 359 | 359 | 739 |
| Bundesliga | 306 | 306 | 306 | 306 | 288 | 288 | 594 |
| Ligue 1 | 306 | 306 | 306 | 306 | 287 | 287 | 593 |
| Eredivisie | 306 | 306 | 306 | 306 | 287 | 287 | 593 |
| Primeira | 306 | 306 | 306 | 306 | 285 | 285 | 591 |
| Scottish Premiership | 228 | 227 | 227 | 228 | 216 | 216 | 443 |

- **2025-26 is complete** (the last match is in May 2026) but every league's missing close rows are in April and May 2026 only. FDUK may fill them later (not checked).
- **Pinnacle closes (`PSCH`) stop after January 2026** in every league (Scottish: October 2025), so only about half of 2025-26 has one. Not needed for the Betfair-close target, but a Pinnacle comparison is limited to 2024-25 plus the first half of 2025-26.
- **Layout change at 2025-26:** William Hill (`WH*`), the 1XBet (`1XB*`) and Betfair Sportsbook (`BF*`) columns are dropped; BetMGM, BetVictor (`BV*`), Coral (`CL*`), Ladbrokes (`LB*`) and Betfred (`BFD*`) are added.
- `BFE*` is the Betfair Exchange, the `BFEC*` columns its closing odds ([FDUK notes](https://football-data.co.uk/notes.txt): closing columns add a `C` after the book code).

## OddsPortal

League pages exist for all eight and parse with `odds_lambda.oddsportal_client.parse_listing`; one upcoming match per league was fetched with `OddsPortalClient` for `1x2` and `over_under_2_5` and returned 17–18 books each, including a Betfair Exchange row:

| League | `league` slug (proposed) | `listing_path` | Upcoming events on 2026-11-05 |
|---|---|---|---|
| Championship | `england-championship` | `/football/england/championship/` | 12 |
| La Liga | `spain-laliga` | `/football/spain/laliga/` | 21 |
| Serie A | `italy-serie-a` | `/football/italy/serie-a/` | 20 |
| Bundesliga | `germany-bundesliga` | `/football/germany/bundesliga/` | 18 |
| Ligue 1 | `france-ligue-1` | `/football/france/ligue-1/` | 18 |
| Eredivisie | `netherlands-eredivisie` | `/football/netherlands/eredivisie/` | 18 |
| Primeira Liga | `portugal-liga-portugal` | `/football/portugal/liga-portugal/` | 9 |
| Scottish Premiership | `scotland-premiership` | `/football/scotland/premiership/` | 13 |

The slugs follow the EPL's `england-premier-league`; the spec's `league` string is ours, only `listing_path` is OddsPortal's. A scrape needs no code beyond a `LeagueSpec` with `markets=["1x2", "over_under_2_5"]`, both of which are in `FEED_MARKETS`.

## Collection effort

Steps are those of [howto-add-sport](../../wiki/collection/howto-add-sport.md). Effort figures are my estimates, not measured.

**Common work, once for all leagues**

- `odds_core.sports.SportKey` is a three-value `Literal`. About 35 files mention `soccer_epl`, and several key dicts by `SportKey` (`LEAGUE_TIMEZONES`, `OVERNIGHT_WINDOWS`, `SPORT_CONFIG`, `_SPORT_SUFFIX_MAP`). Widening it is one line, but whether a league is a new sport key or a `league` field on soccer is a design question for [#779](https://github.com/Acusick1/odds/issues/779).
- `odds_core.team._TEAM_ALIASES` is one flat, sport-blind map of lowercased alias to canonical name, with the canonical set being the Odds API's short forms. Every new league adds its teams there, and any name shared across leagues (the same club spelling, or a name an EPL team already uses) cannot be mapped per league.
- The FDUK loader is EPL-only (`scripts/ingest_football_data_uk.py`: `SPORT_KEY`, `E0.csv`); [#781](https://github.com/Acusick1/odds/issues/781) generalises it. FINAL events with scores come from there, since `fetch-oddsportal-results` is EPL-only.

**Per league**

| Step | Work | Size |
|---|---|---|
| 1–3 `SportKey`, suffix, overnight window | one entry each; an EPL-like `(22, 6)` window fits European kick-offs | trivial |
| 4 `LeagueSpec` | slug and path above; markets exist | trivial |
| 5 `SPORT_CONFIG` | `MATCH_ODDS` and `OVER_UNDER_25` like the EPL; id optional | small, id needs a login |
| 6 Polymarket | optional, skip | none |
| 7 team aliases | distinct teams in FDUK 2024-26: Championship 30, La Liga 23, Serie A 23, Bundesliga 20, Ligue 1 21, Eredivisie 21, Primeira 20, Scottish 14; the four sources (Odds API, OddsPortal, Betfair, FDUK) spell clubs differently | the bulk of the work: 20–30 names per league, each checked against four sources |
| 8 job allowlists | `fetch-odds` lists EPL and NBA and `fetch-scores` EPL only; add each league that forward-tests | small per league |
| 9 Terraform | skip while AWS is dormant | none |
| 10 heartbeats | per job per league | small, mechanical |

- **For the purchase alone**, a league needs steps 1 (the sport key), 7 (aliases, so historical events attach to fixtures) and the FDUK loader. Steps 2–6, 8 and 10 are live collection, which map #774 puts out of scope.
- **Per-league cost is similar across leagues.** Only the alias count differs; the Championship has the most (30) and the Scottish Premiership the fewest (14).

## Planner units, 2024-26

Variant A bands (10 min to KO-2h, 1 h to KO-1d, 6 h to KO-7d), `h2h` + `totals` 2.5, the 10 named books (20 units a call). Fixtures come from the FDUK files (kick-off at UK time, converted to UTC); seasons are `[2024-07-01, 2025-07-01)` and `[2025-07-01, 2026-07-01)`, as in `backfills/odds-api-epl-2020-26.yaml`. The planner is `odds_analytics.backfill_plan.plan`, called from a script because `Manifest.sport` is a three-value literal and the manifest is validated unchanged with `soccer_epl`; the sport does not enter the plan.

| League | Fixtures 24-25 / 25-26 | Calls 24-25 / 25-26 | Units 24-25 | Units 25-26 | Units total |
|---|---|---|---|---|---|
| EPL | 380 / 380 | 5,251 / 5,476 | 105,020 | 109,520 | 214,540 |
| Championship | 552 / 552 | 5,083 / 4,787 | 101,660 | 95,740 | 197,400 |
| La Liga | 380 / 380 | 7,245 / 7,308 | 144,900 | 146,160 | 291,060 |
| Serie A | 380 / 380 | 6,877 / 6,917 | 137,540 | 138,340 | 275,880 |
| Bundesliga | 306 / 306 | 4,609 / 4,719 | 92,180 | 94,380 | 186,560 |
| Ligue 1 | 306 / 306 | 5,324 / 5,170 | 106,480 | 103,400 | 209,880 |
| Eredivisie | 306 / 306 | 5,241 / 5,097 | 104,820 | 101,940 | 206,760 |
| Primeira Liga | 306 / 306 | 6,038 / 6,044 | 120,760 | 120,880 | 241,640 |
| Scottish Premiership | 228 / 228 | 3,615 / 3,883 | 72,300 | 77,660 | 149,960 |

- **Cross-check:** the script's EPL figures (5,251 and 5,476 calls) equal `odds backfill plan --manifest <EPL 2024-26> --cost` run against the production EPL FINAL events, and its La Liga and Championship figures match the map's estimates (~144K and ~97K per season).
- **Cost follows kick-off spread, not fixture count.** The Championship's 552 fixtures cost less than La Liga's 380, and 246 Championship matches kick off at 15:00 against 61 La Liga matches at 15:15 (inferred: the planner shares one call between fixtures with overlapping decision times, so clustered kick-offs are cheap).
- **Estimates:** FDUK fixture lists stand in for FINAL events not yet loaded; counts will shift slightly with the loaded events (postponements, rearranged dates).

## Reproduce

```sh
# FDUK (one request every 4 s; non-www host)
curl -s https://football-data.co.uk/mmz4281/2526/SP1.csv   # codes E1 SP1 I1 D1 F1 N1 P1 SC0, seasons 2425 2526

# EPL cross-check on production events (SELECT only; needs DATABASE_URL)
uv run odds backfill plan --manifest <EPL manifest trimmed to 2024-25 and 2025-26> --cost
```

The per-league plan is `odds_analytics.backfill_plan.plan(manifest, fixtures)`, with `Fixture(i, kickoff_utc)` built from each FDUK row's `Date` and `Time` (`Europe/London`), and `Manifest.model_validate` of the EPL manifest with only the two seasons.

## Open items

- Run `listCompetitions` for the competition ids (free, needs the Betfair login).
- Probe each league's 10-book coverage with one planner call per league on day 1 (160 units).
- Measure O/U 2.5 volume for the non-EPL leagues once Betfair collection covers them; until then the `~` column is an estimate.
