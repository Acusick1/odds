# Live collection for the five new football leagues

Research for [#818](https://github.com/Acusick1/odds/issues/818). Gathered 2026-11-10 at `6bc5c065` on the production host. Builds on [football-league-coverage.md](https://github.com/Acusick1/odds/blob/research/league-coverage/docs/research/football-league-coverage.md) (branch `research/league-coverage`), which gave liquidity, FDUK and planner costs. Read-only throughout: no fetch job ran, production Postgres was read with `SELECT` only, no Odds API call was made. The Betfair probes logged in with the production cert (separate session, read-only `listCompetitions`, `listEvents`, `listMarketTypes`, `listMarketCatalogue`). OddsPortal and ESPN were read with plain GETs: about 330 OddsPortal requests in 7 minutes, with no 429 or challenge.

2026-11-10 is an international break, so every probe saw the 20-22 November round (and Friday 20 November) as the next fixtures.

## Answer

| | Championship | La Liga | Serie A | Bundesliga | Ligue 1 |
|---|---|---|---|---|---|
| SportKey (registered, #779) | `soccer_efl_champ` | `soccer_spain_la_liga` | `soccer_italy_serie_a` | `soccer_germany_bundesliga` | `soccer_france_ligue_one` |
| OddsPortal `listing_path` | `/football/england/championship/` | `/football/spain/laliga/` | `/football/italy/serie-a/` | `/football/germany/bundesliga/` | `/football/france/ligue-1/` |
| Upcoming events listed | 12 | 21 | 20 | 18 | 18 |
| Full scrape, 1x2 + totals | 29 s | 51 s | 48 s | 43 s | 44 s |
| Retail books, `1x2` and totals, next-round event (EPL: 11) | 8 | 10 | 10 | 7 | 8 |
| Betfair Exchange row, `1x2` and totals | 12 of 12 events | 8 of 20 | 8 of 20 | 9 of 18 | 9 of 18 |
| Betfair competition id | `7129730` | `117` | `81` | `59` | `55` |
| Betfair name | English Sky Bet Championship | Spanish La Liga | Italian Serie A | German Bundesliga | French Ligue 1 |
| `MATCH_ODDS` and `OVER_UNDER_25` on each listed event | yes (12) | yes (8) | yes (8) | yes (9) | yes (9) |
| ESPN scoreboard slug | `eng.2` | `esp.1` | `ita.1` | `ger.1` | `fra.1` |
| Betfair spellings with no alias | none | none seen | none seen | `Mgladbach`, `Schalke 04` | `ESTAC Troyes`, `Paris St-G` |

- **Blocker for all five: no schedule feed, so the collectors would stay dormant.** See [Fixture source](#fixture-source). Adding a league to `LEAGUE_SPECS` and `SPORT_CONFIG` alone collects nothing.
- Everything else carries over: the OddsPortal markets, the Betfair job's event matching and its totals 2.5 collection work unchanged.

## OddsPortal

### League pages and books

Each listing parsed with `odds_lambda.oddsportal_client.parse_listing`. `scrape_league` (read-only, no database) was run on all five with `markets=["1x2", "over_under_2_5"]`: zero errors, one snapshot per market per event.

- **Every market of a scraped event carries a Betfair Exchange row** where Betfair has the market open: all 12 Championship events, and 8 to 9 events per other league, which is exactly the number of events Betfair lists in its next-14-day window ([Betfair](#betfair)). Events later than that have no exchange row yet and fewer retail books (3-7 against 7-10), so a decision at KO-60 or KO-30, when Betfair's market is long open, is the case that matters.
- **Retail books on the next-round event (EPL Manchester City v Fulham: 11):** Championship 8, La Liga 10, Serie A 10, Bundesliga 7, Ligue 1 8, counted against `odds_core.bookmakers.UK_RETAIL_BOOKMAKERS`. One event per league, so a sample. Against the EPL's set: the Championship and Ligue 1 lack Betfred and Betway, Bundesliga lacks BetVictor, Betway and Betano, and La Liga and Serie A lack only Spreadex.
- **No William Hill, Coral or Ladbrokes in any league, EPL included**: OddsPortal does not serve them to a GB caller (the earlier research found the same).
- The OddsPortal Betfair row sits in the same `1x2` or `over_under_2_5` snapshot as the retail rows, so the candidate loader's rule of one snapshot carrying both ([fetch-oddsportal-job](../../wiki/collection/fetch-oddsportal-job.md#runs-before-candidate-decisions)) holds. Unverified: that the exchange row's price matches the Betfair job's `betfair` snapshot in each league; only the EPL has been compared.

### Scrape load and Cloudflare

`scrape_league` reads the whole listing every run (two feed requests per upcoming event, 1 s apart; no horizon limit), so a run costs 2N requests for N listed events: 29-51 s per league, against an EPL run of about a minute ([fetch-oddsportal-job](../../wiki/collection/fetch-oddsportal-job.md#runs-before-candidate-decisions)).

- **Measured EPL load:** 34 to 44 distinct OddsPortal runs on matchdays (7-8 November), 14 to 18 on quiet days (`odds_snapshots`, source `oddsportal`, distinct minute).
- **Estimated load with six leagues** (estimate, not measured): the five leagues carry 87 events against the EPL's 20, so a matchday is about 35 runs per league at 40-45 requests, roughly 1,500 requests per league and about 9,000 a day across six, against about 1,900 for the EPL today. Average 0.1 requests per second, but runs of one job's sibling sports run concurrently ([local-scheduler](../../wiki/scheduling/local-scheduler.md#scheduling-primitives)), each with its own 1 s delay.
- **Kick-off clustering at the candidates' decision times.** The leads before KO-60 and KO-30 (`SCRAPE_LEAD` 5 min) for Saturday 21 November, from the listings' kick-off slots (UTC):

| League | Slots | Lead runs |
|---|---|---|
| EPL | 12:30, 15:00, 17:30 | 6 |
| Championship | 12:30, 15:00 | 4 |
| La Liga | 13:00, 15:15, 17:30, 20:00 | 8 |
| Serie A | 14:00, 17:00, 19:45 | 6 |
| Bundesliga | 14:30, 17:30 | 4 |
| Ligue 1 | 16:15, 19:45 | 4 |

  That is 32 lead runs against the EPL's 6. Four leagues (EPL, La Liga, Serie A, Bundesliga) start a lead run in the same minute at 16:25, and EPL, Championship and Bundesliga share 13:55. Up to four concurrent scrapers from one IP is the peak; each run still fits the 5-minute lead (about 50 s).
- **Cloudflare:** none seen in 330 requests over 7 minutes, including five back-to-back full scrapes. That does not test sustained concurrent weekend load, which is unverified. If it matters, the one change the code supports cheaply is a scrape horizon (events within a day or two of kick-off), since far-out events have no exchange row and fewer books, and make up about half of every run's requests. That is a design change to `scrape_league`, not a per-league config.

## Betfair

Event type `1`. `listCompetitions` for football returned 86 competitions; ids below are from that call (verified 2026-11-10), replacing the unverified 3-digit guesses of the earlier research for La Liga, Serie A and Bundesliga (confirmed `117`, `81`, `59`) and filling Ligue 1 and the Championship:

| League | Competition id | Name | Markets in 14 days |
|---|---|---|---|
| English Premier League (existing) | `10932509` | English Premier League | 10 |
| Championship | `7129730` | English Sky Bet Championship | 12 |
| La Liga | `117` | Spanish La Liga | 8 |
| Serie A | `81` | Italian Serie A | 8 |
| Bundesliga | `59` | German Bundesliga | 9 |
| Ligue 1 | `55` | French Ligue 1 | 9 |

- Similar names exist and must not be pinned by hint: `French Premiere Ligue` (`12702178`, the women's league), `Scottish Championship` (`107`), `English National League` (`11086347`).
- **Market types:** every listed event in all five leagues carries `MATCH_ODDS` (runners home, away, "The Draw") and `OVER_UNDER_25` (runners "Under 2.5 Goals", "Over 2.5 Goals"). `OVER_UNDER_05`, `BOTH_TEAMS_TO_SCORE`, `CORRECT_SCORE`, `ASIAN_HANDICAP` and `ALT_TOTAL_GOALS` exist too and the job ignores them.
- **Event names** are "Home v Away" with `" v "`, the same as the EPL's (`name_separator` and `name_order` defaults). Betfair's `open_date` equals OddsPortal's kick-off for every event compared (for example Athletic Bilbao v Espanyol 20:00, Nice v Paris St-G 16:15 on 2026-11-20 and 21).
- **The job carries over unchanged.** A `SPORT_CONFIG` entry per league is the EPL's with a different `sport_key`, `sport_title`, `competition_id` and name hints (`match_odds=BetfairMarketSpec("MATCH_ODDS", Market.ONE_X_TWO)`, `other_markets=(BetfairMarketSpec("OVER_UNDER_25", Market.TOTALS, Decimal("2.5")),)`). The `fetch-betfair-exchange` allowlist is derived from `SPORT_CONFIG`, so no `JOB_SPECS` edit is needed. Use the full names as hints (not "Championship" or "Ligue 1" alone).
- **Cadence and lookahead:** Betfair's market for a fixture was open 10 to 14 days ahead in all six leagues. The default `lookahead_hours` of 168 would have listed no event today in any league (nothing before 20 November), which is harmless because the close run is what matters.
- **Liquidity** (from the earlier research, secondary source): La Liga £359K, Serie A £290K, Bundesliga £205K, Ligue 1 £149K, Championship £42K matched at 1 minute before kick-off; the estimated O/U 2.5 closes are a tenth of that or less, and about £3K for the Championship. The EPL's measured O/U 2.5 median is £59K.

### Team-name gaps

Betfair and OddsPortal spellings were normalised with `odds_core.team.normalize_team_name` and compared with the team names of each sport's stored events (2024-26) and with each other.

| League | Betfair name | Normalises to | OddsPortal name normalises to |
|---|---|---|---|
| Bundesliga | `Mgladbach` | `Mgladbach` | `Borussia Monchengladbach` |
| Bundesliga | `Schalke 04` | `Schalke 04` | `Schalke` |
| Ligue 1 | `Paris St-G` | `Paris St-G` | `Paris Saint Germain` |
| Ligue 1 | `ESTAC Troyes` | `ESTAC Troyes` | `Troyes` |

Four missing aliases, all on the Betfair side. Without them a Betfair event does not match an OddsPortal event of the same fixture (`soccer_*` events match on normalised names within ±2 h), so the match would split into two events. Promoted clubs with no history (Bolton, Lincoln, Frosinone, Elversberg, Paderborn, Le Mans, Malaga, Racing Santander) pass through unchanged and agree across the two sources. OddsPortal's `Dep. A Coruna` has no match in history and no Betfair counterpart was listed that day; unverified. Only a 21-day window was compared, so more gaps are likely as later rounds list.

## Fixture source

**Why it blocks.** `odds_lambda.collection_calendar.published_schedule_kickoff` reads the ESPN fixtures table for `soccer_epl`, the Stats API schedule for MLB, and returns `None` for any other sport. The stored events of the five leagues are all FINAL (production `events`, 612 to 1,104 each, none SCHEDULED). A collector with no fixture within 28 days books a 6-hour check-in and does nothing ([dormancy](../../wiki/scheduling/decision-engine.md#dormancy)), and collection is what creates the events, so the five leagues would never start. NBA hit the same wall and has `odds_lambda.espn_nba_schedule` (#725), which creates events from ESPN's scoreboard.

What can feed it, checked live 2026-11-10:

| Source | Result | Verdict |
|---|---|---|
| ESPN scoreboard, `https://site.api.espn.com/apis/site/v2/sports/soccer/<slug>/scoreboard?dates=YYYYMMDD` | `eng.2`, `esp.1`, `ita.1`, `ger.1`, `fra.1` all return the 21 and 22 November rounds with kick-offs equal to OddsPortal's and Betfair's, final scores and statuses (`STATUS_FULL_TIME`). A `dates=` range returns HTTP 400 (as the existing `fetch_month_scoreboard` notes); a day or `YYYYMM` works. Needs `Accept-Encoding` handling (the response is gzip). | Recommended. Free, same pattern as the NBA precedent, gives results too. |
| OddsPortal listing | Start times and event ids for the next 12-21 matches. | Possible but the same site the scraper hits; one extra request per league per dormancy check. |
| Betfair `listEvents` | Event names and `open_date` for the next 10-14 days only. | Too short a horizon for dormancy, and needs a login. |
| FDUK `fixtures.csv` | One file for all leagues (`Div`, `Date`, `Time`, teams). On 2026-11-10 it still lists the 6-8 November round, not 20-22 November. | Stale; not a schedule. |
| Odds API `/events` | Free in quota terms (documented; the 2026-11-05 test comment in `tests/unit/test_team_normalization.py` lists clubs from it). Not called here. | Unverified this session; fetch-odds is EPL and NBA only. |

Unverified design points: ESPN is not in `odds_lambda.storage.event_kickoff.KICKOFF_AUTHORITY` for these leagues (`_FOOTBALL_ORDER` is OddsPortal, Odds API, FDUK), so a postponement that only ESPN reports would not move an event, and ESPN would need adding to the order (as it is for NBA). ESPN team names (`displayName`) were not compared with the alias map.

## What a league costs to add

Registration is done ([#779](https://github.com/Acusick1/odds/issues/779), [#795](https://github.com/Acusick1/odds/pull/795)): `SportKey`, timezone, kick-off authority, `CLOSES`, candidate markets and the 2024-26 team aliases. What is missing, following [howto-add-sport](../../wiki/collection/howto-add-sport.md):

| Step | Per league | Once |
|---|---|---|
| `LeagueSpec` (slug, path, `markets=["1x2", "over_under_2_5"]`) | 6 lines | |
| `SPORT_CONFIG` entry | 8 lines, id above | |
| `_SPORT_SUFFIX_MAP` | 1 line (needs five new suffixes) | |
| `OVERNIGHT_WINDOWS` | 1 line; the EPL's `(22, 6)` clears the latest kick-off (20:00 UTC) | |
| Heartbeats | 2 entries (`fetch-oddsportal-<suffix>`, `fetch-betfair-exchange-<suffix>`) | |
| `data_collection.sports` (environment) | add the sport key; `fetch-oddsportal` has no allowlist, so it runs for each | |
| Betfair aliases | 2 each for Bundesliga and Ligue 1, others none | |
| Schedule feed | one ESPN slug each | one generalised football sync modelled on `espn_nba_schedule`, a new job allowlist and heartbeat, and the authority-order decision above |
| Tests and wiki pages for the touched symbols | small | `wiki_lint` will list them |

- **Size:** per league about an hour of mechanical work, once the schedule feed exists. The schedule feed is the real task, about a day, with the authority order and the matching of ESPN names to the alias map to settle.
- **Runtime cost** is the load figures above: roughly 30-50 s and 40-45 requests per OddsPortal run, a Betfair login and a handful of requests per Betfair run (cadence 15 min under 3 h, 5 min close run), per league per matchday, and more concurrent scrapers on Saturday afternoons.
- **Cheapest order** if adding fewer than five: La Liga and Serie A (highest liquidity, most retail books), then Bundesliga and Ligue 1 (alias fixes needed), then the Championship (most fixtures, 12 per round, and the least exchange liquidity).

## Reproduce

- OddsPortal: `odds_lambda.jobs.fetch_oddsportal.scrape_league(LeagueSpec(league=..., listing_path=..., sport_key=..., sport_title=..., markets=["1x2", "over_under_2_5"]))` reads and writes nothing to the database.
- Betfair: `odds_lambda.betfair.client.client_from_env()`, `login()`, then `list_competitions("1")` and `trading.betting.list_events` / `list_market_types` / `list_market_catalogue` with `competition_ids=[<id>]`.
- ESPN: `curl -s --compressed "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard?dates=20261121"`.
