# MLB Support Implementation Plan

## Context

The pipeline infrastructure is largely sport-agnostic (Event model has `sport_key`, job routing has `_SPORT_SUFFIX_MAP`, storage accepts sport filters). Adding MLB requires: (1) teaching the OddsPortal scraper about MLB markets, (2) making the results job sport-aware, (3) wiring up job routing/config, (4) updating the MCP server to not be EPL-hardcoded, and (5) writing an MLB agent prompt. No new DB tables needed — `match_briefs` and `paper_trades` are already sport-agnostic via `event_id` FK.

## Key Decisions

- **Agent on-demand for everything except odds** — no scheduled pipelines for pitcher data, weather, lineups. Agent web searches at checkpoint time. Graduate to scheduled if volume warrants it.
- **One MCP server with sport parameters** — tools are functionally identical across sports. Sport-specificity lives in the agent prompt.
- **OddsPortal/OddsHarvester only** — no Odds API quota concerns.
- **Agent triages which games to research** — 5-15 MLB games/day, can't research all deeply. Later: scheduled jobs pre-filter interesting games.

## Implementation Steps

### Step 0: Verify OddsPortal MLB Coverage

Before building anything, verify via Playwright:
- Navigate to `https://www.oddsportal.com/baseball/usa/mlb/`
- Check: games listed, bookmakers present (Pinnacle? bet365? DraftKings?), market tabs
- Navigate to a specific match to see market structure
- Informs which bookmakers to configure as sharp/retail in the agent prompt

#### Step 0 Findings (2026-04-13)

**Games listed**: Yes. Full MLB slate displayed with correct times, team names, and fractional odds columns ("1" home, "2" away). 10+ games visible across today and tomorrow.

**Market tabs available** (on match detail page):
- 1X2 (regulation-time result — draw possible if tied after 9 innings before extras)
- **Home/Away** (primary moneyline market — this is what we use)
- Over/Under
- Asian Handicap
- European Handicap

**Bookmakers present** (17 total on match page):

| Bookmaker | Pipeline Key | Category |
|-----------|-------------|----------|
| bet365 | `bet365` | Retail (sharp-adjacent) |
| Betfair Exchange | `betfair_exchange` | **Sharp (exchange)** |
| William Hill | `williamhill` | Retail |
| Betway | `betway` | Retail |
| bwin | `bwin` | Retail |
| BetMGM | `betmgm` | Retail |
| Betfred | `betfred` | Retail |
| BetVictor | `betvictor` | Retail |
| Paddy Power | `paddypower` | Retail |
| Unibetuk | `unibet_uk` | Retail |
| Betano.uk | `betano` | Retail |
| BetUK | `betuk` | Retail |
| Midnite | `midnite` | Retail |
| SpreadEX | `spreadex` | Retail |
| 10bet | `10bet` | Retail |
| 7Bet | `7bet` | Retail |
| AllBritishCasino | `allbritishcasino` | Retail |

**Notable absences**: Pinnacle, DraftKings, FanDuel, Caesars, BetRivers, PointsBet, Bovada, 1xBet, Marathon Bet. The OddsPortal UK view filters to UK-licensed bookmakers only — US sportsbooks are not shown.

**Sharp reference situation**: Betfair Exchange is the **sole sharp reference** for MLB. Pinnacle is not available (shut down Jan 2026 and was not listed for MLB on OddsPortal UK view even before). Betfair Exchange shows Back/Lay odds with matched amounts (e.g., (109), (21) matched bets). Liquidity appears adequate for price discovery at the match level — needs monitoring over a larger sample.

**Betfair Exchange parsing issue**: OddsHarvester concatenates Back and Lay odds into one string: `"29/5029/50(65)"` instead of just the Back price `"29/50"`. The `oddsportal_adapter` converter will need to handle this — strip the Lay price and matched amount suffix. Same issue exists for EPL but was not noticed because the EPL pipeline uses `odds_history_data` (which stores decimal values separately), not the raw fractional string.

**Test scrape results**: OddsHarvester `run_scraper()` with `sport="baseball"`, `leagues=["mlb"]`, `markets=["home_away"]` returned **10 matches, 0 failures, 100% success rate**. Each match contains `home_away_market` with 17 bookmaker entries. Odds are fractional with keys `"1"` (home) and `"2"` (away), period `"FullIncludingOT"`. Venue data (stadium, city, country) is included.

**New bookmakers to add to `BOOKMAKER_KEY_MAP`**:
- `"7Bet"` -> `"7bet"`
- `"Paddy Power"` -> `"paddypower"`
- `"SpreadEX"` -> `"spreadex"`

**Structural differences from EPL**:
- 2-way market (Home/Away) instead of 3-way (1X2). No draw outcome.
- Betfair Exchange Back/Lay format in raw scrape data (EPL uses `odds_history_data` path instead).
- Odds labels are `"1"` and `"2"` instead of `"1"`, `"X"`, `"2"`.
- `period` is `"FullIncludingOT"` (includes extra innings).
- More games per day (5-15 vs 1-3 for EPL).
- URL structure uses `/baseball/h2h/` prefix instead of `/football/`.

**Implications for Steps 1-6**:
1. **Step 1** is confirmed viable — `_convert_home_away_match()` maps keys `"1"`/`"2"` to home/away outcomes. Must handle Betfair Exchange fractional concatenation.
2. **Step 2** league spec confirmed: `sport="baseball"`, `league="mlb"`, `markets=["home_away"]`. Overnight window: MLB games run ~17:00-04:00 UTC (US afternoon/evening), so `overnight_start_utc=5`, `overnight_resume_utc=14` is correct.
3. **Step 6** agent prompt: Betfair Exchange is the only sharp reference. bet365 is the best retail benchmark. No Pinnacle means no hybrid sharp reference — simpler configuration but less sharp signal than EPL.

### Step 1: OddsPortal Adapter — `home_away` Market Converter

**File:** `packages/odds-lambda/odds_lambda/oddsportal_adapter.py`

OddsHarvester's baseball `home_away` market uses `odds_labels=["1", "2"]` (from `sport_market_registry.py:362`). Scraped data has keys `"1"` (home) and `"2"` (away).

- Add `_convert_home_away_match()` — structurally identical to `_convert_1x2_match()` minus draw. 2 outcomes, market key `"h2h"`.
- Register: `_MARKET_CONVERTERS["home_away"] = _convert_home_away_match`
- Parameterize over/under point value (currently hardcoded to 2.5). Parse from market name: `"over_under_8_5"` → `8.5`.

### Step 2: LeagueSpec Extension + MLB Config

**File:** `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py`

Extend `LeagueSpec`:
- `primary_market: str = "1x2"` — market key for results/closing odds
- `num_outcomes: int = 3` — outcome count for primary market
- `overnight_start_utc: int = 22` / `overnight_resume_utc: int = 6` — per-sport overnight window

Add MLB spec:
```python
LeagueSpec(
    sport="baseball",
    league="mlb",
    sport_key="baseball_mlb",
    sport_title="MLB",
    markets=["home_away"],
    primary_market="home_away",
    num_outcomes=2,
    overnight_start_utc=5,
    overnight_resume_utc=14,
)
```

Parameterize `_apply_overnight_skip` to accept start/resume hours from spec.

### Step 3: Make Results Job Sport-Aware

**File:** `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py`

Currently hardcodes `SPORT_KEY = "soccer_epl"`, `MARKET_KEY = "1x2_market"`, etc.

- Import `_LEAGUE_SPEC_BY_SPORT` from `fetch_oddsportal`
- Look up spec from sport_key in `process_results()`
- Derive `market_key` and `num_outcomes` from spec
- Make `run_harvester_historic()` accept sport/league params

### Step 4: Job Routing & Config

1. `scheduling/jobs.py`: Add `"mlb": "baseball_mlb"` to `_SPORT_SUFFIX_MAP`
2. `config.py` AlertConfig: Add MLB heartbeat expectations
3. `daily_digest.py`: Add `"baseball_mlb": "MLB"` to `_SPORT_DISPLAY_NAMES`

### Step 5: MCP Server Updates

**File:** `packages/odds-mcp/odds_mcp/server.py`

1. Update server instructions to mention MLB
2. Fix `_resolve_sport_meta` fallback (currently assumes football)
3. Fix `refresh_scrape` hardcoded `sport="football"` — look up from `_LEAGUE_SPEC_BY_NAME`

### Step 6: Agent Prompt — AGENT_MLB.md

**File:** `packages/odds-mcp/AGENT_MLB.md` (new)

Self-contained MLB prompt mirroring AGENT.md structure:
- Same information-edge thesis
- MLB-specific tool guidance (league params, bookmaker lists)
- Data sources: MLB.com, ESPN, Baseball Reference, Fangraphs, RotoWire, r/sportsbook
- Two-checkpoint workflow adapted for daily MLB slate:
  - **Checkpoint 1 (morning ~14:00 UTC)**: Triage full slate, identify 3-5 interesting games, deep-research those
  - **Checkpoint 2 (~1h before first pitch)**: Verify starters confirmed, weather update, line movement, decide
- MLB-specific edge types: pitcher scratches, weather on totals, bullpen fatigue, public money loading, reverse line movement

### Step 7: Update `/agent` Command

**File:** `.claude/commands/agent.md`

Add sport routing: `mlb checkpoint1`, `mlb checkpoint2`, etc. Default remains EPL.

### Step 8: Tests

- Unit test for `_convert_home_away_match`
- Unit test for parameterized over/under point
- Unit test for MLB job routing
- Integration test for MLB league ingestion

## Deferred

- **MLB totals**: Start with moneyline only. Add totals after verifying OddsPortal line presentation.
- **ML features**: No feature extractors for MLB. Agent information-edge approach first.
- **Terraform/AWS**: Add MLB `sport_configs` when deploying to production.
- **NBA reactivation**: Same pattern — add LeagueSpec, agent prompt, sport suffix. After MLB validates multi-sport.
