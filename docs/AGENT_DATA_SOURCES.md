# Agent Data Sources

Data sources for the betting agent workflow, categorised by collection mode and current status. See [BETTING_AGENT.md](BETTING_AGENT.md) for the agent architecture and [ARCHITECTURE.md](ARCHITECTURE.md) for the core pipeline data sources.

## Scheduled Collection (always-on)

These feed the agent's DB-backed context. Collected on a schedule, available via MCP tools.

### Already built

| Source | What it provides | Status | Key files |
|--------|-----------------|--------|-----------|
| **OddsPortal** | Multi-bookmaker odds (bet365, betway, betfred, betvictor, bwin), opening + closing snapshots | Active, hourly on AWS Lambda | `jobs/fetch_oddsportal.py`, `oddsportal_common.py` |
| **The Odds API** | US bookmaker odds, live polling | Active but quota-limited (500 units/month) | `data_fetcher.py`, `jobs/fetch_odds.py` |
| **football-data.co.uk** | Historical EPL (11 seasons), Pinnacle + Betfair Exchange closing odds | Loaded, not live | `storage/fduk_writer.py` |
| **Polymarket** | Volume, orderbook depth, bid/ask imbalance, price velocity | Full pipeline built, **deprioritized** — EPL volume thin ($10-100K/match) and AMM-driven | See [POLYMARKET.md](POLYMARKET.md) |
| **ESPN fixtures/lineups** | All-competition fixture schedule, starting XI data | Models + reader/writer exist, manual ingestion scripts | `scripts/ingest_espn_fixtures.py`, `scripts/ingest_espn_lineups.py` |

### Viable, not yet built

| Source | What it provides | API | Free tier | Notes |
|--------|-----------------|-----|-----------|-------|
| **Smarkets** | Exchange orderbook + matched amounts, EPL markets | REST, `docs.smarkets.com` | Free with account | Thinner liquidity than Betfair (~1/10th) but includes volume data for free. Secondary exchange perspective. |

## On-Demand Agent Research (browse at evaluation time)

These are not collected on a schedule. The agent searches/browses them during checkpoint evaluation using web search and Playwright.

| Source | What it provides | How the agent uses it |
|--------|-----------------|----------------------|
| **BBC Sport** | Confirmed lineups, match previews, injury updates | Browse for team news at Checkpoint 2 |
| **Club websites / Twitter** | Official lineup announcements, press conference quotes | Primary source for lineup confirmation |
| **RotoWire** | Predicted + confirmed lineups (HTML) | Cross-reference lineup data |
| **Understat** | xG, shot maps, team form context | Match preview context at Checkpoint 1 |
| **Reddit r/soccer** | Match thread discussion, qualitative injury/team news | Supplementary context, sentiment signal |
| **OddsShark** | Consensus picks (direction) | Crude public sentiment proxy |

## Not Worth Pursuing

| Source | Why |
|--------|-----|
| **Betfair Exchange volume** | Requires £499 Live App Key + active betting. Free tier gives prices only, no volume. |
| **Twitter/X API** | $100-5000/month since 2023 pricing changes. Economically unviable. |
| **Action Network / BetQL** | US sportsbook data (FanDuel, DraftKings). Wrong market for EPL — the "public" they track isn't the public that moves EPL lines. |
| **Accumulator popularity** | No source publishes actual most-backed acca legs with volume data. Footy Accumulators etc. are tipster output, not crowd data. |
| **Tipster consensus** (Predictz, Forebet) | Model-driven predictions that lag sharp money. No signal for anticipating line movement. |
| **FPL API ownership/transfers** | Tested as XGBoost features (2026-03-20): R²=-0.017 with sliding-380, halved R² with expanding. Market already prices in squad availability. Transfer spikes lag actual news sources. Agent web search surfaces the same information faster. |
| **API-Football** | Free tier blocks current-season (2025-26) fixture lookup — restricted to seasons 2022-2024. Lineups endpoint works with any fixture ID but the fixtures search needed to obtain them is gated. Pro plan (~$20/month) required. PR #296 abandoned. |
| **football-data.org** | Different site from football-data.co.uk (which we use for historical data). Lineups paywalled at €29/month (Deep Data plan). Even paid, lineup timing is undocumented — no guarantee data populates pre-KO vs at match start. |
| **ESPN API (for lineups)** | ESPN summary endpoint populates lineups around KO-60min but data is **non-monotonic and unreliable** — lineups appeared at KO-60 then vanished by KO-45 in CL testing (2026-04-14). EPL test (2026-04-13) showed nothing at KO-59. Returns 403 from cloud/datacenter IPs. Not viable as a structured lineup source. |
| **Polymarket** | EPL match volume thin ($10-100K/match) and AMM-driven — orderbook reflects market maker parameters, not genuine public sentiment. Not accessible from UK. Full pipeline built if liquidity improves. |

## Lineup Data Source Conclusion (2026-04-13)

Every structured lineup API investigated is either paywalled, blocks current-season data on free tier, or has unconfirmed pre-KO timing. The agent's existing web search + Playwright is the best option: BBC Sport publishes confirmed lineups within minutes of the official team sheet (KO-60), and the agent can read those directly. No new infrastructure needed.

## Priority for Agent Workflow

The minimal viable agent needs only what's already built: OddsPortal odds in DB + web search + Playwright for on-demand research. All structured data source integrations investigated (Polymarket, FPL, API-Football, football-data.org, ESPN) have been ruled out for various reasons documented above.

Enhancement priority after the core loop is working:

1. **Smarkets integration** — secondary exchange volume signal, free with account. Only remaining viable data source not yet explored.
