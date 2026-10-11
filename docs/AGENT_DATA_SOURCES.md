# Agent Data Sources

The data sources the betting agent uses, by collection mode, and the sources that were
evaluated and rejected. See [BETTING_AGENT.md](BETTING_AGENT.md) for the agent architecture
and [ARCHITECTURE.md](ARCHITECTURE.md#data-sources) for the core pipeline data sources.

## Scheduled collection

Collected by scheduled jobs or ingestion scripts into the database, and available to the
agent through MCP tools. Where each job runs is in the
[Job Deployment Matrix](DEPLOYMENT.md#job-deployment-matrix).

| Source | What it provides | Collection | Key files |
|--------|-----------------|------------|-----------|
| **OddsPortal** | Multi-bookmaker UK odds (bet365, Betway, Betfred, BetVictor, bwin and others in `BOOKMAKER_KEY_MAP`), upcoming and closing snapshots | Self-scheduling by kickoff proximity (30 min–2 h) | `jobs/fetch_oddsportal.py`, `oddsportal_common.py` |
| **Betfair Exchange** | Best back/lay Match Odds (sharp reference) | Self-scheduling by kickoff proximity | `jobs/fetch_betfair_exchange.py`, `betfair/` |
| **The Odds API** | US bookmaker odds | Self-scheduling by fetch tier; quota-limited (500 units/month per free key) | `data_fetcher.py`, `jobs/fetch_odds.py` |
| **football-data.co.uk** | Historical EPL (11 seasons), Pinnacle + Betfair Exchange closing odds | Manual ingestion | `scripts/ingest_football_data_uk.py` |
| **ESPN fixtures** | All-competition EPL fixture schedule | Daily job | `jobs/fetch_espn_fixtures.py` (also `scripts/ingest_espn_fixtures.py`) |
| **ESPN lineups** | Starting XI data | Manual ingestion | `scripts/ingest_espn_lineups.py` |
| **MLB Stats API** | MLB probable pitchers | Daily job | `jobs/fetch_mlb_probables.py` |
| **Polymarket** | Volume, order book depth, bid/ask imbalance, price velocity | Pipeline built, deprioritized | See [POLYMARKET.md](POLYMARKET.md) |

Paths under `jobs/` and `betfair/` are relative to `packages/odds-lambda/odds_lambda/`.

## On-demand agent research

Not collected on a schedule. The agent searches and browses these during each wake-up using
web search and Playwright.

| Source | What it provides | How the agent uses it |
|--------|-----------------|----------------------|
| **BBC Sport** | Confirmed lineups, match previews, injury updates | Team news, lineup confirmation close to kickoff |
| **Club websites / Twitter** | Official lineup announcements, press conference quotes | Primary source for lineup confirmation |
| **RotoWire** | Predicted + confirmed lineups (HTML) | Cross-reference lineup data |
| **Understat** | xG, shot maps, team form context | Match preview context |
| **Reddit r/soccer** | Match thread discussion, qualitative injury/team news | Supplementary context, sentiment signal |
| **OddsShark** | Consensus picks (direction) | Crude public sentiment proxy |

## Evaluated and rejected

| Source | Why |
|--------|-----|
| **Betfair Exchange volume** | Requires a £499 Live App Key and active betting. The free tier gives prices only, no volume. |
| **Twitter/X API** | $100–5,000/month under the 2023 pricing. Economically unviable. |
| **Action Network / BetQL** | US sportsbook data (FanDuel, DraftKings). The "public" they track is not the public that moves EPL lines. |
| **Accumulator popularity** | No source publishes actual most-backed acca legs with volume data. Footy Accumulators and similar are tipster output, not crowd data. |
| **Tipster consensus** (Predictz, Forebet) | Model-driven predictions that lag sharp money; no signal for anticipating line movement. |
| **FPL API ownership/transfers** | As XGBoost features (2026-03-20), R² = −0.017 under sliding-380 and halved R² under expanding (see [MODELING.md](MODELING.md#epl-feature-group-ablation)). Transfer spikes lag the news sources the agent searches directly. |
| **API-Football** | The free tier blocks current-season (2025-26) fixture lookup, restricting it to 2022–2024. The lineups endpoint needs fixture IDs from the gated search. The Pro plan (~$20/month) is required. |
| **football-data.org** | A different site from football-data.co.uk. Lineups are paywalled at €29/month (Deep Data plan), and lineup timing is undocumented even when paid. |
| **ESPN API (for lineups)** | The summary endpoint populates lineups around KO-60 but is non-monotonic: lineups appeared at KO-60 then vanished by KO-45 in Champions League testing (2026-04-14), and an EPL test (2026-04-13) showed nothing at KO-59. Returns 403 from cloud/datacenter IPs. |
| **Polymarket** | EPL match volume is thin ($10–100K/match) and AMM-driven, so the order book reflects market maker parameters, not public sentiment. Not accessible from the UK. |

## Lineup data (2026-04-13)

Every structured lineup API investigated is paywalled, blocks current-season data on its
free tier, or has unconfirmed pre-KO timing. The agent's web search and Playwright are the
lineup source: BBC Sport publishes confirmed lineups within minutes of the official team
sheet (KO-60), and the agent reads them directly.
