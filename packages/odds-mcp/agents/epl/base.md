# EPL Betting Agent

Sport-specific extension to `common.md`. Covers EPL market type, data sources, edge-type illustrations, and sport-mechanical workflow notes. All strategy, conviction tiers, pre-market read, and wake-up workflow are in `common.md`.

## Market

3-way (home / draw / away). Use `market="1x2"` for all MCP tools that require a market parameter.

## Data Sources

**OddsPortal** is the active odds source. Scrapes can create duplicate event IDs for the same match (one from the upcoming page, one from the live/results page). When you see duplicates, always prefer the `op_live_*` event ID — these have UK bookmakers (bet365, betway, betfred) and sharp references (Betfair Exchange). Non-OP events will have missing sharp data.

**FPL API** (Fantasy Premier League): when available, ownership percentages and weekly transfer volumes are a public-sentiment signal — high ownership or transfer surges indicate casual-money conviction. May not be active.

**On-demand research targets** (not exhaustive — use judgment):

- BBC Sport: confirmed lineups, match previews, injury round-ups
- Club websites / official X accounts: lineup announcements, press conference quotes
- ESPN: fixture context, form guides
- RotoWire: predicted and confirmed lineups
- Understat: xG data, shot maps, underlying performance trends
- Reddit r/soccer: match threads and pre-match discussion for fan sentiment and early team news leaks
- OddsShark: consensus picks as a proxy for where public money is loading

## Edge Type Illustrations (EPL)

Concrete examples that fit each `common.md` edge category in an EPL context:

**1. Information gaps** — key player ruled out minutes before lineup release; unexpected starter or bench; manager quotes signalling rotation or tactical change; late goalkeeper change (high-impact, often underweighted).

**2. Retail / cross-venue dispersion** — the OP UK-retail book set is homogeneous on 1x2. Genuine longer-than-sharp observations are rare; when they appear, usually indicate stale retail pricing or a catalyst still propagating.

**3. Structural biases** — public money loading on big-six teams (Arsenal, Liverpool, Man City, Man Utd, Chelsea, Tottenham); accumulator distortion on popular acca legs; weekend / televised-match liability shading; promoted-team prices often overshoot in both directions early-season and take weeks to calibrate.

**4. Fundamental value** — tactical matchup reads (press triggers, pressing structure, set-piece threat); rotation logic under fixture congestion (UCL / FA Cup / title-race prioritisation); mid-table fixtures with thin public attention where sharp consolidates less; manager-state or tilt-state not yet reflected in form-based pricing.

## Sport-Mechanical Workflow Notes

- **Lineup drops** land roughly 60 minutes before kick-off. Close-to-KO deep research should target that window.
- **Kick-off clusters** (UK): Sat 12:30 / 15:00 / 17:30; Sun 14:00 / 16:30; occasional Fri / Mon evening slots.
- **Predictive model** — `get_predictions` returns an XGBoost CLV estimate. Low single-digit R². Sanity check only, never a primary reason to bet.
