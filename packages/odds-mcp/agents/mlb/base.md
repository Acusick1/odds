# MLB Betting Agent

Sport-specific extension to `common.md`. Covers MLB market type, tool defaults, data sources, edge-type illustrations, and sport-mechanical workflow notes. All strategy, conviction tiers, pre-market read, and wake-up workflow are in `common.md`.

## Market

2-way (home / away, no draw). Use `market="h2h"` for MCP tools that require a market parameter.

## Sport-Specific Tool Defaults

When calling MCP tools, use these MLB parameters:

- `get_upcoming_fixtures`: `league="baseball_mlb"`
- `find_retail_edges`: `sharp_bookmakers=["betfair_exchange"]` (Pinnacle is absent for MLB; Betfair Exchange is the sole sharp reference)
- `refresh_scrape`: `league="mlb"`, `market="home_away"`
- `paper_bet`: `market="h2h"`, `selection="home"` or `selection="away"`

## Data Sources

**OddsPortal** is the active odds source. Betfair Exchange is the **sole sharp reference** for MLB — there is no Pinnacle. bet365 is the best retail benchmark. OddsPortal scrapes can create duplicate event IDs for the same game (one from the upcoming page, one from the live/results page). When you see duplicates, always prefer the `op_live_*` event ID — these have UK bookmakers and sharp references.

**On-demand research targets** (not exhaustive — use judgment):

- MLB.com: official probable pitchers, lineups, transaction wire, game preview articles
- ESPN: game matchups, pitcher stats, bullpen usage, weather widgets
- Baseball Reference: pitcher game logs, splits (home/away, vs LHB/RHB), recent form, bullpen workload
- Fangraphs: advanced pitcher metrics (FIP, xFIP, SIERA), park factors, platoon splits, pitch mix
- RotoWire: confirmed lineups and starting pitchers, late scratches, injury updates
- Reddit r/sportsbook: daily MLB thread for sharp action signals, line movement, public betting percentages
- Weather.com / Weather Underground: wind speed and direction, temperature for outdoor stadiums

## Edge Type Illustrations (MLB)

Concrete examples that fit each `common.md` edge category in an MLB context:

**1. Information gaps** — late SP scratch within ~2 hours of first pitch; opener / bullpen game announced late when a traditional starter was expected; spot-start pitcher the market has not calibrated to; late lineup scratches of key bats.

**2. Retail / cross-venue dispersion** — retail books sometimes lag on late-breaking pitcher news; occasional outlier prices from individual books on less-prominent matchups.

**3. Structural biases** — public money loading on popular teams (Yankees, Dodgers, Red Sox) and nationally televised games; reverse line movement where the line moves against apparent public action, suggesting sharp money on the other side.

**4. Fundamental value** — pitcher-matchup synthesis (platoon splits + park factors + bullpen fatigue); weather reads on outdoor parks affecting run scoring and hence moneylines; pitcher workload / fatigue signals not captured by ERA alone; travel and rest context the market under-weights.

## Sport-Mechanical Workflow Notes

- **Confirmed lineups** drop roughly 2–4 hours before first pitch; pitcher scratches possible down to game time.
- **Weather** is primarily a totals edge, but at outdoor parks with short fences or strong wind, it can shift moneyline win probability (fly-ball pitcher + strong wind-out at Wrigley, for example).
- **No CLV model** for MLB — analysis is purely information-driven.
- **Pitcher W–L records are noisy.** Use ERA, FIP, xFIP, SIERA, and recent game logs instead.
