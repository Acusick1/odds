# TODO

IMPORTANT: This doc is only for items to be done. Upon item completion, any necessary information should be persisted elsewhere, and the item removed completely from this doc.

## Betting Agent

Follow-up MLB MCP tools (planned after `get_probable_pitchers` ships in #368):

- [ ] `get_lineup_status` — surface confirmed-vs-projected lineup state and late scratches around first pitch. MLB Stats API exposes `/game/{gamePk}/boxscore` once lineups post (~2-4h pre-game); structure mirrors `get_probable_pitchers` (snapshot table, write-through MCP tool).
- [ ] `get_transactions_il` — recent transactions and IL moves (`/transactions`, `/teams/{id}/roster?rosterType=fullSeason`). Useful for catching late-breaking roster impact the agent currently web-searches for.
- [ ] `get_pitcher_stats` — season + last-N starts for a pitcher (`/people/{id}/stats?stats=gameLog,season,statsSingleSeason`). **Caveat**: FIP / xFIP / SIERA are Fangraphs-only and **not** in MLB Stats API; only standard stat lines (ERA, K/9, BB/9, WHIP, IP, etc.) are available. Document this in the docstring so we don't promise stats we can't deliver.
- [ ] `get_weather` — wind speed/direction and temperature for outdoor parks. No MLB Stats API equivalent; needs a separate weather provider integration (Open-Meteo or NOAA). Primarily a totals edge but moves moneylines at extreme conditions.
