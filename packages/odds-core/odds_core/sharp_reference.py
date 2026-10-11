"""Sharp bookmaker reference used by the betting agent (MCP tools and paper trading)."""

from __future__ import annotations

# Priority order: the first bookmaker quoting an outcome supplies its sharp price.
# Differs on purpose from the XGBoost model's Pinnacle-only DEFAULT_SHARP_BOOKMAKERS
# (odds_analytics.feature_extraction), which is a separate model setting.
AGENT_SHARP_BOOKMAKERS: tuple[str, ...] = ("pinnacle", "betfair_exchange")
