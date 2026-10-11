"""Sharp bookmaker reference used by the betting agent (MCP tools and paper trading)."""

from __future__ import annotations

# Priority order, highest first: earlier bookmakers are preferred over later ones.
# Differs on purpose from the XGBoost model's Pinnacle-only DEFAULT_SHARP_BOOKMAKERS
# (odds_analytics.feature_extraction), which is a separate model setting.
AGENT_SHARP_BOOKMAKERS: tuple[str, ...] = ("pinnacle", "betfair_exchange")
