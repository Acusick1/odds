"""Per-sport Polymarket discovery config."""

from __future__ import annotations

from dataclasses import dataclass
from re import Pattern, compile, escape

from odds_core.sports import SportKey


@dataclass(frozen=True)
class PolymarketSportConfig:
    sport_key: SportKey
    sport_title: str
    series_id: str
    tag_id: str | None
    ticker_prefix: str
    three_way_markets: bool
    market_key: str

    @property
    def match_slug(self) -> Pattern[str]:
        return compile(
            rf"^{escape(self.ticker_prefix)}-(?P<home>[a-z0-9]+)-(?P<away>[a-z0-9]+)-\d{{4}}-\d{{2}}-\d{{2}}$"
        )


POLYMARKET_SPORTS: dict[SportKey, PolymarketSportConfig] = {
    "basketball_nba": PolymarketSportConfig(
        sport_key="basketball_nba",
        sport_title="NBA",
        series_id="10345",
        tag_id="100639",
        ticker_prefix="nba",
        three_way_markets=False,
        market_key="h2h",
    ),
    "soccer_epl": PolymarketSportConfig(
        sport_key="soccer_epl",
        sport_title="EPL",
        series_id="10188",
        tag_id=None,
        ticker_prefix="epl",
        three_way_markets=True,
        market_key="1x2",
    ),
}

MATCH_ODDS_SPORTS: tuple[SportKey, ...] = tuple(
    key for key, cfg in POLYMARKET_SPORTS.items() if cfg.three_way_markets
)
