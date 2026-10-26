"""Bookmaker roles shared by validators, feature selection and the MCP tools."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from odds_core.snapshot_sources import SnapshotKind
from odds_core.vocabulary import PRICE_SOURCES, Source

EXCHANGE_BOOKMAKERS: frozenset[str] = frozenset(
    {"betfair_exchange", "betdaq", "smarkets", "matchbook"}
)
PREDICTION_MARKET_BOOKMAKERS: frozenset[str] = frozenset({"polymarket"})
# Venues that quote no meaningful overround.
ZERO_MARGIN_BOOKMAKERS: frozenset[str] = EXCHANGE_BOOKMAKERS | PREDICTION_MARKET_BOOKMAKERS
# Venues a UK bettor cannot bet on, so never a retail price to find edges against.
NON_BETTABLE_BOOKMAKERS: frozenset[str] = frozenset({"pinnacle", "polymarket"})

# Keys different sources use for one book, mapped to the key used everywhere else.
BOOKMAKER_ALIASES: dict[str, str] = {
    "betfair_ex_uk": "betfair_exchange",
    "betfair": "betfair_sportsbook",
    "betfair_sb_uk": "betfair_sportsbook",
    "betfred_uk": "betfred",
    "betmgmuk": "betmgm",
    "ladbrokes_uk": "ladbrokes",
    "sport888": "888sport",
    "7betuk": "7bet",
}
# UK-licensed sportsbooks a UK bettor can take a price from, by canonical key.
UK_RETAIL_BOOKMAKERS: frozenset[str] = frozenset(
    {
        "10bet",
        "7bet",
        "888sport",
        "allbritishcasino",
        "bet365",
        "betano",
        "betfair_sportsbook",
        "betfred",
        "betmgm",
        "betuk",
        "betvictor",
        "betway",
        "boylesports",
        "bwin",
        "coral",
        "ladbrokes",
        "midnite",
        "paddypower",
        "skybet",
        "spreadex",
        "unibet_uk",
        "williamhill",
    }
)


# The Odds API names US BetMGM ``betmgm``, the key OddsPortal and FDUK give BetMGM UK.
ODDS_API_KEY_ALIASES: dict[str, str] = {"betmgm": "betmgm_us"}


def canonical_bookmaker_key(key: str, kind: SnapshotKind) -> str:
    if kind is SnapshotKind.ODDS_API and key in ODDS_API_KEY_ALIASES:
        return ODDS_API_KEY_ALIASES[key]
    return BOOKMAKER_ALIASES.get(key, key)


# Odds API keys that differ from the pipeline's bookmaker keys, rewritten in its responses as
# they are fetched. Kept apart from the read-time maps above.
ODDS_API_BOOKMAKER_KEY_MAP: dict[str, str] = {
    "betfair_ex_uk": "betfair_exchange",
    "betfair_ex_eu": "betfair_exchange",
    "ladbrokes_uk": "ladbrokes",
    "betfred_uk": "betfred",
    "sport888": "888sport",
}

# Per source, every key a source names a book by mapped to the book's canonical key; price rows
# store only canonical keys.
BOOKMAKER_ALIASES_BY_SOURCE: Mapping[Source, Mapping[str, str]] = MappingProxyType(
    {
        source: MappingProxyType(
            {**BOOKMAKER_ALIASES, **ODDS_API_BOOKMAKER_KEY_MAP, **ODDS_API_KEY_ALIASES}
            if source is Source.ODDS_API
            else dict(BOOKMAKER_ALIASES)
        )
        for source in PRICE_SOURCES
    }
)


def canonical_bookmaker(source: Source, key: str) -> str:
    """The canonical key of the book ``source`` names ``key``."""
    return BOOKMAKER_ALIASES_BY_SOURCE[source].get(key, key)


def is_reference_only_snapshot(raw_data: dict[str, Any] | None) -> bool:
    """True when every bookmaker in the snapshot is a prediction market."""
    keys = {bm.get("key") for bm in (raw_data or {}).get("bookmakers", [])}
    return bool(keys) and keys <= PREDICTION_MARKET_BOOKMAKERS
