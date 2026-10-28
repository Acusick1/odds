"""Adapt Betfair Exchange ``BetfairBook`` into the envelope shape the writers store.

The pipeline's canonical bookmaker entry is::

    {
      "key": "betfair_exchange",
      "title": "Betfair Exchange",
      "last_update": "2026-04-25T14:35:00+00:00",
      "markets": [
        {
          "key": "1x2" | "h2h" | "totals",
          "outcomes": [{"name": <canonical team> | "Draw", "price_decimal": <best back>}, ...]
          # totals: [{"name": "Over" | "Under", "point": 2.5, "price_decimal": ...}, ...]
        }
      ],
      "betfair_meta": {  # optional, supplemental
        "market_id": "1.123",
        "market_status": "OPEN",
        "market_start_time": "2026-04-25T14:00:00+00:00",  # Betfair's, not the event's
        "inplay": false,
        "total_matched": 385702,
        "runners": [
            {"name": "Liverpool",
             "best_back": 1.64, "best_lay": 1.65,
             "back_size": 292, "lay_size": 314,
             "last_price_traded": 1.64}
        ],
      }
    }
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import structlog
from odds_core.odds_math import PRICE_DECIMAL_KEY
from odds_core.team import normalize_team_name
from odds_core.vocabulary import Market, PriceOutcome

from odds_lambda.betfair.client import BetfairBook, BetfairRunner
from odds_lambda.betfair.constants import BetfairMarketSpec, SportBetfairConfig

logger = structlog.get_logger()

DRAW_RUNNER_NAMES: frozenset[str] = frozenset({"the draw", "draw"})

# Betfair appends the probable pitchers to baseball event names ("(TBD)",
# "(Cole)") and a game number to doubleheaders ("Game 1", "- Game 2"), and cuts
# the name at a fixed length, so a half can end mid-word ("Kansas City Royal")
# or with an unclosed bracket ("Tampa Bay Rays (D Rasm"). Runner names are
# always the full team names, so they decide the teams; these strippers only
# serve a half no runner matches. They live here rather than in
# odds_core.team because trailing parens can legitimately disambiguate teams in
# other contexts (e.g. "(W)" for women's).
_PAREN_ANNOTATION_RE = re.compile(r"\s*\([^)]*\)?\s*$")
_GAME_SUFFIX_RE = re.compile(r"\s*(?:-\s*)?Game\s*\d*\s*$")
_GAME_NUMBER_RE = re.compile(r"\bGame\s*(\d+)\b")
_TOTALS_RUNNER_RE = re.compile(r"(over|under)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE)


def strip_pitcher_annotation(name: str) -> str:
    """Remove a trailing, possibly truncated, parenthetical (e.g. '(TBD)' or '(S Cecc') from a name."""
    return _PAREN_ANNOTATION_RE.sub("", name).strip()


def doubleheader_game_number(event_name: str) -> int | None:
    """The N of a doubleheader's "Game N" label in a Betfair event name, if it carries one."""
    match = _GAME_NUMBER_RE.search(event_name)
    return int(match.group(1)) if match else None


def clean_team_name(name: str) -> str:
    """Remove trailing pitcher annotations and doubleheader game suffixes."""
    previous = None
    while previous != name:
        previous = name
        name = _GAME_SUFFIX_RE.sub("", strip_pitcher_annotation(name)).strip()
    return name


def _normalize_runner_name(name: str) -> str:
    """Strip Betfair's pitcher annotation, then delegate to canonical normalization."""
    return normalize_team_name(strip_pitcher_annotation(name))


def _match_runner(half: str, team_runner_names: Sequence[str]) -> str | None:
    """The one runner a possibly annotated or truncated event-name half names."""
    lowered = half.strip().lower()
    cleaned = clean_team_name(half).lower()
    matches = [
        name
        for name in team_runner_names
        if lowered.startswith(name.lower()) or (cleaned and name.lower().startswith(cleaned))
    ]
    return matches[0] if len(matches) == 1 else None


def canonical_teams(home_half: str, away_half: str, runner_names: Sequence[str]) -> tuple[str, str]:
    """Canonical (home, away) for event-name halves, preferring the full runner names."""
    team_runner_names = [n for n in runner_names if n and not _is_draw(n)]
    home_runner = _match_runner(home_half, team_runner_names)
    away_runner = _match_runner(away_half, team_runner_names)
    if home_runner is not None and home_runner == away_runner:
        home_runner = away_runner = None
    return (
        normalize_team_name(home_runner or clean_team_name(home_half)),
        normalize_team_name(away_runner or clean_team_name(away_half)),
    )


def _is_draw(runner_name: str) -> bool:
    return runner_name.strip().lower() in DRAW_RUNNER_NAMES


def resolve_teams(
    book: BetfairBook,
    sport_cfg: SportBetfairConfig,
) -> tuple[str, str] | None:
    """Resolve (home_team, away_team) from the Betfair event name and runners.

    The event name only decides which team is home; the teams themselves come
    from the full runner names when each half matches exactly one of them.

    Soccer event names: ``"<Home> v <Away>"``.
    Baseball event names: ``"<Away> @ <Home>"`` (with optional pitcher
    annotations like ``"(TBD)"``).

    Returns canonical (home, away) pair, or None if parsing fails.
    """
    sep = sport_cfg.name_separator
    parts = [p.strip() for p in book.betfair_event_name.split(sep)]
    if len(parts) != 2:
        logger.warning(
            "betfair_unparseable_event_name",
            sport=sport_cfg.sport_key,
            betfair_event_id=book.betfair_event_id,
            name=book.betfair_event_name,
            separator=sep,
        )
        return None

    if sport_cfg.name_order == "home_first":
        home_raw, away_raw = parts
    else:
        away_raw, home_raw = parts
    return canonical_teams(home_raw, away_raw, [r.runner_name for r in book.runners])


def totals_runner(name: str) -> tuple[PriceOutcome, Decimal] | None:
    """The side and goal line of an over/under runner ("Over 2.5 Goals"), or None."""
    match = _TOTALS_RUNNER_RE.match(name.strip())
    if match is None:
        return None
    side = PriceOutcome.OVER if match.group(1).lower() == "over" else PriceOutcome.UNDER
    return side, Decimal(match.group(2))


def _team_outcome(
    runner_name: str, market: Market, home_team: str, away_team: str
) -> dict[str, Any]:
    if market is Market.ONE_X_TWO and _is_draw(runner_name):
        return {"name": "Draw"}
    canonical = _normalize_runner_name(runner_name)
    if canonical not in (home_team, away_team):
        # Fallback: use the canonical (may not equal home/away if
        # alias incomplete, but downstream sees a stable name).
        logger.warning(
            "betfair_runner_name_fallback",
            runner_name=runner_name,
            normalized=canonical,
            home_team=home_team,
            away_team=away_team,
        )
    return {"name": canonical}


def _runner_to_outcome(
    runner: BetfairRunner,
    spec: BetfairMarketSpec,
    home_team: str,
    away_team: str,
) -> dict[str, Any] | None:
    """Build one outcome dict from a Betfair runner.

    Skips runners with no best-back price, and over/under runners whose name carries no line
    or a line other than the spec's.
    """
    if runner.best_back is None:
        return None
    name_in = runner.runner_name or ""
    market = spec.market
    if market is Market.TOTALS:
        read = totals_runner(name_in)
        if read is None:
            logger.warning("betfair_totals_runner_unparsed", runner_name=name_in)
            return None
        side, line = read
        if line != spec.line:
            logger.warning(
                "betfair_totals_runner_off_line",
                runner_name=name_in,
                line=str(line),
                spec_line=str(spec.line),
            )
            return None
        outcome = {"name": side.value.capitalize(), "point": float(line)}
    else:
        outcome = _team_outcome(name_in, market, home_team, away_team)
    return {**outcome, PRICE_DECIMAL_KEY: runner.best_back}


def betfair_book_to_bookmaker_entry(
    book: BetfairBook,
    spec: BetfairMarketSpec,
    home_team: str,
    away_team: str,
    snapshot_time: datetime | None = None,
) -> dict[str, Any] | None:
    """Convert a ``BetfairBook`` of the ``spec`` market into a pipeline ``bookmakers[]`` entry.

    Returns None if the book has fewer usable outcomes than the market has runners.
    """
    snapshot_time = snapshot_time or datetime.now(UTC)
    outcomes: list[dict[str, Any]] = []
    for r in book.runners:
        out = _runner_to_outcome(r, spec, home_team, away_team)
        if out is not None:
            outcomes.append(out)

    if len(outcomes) < spec.runner_count:
        logger.warning(
            "betfair_book_missing_outcomes",
            market_id=book.market_id,
            market_type=book.market_type,
            got=len(outcomes),
            expected=spec.runner_count,
            event_name=book.betfair_event_name,
        )
        return None

    return {
        "key": "betfair_exchange",
        "title": "Betfair Exchange",
        "last_update": snapshot_time.isoformat(),
        "markets": [{"key": spec.market.value, "outcomes": outcomes}],
        "betfair_meta": {
            "market_id": book.market_id,
            "market_status": book.market_status,
            "market_start_time": (
                book.market_start_time.isoformat() if book.market_start_time else None
            ),
            "inplay": book.inplay,
            "total_matched": book.total_matched,
            "runners": [asdict(r) for r in book.runners],
        },
    }
