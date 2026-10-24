"""Adapt Betfair Exchange ``BetfairBook`` into pipeline ``raw_data`` shape.

The pipeline's canonical bookmaker entry is::

    {
      "key": "betfair_exchange",
      "title": "Betfair Exchange",
      "last_update": "2026-04-25T14:35:00+00:00",
      "markets": [
        {
          "key": "1x2" | "h2h",
          "outcomes": [{"name": <canonical team>, "price_decimal": <best back>}, ...]
        }
      ],
      "betfair_meta": {  # optional, supplemental
        "market_id": "1.123",
        "market_status": "OPEN",
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
from typing import Any

import structlog
from odds_core.odds_math import PRICE_DECIMAL_KEY
from odds_core.team import normalize_team_name

from odds_lambda.betfair.client import BetfairBook, BetfairRunner
from odds_lambda.betfair.constants import SportBetfairConfig

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


def _runner_to_outcome(
    runner: BetfairRunner,
    home_team: str,
    away_team: str,
    has_draw: bool,
) -> dict[str, Any] | None:
    """Build one outcome dict from a Betfair runner.

    Skips runners with no best-back price.
    """
    if runner.best_back is None:
        return None

    name_in = runner.runner_name or ""
    if has_draw and _is_draw(name_in):
        outcome_name = "Draw"
    else:
        canonical = _normalize_runner_name(name_in)
        if canonical == home_team:
            outcome_name = home_team
        elif canonical == away_team:
            outcome_name = away_team
        else:
            # Fallback: use the canonical (may not equal home/away if
            # alias incomplete, but downstream sees a stable name).
            logger.warning(
                "betfair_runner_name_fallback",
                runner_name=name_in,
                normalized=canonical,
                home_team=home_team,
                away_team=away_team,
            )
            outcome_name = canonical

    return {"name": outcome_name, PRICE_DECIMAL_KEY: runner.best_back}


def betfair_book_to_bookmaker_entry(
    book: BetfairBook,
    sport_cfg: SportBetfairConfig,
    home_team: str,
    away_team: str,
    snapshot_time: datetime | None = None,
) -> dict[str, Any] | None:
    """Convert a ``BetfairBook`` into a pipeline ``bookmakers[]`` entry.

    Returns None if the book has no usable outcomes (all prices missing).
    """
    snapshot_time = snapshot_time or datetime.now(UTC)
    outcomes: list[dict[str, Any]] = []
    for r in book.runners:
        out = _runner_to_outcome(r, home_team, away_team, sport_cfg.has_draw)
        if out is not None:
            outcomes.append(out)

    expected_outcomes = 3 if sport_cfg.has_draw else 2
    if len(outcomes) < expected_outcomes:
        logger.warning(
            "betfair_book_missing_outcomes",
            market_id=book.market_id,
            sport=sport_cfg.sport_key,
            got=len(outcomes),
            expected=expected_outcomes,
            event_name=book.betfair_event_name,
        )
        return None

    return {
        "key": "betfair_exchange",
        "title": "Betfair Exchange",
        "last_update": snapshot_time.isoformat(),
        "markets": [{"key": sport_cfg.market_key, "outcomes": outcomes}],
        "betfair_meta": {
            "market_id": book.market_id,
            "market_status": book.market_status,
            "inplay": book.inplay,
            "total_matched": book.total_matched,
            "runners": [asdict(r) for r in book.runners],
        },
    }


def build_raw_data(bookmaker_entry: dict[str, Any]) -> dict[str, Any]:
    """Wrap a single bookmaker entry into a snapshot ``raw_data`` payload."""
    return {"bookmakers": [bookmaker_entry], "source": "betfair_api"}
