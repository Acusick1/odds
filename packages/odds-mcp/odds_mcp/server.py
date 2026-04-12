"""FastMCP server exposing betting pipeline tools for AI agents.

Thin wrappers over existing DB queries, jobs, and paper trading logic.
All tools are stateless and use async_session_maker() for DB access.
"""

import math
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import structlog
from fastmcp import FastMCP
from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import TabularFeatureExtractor
from odds_analytics.sequence_loader import extract_odds_from_snapshot
from odds_analytics.utils import calculate_implied_probability
from odds_core.database import async_session_maker
from odds_core.match_brief_models import BriefCheckpoint, MatchBrief, SharpPriceMap
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade
from odds_core.prediction_models import Prediction
from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPECS
from odds_lambda.paper_trading import (
    get_open_trades,
    get_portfolio_summary,
    place_trade,
    settle_trades,
)
from odds_lambda.storage.readers import OddsReader
from sqlalchemy import func, select

logger = structlog.get_logger()

mcp = FastMCP(
    "odds-mcp",
    instructions=(
        "Betting odds pipeline tools. Use these to inspect fixtures, odds, predictions, "
        "and manage paper trades for EPL football. All times are UTC."
    ),
)

# Build league lookup from canonical LEAGUE_SPECS
_LEAGUE_SPEC_BY_NAME: dict[str, Any] = {spec.league: spec for spec in LEAGUE_SPECS}

# Hybrid sharp reference matching production defaults.
# Duplicated from feature_extraction.py DEFAULT_SHARP_BOOKMAKERS / DEFAULT_RETAIL_BOOKMAKERS
# with EPL-specific overrides — keep in sync.
_DEFAULT_SHARP_BOOKMAKERS = ["pinnacle", "betfair_exchange"]
_DEFAULT_RETAIL_BOOKMAKERS = ["bet365", "betway", "betfred"]


def _event_to_dict(event: Event) -> dict[str, Any]:
    """Serialize an Event to a JSON-safe dict."""
    return {
        "id": event.id,
        "sport_key": event.sport_key,
        "sport_title": event.sport_title,
        "home_team": event.home_team,
        "away_team": event.away_team,
        "commence_time": event.commence_time.isoformat(),
        "status": event.status.value,
        "home_score": event.home_score,
        "away_score": event.away_score,
        "completed_at": event.completed_at.isoformat() if event.completed_at else None,
    }


def _odds_to_dict(odds: Odds) -> dict[str, Any]:
    """Serialize an Odds object to a JSON-safe dict."""
    return {
        "bookmaker_key": odds.bookmaker_key,
        "bookmaker_title": odds.bookmaker_title,
        "market_key": odds.market_key,
        "outcome_name": odds.outcome_name,
        "price": odds.price,
        "point": odds.point,
    }


def _snapshot_to_dict(
    snapshot: OddsSnapshot,
    *,
    include_raw_data: bool = False,
    extracted_odds: list[Odds] | None = None,
) -> dict[str, Any]:
    """Serialize an OddsSnapshot to a JSON-safe dict.

    Args:
        snapshot: The snapshot to serialize.
        include_raw_data: If True, include the full raw_data blob (for single-snapshot views).
        extracted_odds: If provided, include structured odds instead of raw_data.
    """
    result: dict[str, Any] = {
        "id": snapshot.id,
        "event_id": snapshot.event_id,
        "snapshot_time": snapshot.snapshot_time.isoformat(),
        "created_at": snapshot.created_at.isoformat(),
        "bookmaker_count": snapshot.bookmaker_count,
        "fetch_tier": snapshot.fetch_tier,
        "hours_until_commence": snapshot.hours_until_commence,
    }
    if include_raw_data:
        result["raw_data"] = snapshot.raw_data
    if extracted_odds is not None:
        result["odds"] = [_odds_to_dict(o) for o in extracted_odds]
    return result


def _trade_to_dict(trade: PaperTrade) -> dict[str, Any]:
    """Serialize a PaperTrade to a JSON-safe dict."""
    return {
        "id": trade.id,
        "event_id": trade.event_id,
        "market": trade.market,
        "selection": trade.selection,
        "bookmaker": trade.bookmaker,
        "odds": trade.odds,
        "stake": trade.stake,
        "reasoning": trade.reasoning,
        "confidence": trade.confidence,
        "bankroll_before": trade.bankroll_before,
        "bankroll_after": trade.bankroll_after,
        "placed_at": trade.placed_at.isoformat() if trade.placed_at else None,
        "settled_at": trade.settled_at.isoformat() if trade.settled_at else None,
        "result": trade.result.value if trade.result else None,
        "pnl": trade.pnl,
    }


def _safe_float(val: Any) -> float | None:
    """Convert numpy/float values, returning None for NaN."""
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


def _resolve_sport_meta(league: str) -> tuple[str, str]:
    """Map league name to pipeline sport_key and sport_title using LEAGUE_SPECS."""
    spec = _LEAGUE_SPEC_BY_NAME.get(league)
    if spec is not None:
        return spec.sport_key, spec.sport_title
    return f"football_{league.replace('-', '_')}", league.title()


@mcp.tool()
async def get_upcoming_fixtures(
    league: str = "soccer_epl",
    days_ahead: int = 7,
) -> list[dict[str, Any]]:
    """Get upcoming scheduled fixtures for a league.

    Args:
        league: Sport key (e.g. "soccer_epl").
        days_ahead: How many days ahead to look (1-30).

    Returns:
        List of fixture dicts with id, teams, commence_time, status.
    """
    days_ahead = max(1, min(days_ahead, 30))
    now = datetime.now(UTC)
    end = now + timedelta(days=days_ahead)

    async with async_session_maker() as session:
        reader = OddsReader(session)
        events = await reader.get_events_by_date_range(
            start_date=now,
            end_date=end,
            sport_key=league,
            status=EventStatus.SCHEDULED,
        )

    return [_event_to_dict(e) for e in events]


@mcp.tool()
async def get_current_odds(
    event_id: str,
    include_raw_data: bool = False,
) -> dict[str, Any]:
    """Get the latest odds snapshot for an event, showing current bookmaker prices.

    Args:
        event_id: Event identifier.
        include_raw_data: If True, also include the full raw_data JSON blob.

    Returns:
        Dict with event info and the latest snapshot with structured odds.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        snapshot = await reader.get_latest_snapshot(event_id, market="h2h")
        if snapshot is None:
            return {
                "event": _event_to_dict(event),
                "snapshot": None,
                "message": "No odds snapshots available for this event",
            }

    odds = extract_odds_from_snapshot(snapshot, event_id, market="h2h")
    return {
        "event": _event_to_dict(event),
        "snapshot": _snapshot_to_dict(
            snapshot, include_raw_data=include_raw_data, extracted_odds=odds
        ),
    }


@mcp.tool()
async def get_odds_history(event_id: str) -> dict[str, Any]:
    """Get the full odds movement timeline for an event (all snapshots).

    Returns structured bookmaker odds per snapshot instead of raw JSON blobs
    to keep response size manageable.

    Args:
        event_id: Event identifier.

    Returns:
        Dict with event info and chronologically ordered list of snapshots.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        snapshots = await reader.get_snapshots_for_event(event_id)

    serialized = []
    for s in snapshots:
        odds = extract_odds_from_snapshot(s, event_id, market="h2h")
        serialized.append(_snapshot_to_dict(s, extracted_odds=odds))

    return {
        "event": _event_to_dict(event),
        "snapshot_count": len(snapshots),
        "snapshots": serialized,
    }


@mcp.tool()
async def refresh_scrape(
    league: str = "england-premier-league",
    market: str = "1x2",
) -> dict[str, Any]:
    """Trigger an on-demand OddsPortal scrape for a league and market.

    This runs the full scrape-and-ingest pipeline: scrapes OddsPortal,
    converts odds, matches/creates events, and stores snapshots.

    Args:
        league: OddsHarvester league name (e.g. "england-premier-league").
        market: Market to scrape (e.g. "1x2", "over_under_2_5").

    Returns:
        Dict with ingestion statistics (matches scraped, events matched, etc).
    """
    from odds_lambda.jobs.fetch_oddsportal import LeagueSpec, ingest_league

    sport_key, sport_title = _resolve_sport_meta(league)

    spec = LeagueSpec(
        sport="football",
        league=league,
        sport_key=sport_key,
        sport_title=sport_title,
        markets=[market],
    )

    try:
        stats = await ingest_league(spec)
    except Exception as e:
        logger.error("refresh_scrape_failed", league=league, error=str(e), exc_info=True)
        return {"error": str(e), "error_type": type(e).__name__}

    return {
        "league": stats.league,
        "matches_scraped": stats.matches_scraped,
        "matches_converted": stats.matches_converted,
        "events_matched": stats.events_matched,
        "events_created": stats.events_created,
        "snapshots_stored": stats.snapshots_stored,
        "errors": stats.errors,
    }


@mcp.tool()
async def get_predictions(
    event_id: str,
    limit: int = 5,
    since_hours: float | None = None,
) -> dict[str, Any]:
    """Get pre-scored CLV predictions for an event (most recent first).

    Reads predictions stored by the scoring pipeline (not on-demand inference).
    Returns up to `limit` predictions, newest first. Use `since_hours` to
    restrict to recent predictions only.

    Args:
        event_id: Event identifier.
        limit: Max predictions to return, newest first. Clamped to 1 minimum. Default 5.
        since_hours: If set, only return predictions created in the last N hours.

    Returns:
        Dict with event info, total matching count, and the limited list.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        query = select(Prediction).where(Prediction.event_id == event_id)

        if since_hours is not None:
            cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
            query = query.where(Prediction.created_at >= cutoff)

        # Count matching rows (respects since_hours filter)
        count_query = query.with_only_columns(func.count(Prediction.id))
        count_result = await session.execute(count_query)
        total_matching = count_result.scalar() or 0

        query = query.order_by(Prediction.created_at.desc()).limit(max(1, limit))
        pred_result = await session.execute(query)
        predictions = list(pred_result.scalars().all())

    return {
        "event": _event_to_dict(event),
        "total_matching": total_matching,
        "returned": len(predictions),
        "predictions": [
            {
                "id": p.id,
                "snapshot_id": p.snapshot_id,
                "model_name": p.model_name,
                "model_version": p.model_version,
                "predicted_clv": p.predicted_clv,
                "created_at": p.created_at.isoformat(),
            }
            for p in predictions
        ],
    }


@mcp.tool()
async def get_event_features(
    event_id: str,
    outcome: Literal["home", "away"] = "home",
    sharp_bookmakers: list[str] | None = None,
    retail_bookmakers: list[str] | None = None,
) -> dict[str, Any]:
    """Get tabular features for an event's latest snapshot.

    Extracts tabular features only (bookmaker odds, implied probabilities,
    market consensus) plus hours_until_event. Does not include other feature
    groups (standings, schedule, match_stats) which require additional data
    sources not available via this tool.

    Args:
        event_id: Event identifier.
        outcome: Which outcome to extract features for ("home" or "away").
        sharp_bookmakers: Sharp bookmaker keys (default: ["pinnacle", "betfair_exchange"]).
        retail_bookmakers: Retail bookmaker keys (default: ["bet365", "betway", "betfred"]).

    Returns:
        Dict with event info and feature name/value pairs.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        snapshot = await reader.get_latest_snapshot(event_id, market="h2h")
        if snapshot is None:
            return {
                "event": _event_to_dict(event),
                "features": None,
                "message": "No snapshots available for feature extraction",
            }

    outcome_name = event.home_team if outcome == "home" else event.away_team

    try:
        features = _extract_features_for_event(
            event,
            snapshot,
            outcome_name=outcome_name,
            sharp_bookmakers=sharp_bookmakers or _DEFAULT_SHARP_BOOKMAKERS,
            retail_bookmakers=retail_bookmakers or _DEFAULT_RETAIL_BOOKMAKERS,
        )
    except Exception as e:
        logger.error("feature_extraction_failed", event_id=event_id, error=str(e), exc_info=True)
        return {
            "event": _event_to_dict(event),
            "features": None,
            "message": f"Feature extraction failed: {e}",
        }

    return {
        "event": _event_to_dict(event),
        "snapshot_time": snapshot.snapshot_time.isoformat(),
        "features": features,
    }


def _extract_features_for_event(
    event: Event,
    snapshot: OddsSnapshot,
    *,
    outcome_name: str,
    sharp_bookmakers: list[str],
    retail_bookmakers: list[str],
) -> dict[str, float | None]:
    """Extract feature dict from an event and its latest snapshot."""
    odds = extract_odds_from_snapshot(snapshot, event.id, market="h2h")
    if not odds:
        return {"error": "No h2h odds data in snapshot"}

    backtest_event = BacktestEvent(
        id=event.id,
        commence_time=event.commence_time,
        home_team=event.home_team,
        away_team=event.away_team,
        home_score=0,
        away_score=0,
        status=event.status,
    )

    extractor = TabularFeatureExtractor(
        sharp_bookmakers=sharp_bookmakers,
        retail_bookmakers=retail_bookmakers,
    )
    tab_feats = extractor.extract_features(
        event=backtest_event,
        odds_data=odds,
        outcome=outcome_name,
        market="h2h",
    )

    feature_names = extractor.get_feature_names()
    feature_array = tab_feats.to_array()

    features = {
        f"tab_{name}": _safe_float(val)
        for name, val in zip(feature_names, feature_array, strict=True)
    }

    hours_until = (event.commence_time - snapshot.snapshot_time).total_seconds() / 3600
    features["hours_until_event"] = round(hours_until, 2)

    return features


@mcp.tool()
async def paper_bet(
    event_id: str,
    market: str,
    selection: str,
    odds: int,
    stake: float,
    reasoning: str = "",
    bookmaker: str = "best_available",
    confidence: float | None = None,
    bankroll: float | None = None,
) -> dict[str, Any]:
    """Place a paper (simulated) bet on an event.

    Args:
        event_id: Event identifier.
        market: Market type (e.g. "h2h").
        selection: "home", "draw", or "away".
        odds: American odds at time of placement (e.g. -110, +150).
        stake: Stake amount.
        reasoning: Why this bet is being placed.
        bookmaker: Bookmaker key (default "best_available").
        confidence: Agent confidence score (0.0 to 1.0, optional).
        bankroll: Current bankroll. If omitted, computed from portfolio state.

    Returns:
        Dict with the placed trade details.
    """
    async with async_session_maker() as session:
        event_result = await session.execute(select(Event).where(Event.id == event_id))
        event = event_result.scalar_one_or_none()
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        if event.status != EventStatus.SCHEDULED:
            return {"error": f"Event is '{event.status.value}', not 'scheduled'. Cannot place bet."}

        if bankroll is None:
            portfolio = await get_portfolio_summary(session)
            bankroll = portfolio.current_bankroll

        try:
            trade = await place_trade(
                session,
                event_id=event_id,
                market=market,
                selection=selection,
                bookmaker=bookmaker,
                odds=odds,
                stake=stake,
                bankroll=bankroll,
                reasoning=reasoning or None,
                confidence=confidence,
            )
            await session.commit()
        except ValueError as e:
            return {"error": str(e), "error_type": "ValueError"}
        except Exception as e:
            logger.error("paper_bet_failed", event_id=event_id, error=str(e), exc_info=True)
            return {"error": str(e), "error_type": type(e).__name__}

    return {
        "status": "placed",
        "trade": _trade_to_dict(trade),
    }


@mcp.tool()
async def get_portfolio(initial_bankroll: float = 1000.0) -> dict[str, Any]:
    """Get paper trading portfolio summary: open bets, P&L, bankroll.

    Args:
        initial_bankroll: Starting bankroll for ROI calculation.

    Returns:
        Dict with portfolio metrics and list of open trades.
    """
    async with async_session_maker() as session:
        portfolio = await get_portfolio_summary(session, initial_bankroll=initial_bankroll)
        open_trades = await get_open_trades(session)

    return {
        "current_bankroll": portfolio.current_bankroll,
        "total_trades": portfolio.total_trades,
        "settled_trades": portfolio.settled_trades,
        "open_trades": portfolio.open_trades,
        "total_pnl": portfolio.total_pnl,
        "total_staked": portfolio.total_staked,
        "roi": round(portfolio.roi, 2),
        "record": {
            "wins": portfolio.win_count,
            "losses": portfolio.loss_count,
            "pushes": portfolio.push_count,
        },
        "open_trade_details": [_trade_to_dict(t) for t in open_trades],
    }


@mcp.tool()
async def settle_bets() -> dict[str, Any]:
    """Settle all paper trades whose events have final scores.

    Idempotent: only settles trades that haven't been settled yet.
    Determines win/loss by comparing selection against actual match result.

    Returns:
        Dict with count of settled trades and their details.
    """
    async with async_session_maker() as session:
        settled = await settle_trades(session)
        await session.commit()

        total_pnl = sum(t.pnl for t in settled if t.pnl is not None)
        serialized = [_trade_to_dict(t) for t in settled]

    return {
        "settled_count": len(settled),
        "total_pnl": total_pnl,
        "settled_trades": serialized,
    }


def _snapshot_sharp_prices(
    odds_list: list[Odds],
    sharp_bookmakers: list[str],
) -> SharpPriceMap:
    """Extract current sharp bookmaker prices from an odds list.

    Uses per-outcome priority fallback: each outcome independently falls through
    to the next sharp bookmaker if a higher-priority one lacks that outcome.

    Returns a dict keyed by outcome name, each containing the sharp bookmaker key,
    American odds price, and implied probability.
    """
    result: dict[str, dict[str, Any]] = {}
    for o in odds_list:
        if o.outcome_name in result:
            continue
        # Find highest-priority sharp bookmaker that has this outcome
        for bm_key in sharp_bookmakers:
            bm_match = [
                x
                for x in odds_list
                if x.bookmaker_key == bm_key and x.outcome_name == o.outcome_name
            ]
            if bm_match:
                result[o.outcome_name] = {
                    "bookmaker": bm_key,
                    "price": bm_match[0].price,
                    "implied_prob": round(calculate_implied_probability(bm_match[0].price), 6),
                }
                break
    return result


@mcp.tool()
async def save_match_brief(
    event_id: str,
    brief_text: str,
    checkpoint: Literal["context", "decision"],
) -> dict[str, Any]:
    """Save a structured analysis brief for an event at a workflow checkpoint.

    Automatically snapshots current sharp bookmaker prices at save time.
    Multiple briefs per event+checkpoint are allowed (agent may re-evaluate).

    Args:
        event_id: Event identifier.
        brief_text: Freeform brief content (structure controlled by agent prompt).
        checkpoint: Workflow checkpoint: "context" (day before) or "decision" (KO-90min).

    Returns:
        Dict with saved brief details including snapshotted sharp prices.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        # Snapshot sharp prices from the latest odds
        sharp_prices: SharpPriceMap | None = None
        snapshot = await reader.get_latest_snapshot(event_id, market="h2h")
        if snapshot is not None:
            odds = extract_odds_from_snapshot(snapshot, event_id, market="h2h")
            if odds:
                sharp_prices = _snapshot_sharp_prices(odds, _DEFAULT_SHARP_BOOKMAKERS)

        brief = MatchBrief(
            event_id=event_id,
            checkpoint=BriefCheckpoint(checkpoint),
            brief_text=brief_text,
            sharp_price_at_brief=sharp_prices,
        )
        session.add(brief)
        await session.commit()
        await session.refresh(brief)

    return {
        "id": brief.id,
        "event_id": brief.event_id,
        "checkpoint": brief.checkpoint.value,
        "brief_text": brief.brief_text,
        "sharp_price_at_brief": brief.sharp_price_at_brief,
        "created_at": brief.created_at.isoformat(),
    }


@mcp.tool()
async def get_match_brief(
    event_id: str,
    checkpoint: Literal["context", "decision"] | None = None,
) -> dict[str, Any]:
    """Retrieve saved match briefs for an event.

    Returns all briefs for the event, newest first. Optionally filtered by checkpoint.
    Returns empty gracefully when no brief exists.

    Args:
        event_id: Event identifier.
        checkpoint: If set, only return briefs for this checkpoint.

    Returns:
        Dict with event info and list of matching briefs (newest first).
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        query = select(MatchBrief).where(MatchBrief.event_id == event_id)
        if checkpoint is not None:
            query = query.where(MatchBrief.checkpoint == BriefCheckpoint(checkpoint))
        query = query.order_by(MatchBrief.created_at.desc())

        result = await session.execute(query)
        briefs = list(result.scalars().all())

    return {
        "event": _event_to_dict(event),
        "brief_count": len(briefs),
        "briefs": [
            {
                "id": b.id,
                "checkpoint": b.checkpoint.value,
                "brief_text": b.brief_text,
                "sharp_price_at_brief": b.sharp_price_at_brief,
                "created_at": b.created_at.isoformat(),
            }
            for b in briefs
        ],
    }


@mcp.tool()
async def get_sharp_soft_spread(
    event_id: str,
    sharp_bookmakers: list[str] | None = None,
    retail_bookmakers: list[str] | None = None,
) -> dict[str, Any]:
    """Get sharp vs soft bookmaker price divergence for an event.

    Returns the sharp reference price, per-bookmaker soft prices, and implied
    probability divergence for each outcome (home/draw/away).

    Args:
        event_id: Event identifier.
        sharp_bookmakers: Sharp bookmaker keys (default: ["pinnacle", "betfair_exchange"]).
        retail_bookmakers: Retail bookmaker keys (default: ["bet365", "betway", "betfred"]).

    Returns:
        Dict with per-outcome sharp price, soft prices, and divergence values.
    """
    sharp_bms = sharp_bookmakers or _DEFAULT_SHARP_BOOKMAKERS
    retail_bms = retail_bookmakers or _DEFAULT_RETAIL_BOOKMAKERS

    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        snapshot = await reader.get_latest_snapshot(event_id, market="h2h")
        if snapshot is None:
            return {
                "event": _event_to_dict(event),
                "spread": None,
                "message": "No odds snapshots available for this event",
            }

        # Extract odds and snapshot metadata inside session while ORM objects are live
        odds = extract_odds_from_snapshot(snapshot, event_id, market="h2h")
        snapshot_time_iso = snapshot.snapshot_time.isoformat()
        event_dict = _event_to_dict(event)

    if not odds:
        return {
            "event": event_dict,
            "spread": None,
            "message": "No h2h odds in latest snapshot",
        }

    # Reuse shared sharp price extraction
    sharp_by_outcome = _snapshot_sharp_prices(odds, sharp_bms)

    # Group odds by outcome for retail lookup
    outcomes: dict[str, list[Odds]] = {}
    for o in odds:
        outcomes.setdefault(o.outcome_name, []).append(o)

    spread: dict[str, dict[str, Any]] = {}
    for outcome_name, outcome_odds in outcomes.items():
        sharp_entry = sharp_by_outcome.get(outcome_name)
        sharp_prob = sharp_entry["implied_prob"] if sharp_entry else None

        # Collect retail bookmaker prices
        soft_prices: list[dict[str, Any]] = []
        for bm_key in retail_bms:
            bm_match = [o for o in outcome_odds if o.bookmaker_key == bm_key]
            if not bm_match:
                continue
            retail_price = bm_match[0].price
            retail_prob = round(calculate_implied_probability(retail_price), 6)
            divergence = round(retail_prob - sharp_prob, 6) if sharp_prob is not None else None
            soft_prices.append(
                {
                    "bookmaker": bm_key,
                    "price": retail_price,
                    "implied_prob": retail_prob,
                    "divergence": divergence,
                }
            )

        spread[outcome_name] = {
            "sharp": sharp_entry or {"bookmaker": None, "price": None, "implied_prob": None},
            "soft": soft_prices,
        }

    return {
        "event": event_dict,
        "snapshot_time": snapshot_time_iso,
        "spread": spread,
    }


@mcp.tool()
async def get_confirmed_lineups(event_id: str) -> dict[str, Any]:
    """Get confirmed lineups for an EPL match from API-Football.

    Returns formation, starting XI (name, position, number), substitutes,
    and coach for both teams. Lineups are typically available ~55-75 min
    before kickoff.

    Results are persisted to DB. If lineups were previously fetched for this
    fixture, returns the cached version without hitting the API.

    Args:
        event_id: Event identifier (e.g. "op_live_ARS_CHE_2026-04-13").

    Returns:
        Dict with lineup data per team, or a message if lineups are not yet available.
    """
    from odds_core.epl_data_models import ApiFootballLineup
    from odds_lambda.api_football_client import fetch_lineups_for_event

    async with async_session_maker() as session:
        # Look up the event
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        # Check for cached lineups first
        cached_query = select(ApiFootballLineup).where(ApiFootballLineup.event_id == event_id)
        cached_result = await session.execute(cached_query)
        cached = list(cached_result.scalars().all())

        if cached:
            return {
                "event": _event_to_dict(event),
                "source": "cached",
                "lineups": [_lineup_to_dict(row) for row in cached],
            }

    # Fetch from API-Football
    try:
        result = await fetch_lineups_for_event(
            event_id=event_id,
            home_team=event.home_team,
            away_team=event.away_team,
            commence_time=event.commence_time,
        )
    except ValueError as e:
        return {
            "event": _event_to_dict(event),
            "error": str(e),
            "message": "API_FOOTBALL_KEY is not configured. Set the environment variable.",
        }
    except Exception as e:
        logger.error("get_confirmed_lineups_failed", event_id=event_id, error=str(e), exc_info=True)
        return {
            "event": _event_to_dict(event),
            "error": str(e),
            "error_type": type(e).__name__,
        }

    if not result.get("available"):
        return {
            "event": _event_to_dict(event),
            **result,
        }

    # Persist to DB
    fixture_id = result["fixture_id"]
    lineups_data = result["lineups"]

    async with async_session_maker() as session:
        rows = []
        for team in lineups_data:
            row = ApiFootballLineup(
                event_id=event_id,
                fixture_id=fixture_id,
                team_name=team["team_name"],
                team_id=team["team_id"],
                formation=team["formation"],
                coach=team["coach"],
                start_xi=team["start_xi"],
                substitutes=team["substitutes"],
            )
            rows.append(row)

        # Upsert: delete existing rows for this fixture, then insert fresh
        from sqlalchemy import delete

        await session.execute(
            delete(ApiFootballLineup).where(ApiFootballLineup.fixture_id == fixture_id)
        )
        session.add_all(rows)
        await session.commit()

        # Re-read to get DB-assigned IDs
        for row in rows:
            await session.refresh(row)

        return {
            "event": _event_to_dict(event),
            "source": "api_football",
            "fixture_id": fixture_id,
            "lineups": [_lineup_to_dict(row) for row in rows],
        }


def _lineup_to_dict(row: Any) -> dict[str, Any]:
    """Serialize an ApiFootballLineup row to a JSON-safe dict."""
    return {
        "team_name": row.team_name,
        "team_id": row.team_id,
        "formation": row.formation,
        "coach": row.coach,
        "start_xi": row.start_xi,
        "substitutes": row.substitutes,
        "fetched_at": row.fetched_at.isoformat() if row.fetched_at else None,
    }


if __name__ == "__main__":
    mcp.run()
