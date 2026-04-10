"""FastMCP server exposing betting pipeline tools for AI agents.

Thin wrappers over existing DB queries, jobs, and paper trading logic.
All tools are stateless and use async_session_maker() for DB access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastmcp import FastMCP
from odds_core.database import async_session_maker
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade
from odds_core.prediction_models import Prediction
from odds_lambda.paper_trading import (
    get_open_trades,
    get_portfolio_summary,
    place_trade,
    settle_trades,
)
from odds_lambda.storage.readers import OddsReader
from sqlalchemy import select

logger = structlog.get_logger()

mcp = FastMCP(
    "odds-mcp",
    instructions=(
        "Betting odds pipeline tools. Use these to inspect fixtures, odds, predictions, "
        "and manage paper trades for EPL football. All times are UTC."
    ),
)


def _event_to_dict(event: Event) -> dict[str, Any]:
    """Serialize an Event to a JSON-safe dict."""
    return {
        "id": event.id,
        "sport_key": event.sport_key,
        "home_team": event.home_team,
        "away_team": event.away_team,
        "commence_time": event.commence_time.isoformat(),
        "status": event.status.value,
        "home_score": event.home_score,
        "away_score": event.away_score,
    }


def _snapshot_to_dict(snapshot: OddsSnapshot) -> dict[str, Any]:
    """Serialize an OddsSnapshot to a JSON-safe dict."""
    return {
        "id": snapshot.id,
        "event_id": snapshot.event_id,
        "snapshot_time": snapshot.snapshot_time.isoformat(),
        "bookmaker_count": snapshot.bookmaker_count,
        "fetch_tier": snapshot.fetch_tier,
        "hours_until_commence": snapshot.hours_until_commence,
        "raw_data": snapshot.raw_data,
    }


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


# ---------------------------------------------------------------------------
# Tool 1: get_upcoming_fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool 2: get_current_odds
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_current_odds(event_id: str) -> dict[str, Any]:
    """Get the latest odds snapshot for an event, showing current bookmaker prices.

    Args:
        event_id: Event identifier.

    Returns:
        Dict with event info and the latest snapshot's raw bookmaker odds data.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        snapshot = await reader.get_latest_snapshot(event_id)
        if snapshot is None:
            return {
                "event": _event_to_dict(event),
                "snapshot": None,
                "message": "No odds snapshots available for this event",
            }

    return {
        "event": _event_to_dict(event),
        "snapshot": _snapshot_to_dict(snapshot),
    }


# ---------------------------------------------------------------------------
# Tool 3: get_odds_history
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_odds_history(event_id: str) -> dict[str, Any]:
    """Get the full odds movement timeline for an event (all snapshots).

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

    return {
        "event": _event_to_dict(event),
        "snapshot_count": len(snapshots),
        "snapshots": [_snapshot_to_dict(s) for s in snapshots],
    }


# ---------------------------------------------------------------------------
# Tool 4: refresh_scrape
# ---------------------------------------------------------------------------


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
        return {"error": str(e)}

    return {
        "league": stats.league,
        "matches_scraped": stats.matches_scraped,
        "matches_converted": stats.matches_converted,
        "events_matched": stats.events_matched,
        "events_created": stats.events_created,
        "snapshots_stored": stats.snapshots_stored,
        "errors": stats.errors,
    }


def _resolve_sport_meta(league: str) -> tuple[str, str]:
    """Map league name to pipeline sport_key and sport_title."""
    mapping: dict[str, tuple[str, str]] = {
        "england-premier-league": ("soccer_epl", "EPL"),
        "nba": ("basketball_nba", "NBA"),
    }
    return mapping.get(league, (f"football_{league.replace('-', '_')}", league.title()))


# ---------------------------------------------------------------------------
# Tool 5: get_model_prediction
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_model_prediction(event_id: str) -> dict[str, Any]:
    """Get CLV model predictions for an event.

    Returns all predictions (one per snapshot scored) for the given event,
    ordered by creation time. If no predictions exist, returns an empty list.

    Args:
        event_id: Event identifier.

    Returns:
        Dict with event info and list of prediction records.
    """
    async with async_session_maker() as session:
        event_result = await session.execute(select(Event).where(Event.id == event_id))
        event = event_result.scalar_one_or_none()
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        pred_result = await session.execute(
            select(Prediction)
            .where(Prediction.event_id == event_id)
            .order_by(Prediction.created_at)
        )
        predictions = list(pred_result.scalars().all())

    return {
        "event": _event_to_dict(event),
        "prediction_count": len(predictions),
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


# ---------------------------------------------------------------------------
# Tool 6: get_event_features
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_event_features(event_id: str) -> dict[str, Any]:
    """Get the feature vector for an event's latest snapshot.

    Extracts tabular features (bookmaker odds, implied probabilities,
    market consensus) plus hours_until_event. Uses the same feature
    extraction pipeline as the scoring job.

    Args:
        event_id: Event identifier.

    Returns:
        Dict with event info and feature name/value pairs.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        snapshot = await reader.get_latest_snapshot(event_id)
        if snapshot is None:
            return {
                "event": _event_to_dict(event),
                "features": None,
                "message": "No snapshots available for feature extraction",
            }

    # Feature extraction is sync and doesn't need a session
    try:
        features = _extract_features_for_event(event, snapshot)
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


def _extract_features_for_event(event: Event, snapshot: OddsSnapshot) -> dict[str, float | None]:
    """Extract feature dict from an event and its latest snapshot."""
    from odds_analytics.backtesting import BacktestEvent
    from odds_analytics.feature_extraction import TabularFeatureExtractor
    from odds_analytics.sequence_loader import extract_odds_from_snapshot

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

    extractor = TabularFeatureExtractor()
    tab_feats = extractor.extract_features(
        event=backtest_event,
        odds_data=odds,
        outcome=event.home_team,
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


def _safe_float(val: Any) -> float | None:
    """Convert numpy/float values, returning None for NaN."""
    import math

    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 6)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Tool 7: paper_bet
# ---------------------------------------------------------------------------


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
        # Validate event exists
        event_result = await session.execute(select(Event).where(Event.id == event_id))
        event = event_result.scalar_one_or_none()
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        # Compute bankroll from portfolio if not provided
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
            return {"error": str(e)}

    return {
        "status": "placed",
        "trade": _trade_to_dict(trade),
    }


# ---------------------------------------------------------------------------
# Tool 8: get_portfolio
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool 9: settle_bets
# ---------------------------------------------------------------------------


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

    return {
        "settled_count": len(settled),
        "total_pnl": total_pnl,
        "settled_trades": [_trade_to_dict(t) for t in settled],
    }
