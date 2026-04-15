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
from odds_core.agent_wakeup_models import AgentWakeup
from odds_core.database import async_session_maker
from odds_core.match_brief_models import BriefCheckpoint, MatchBrief, SharpPriceMap
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.paper_trade_models import PaperTrade
from odds_core.prediction_models import Prediction
from odds_core.scrape_job_models import ScrapeJob, ScrapeJobStatus
from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPEC_BY_NAME
from odds_lambda.paper_trading import (
    get_open_trades,
    get_portfolio_summary,
    place_trade,
    settle_trades,
)
from odds_lambda.scheduling.backends import BackendUnavailableError, get_scheduler_backend
from odds_lambda.storage.readers import OddsReader
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

logger = structlog.get_logger()

MarketKey = Literal["h2h", "1x2", "totals", "spreads"]

mcp = FastMCP(
    "odds-mcp",
    instructions=(
        "Betting odds pipeline tools. Use these to inspect fixtures, odds, predictions, "
        "and manage paper trades across sports (EPL football, MLB baseball). "
        "All times are UTC."
    ),
)

# Re-export canonical lookup built where LEAGUE_SPECS is defined
_LEAGUE_SPEC_BY_NAME = LEAGUE_SPEC_BY_NAME

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
    market: MarketKey,
    include_raw_data: bool = False,
) -> dict[str, Any]:
    """Get the latest odds snapshot for an event, showing current bookmaker prices.

    Args:
        event_id: Event identifier.
        market: Market type — "h2h" (2-way moneyline), "1x2" (3-way with
            draw), "totals", or "spreads".
        include_raw_data: If True, also include the full raw_data JSON blob.

    Returns:
        Dict with event info and the latest snapshot with structured odds.
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        snapshot = await reader.get_latest_snapshot(event_id, market=market)
        if snapshot is None:
            return {
                "event": _event_to_dict(event),
                "snapshot": None,
                "message": "No odds snapshots available for this event",
            }

    odds = extract_odds_from_snapshot(snapshot, event_id, market=market)
    return {
        "event": _event_to_dict(event),
        "snapshot": _snapshot_to_dict(
            snapshot, include_raw_data=include_raw_data, extracted_odds=odds
        ),
    }


@mcp.tool()
async def get_odds_history(event_id: str, market: MarketKey) -> dict[str, Any]:
    """Get the full odds movement timeline for an event (all snapshots).

    Returns structured bookmaker odds per snapshot instead of raw JSON blobs
    to keep response size manageable.

    Args:
        event_id: Event identifier.
        market: Market type — "h2h", "1x2", "totals", or "spreads".

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
        odds = extract_odds_from_snapshot(s, event_id, market=market)
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
    """Enqueue an on-demand OddsPortal scrape for a league and market.

    Creates a job that will be picked up by the background worker process
    (``odds worker start``). Returns the job ID immediately without
    launching Playwright in the MCP server process.

    If a pending or running job already exists for the same league+market,
    returns that existing job instead of creating a duplicate.

    Args:
        league: OddsHarvester league name (e.g. "england-premier-league").
        market: Market to scrape (e.g. "1x2", "over_under_2_5").

    Returns:
        Dict with job ID and status. Use ``get_scrape_status`` to poll for results.
    """
    known_spec = _LEAGUE_SPEC_BY_NAME.get(league)
    if known_spec is None:
        return {
            "error": f"Unknown league '{league}'. Known leagues: {sorted(_LEAGUE_SPEC_BY_NAME.keys())}",
            "error_type": "ValueError",
        }

    async with async_session_maker() as session:
        # Check for existing pending/running job for same league+market
        existing_query = (
            select(ScrapeJob)
            .where(
                ScrapeJob.league == league,
                ScrapeJob.market == market,
                ScrapeJob.status.in_([ScrapeJobStatus.PENDING, ScrapeJobStatus.RUNNING]),
            )
            .order_by(ScrapeJob.created_at.desc())
            .limit(1)
        )
        result = await session.execute(existing_query)
        existing = result.scalar_one_or_none()

        if existing is not None:
            return {
                "job_id": existing.id,
                "status": existing.status.value,
                "league": existing.league,
                "market": existing.market,
                "created_at": existing.created_at.isoformat(),
                "message": "Existing job found for this league+market",
            }

        job = ScrapeJob(league=league, market=market)
        session.add(job)
        await session.commit()
        await session.refresh(job)

    return {
        "job_id": job.id,
        "status": job.status.value,
        "league": job.league,
        "market": job.market,
        "created_at": job.created_at.isoformat(),
        "message": "Job enqueued. Run 'odds worker start' to process.",
    }


@mcp.tool()
async def get_scrape_status(job_id: int) -> dict[str, Any]:
    """Check the status and results of a scrape job.

    Args:
        job_id: Scrape job ID returned by ``refresh_scrape``.

    Returns:
        Dict with job status, timestamps, and ingestion stats (when completed).
    """
    async with async_session_maker() as session:
        result = await session.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
        job = result.scalar_one_or_none()

    if job is None:
        return {"error": f"Scrape job {job_id} not found"}

    response: dict[str, Any] = {
        "job_id": job.id,
        "league": job.league,
        "market": job.market,
        "status": job.status.value,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    if job.status == ScrapeJobStatus.COMPLETED:
        response["results"] = {
            "matches_scraped": job.matches_scraped,
            "matches_converted": job.matches_converted,
            "events_matched": job.events_matched,
            "events_created": job.events_created,
            "snapshots_stored": job.snapshots_stored,
        }

    if job.status == ScrapeJobStatus.FAILED:
        response["error_message"] = job.error_message

    return response


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
    market: MarketKey,
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
        market: Market type — "h2h", "1x2", "totals", or "spreads".
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

        snapshot = await reader.get_latest_snapshot(event_id, market=market)
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
            market=market,
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
    market: MarketKey,
    outcome_name: str,
    sharp_bookmakers: list[str],
    retail_bookmakers: list[str],
) -> dict[str, float | None]:
    """Extract feature dict from an event and its latest snapshot."""
    odds = extract_odds_from_snapshot(snapshot, event.id, market=market)
    if not odds:
        return {"error": f"No {market} odds data in snapshot"}

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
        market=market,
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


@mcp.tool()
async def save_match_brief(
    event_id: str,
    market: MarketKey,
    brief_text: str,
    checkpoint: Literal["context", "decision"],
) -> dict[str, Any]:
    """Save a structured analysis brief for an event at a workflow checkpoint.

    Automatically snapshots current sharp bookmaker prices at save time.
    Multiple briefs per event+checkpoint are allowed (agent may re-evaluate).

    Args:
        event_id: Event identifier.
        market: Market type — "h2h", "1x2", "totals", or "spreads".
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

        # Snapshot sharp prices via lookback across recent snapshots.
        # Anchor on the latest snapshot and use a generous window — we just
        # want the best available price for stamping the brief.
        latest_snapshot = await reader.get_latest_snapshot(event_id, market=market)
        if latest_snapshot is not None:
            sharp_result = await reader.get_sharp_prices(
                event_id,
                market=market,
                sharp_bookmakers=_DEFAULT_SHARP_BOOKMAKERS,
                lookback_hours=24.0,
                now=latest_snapshot.snapshot_time,
            )
            sharp_prices: SharpPriceMap | None = sharp_result.prices or None
        else:
            sharp_prices = None

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
    market: MarketKey,
    sharp_bookmakers: list[str] | None = None,
    retail_bookmakers: list[str] | None = None,
    sharp_lookback_hours: float = 2.0,
) -> dict[str, Any]:
    """Get sharp vs soft bookmaker price divergence for an event.

    Sharp prices are resolved via a time-windowed lookback across recent
    snapshots so that a missing sharp bookmaker in the latest scrape does not
    discard a perfectly good price from a nearby snapshot.  Retail prices
    always come from the single latest snapshot.

    Args:
        event_id: Event identifier.
        market: Market type — "h2h", "1x2", "totals", or "spreads".
        sharp_bookmakers: Sharp bookmaker keys (default: ["pinnacle", "betfair_exchange"]).
        retail_bookmakers: Retail bookmaker keys (default: ["bet365", "betway", "betfred"]).
        sharp_lookback_hours: How far back to search for sharp prices (default 2.0 h).

    Returns:
        Dict with per-outcome sharp price (with source snapshot time),
        soft prices, and divergence values.
    """
    sharp_bms = sharp_bookmakers or _DEFAULT_SHARP_BOOKMAKERS
    retail_bms = retail_bookmakers or _DEFAULT_RETAIL_BOOKMAKERS

    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        # Retail prices: latest snapshot only
        snapshot = await reader.get_latest_snapshot(event_id, market=market)
        if snapshot is None:
            return {
                "event": _event_to_dict(event),
                "spread": None,
                "message": "No odds snapshots available for this event",
            }

        # Sharp prices: lookback across recent snapshots, anchored on the
        # latest snapshot time so the window is data-relative, not clock-relative.
        sharp_result = await reader.get_sharp_prices(
            event_id,
            market=market,
            sharp_bookmakers=sharp_bms,
            lookback_hours=sharp_lookback_hours,
            now=snapshot.snapshot_time,
        )

        # Extract retail odds and metadata while ORM objects are live
        odds = extract_odds_from_snapshot(snapshot, event_id, market=market)
        snapshot_time_iso = snapshot.snapshot_time.isoformat()
        event_dict = _event_to_dict(event)

    if not odds:
        return {
            "event": event_dict,
            "spread": None,
            "message": f"No {market} odds in latest snapshot",
        }

    sharp_by_outcome = sharp_result.prices

    # Group odds by outcome for retail lookup
    outcomes: dict[str, list[Odds]] = {}
    for o in odds:
        outcomes.setdefault(o.outcome_name, []).append(o)

    spread: dict[str, dict[str, Any]] = {}
    for outcome_name, outcome_odds in outcomes.items():
        sharp_entry = sharp_by_outcome.get(outcome_name)
        sharp_prob = sharp_entry["implied_prob"] if sharp_entry else None

        # Build sharp block with source snapshot metadata
        if sharp_entry:
            meta = sharp_result.meta.get(outcome_name)
            sharp_block: dict[str, Any] = {
                **sharp_entry,
                "snapshot_time": meta.snapshot_time.isoformat() if meta else None,
                "age_seconds": meta.age_seconds if meta else None,
            }
        else:
            sharp_block = {
                "bookmaker": None,
                "price": None,
                "implied_prob": None,
                "snapshot_time": None,
                "age_seconds": None,
            }

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
            "sharp": sharp_block,
            "soft": soft_prices,
        }

    return {
        "event": event_dict,
        "snapshot_time": snapshot_time_iso,
        "spread": spread,
    }


@mcp.tool()
async def get_scheduled_jobs(
    sport: str | None = None,
) -> dict[str, Any]:
    """List all currently scheduled jobs from the scheduler backend.

    Returns job name, next run time, and status for each job. Optionally
    filter by sport key (substring match on job name).

    If the scheduler backend is not running (e.g. local APScheduler not
    started), returns an informative message rather than an error.

    Args:
        sport: Optional sport key to filter jobs (e.g. "soccer_epl").
               Matches as substring against job names.

    Returns:
        Dict with list of scheduled jobs or an informative message.
    """
    try:
        backend = get_scheduler_backend()
        jobs = await backend.get_scheduled_jobs()
    except BackendUnavailableError as e:
        return {
            "jobs": [],
            "message": f"Scheduler backend unavailable: {e}",
        }
    except Exception as e:
        logger.warning("get_scheduled_jobs_failed", error=str(e))
        return {
            "jobs": [],
            "message": (
                f"Could not query scheduler: {e}. "
                "The local APScheduler backend requires the scheduler to be "
                "running in a separate process (odds scheduler start)."
            ),
        }

    if sport:
        jobs = [j for j in jobs if sport in j.job_name]

    return {
        "job_count": len(jobs),
        "jobs": [
            {
                "job_name": j.job_name,
                "next_run_time": j.next_run_time.isoformat() if j.next_run_time else None,
                "status": j.status.value,
            }
            for j in jobs
        ],
    }


@mcp.tool()
async def schedule_next_wakeup(
    sport: str,
    delay_hours: float,
    reason: str,
) -> dict[str, Any]:
    """Request the agent's next wake-up at now + delay_hours.

    Writes an upsert to the agent_wakeups table (one active row per sport).
    The agent_run job module reads this after the agent subprocess exits and
    reschedules if the requested time is sooner than the default.

    Args:
        sport: Sport key (e.g. "soccer_epl", "baseball_mlb").
        delay_hours: Hours from now until the next wake-up (0.5 to 168).
        reason: Why this wake-up is being scheduled (shown in logs).

    Returns:
        Dict confirming the scheduled wake-up with the requested UTC time.
    """
    now = datetime.now(UTC)
    delay_hours = max(0.5, min(delay_hours, 168.0))
    requested_time = now + timedelta(hours=delay_hours)

    async with async_session_maker() as session:
        stmt = (
            pg_insert(AgentWakeup)
            .values(
                sport_key=sport,
                requested_time=requested_time,
                reason=reason,
                created_at=now,
            )
            .on_conflict_do_update(
                constraint="uq_agent_wakeup_sport_key",
                set_={
                    "requested_time": requested_time,
                    "reason": reason,
                    "created_at": now,
                    "consumed_at": None,
                },
            )
        )
        await session.execute(stmt)
        await session.commit()

    return {
        "sport": sport,
        "requested_time": requested_time.isoformat(),
        "delay_hours": delay_hours,
        "reason": reason,
    }


if __name__ == "__main__":
    mcp.run()
