"""FastMCP server exposing betting pipeline tools for AI agents.

Thin wrappers over existing DB queries, jobs, and paper trading logic.
All tools are stateless and use async_session_maker() for DB access.
"""

import json
import math
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import structlog
from fastmcp import FastMCP
from odds_analytics.backtesting import BacktestEvent
from odds_analytics.feature_extraction import TabularFeatureExtractor
from odds_analytics.utils import per_book_market_holds
from odds_core.agent_wakeup_models import AgentWakeup
from odds_core.database import async_session_maker
from odds_core.match_brief_models import BriefDecision, MatchBrief, SharpPriceMap
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.odds_math import calculate_implied_probability
from odds_core.paper_trade_models import PaperTrade
from odds_core.prediction_models import Prediction
from odds_core.snapshot_utils import extract_odds_from_snapshot
from odds_core.sports import SportKey
from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPEC_BY_NAME
from odds_lambda.paper_trading import (
    get_open_trades,
    get_portfolio_summary,
    place_trade,
    settle_trades,
)
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
# Duplicated from feature_extraction.py DEFAULT_SHARP_BOOKMAKERS with EPL-specific overrides — keep in sync.
_DEFAULT_SHARP_BOOKMAKERS = ["pinnacle", "betfair_exchange"]
# Used only by get_event_features for TabularFeatureExtractor. Distinct from
# feature_extraction.DEFAULT_RETAIL_BOOKMAKERS (US books for the NBA model) — these are
# UK books chosen to match the not-yet-deployed EPL model. When an EPL model ships, this
# default must match whatever retail set it was trained on or features will silently drift.
_DEFAULT_RETAIL_BOOKMAKERS = ["bet365", "betway", "betfred", "betvictor", "bwin"]


def _coerce_bookmaker_list(value: str | list[str] | None) -> list[str] | None:
    # Claude Code's MCP client stringifies array params (anthropics/claude-code#22394).
    # Accept JSON-array strings and comma-separated strings so tools stay callable.
    # Remove once the upstream client is fixed.
    if value is None or isinstance(value, list):
        return value
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON array in bookmaker list: {stripped!r}") from e
        if not isinstance(parsed, list):
            raise ValueError(
                f"Expected JSON array for bookmaker list, got {type(parsed).__name__}: {stripped!r}"
            )
        return [str(item) for item in parsed]
    return [item.strip() for item in stripped.split(",") if item.strip()]


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
    league: SportKey = "soccer_epl",
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
    """Trigger an on-demand OddsPortal scrape for a league and market.

    Adds a job to the shared APScheduler data store. If the scheduler
    process is running (``odds scheduler start``), it picks the job up
    immediately via Postgres LISTEN/NOTIFY.

    Args:
        league: OddsHarvester league name (e.g. "england-premier-league").
        market: Market to scrape (e.g. "1x2", "over_under_2_5").

    Returns:
        Dict confirming the job was submitted, or an error.
    """
    import dataclasses

    known_spec = _LEAGUE_SPEC_BY_NAME.get(league)
    if known_spec is None:
        return {
            "error": f"Unknown league '{league}'. Known leagues: {sorted(_LEAGUE_SPEC_BY_NAME.keys())}",
            "error_type": "ValueError",
        }

    spec = dataclasses.replace(known_spec, markets=[market])

    try:
        from apscheduler import SchedulerRole
        from odds_lambda.jobs.fetch_oddsportal import ingest_league
        from odds_lambda.scheduling.backends.local import build_scheduler

        async with build_scheduler(role=SchedulerRole.scheduler) as scheduler:
            job_id = await scheduler.add_job(ingest_league, args=[spec])

    except Exception as e:
        logger.error(
            "refresh_scrape_failed", league=league, market=market, error=str(e), exc_info=True
        )
        return {
            "error": f"Failed to submit scrape job: {e}",
            "error_type": type(e).__name__,
        }

    return {
        "job_id": str(job_id),
        "league": league,
        "market": market,
        "message": "Job submitted to scheduler via LISTEN/NOTIFY.",
    }


@mcp.tool()
async def get_scrape_status(
    job_id: str | None = None,
) -> dict[str, Any]:
    """Check scheduler status, list pending scrape jobs, and surface results.

    Without ``job_id``: returns the current list of pending/running scrape
    jobs in the APScheduler data store.

    With ``job_id`` (returned from ``refresh_scrape``): also looks up the
    completed result and returns ``matches_scraped``, ``events_matched``,
    ``snapshots_stored``, ``errors``, and job outcome. Results are retained
    by APScheduler for a bounded window after completion.

    Args:
        job_id: Optional UUID string. When provided, the tool reports the
                job's lifecycle state (pending, running, completed, failed)
                and, for completed jobs, the ingestion stats.

    Returns:
        Dict with scheduler status, pending jobs, and (if job_id is given)
        the specific job's outcome.
    """
    import dataclasses
    from uuid import UUID

    try:
        from apscheduler import SchedulerRole
        from odds_lambda.scheduling.backends.local import build_scheduler

        async with build_scheduler(role=SchedulerRole.scheduler) as scheduler:
            jobs = await scheduler.get_jobs()

            pending = [j for j in jobs if "ingest_league" in j.task_id]
            if job_id is not None:
                pending = [j for j in pending if str(j.id) == job_id]

            pending_jobs = [
                {
                    "job_id": str(j.id),
                    "task_id": j.task_id,
                    "created_at": j.created_at.isoformat() if j.created_at else None,
                    "state": "running" if j.acquired_by else "pending",
                }
                for j in pending
            ]

            job_result: dict[str, Any] | None = None
            if job_id is not None and not pending_jobs:
                result = await scheduler.get_job_result(UUID(job_id), wait=False)
                if result is not None:
                    stats = result.return_value
                    job_result = {
                        "job_id": job_id,
                        "state": "completed",
                        "outcome": result.outcome.name,
                        "started_at": result.started_at.isoformat() if result.started_at else None,
                        "finished_at": result.finished_at.isoformat(),
                        "stats": dataclasses.asdict(stats)
                        if dataclasses.is_dataclass(stats)
                        else None,
                        "exception": repr(result.exception) if result.exception else None,
                    }
                else:
                    job_result = {"job_id": job_id, "state": "unknown"}

    except Exception as e:
        logger.warning("get_scrape_status_failed", error=str(e))
        return {
            "status": "unavailable",
            "message": f"Could not query scheduler: {e}",
            "jobs": [],
        }

    response: dict[str, Any] = {
        "status": "ok",
        "pending_scrape_jobs": len(pending_jobs),
        "jobs": pending_jobs,
    }
    if job_result is not None:
        response["result"] = job_result
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
    sharp_bookmakers: str | list[str] | None = None,
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
            sharp_bookmakers=_coerce_bookmaker_list(sharp_bookmakers) or _DEFAULT_SHARP_BOOKMAKERS,
            retail_bookmakers=_DEFAULT_RETAIL_BOOKMAKERS,
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
    decision: Literal["watching", "bet", "skip"],
    summary: str,
    brief_text: str,
) -> dict[str, Any]:
    """Save a structured analysis brief for an event.

    Automatically snapshots current sharp bookmaker prices at save time.
    Briefs are append-only — each call creates a new row. The agent loads
    all previous briefs for a match (newest first) and builds on them.

    Args:
        event_id: Event identifier.
        market: Market type — "h2h", "1x2", "totals", or "spreads".
        decision: Agent decision — "watching", "bet", or "skip".
        summary: One-line summary for triage views (keep under ~100 chars).
        brief_text: Freeform brief content (structure controlled by agent prompt).

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
            decision=BriefDecision(decision),
            summary=summary,
            brief_text=brief_text,
            sharp_price_at_brief=sharp_prices,
        )
        session.add(brief)
        await session.commit()
        await session.refresh(brief)

    return {
        "id": brief.id,
        "event_id": brief.event_id,
        "decision": brief.decision.value,
        "summary": brief.summary,
        "brief_text": brief.brief_text,
        "sharp_price_at_brief": brief.sharp_price_at_brief,
        "created_at": brief.created_at.isoformat(),
    }


@mcp.tool()
async def get_match_brief(
    event_id: str,
    limit: int | None = None,
) -> dict[str, Any]:
    """Retrieve saved match briefs for an event.

    Returns briefs for the event, newest first (append-only history).
    Returns empty gracefully when no brief exists.

    Args:
        event_id: Event identifier.
        limit: Max number of briefs to return (newest first). Returns all if omitted.

    Returns:
        Dict with event info and list of matching briefs (newest first).
    """
    async with async_session_maker() as session:
        reader = OddsReader(session)
        event = await reader.get_event_by_id(event_id)
        if event is None:
            return {"error": f"Event '{event_id}' not found"}

        query = (
            select(MatchBrief)
            .where(MatchBrief.event_id == event_id)
            .order_by(MatchBrief.created_at.desc())
        )
        if limit is not None:
            query = query.limit(limit)

        result = await session.execute(query)
        briefs = list(result.scalars().all())

    return {
        "event": _event_to_dict(event),
        "brief_count": len(briefs),
        "briefs": [
            {
                "id": b.id,
                "decision": b.decision.value,
                "summary": b.summary,
                "brief_text": b.brief_text,
                "sharp_price_at_brief": b.sharp_price_at_brief,
                "created_at": b.created_at.isoformat(),
            }
            for b in briefs
        ],
    }


@mcp.tool()
async def get_slate_briefs(
    league: SportKey = "soccer_epl",
    days_ahead: int = 7,
) -> dict[str, Any]:
    """Get latest brief status for all upcoming fixtures in a league.

    Returns one entry per event with the most recent brief's decision,
    summary, and timestamp. Events with no brief are included with
    decision=None. Use this for slate triage before pulling full briefs.

    Args:
        league: Sport key (e.g. "soccer_epl").
        days_ahead: How many days ahead to look (1-30).

    Returns:
        Dict with list of events and their latest brief status.
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

        # Fetch latest brief per event using DISTINCT ON
        latest_by_event: dict[str, Any] = {}
        if events:
            event_ids = [e.id for e in events]
            result = await session.execute(
                select(MatchBrief)
                .where(MatchBrief.event_id.in_(event_ids))  # type: ignore[union-attr]
                .distinct(MatchBrief.event_id)
                .order_by(MatchBrief.event_id, MatchBrief.created_at.desc())
            )
            latest_by_event = {
                b.event_id: {
                    "decision": b.decision.value,
                    "summary": b.summary,
                    "created_at": b.created_at.isoformat(),
                }
                for b in result.scalars()
            }

        entries = [
            {
                "event": _event_to_dict(e),
                "latest_brief": latest_by_event.get(e.id),
            }
            for e in events
        ]

    return {
        "fixture_count": len(entries),
        "fixtures": entries,
    }


@mcp.tool()
async def get_sharp_soft_spread(
    event_id: str,
    market: MarketKey,
    sharp_bookmakers: str | list[str] | None = None,
    sharp_lookback_hours: float = 2.0,
) -> dict[str, Any]:
    """Get sharp vs soft bookmaker price divergence for an event.

    Sharp prices are resolved via a time-windowed lookback across recent
    snapshots so that a missing sharp bookmaker in the latest scrape does not
    discard a perfectly good price from a nearby snapshot. Retail prices
    always come from the single latest snapshot and cover every non-sharp
    bookmaker in it. For h2h / 1x2 markets each retail entry also reports
    that book's market hold, so callers can filter high-margin books.

    Args:
        event_id: Event identifier.
        market: Market type — "h2h", "1x2", "totals", or "spreads".
        sharp_bookmakers: Sharp bookmaker keys (default: ["pinnacle", "betfair_exchange"]).
        sharp_lookback_hours: How far back to search for sharp prices (default 2.0 h).

    Returns:
        Dict with per-outcome sharp price (with source snapshot time),
        soft prices, and divergence values. Each soft entry includes
        ``market_hold`` (h2h / 1x2 only; null for multi-line markets).
    """
    sharp_bms = _coerce_bookmaker_list(sharp_bookmakers) or _DEFAULT_SHARP_BOOKMAKERS

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

    # Retail set is every non-sharp book in the snapshot.
    sharp_set = set(sharp_bms)
    retail_bms = sorted({o.bookmaker_key for o in odds if o.bookmaker_key not in sharp_set})

    # Per-book market hold is only meaningful for single-line markets; totals / spreads
    # span multiple lines, so computing it event-wide would conflate them.
    book_hold: dict[str, float] | None = None
    if market in ("h2h", "1x2"):
        book_hold = per_book_market_holds(
            ((o.bookmaker_key, o.outcome_name, o.price) for o in odds),
            required_outcomes=outcomes.keys(),
        )

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
            market_hold = (
                round(book_hold[bm_key], 6)
                if book_hold is not None and bm_key in book_hold
                else None
            )
            soft_prices.append(
                {
                    "bookmaker": bm_key,
                    "price": retail_price,
                    "implied_prob": retail_prob,
                    "divergence": divergence,
                    "market_hold": market_hold,
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
    sport: SportKey | None = None,
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
        from apscheduler import SchedulerRole
        from odds_lambda.scheduling.backends.local import build_scheduler

        async with build_scheduler(role=SchedulerRole.scheduler) as scheduler:
            schedules = await scheduler.get_schedules()
    except Exception as e:
        logger.warning("get_scheduled_jobs_failed", error=str(e))
        return {
            "jobs": [],
            "message": f"Could not query scheduler: {e}",
        }

    jobs = [
        {
            "job_name": s.id,
            "next_run_time": s.next_fire_time.isoformat() if s.next_fire_time else None,
            "status": "scheduled",
        }
        for s in schedules
    ]

    if sport:
        jobs = [j for j in jobs if sport in j["job_name"]]

    return {
        "job_count": len(jobs),
        "jobs": jobs,
    }


@mcp.tool()
async def schedule_next_wakeup(
    sport: SportKey,
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
