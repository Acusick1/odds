"""Runs the historical Odds API calls an acquisition manifest plans.

Each call is paid for once and everything it returns that resolves to a stored event is stored,
at the snapshot time the API reports.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import structlog
from odds_core.database import async_session_maker
from odds_core.models import FetchLog, OddsSnapshot
from odds_core.time import utc_isoformat
from odds_core.utils import is_three_way_sport
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.live_events import DroppedGame, alert_dropped_games, lock_order
from odds_lambda.odds_api_events import game_row, odds_api_finder, resolve_odds_api_event
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_payload
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backfill_plan import Call, Manifest

logger = structlog.get_logger()

CallStatus = Literal["fetched", "skipped", "would_fetch", "failed"]


@dataclass
class BackfillResult:
    """What a run did: ``calls_made`` paid calls, ``calls_skipped`` calls whose fixtures were
    already stored, ``would_fetch`` calls a dry run did not make."""

    calls_made: int = 0
    calls_skipped: int = 0
    would_fetch: int = 0
    units_used: int = 0
    snapshots_stored: int = 0
    fixtures_stored: int = 0
    quota_remaining: int | None = None
    dropped_games: list[DroppedGame] = field(default_factory=list)


@dataclass(frozen=True)
class CallProgress:
    """One call's outcome, reported as the run goes."""

    time: datetime
    status: CallStatus
    fixtures_stored: int = 0
    units_used: int | None = None
    quota_remaining: int | None = None
    message: str | None = None


class BackfillExecutor:
    """Executes a manifest's planned calls against the Odds API."""

    def __init__(
        self,
        manifest: Manifest,
        client: TheOddsAPIClient | None = None,
        session_factory: Callable[[], Any] | None = None,
        skip_existing: bool = True,
        dry_run: bool = False,
        rate_limit_seconds: float = 1.0,
    ):
        """
        Args:
            manifest: Fixes the sport (whose key pays), markets, and bookmakers or regions
            client: API client (one is created and closed with the executor if not given)
            session_factory: Creates database sessions (defaults to ``async_session_maker``)
            skip_existing: Skip a call when every fixture it serves already has the snapshot it
                would return
            dry_run: Make no API call and write nothing
            rate_limit_seconds: Pause after each paid call
        """
        self.manifest = manifest
        self._client = client
        self._owns_client = client is None
        self._session_factory = session_factory or async_session_maker
        self.skip_existing = skip_existing
        self.dry_run = dry_run
        self.rate_limit_seconds = rate_limit_seconds

    async def __aenter__(self) -> "BackfillExecutor":
        if self._owns_client:
            self._client = TheOddsAPIClient()
            await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._owns_client and self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def execute(
        self,
        calls: list[Call],
        progress_callback: Callable[[CallProgress], None] | None = None,
    ) -> BackfillResult:
        """Run ``calls`` in order, stopping at the first exception.

        Each call is paid, and a rerun skips the calls already stored, so a failure ends the run
        after reporting it rather than spending on.
        """
        result = BackfillResult()

        def report(progress: CallProgress) -> None:
            if progress_callback:
                progress_callback(progress)

        try:
            for call in calls:
                try:
                    progress = await self._run_call(call, result)
                except Exception as e:
                    logger.error("backfill_call_failed", call_time=utc_isoformat(call.time))
                    report(CallProgress(call.time, "failed", message=str(e)))
                    raise
                report(progress)
                if progress.status == "fetched":
                    await asyncio.sleep(self.rate_limit_seconds)
        finally:
            await alert_dropped_games("odds-backfill", result.dropped_games)
        return result

    async def _run_call(self, call: Call, result: BackfillResult) -> CallProgress:
        if self.skip_existing and await self._already_stored(call):
            result.calls_skipped += 1
            return CallProgress(call.time, "skipped", message="Already stored")
        if self.dry_run:
            result.would_fetch += 1
            return CallProgress(call.time, "would_fetch")

        client = self._client
        if client is None:
            raise RuntimeError("BackfillExecutor client not initialized")
        try:
            return await self._fetch_and_store(call, result, client)
        except Exception as e:
            await self._log_failed_fetch(client, result, e)
            raise

    async def _fetch_and_store(
        self, call: Call, result: BackfillResult, client: TheOddsAPIClient
    ) -> CallProgress:
        manifest = self.manifest
        response = await client.get_historical_odds(
            sport=manifest.sport,
            date=utc_isoformat(call.time),
            markets=[str(market.key) for market in manifest.markets],
            regions=list(manifest.regions) or None,
            bookmakers=list(manifest.bookmakers) or None,
        )
        result.calls_made += 1
        result.units_used += response.units_used or 0
        result.quota_remaining = response.quota_remaining
        logger.info(
            "backfill_call_fetched",
            call_time=utc_isoformat(call.time),
            snapshot_time=utc_isoformat(response.snapshot_time),
            units_used=response.units_used,
            quota_remaining=response.quota_remaining,
        )

        stored = 0
        async with self._session_factory() as session:
            writer = OddsWriter(session)
            finder = await odds_api_finder(manifest.sport, response.events)
            payload_id: int | None = None
            # The resolver takes fixture locks, which must be acquired in ascending order.
            fixtures = sorted(
                zip(response.events, response.raw_events_data, strict=True),
                key=lambda pair: lock_order(manifest.sport, game_row(pair[0])),
            )
            for event, raw in fixtures:
                found = await resolve_odds_api_event(finder, session, event)
                if isinstance(found, DroppedGame):
                    result.dropped_games.append(found)
                    continue
                if payload_id is None:
                    payload_id = await store_payload(session, response.raw_events_data)
                envelope = await writer.store_envelope(
                    event_id=found.event.id,
                    envelope=raw,
                    snapshot_time=response.snapshot_time,
                    shape=SnapshotShape(Source.ODDS_API, TimeBasis.OBSERVED, payload_id),
                )
                result.snapshots_stored += envelope.snapshots_written
                stored += 1
            await writer.log_fetch(
                FetchLog(
                    sport_key=manifest.sport,
                    events_count=len(response.events),
                    bookmakers_count=len(
                        {
                            b["key"]
                            for raw in response.raw_events_data
                            for b in raw.get("bookmakers", [])
                        }
                    ),
                    success=True,
                    api_quota_remaining=response.quota_remaining,
                    api_quota_used=response.quota_used,
                    response_time_ms=response.response_time_ms,
                )
            )
            await session.commit()
        result.fixtures_stored += stored
        return CallProgress(
            call.time,
            "fetched",
            fixtures_stored=stored,
            units_used=response.units_used,
            quota_remaining=response.quota_remaining,
        )

    async def _log_failed_fetch(
        self, client: TheOddsAPIClient, result: BackfillResult, error: Exception
    ) -> None:
        """Record a failed call's fetch log in its own transaction, so the key's last balance
        survives the call's rollback."""
        result.quota_remaining = client.quota_remaining
        async with self._session_factory() as session:
            await OddsWriter(session).log_fetch(
                FetchLog(
                    sport_key=self.manifest.sport,
                    events_count=0,
                    bookmakers_count=0,
                    success=False,
                    error_message=f"{type(error).__name__}: {error}",
                    api_quota_remaining=client.quota_remaining,
                    api_quota_used=client.quota_used,
                )
            )
            await session.commit()

    async def _already_stored(self, call: Call) -> bool:
        """Whether every fixture the call serves has the snapshot it would return: an Odds API
        snapshot of each requested market in ``(call time - grid step, call time]``."""
        wanted = {target.fixture.event_id for target in call.serves}
        if not wanted:
            return True
        # Soccer's three-way h2h is stored as 1x2 (odds_core.utils.relabel_three_way_markets).
        markets = {
            Market.ONE_X_TWO
            if market.key == Market.H2H and is_three_way_sport(self.manifest.sport)
            else Market(market.key)
            for market in self.manifest.markets
        }
        step = self.manifest.grid_step_at(call.time)
        async with self._session_factory() as session:
            stored = await _stored_markets(session, wanted, call.time - step, call.time)
        return all(markets <= stored.get(event_id, set()) for event_id in wanted)


async def _stored_markets(
    session: AsyncSession, event_ids: set[int], after: datetime, until: datetime
) -> dict[int, set[Market]]:
    rows = await session.execute(
        select(OddsSnapshot.event_id, OddsSnapshot.market)
        .where(
            OddsSnapshot.event_id.in_(event_ids),  # type: ignore[attr-defined]
            OddsSnapshot.source == Source.ODDS_API,
            OddsSnapshot.snapshot_time > after,
            OddsSnapshot.snapshot_time <= until,
        )
        .distinct()
    )
    stored: dict[int, set[Market]] = {}
    for event_id, market in rows.all():
        if market is not None:
            stored.setdefault(event_id, set()).add(market)
    return stored
