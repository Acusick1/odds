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
from odds_core.config import get_settings
from odds_core.database import async_session_maker
from odds_core.models import FetchLog, OddsSnapshot
from odds_core.time import utc_isoformat
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.live_events import DroppedGame, alert_dropped_games, lock_order
from odds_lambda.odds_api_events import (
    attach_historical_odds_api_event,
    game_row,
    odds_api_finder,
)
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_payload
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.backfill_checks import (
    ResponseCheck,
    book_judged_lead,
    check_response,
    stored_market,
)
from odds_analytics.backfill_plan import Call, Manifest

logger = structlog.get_logger()

CallStatus = Literal["fetched", "skipped", "would_fetch", "failed"]


@dataclass
class BackfillResult:
    """What a run did: ``calls_made`` paid calls, ``calls_skipped`` calls whose fixtures were
    already stored, ``would_fetch`` calls a dry run did not make. ``dropped_games`` holds each
    game the run dropped once, with the number of calls that listed it."""

    calls_made: int = 0
    calls_skipped: int = 0
    would_fetch: int = 0
    units_used: int = 0
    snapshots_stored: int = 0
    fixtures_stored: int = 0
    quota_remaining: int | None = None
    dropped_games: dict[DroppedGame, int] = field(default_factory=dict)


@dataclass(frozen=True)
class CallProgress:
    """One call's outcome, reported as the run goes."""

    time: datetime
    status: CallStatus
    fixtures_stored: int = 0
    units_used: int | None = None
    quota_remaining: int | None = None
    message: str | None = None


class BackfillStopped(Exception):
    """A guard ended the run; ``result`` is what it did before."""

    def __init__(self, reason: str, result: BackfillResult) -> None:
        super().__init__(reason)
        self.result = result


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
        max_units: int | None = None,
        max_consecutive_gaps: int = 10,
    ):
        """
        Args:
            manifest: Fixes the sport, markets, and bookmakers or regions
            client: API client (one spending ``APIConfig.acquisition_key`` is created and closed
                with the executor if not given, except in a dry run, which needs none)
            session_factory: Creates database sessions (defaults to ``async_session_maker``)
            skip_existing: Skip a call when any fixture it serves already has the snapshot it
                would return
            dry_run: Make no API call and write nothing
            rate_limit_seconds: Pause after each paid call
            max_units: Stop before a call that would take the run's billed units past this
            max_consecutive_gaps: Stop after this many consecutive fetched calls that return no
                events or lack a required bookmaker on a market
        """
        if max_consecutive_gaps < 1:
            raise ValueError(f"max_consecutive_gaps must be at least 1, got {max_consecutive_gaps}")
        self.manifest = manifest
        self._client = client
        self._owns_client = client is None
        self._session_factory = session_factory or async_session_maker
        self.skip_existing = skip_existing
        self.dry_run = dry_run
        self.rate_limit_seconds = rate_limit_seconds
        self.max_units = max_units
        self.max_consecutive_gaps = max_consecutive_gaps

    async def __aenter__(self) -> "BackfillExecutor":
        if self._owns_client and not self.dry_run:
            self._client = TheOddsAPIClient(api_key=get_settings().api.acquisition_key())
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
        """Run ``calls`` in order, stopping at the first exception or guard.

        Each call is paid, and a rerun skips the calls already stored, so a failure ends the run
        after reporting it rather than spending on. A guard raises ``BackfillStopped`` when a
        non-empty call's billed units differ from the manifest's or are missing (an empty one
        bills nothing), when a call returns a snapshot other than the planned one
        (``ResponseCheck.lag_ok``), when the next call would pass ``max_units``, or after
        ``max_consecutive_gaps`` consecutive sparse responses. A call that trips a guard is stored
        first, since it is paid.
        """
        result = BackfillResult()
        sparse_streak = 0

        def report(progress: CallProgress) -> None:
            if progress_callback:
                progress_callback(progress)

        try:
            for call in calls:
                try:
                    progress, check = await self._run_call(call, result)
                except BackfillStopped:
                    raise
                except Exception as e:
                    logger.error("backfill_call_failed", call_time=utc_isoformat(call.time))
                    report(CallProgress(call.time, "failed", message=str(e)))
                    raise
                report(progress)
                if check is not None:
                    sparse_streak = sparse_streak + 1 if check.sparse_reason else 0
                    self._guard(check, sparse_streak, result)
                if progress.status == "fetched":
                    await asyncio.sleep(self.rate_limit_seconds)
        finally:
            await alert_dropped_games("odds-backfill", list(result.dropped_games))
        return result

    def _guard(self, check: ResponseCheck, sparse_streak: int, result: BackfillResult) -> None:
        if not check.units_ok:
            raise BackfillStopped(
                f"call billed {check.units} units, expected "
                f"{'' if check.events else '0 or '}{self.manifest.probe.units_per_call}",
                result,
            )
        if not check.lag_ok:
            raise BackfillStopped(
                f"call for the {utc_isoformat(check.modelled)} snapshot returned the "
                f"{utc_isoformat(check.snapshot_time)} snapshot",
                result,
            )
        if sparse_streak >= self.max_consecutive_gaps:
            raise BackfillStopped(
                f"{sparse_streak} consecutive sparse responses, the last {check.sparse_reason}",
                result,
            )

    async def probe(self) -> list[ResponseCheck]:
        """Request the manifest's probe snapshots, store what they return as a run does, and
        judge each response against the manifest. An erroring call stops the probe."""
        result = BackfillResult()
        client = self._require_client()
        checks: list[ResponseCheck] = []
        for snapshot in self.manifest.probe.snapshots:
            try:
                _, check = await self._fetch_and_store(
                    Call(snapshot, ()), result, client, judge_every_event=True
                )
            except Exception as e:
                await self._log_failed_fetch(client, result, e)
                raise
            checks.append(check)
            await asyncio.sleep(self.rate_limit_seconds)
        await alert_dropped_games("odds-backfill", list(result.dropped_games))
        return checks

    def _require_client(self) -> TheOddsAPIClient:
        if self._client is None:
            raise RuntimeError("BackfillExecutor client not initialized")
        return self._client

    async def _run_call(
        self, call: Call, result: BackfillResult
    ) -> tuple[CallProgress, ResponseCheck | None]:
        if self.skip_existing and await self._already_stored(call):
            result.calls_skipped += 1
            return CallProgress(call.time, "skipped", message="Already stored"), None
        if self.dry_run:
            result.would_fetch += 1
            return CallProgress(call.time, "would_fetch"), None

        client = self._require_client()
        if (
            self.max_units is not None
            and result.units_used + self.manifest.units_per_call > self.max_units
        ):
            raise BackfillStopped(
                f"the next call would pass --max-units {self.max_units} "
                f"({result.units_used} billed)",
                result,
            )
        try:
            return await self._fetch_and_store(call, result, client)
        except Exception as e:
            await self._log_failed_fetch(client, result, e)
            raise

    async def _fetch_and_store(
        self,
        call: Call,
        result: BackfillResult,
        client: TheOddsAPIClient,
        judge_every_event: bool = False,
    ) -> tuple[CallProgress, ResponseCheck]:
        manifest = self.manifest
        response = await client.get_historical_odds(
            sport=manifest.sport,
            date=utc_isoformat(manifest.request_time(call.time)),
            markets=[str(market) for market in manifest.markets],
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
        attached: dict[int, dict[str, Any]] = {}
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
                found = await attach_historical_odds_api_event(finder, session, event)
                if isinstance(found, DroppedGame):
                    result.dropped_games[found] = result.dropped_games.get(found, 0) + 1
                    continue
                if payload_id is None:
                    payload_id = await store_payload(session, response.raw_events_data)
                envelope = await writer.store_envelope(
                    event_id=found.event.id,
                    envelope=raw,
                    snapshot_time=response.snapshot_time,
                    shape=SnapshotShape(Source.ODDS_API, TimeBasis.SOURCE_STAMPED, payload_id),
                )
                result.snapshots_stored += envelope.snapshots_written
                attached[found.event.id] = raw
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
                    response_time_ms=response.response_time_ms,
                )
            )
            await session.commit()
        result.fixtures_stored += stored
        progress = CallProgress(
            call.time,
            "fetched",
            fixtures_stored=stored,
            units_used=response.units_used,
            quota_remaining=response.quota_remaining,
        )
        served = {target.fixture.event_id for target in call.serves}
        horizon = book_judged_lead(manifest)
        near = {
            target.fixture.event_id
            for target in call.serves
            if target.fixture.kickoff - response.snapshot_time <= horizon
        }
        judged = (
            response.raw_events_data
            if judge_every_event
            else [raw for event_id, raw in attached.items() if event_id in near] or None
        )
        check = check_response(
            manifest,
            response,
            call.time,
            judged_events=judged,
            served=len(served),
            served_stored=len(served & attached.keys()),
        )
        return progress, check

    async def _log_failed_fetch(
        self, client: TheOddsAPIClient, result: BackfillResult, error: Exception
    ) -> None:
        """Record a failed call's fetch log in its own transaction, so it survives the call's
        rollback."""
        result.quota_remaining = client.quota_remaining
        async with self._session_factory() as session:
            await OddsWriter(session).log_fetch(
                FetchLog(
                    sport_key=self.manifest.sport,
                    events_count=0,
                    bookmakers_count=0,
                    success=False,
                    error_message=f"{type(error).__name__}: {error}",
                )
            )
            await session.commit()

    async def _already_stored(self, call: Call) -> bool:
        """Whether any fixture the call serves already has a historical Odds API snapshot
        (``source_stamped``, never a live fetch) of a requested market in
        ``(request time - grid step, request time]``, where the snapshot the request returns
        falls. A served fixture the response does not list is one the API did not list at that
        time, and fetching again will not change that."""
        wanted = {target.fixture.event_id for target in call.serves}
        if not wanted:
            return True
        markets = {stored_market(self.manifest, market) for market in self.manifest.markets}
        step = self.manifest.grid_step_at(call.time)
        requested = self.manifest.request_time(call.time)
        async with self._session_factory() as session:
            stored = await _stored_markets(session, wanted, requested - step, requested)
        return any(markets & stored.get(event_id, set()) for event_id in wanted)


async def _stored_markets(
    session: AsyncSession, event_ids: set[int], after: datetime, until: datetime
) -> dict[int, set[Market]]:
    rows = await session.execute(
        select(OddsSnapshot.event_id, OddsSnapshot.market)
        .where(
            OddsSnapshot.event_id.in_(event_ids),  # type: ignore[attr-defined]
            OddsSnapshot.source == Source.ODDS_API,
            OddsSnapshot.time_basis == TimeBasis.SOURCE_STAMPED,
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
