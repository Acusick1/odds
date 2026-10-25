"""Database write operations for odds data."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import structlog
from odds_core.event_ids import LIVE_EVENT_ID_PREFIX
from odds_core.event_source import canonical_event_id
from odds_core.models import DataQualityLog, Event, EventStatus, FetchLog, Odds, OddsSnapshot
from odds_core.odds_math import outcome_price_decimal
from odds_core.team import normalize_team_name, team_abbrev
from odds_core.time import ensure_utc, parse_api_datetime
from sqlalchemy import and_, case, literal, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.mlb_game_resolver import MLB_SPORT_KEY, ResolvedGame
from odds_lambda.storage.event_merge import (
    find_match_candidates,
    find_match_id,
    follow_schedule,
)
from odds_lambda.storage.validators import OddsValidator
from odds_lambda.tier_utils import calculate_hours_until_commence, calculate_tier_from_timestamps

logger = structlog.get_logger()


class OddsWriter:
    """Handles all write operations to the database."""

    LIVE_EVENT_STALE_THRESHOLD = timedelta(hours=1)

    def __init__(self, session: AsyncSession):
        """
        Initialize writer with database session.

        Args:
            session: Async database session
        """
        self.session = session
        self.validator = OddsValidator()

    async def upsert_event(self, event: Event) -> Event:
        """
        Insert or update an event.

        Args:
            event: Event instance to upsert

        Returns:
            Created or updated Event instance

        Example:
            event = Event(
                id="abc123",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20),
                home_team="Lakers",
                away_team="Celtics"
            )
            result = await writer.upsert_event(event)
        """
        # Check if event exists
        result = await self.session.execute(select(Event).where(Event.id == event.id))
        existing_event = result.scalar_one_or_none()

        if existing_event:
            # Update existing event with new data
            existing_event.sport_key = event.sport_key
            existing_event.sport_title = event.sport_title
            existing_event.home_team = event.home_team
            existing_event.away_team = event.away_team
            existing_event.commence_time = event.commence_time
            existing_event.updated_at = datetime.now(UTC)

            logger.info("event_updated", event_id=event.id)
            return existing_event
        else:
            # Create new event
            self.session.add(event)
            logger.info("event_created", event_id=event.id)
            return event

    @staticmethod
    def _build_event_id(home_team: str, away_team: str, match_date: datetime) -> str:
        """Generate deterministic event ID for OddsPortal-sourced events.

        Includes the match start time down to the minute so same-day
        doubleheaders produce distinct ids. The ±2h match window in
        find_or_create_event handles minor jitter across re-scrapes.
        """
        home_abbrev = team_abbrev(home_team)
        away_abbrev = team_abbrev(away_team)
        date_str = match_date.strftime("%Y-%m-%dT%H%M")
        return f"{LIVE_EVENT_ID_PREFIX}{home_abbrev}_{away_abbrev}_{date_str}"

    async def find_unscheduled_mlb_event(
        self, home_team: str, away_team: str, match_date: datetime, game_number: int | None
    ) -> str | None:
        """The MLB event of a listed game while the MLB schedule is unavailable: the one event of
        the pair within ±2 h. A doubleheader game (``game_number`` set) or a window holding
        several events matches nothing, since without the schedule it could be another game."""
        if game_number is not None:
            return None
        candidates = await find_match_candidates(
            self.session,
            normalize_team_name(home_team),
            normalize_team_name(away_team),
            match_date,
            MLB_SPORT_KEY,
        )
        return candidates[0].id if len(candidates) == 1 else None

    async def find_or_create_event(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        sport_key: str,
        sport_title: str,
        *,
        mlb_game: ResolvedGame | None = None,
    ) -> tuple[str, bool]:
        """Find the event for a listed game or create it.

        An MLB game must come resolved (``mlb_game``): see ``_find_or_create_game``. Other
        sports match on teams within ±2 h.

        Returns:
            (event_id, was_created)
        """
        home_team = normalize_team_name(home_team)
        away_team = normalize_team_name(away_team)
        if sport_key == MLB_SPORT_KEY:
            if mlb_game is None:
                raise ValueError(f"an MLB event needs its gamePk: {home_team} v {away_team}")
            return await self._find_or_create_game(
                home_team, away_team, match_date, sport_title, mlb_game
            )

        existing_id = await find_match_id(self.session, home_team, away_team, match_date, sport_key)
        if existing_id is not None:
            return existing_id, False
        event_id = self._build_event_id(home_team, away_team, match_date)
        if await self.session.get(Event, event_id) is not None:
            return event_id, False
        self._add_event(event_id, home_team, away_team, match_date, sport_key, sport_title)
        return event_id, True

    async def _event_by_game_pk(self, game_pk: int) -> Event | None:
        result = await self.session.execute(select(Event).where(Event.mlb_game_pk == game_pk))
        return result.scalar_one_or_none()

    async def _find_or_create_game(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        sport_title: str,
        game: ResolvedGame,
    ) -> tuple[str, bool]:
        """The event holding the game's ``gamePk``; else a claimable event, which takes it; else a
        new event created with it. The event's ``commence_time`` follows the game's scheduled
        start when the schedule has one."""
        anchor = game.start or match_date
        held = await self._event_by_game_pk(game.game_pk)
        if held is not None:
            await follow_schedule(self.session, held, game.game_pk, game.start)
            return held.id, False

        claimable = await self._claimable_event(home_team, away_team, anchor, game)
        # Read before the savepoint: a rollback expires the claimed event's attributes.
        claimable_id = claimable.id if claimable is not None else None
        # A concurrent run may take the gamePk first; the savepoint keeps its unique-index
        # violation from aborting the rest of this run's batch.
        try:
            async with self.session.begin_nested():
                if claimable is not None:
                    claimable.assign_game_pk(game.game_pk)
                    await follow_schedule(self.session, claimable, game.game_pk, game.start)
                    outcome = (claimable.id, False)
                else:
                    outcome = (
                        await self._create_game_event(
                            home_team, away_team, anchor, sport_title, game.game_pk
                        ),
                        True,
                    )
        except IntegrityError:
            held = await self._event_by_game_pk(game.game_pk)
            if held is None:
                raise
            logger.warning(
                "mlb_game_pk_duplicate",
                game_pk=game.game_pk,
                event_id=held.id,
                duplicate_id=claimable_id,
            )
            await follow_schedule(self.session, held, game.game_pk, game.start)
            return held.id, False
        return outcome

    async def _claimable_event(
        self, home_team: str, away_team: str, anchor: datetime, game: ResolvedGame
    ) -> Event | None:
        """The nearest same-pair event within ±2 h of ``anchor``, when it holds no ``gamePk`` and
        no other game of the pair that day is scheduled nearer to it."""
        candidates = await find_match_candidates(
            self.session, home_team, away_team, anchor, MLB_SPORT_KEY
        )
        if not candidates:
            return None

        def gap(event: Event, start: datetime) -> timedelta:
            return abs(ensure_utc(event.commence_time) - ensure_utc(start))

        nearest_gap = min(gap(e, anchor) for e in candidates)
        nearest_ids = [e.id for e in candidates if gap(e, anchor) == nearest_gap]
        nearest = next(e for e in candidates if e.id == canonical_event_id(nearest_ids))
        if nearest.mlb_game_pk is not None:
            return None
        if any(gap(nearest, start) < nearest_gap for start in game.sibling_starts):
            return None
        return nearest

    async def _create_game_event(
        self, home_team: str, away_team: str, start: datetime, sport_title: str, game_pk: int
    ) -> str:
        event_id = self._build_event_id(home_team, away_team, start)
        taken = await self.session.get(Event, event_id)
        if taken is not None:
            logger.warning(
                "mlb_event_id_collision",
                event_id=event_id,
                held_game_pk=taken.mlb_game_pk,
                game_pk=game_pk,
            )
            event_id = f"{event_id}_{game_pk}"
        self._add_event(event_id, home_team, away_team, start, MLB_SPORT_KEY, sport_title, game_pk)
        return event_id

    def _add_event(
        self,
        event_id: str,
        home_team: str,
        away_team: str,
        match_date: datetime,
        sport_key: str,
        sport_title: str,
        mlb_game_pk: int | None = None,
    ) -> None:
        if event_id.startswith(LIVE_EVENT_ID_PREFIX):
            stale_cutoff = datetime.now(UTC) - self.LIVE_EVENT_STALE_THRESHOLD
            if ensure_utc(match_date) < stale_cutoff:
                raise ValueError(
                    "Refusing to create stale live-scrape event: "
                    f"home={home_team} away={away_team} "
                    f"match_date={match_date.isoformat()}"
                )
        self.session.add(
            Event(
                id=event_id,
                sport_key=sport_key,
                sport_title=sport_title,
                commence_time=match_date,
                home_team=home_team,
                away_team=away_team,
                status=EventStatus.SCHEDULED,
                mlb_game_pk=mlb_game_pk,
            )
        )

    async def store_odds_snapshot(
        self,
        event_id: str,
        raw_data: dict,
        snapshot_time: datetime | None = None,
        validate: bool = True,
    ) -> tuple[OddsSnapshot, list[Odds]]:
        """
        Store odds snapshot with hybrid storage approach.

        Args:
            event_id: Event identifier
            raw_data: Complete API response for the event
            snapshot_time: Time of snapshot (defaults to now)
            validate: Whether to run validation checks

        Returns:
            Tuple of (OddsSnapshot, list of normalized Odds records)

        Storage:
            - Raw JSONB snapshot for debugging
            - Normalized odds records for querying
            - Fetch tier and hours_until_commence for validation and ML features
        """
        snapshot_time = ensure_utc(snapshot_time or datetime.now(UTC))

        # Get event to calculate tier
        result = await self.session.execute(select(Event).where(Event.id == event_id))
        event = result.scalar_one_or_none()

        # Always calculate tier and hours_until_commence per-event from timestamps
        tier_value = None
        hours_until = None
        if event:
            event_commence = ensure_utc(event.commence_time)
            hours_until = calculate_hours_until_commence(snapshot_time, event_commence)
            calculated_tier = calculate_tier_from_timestamps(snapshot_time, event_commence)
            tier_value = calculated_tier.value

        # Validate if enabled
        if validate:
            is_valid, warnings = self.validator.validate_odds_snapshot(raw_data, event_id)
            if warnings:
                quality_log = DataQualityLog(
                    event_id=event_id,
                    severity="warning" if is_valid else "error",
                    issue_type="validation_warnings",
                    description=f"Validation found {len(warnings)} issues",
                    raw_data={"warnings": warnings[:10]},  # Store first 10
                )
                await self.log_data_quality_issue(quality_log)

        # Store raw snapshot
        bookmakers = raw_data.get("bookmakers", [])
        if isinstance(raw_data, list):
            # Find specific event in list
            event_data = next((e for e in raw_data if e.get("id") == event_id), {})
            bookmakers = event_data.get("bookmakers", [])

        snapshot = OddsSnapshot(
            event_id=event_id,
            snapshot_time=snapshot_time,
            raw_data=raw_data,
            bookmaker_count=len(bookmakers),
            fetch_tier=tier_value,
            hours_until_commence=hours_until,
        )
        self.session.add(snapshot)

        # Parse and store normalized odds
        odds_records = await self._parse_and_store_odds(
            event_id=event_id,
            bookmakers=bookmakers,
            snapshot_time=snapshot_time,
        )

        logger.info(
            "odds_snapshot_stored",
            event_id=event_id,
            bookmakers=len(bookmakers),
            odds_records=len(odds_records),
            fetch_tier=tier_value,
            hours_until_commence=hours_until,
        )

        return snapshot, odds_records

    async def _parse_and_store_odds(
        self,
        event_id: str,
        bookmakers: list[dict],
        snapshot_time: datetime,
    ) -> list[Odds]:
        """
        Parse bookmaker data and create normalized odds records.

        Args:
            event_id: Event identifier
            bookmakers: List of bookmaker data from API
            snapshot_time: Time of snapshot

        Returns:
            List of created Odds records
        """
        odds_records = []

        for bookmaker in bookmakers:
            bookmaker_key = str(bookmaker.get("key") or "").strip()
            if not bookmaker_key:
                continue  # Skip malformed bookmaker entries

            bookmaker_title = str(bookmaker.get("title") or bookmaker_key)

            # Parse last_update timestamp
            last_update = snapshot_time
            last_update_str = bookmaker.get("last_update")
            if isinstance(last_update_str, str):
                try:
                    last_update = parse_api_datetime(last_update_str)
                except Exception:
                    last_update = snapshot_time

            # Process each market
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                # Process each outcome
                for outcome in market.get("outcomes", []):
                    odds = Odds(
                        event_id=event_id,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market_key=market_key,
                        outcome_name=outcome.get("name"),
                        price_decimal=outcome_price_decimal(outcome),
                        point=outcome.get("point"),  # Will be None for h2h
                        odds_timestamp=snapshot_time,
                        last_update=last_update,
                        is_valid=True,
                    )
                    self.session.add(odds)
                    odds_records.append(odds)

        return odds_records

    async def update_event_status(
        self,
        event_id: str,
        status: EventStatus,
        home_score: int | None = None,
        away_score: int | None = None,
    ) -> Event | None:
        """
        Update event status and scores.

        Args:
            event_id: Event identifier
            status: New event status
            home_score: Final home team score (if completed)
            away_score: Final away team score (if completed)

        Returns:
            Updated Event or None if not found

        Raises:
            ValueError: If ``status`` is FINAL and either score is missing, or the stored
                status cannot become ``status`` (``EventStatus.can_become``)
        """
        if status == EventStatus.FINAL and (home_score is None or away_score is None):
            raise ValueError(f"event {event_id} cannot be FINAL without both scores")

        result = await self.session.execute(select(Event).where(Event.id == event_id))
        event = result.scalar_one_or_none()

        if not event:
            logger.warning("event_not_found", event_id=event_id)
            return None

        if not event.status.can_become(status):
            raise ValueError(
                f"event {event_id} cannot move from {event.status.value} to {status.value}"
            )

        event.status = status
        event.updated_at = datetime.now(UTC)

        if status == EventStatus.FINAL:
            event.home_score = home_score
            event.away_score = away_score
            event.completed_at = datetime.now(UTC)

        logger.info(
            "event_status_updated",
            event_id=event_id,
            status=status.value,
            home_score=home_score,
            away_score=away_score,
        )

        return event

    async def log_fetch(self, fetch_log: FetchLog) -> FetchLog:
        """
        Log an API fetch operation.

        Args:
            fetch_log: FetchLog instance to persist

        Returns:
            Created FetchLog record

        Example:
            fetch_log = FetchLog(
                sport_key="basketball_nba",
                events_count=10,
                bookmakers_count=8,
                success=True,
                api_quota_remaining=15000
            )
            result = await writer.log_fetch(fetch_log)
        """
        self.session.add(fetch_log)

        logger.info(
            "fetch_logged",
            sport_key=fetch_log.sport_key,
            events_count=fetch_log.events_count,
            success=fetch_log.success,
            quota_remaining=fetch_log.api_quota_remaining,
        )

        return fetch_log

    async def log_data_quality_issue(self, quality_log: DataQualityLog) -> DataQualityLog:
        """
        Log a data quality issue.

        Args:
            quality_log: DataQualityLog instance to persist

        Returns:
            Created DataQualityLog record

        Example:
            quality_log = DataQualityLog(
                event_id="abc123",
                severity="warning",
                issue_type="missing_data",
                description="Bookmaker X missing from response"
            )
            result = await writer.log_data_quality_issue(quality_log)
        """
        self.session.add(quality_log)

        logger.warning(
            "data_quality_issue",
            event_id=quality_log.event_id,
            severity=quality_log.severity,
            issue_type=quality_log.issue_type,
            description=quality_log.description,
        )

        return quality_log

    async def bulk_insert_odds(self, odds_records: list[Odds]) -> int:
        """
        Bulk insert odds records for efficient backfill.

        Args:
            odds_records: List of Odds instances

        Returns:
            Number of records inserted

        Note:
            Uses PostgreSQL INSERT ... ON CONFLICT DO NOTHING for efficiency

        Example:
            odds = [
                Odds(event_id="abc", bookmaker_key="fanduel", ...),
                Odds(event_id="abc", bookmaker_key="draftkings", ...),
            ]
            count = await writer.bulk_insert_odds(odds)
        """
        if not odds_records:
            return 0

        # Convert Odds instances to dicts for bulk insert
        odds_dicts = [odd.model_dump(exclude_unset=True) for odd in odds_records]

        stmt = insert(Odds).values(odds_dicts)
        stmt = stmt.on_conflict_do_nothing()

        result = await self.session.execute(stmt)
        count = result.rowcount

        logger.info("bulk_odds_inserted", count=count)
        return count

    async def bulk_upsert_events(
        self, events: list[Event], now: datetime | None = None
    ) -> dict[str, int]:
        """
        Bulk upsert event records for efficient backfill operations.

        Args:
            events: List of Event instances to insert or update
            now: Clock for the LIVE exception below (defaults to the current time)

        Returns:
            Dictionary with keys 'inserted' and 'updated' containing approximate counts

        Note:
            Uses PostgreSQL INSERT ... ON CONFLICT DO UPDATE pattern.
            Preserves created_at timestamps on updates, updates updated_at field.
            Status, scores and completed_at are never overwritten, because the incoming
            events are discovery records (scheduled, no scores). The exceptions
            are a POSTPONED event whose kick-off moved and a LIVE event whose incoming
            kick-off is after ``now``, which return to SCHEDULED.

        Example:
            events = [
                Event(id="abc", sport_key="basketball_nba", ...),
                Event(id="def", sport_key="basketball_nba", ...),
            ]
            result = await writer.bulk_upsert_events(events)
            # result = {'inserted': 1, 'updated': 1}
        """
        if not events:
            logger.info("bulk_events_upserted", inserted=0, updated=0)
            return {"inserted": 0, "updated": 0}

        # Deduplicate by event ID (API returns same event across multiple dates)
        seen: dict[str, Event] = {}
        for event in events:
            seen[event.id] = event
        events = list(seen.values())

        # Determine which events already exist
        event_ids = [event.id for event in events]
        result = await self.session.execute(select(Event.id).where(Event.id.in_(event_ids)))
        existing_ids = {row[0] for row in result.fetchall()}

        # Calculate counts
        updated_count = len(existing_ids)
        inserted_count = len(events) - updated_count

        # Convert Event instances to dicts for bulk upsert
        # Use model_dump() to include all fields with defaults
        event_dicts = [event.model_dump() for event in events]

        # Build upsert statement
        stmt = insert(Event).values(event_dicts)

        events_table = Event.__table__
        result_columns = {"status", "home_score", "away_score", "completed_at"}
        # A discovery record carries no gamePk or mark, so stored ones are kept.
        kept_columns = {"created_at", "mlb_game_pk", "no_game_pk_reason", *result_columns}
        set_ = {
            col.name: stmt.excluded[col.name]
            for col in events_table.columns
            if not col.primary_key and col.name not in kept_columns
        }
        now = now or datetime.now(UTC)
        rescheduled = or_(
            and_(
                events_table.c.status == EventStatus.POSTPONED,
                events_table.c.commence_time != stmt.excluded.commence_time,
            ),
            and_(
                events_table.c.status == EventStatus.LIVE,
                stmt.excluded.commence_time > now,
            ),
        )
        set_["status"] = case(
            (rescheduled, literal(EventStatus.SCHEDULED, events_table.c.status.type)),
            else_=events_table.c.status,
        )
        # Override updated_at to current time (SQLAlchemy doesn't auto-apply Python defaults)
        set_["updated_at"] = datetime.now(UTC)

        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_=set_,
        )

        await self.session.execute(stmt)

        logger.info(
            "bulk_events_upserted",
            inserted=inserted_count,
            updated=updated_count,
            total=len(events),
        )

        return {"inserted": inserted_count, "updated": updated_count}
