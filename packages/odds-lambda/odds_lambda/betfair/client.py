"""Betfair Exchange API client wrapping ``betfairlightweight``.

Read-only data access via the delayed application key. Provides:
  - ``BetfairExchangeClient.connect()`` — login (cert-based or interactive)
  - ``list_events(sport_cfg, lookahead_hours)``
  - ``list_match_odds_books(sport_cfg, event_ids)``

Returned dataclasses (``BetfairEvent``, ``BetfairBook``) are sport-agnostic
and feed the adapter layer.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import betfairlightweight
import structlog
from betfairlightweight import filters
from betfairlightweight.exceptions import APIError, BetfairError

from odds_lambda.betfair.constants import SportBetfairConfig

logger = structlog.get_logger()


@dataclass(frozen=True)
class BetfairEvent:
    """Subset of Betfair Event we care about."""

    betfair_event_id: str
    name: str
    open_date: datetime
    market_count: int


@dataclass(frozen=True)
class BetfairRunner:
    """One side of a Match Odds market."""

    selection_id: int
    runner_name: str
    best_back: float | None
    best_lay: float | None
    back_size: float | None
    lay_size: float | None
    last_price_traded: float | None


@dataclass(frozen=True)
class BetfairBook:
    """Full price snapshot for a single MATCH_ODDS market on one event."""

    market_id: str
    betfair_event_id: str
    betfair_event_name: str
    market_start_time: datetime
    market_status: str
    inplay: bool
    total_matched: float | None
    runners: list[BetfairRunner]


class BetfairExchangeClient:
    """Thin wrapper around ``betfairlightweight.APIClient``.

    Login mode is determined by whether a cert path is provided:
        - ``cert_file`` and ``cert_key`` set → non-interactive cert login
          (production: bypasses 2FA, suitable for unattended/Lambda use).
        - both unset → interactive login (development only; will prompt
          for 2FA if account has it enabled).
    """

    def __init__(
        self,
        username: str,
        password: str,
        app_key: str,
        cert_file: str | None = None,
        cert_key: str | None = None,
    ) -> None:
        self._username = username
        self._password = password
        self._app_key = app_key
        self._cert_file = cert_file
        self._cert_key = cert_key

        kwargs: dict[str, Any] = {
            "username": username,
            "password": password,
            "app_key": app_key,
        }
        if cert_file and cert_key:
            kwargs["cert_files"] = (cert_file, cert_key)
        self._trading = betfairlightweight.APIClient(**kwargs)

    @property
    def trading(self) -> betfairlightweight.APIClient:
        return self._trading

    def login(self) -> None:
        if self._cert_file and self._cert_key:
            logger.info("betfair_login_cert")
            self._trading.login()
        else:
            logger.warning(
                "betfair_login_interactive",
                note="cert paths not configured; falling back to identitysso login",
            )
            self._trading.login_interactive()

    def keep_alive(self) -> None:
        self._trading.keep_alive()

    def logout(self) -> None:
        try:
            self._trading.logout()
        except (BetfairError, APIError) as e:  # pragma: no cover — best-effort cleanup
            logger.warning("betfair_logout_failed", error=str(e))

    # ---------------------------------------------------------------- discovery

    def list_competitions(self, event_type_id: str) -> list[dict[str, Any]]:
        """Return all competitions for a sport (id, name, market_count)."""
        market_filter = filters.market_filter(event_type_ids=[event_type_id])
        comps = self._trading.betting.list_competitions(filter=market_filter)
        return [
            {
                "id": c.competition.id,
                "name": c.competition.name,
                "market_count": c.market_count,
            }
            for c in comps
        ]

    def resolve_competition_id(self, sport_cfg: SportBetfairConfig) -> str:
        """Return a single competition_id for the given sport.

        Strategy:
          1. If ``sport_cfg.competition_id`` is set, return it directly.
          2. Otherwise list competitions for the sport's event_type_id and pick
             the first one whose name matches any of the configured hints.
          3. Raise ValueError if nothing matches.
        """
        if sport_cfg.competition_id:
            return sport_cfg.competition_id

        comps = self.list_competitions(sport_cfg.event_type_id)
        for comp in comps:
            name_lc = (comp["name"] or "").lower()
            for hint in sport_cfg.competition_name_hints:
                if hint.lower() in name_lc:
                    logger.info(
                        "betfair_resolved_competition",
                        sport=sport_cfg.sport_key,
                        competition_id=comp["id"],
                        competition_name=comp["name"],
                    )
                    return str(comp["id"])

        # Surface the candidates to ease debugging
        candidates = ", ".join(f"{c['id']}={c['name']}" for c in comps[:20])
        raise ValueError(
            f"No matching competition for {sport_cfg.sport_key} "
            f"(hints={sport_cfg.competition_name_hints}). Candidates: {candidates}"
        )

    def list_events(
        self,
        sport_cfg: SportBetfairConfig,
        lookahead_hours: int,
    ) -> list[BetfairEvent]:
        """List upcoming events for a sport+competition within a window."""
        competition_id = self.resolve_competition_id(sport_cfg)

        now = datetime.now(UTC)
        time_range = filters.time_range(
            from_=now.isoformat(),
            to=(now + timedelta(hours=lookahead_hours)).isoformat(),
        )
        market_filter = filters.market_filter(
            event_type_ids=[sport_cfg.event_type_id],
            competition_ids=[competition_id],
            market_start_time=time_range,
        )
        events = self._trading.betting.list_events(filter=market_filter)

        out: list[BetfairEvent] = []
        for ev in events:
            open_date = ev.event.open_date
            if open_date.tzinfo is None:
                open_date = open_date.replace(tzinfo=UTC)
            out.append(
                BetfairEvent(
                    betfair_event_id=str(ev.event.id),
                    name=ev.event.name,
                    open_date=open_date,
                    market_count=ev.market_count,
                )
            )
        out.sort(key=lambda e: e.open_date)
        return out

    # --------------------------------------------------------------------- books

    def list_match_odds_books(
        self,
        sport_cfg: SportBetfairConfig,
        events: list[BetfairEvent],
    ) -> list[BetfairBook]:
        """Fetch best back/lay for the configured market type for given events."""
        if not events:
            return []

        event_ids = [e.betfair_event_id for e in events]
        cat_filter = filters.market_filter(
            event_type_ids=[sport_cfg.event_type_id],
            event_ids=event_ids,
            market_type_codes=[sport_cfg.market_type_code],
        )
        catalogues = self._trading.betting.list_market_catalogue(
            filter=cat_filter,
            market_projection=["RUNNER_DESCRIPTION", "EVENT", "MARKET_START_TIME"],
            max_results=200,
        )

        if not catalogues:
            return []

        market_ids = [c.market_id for c in catalogues]
        # 200-point weight budget. EX_BEST_OFFERS = 5/market at depth=3.
        # 40 markets safe per call; chunk to be conservative.
        BATCH = 25
        price_proj = filters.price_projection(price_data=["EX_BEST_OFFERS"])
        book_by_id: dict[str, Any] = {}
        for i in range(0, len(market_ids), BATCH):
            batch = market_ids[i : i + BATCH]
            books = self._trading.betting.list_market_book(
                market_ids=batch,
                price_projection=price_proj,
            )
            for b in books:
                book_by_id[b.market_id] = b

        results: list[BetfairBook] = []
        for cat in catalogues:
            book = book_by_id.get(cat.market_id)
            if book is None:
                continue

            runners: list[BetfairRunner] = []
            for runner in book.runners:
                runner_name = ""
                for cr in cat.runners:
                    if cr.selection_id == runner.selection_id:
                        runner_name = cr.runner_name or ""
                        break
                ex = runner.ex
                back = ex.available_to_back[0] if ex and ex.available_to_back else None
                lay = ex.available_to_lay[0] if ex and ex.available_to_lay else None
                runners.append(
                    BetfairRunner(
                        selection_id=runner.selection_id,
                        runner_name=runner_name,
                        best_back=float(back.price) if back else None,
                        best_lay=float(lay.price) if lay else None,
                        back_size=float(back.size) if back else None,
                        lay_size=float(lay.size) if lay else None,
                        last_price_traded=(
                            float(runner.last_price_traded)
                            if runner.last_price_traded is not None
                            else None
                        ),
                    )
                )

            event_name = cat.event.name if cat.event else ""
            event_id = str(cat.event.id) if cat.event else ""
            start = cat.market_start_time
            if start and start.tzinfo is None:
                start = start.replace(tzinfo=UTC)

            results.append(
                BetfairBook(
                    market_id=cat.market_id,
                    betfair_event_id=event_id,
                    betfair_event_name=event_name,
                    market_start_time=start or datetime.now(UTC),
                    market_status=str(book.status or ""),
                    inplay=bool(book.inplay),
                    total_matched=(
                        float(book.total_matched) if book.total_matched is not None else None
                    ),
                    runners=runners,
                )
            )

        return results


@asynccontextmanager
async def betfair_session(
    *,
    username: str | None,
    password: str | None,
    app_key: str | None,
    cert_file: str | None = None,
    cert_key: str | None = None,
) -> AsyncIterator[BetfairExchangeClient]:
    """Context manager: login, yield client, logout on exit."""
    if not (username and password and app_key):
        raise ValueError(
            "Betfair credentials missing: BETFAIR_USERNAME / BETFAIR_PASSWORD / BETFAIR_APP_KEY"
        )

    client = BetfairExchangeClient(
        username=username,
        password=password,
        app_key=app_key,
        cert_file=cert_file,
        cert_key=cert_key,
    )
    client.login()
    try:
        yield client
    finally:
        client.logout()


def client_from_env() -> BetfairExchangeClient:
    """Convenience constructor for scripts: read ``BETFAIR_*`` from env."""
    return BetfairExchangeClient(
        username=os.environ["BETFAIR_USERNAME"],
        password=os.environ["BETFAIR_PASSWORD"],
        app_key=os.environ["BETFAIR_APP_KEY"],
        cert_file=os.environ.get("BETFAIR_CERT_FILE"),
        cert_key=os.environ.get("BETFAIR_CERT_KEY"),
    )
