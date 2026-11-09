"""The FastAPI app: server-rendered pages over cached, read-only report data."""

from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from odds_analytics.candidate_report import candidate_report
from odds_core.candidates import candidates_dir, load_candidates
from odds_core.config import Settings, get_settings

from odds_web.cache import Cached, Failure, SingleFlightCache
from odds_web.closes import WEEKS as CLOSE_WEEKS
from odds_web.closes import CloseWeeks, close_coverage
from odds_web.coverage import (
    BUCKET_MINUTES,
    PAST,
    UPCOMING,
    CandidateCoverage,
    candidate_coverage,
    coverage_candidates,
)
from odds_web.db import WebDatabase, web_database_from_settings
from odds_web.portfolio import AXIS_PERCENT, FOREST_WIDTH, PortfolioRow, portfolio_rows
from odds_web.status import StatusStrip, load_status

PACKAGE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Section[T]:
    """A page section's data and when it was loaded, and why the last refresh failed if it did."""

    value: T | None
    as_of: datetime | None
    failure: Failure | None

    @classmethod
    def of(cls, cached: Cached[T]) -> Section[T]:
        return cls(cached.value, cached.as_of, cached.failure)


def _signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _clock(value: datetime | None) -> str:
    return "never" if value is None else value.strftime("%H:%M UTC")


def _stamp(value: datetime | None) -> str:
    return "never" if value is None else value.strftime("%Y-%m-%d %H:%M UTC")


def _ago(value: datetime | None, now: datetime) -> str:
    """How long before ``now`` ``value`` was, in its largest whole unit."""
    if value is None:
        return "never"
    seconds = max(0, int((now - value).total_seconds()))
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def create_app(
    database: WebDatabase,
    registry_dir: Path | None = None,
    cache: SingleFlightCache | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """The app over ``database``, whose sessions every reader takes; ``settings`` name the jobs,
    sources and thresholds the status strip judges by."""
    # One thread, so at most one bootstrap runs at a time.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bootstrap")
    cache = cache or SingleFlightCache()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="odds-web", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    templates.env.filters["signed"] = _signed
    templates.env.filters["clock"] = _clock
    templates.env.filters["stamp"] = _stamp
    templates.env.globals.update(
        ago=_ago,
        close_weeks=CLOSE_WEEKS,
        bucket_minutes=BUCKET_MINUTES,
        coverage_past_days=PAST.days,
        coverage_upcoming_hours=int(UPCOMING.total_seconds() // 3600),
    )
    # Paths, not absolute URLs: behind `tailscale serve` the app sees http while the page is https.
    templates.env.globals.update(
        axis_percent=f"{AXIS_PERCENT:g}", forest_width=FOREST_WIDTH, url_path=app.url_path_for
    )

    async def load_portfolio() -> list[PortfolioRow]:
        registry = load_candidates(registry_dir or candidates_dir())
        async with database.session_maker() as session:
            summaries = await candidate_report(session, registry, executor=executor)
        return portfolio_rows(summaries, registry)

    async def portfolio_section() -> Section[list[PortfolioRow]]:
        return Section.of(await cache.get("portfolio", load_portfolio))

    async def load_status_strip() -> StatusStrip:
        async with database.session_maker() as session:
            return await load_status(session, settings or get_settings(), datetime.now(UTC))

    async def load_coverage() -> list[CandidateCoverage]:
        registry = load_candidates(registry_dir or candidates_dir())
        now = datetime.now(UTC)
        async with database.session_maker() as session:
            return [
                await candidate_coverage(session, candidate, now)
                for candidate in coverage_candidates(registry)
            ]

    async def load_closes() -> list[CloseWeeks]:
        async with database.session_maker() as session:
            return await close_coverage(session, datetime.now(UTC))

    async def status_section() -> Section[StatusStrip]:
        return Section.of(await cache.get("status", load_status_strip))

    async def coverage_section() -> Section[list[CandidateCoverage]]:
        return Section.of(await cache.get("coverage", load_coverage))

    async def closes_section() -> Section[list[CloseWeeks]]:
        return Section.of(await cache.get("closes", load_closes))

    @app.get("/fragments/status", response_class=HTMLResponse)
    async def status_fragment(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_status.html.j2", {"status": await status_section()}
        )

    @app.get("/fragments/coverage", response_class=HTMLResponse)
    async def coverage_fragment(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_coverage.html.j2", {"coverage": await coverage_section()}
        )

    @app.get("/fragments/closes", response_class=HTMLResponse)
    async def closes_fragment(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_closes.html.j2", {"closes": await closes_section()}
        )

    @app.get("/data", response_class=HTMLResponse)
    async def data_health(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "data.html.j2",
            {"coverage": await coverage_section(), "closes": await closes_section()},
        )

    @app.get("/", response_class=HTMLResponse)
    async def portfolio(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "portfolio.html.j2", {"portfolio": await portfolio_section()}
        )

    @app.get("/fragments/portfolio", response_class=HTMLResponse)
    async def portfolio_fragment(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_portfolio.html.j2", {"portfolio": await portfolio_section()}
        )

    return app


def create_app_from_settings() -> FastAPI:
    """The app on ``WEB_DATABASE_URL``; raises ``WebDatabaseNotConfigured`` when it is unset."""
    return create_app(web_database_from_settings())
