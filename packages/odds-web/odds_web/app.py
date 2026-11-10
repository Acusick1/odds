"""The FastAPI app: server-rendered pages over cached, read-only report data."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from odds_analytics.candidate_report import CandidateReport, candidate_report
from odds_core.candidates import candidates_dir, load_candidates
from odds_core.config import Settings, get_settings

from odds_web.cache import Cached, Failure, SingleFlightCache
from odds_web.candidate_page import CandidatePage, build_candidate_page, has_decisions
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
from odds_web.decisions import DecisionHistory, load_decision_history
from odds_web.ledger import LedgerPage, load_ledger_page
from odds_web.portfolio import AXIS_PERCENT, FOREST_WIDTH, PortfolioRow, portfolio_rows
from odds_web.status import StatusStrip, load_status

PACKAGE_DIR = Path(__file__).parent

PageQuery = Annotated[int, Query(ge=1, le=100_000)]


@dataclass(frozen=True)
class Section[T]:
    """A page section's data and when it was loaded, and why the last refresh failed if it did."""

    value: T | None
    as_of: datetime | None
    failure: Failure | None

    @classmethod
    def of(cls, cached: Cached[T]) -> Section[T]:
        return cls(cached.value, cached.as_of, cached.failure)

    def map[U](self, convert: Callable[[T], U]) -> Section[U]:
        return Section(
            None if self.value is None else convert(self.value), self.as_of, self.failure
        )


def _signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _percent(value: float | None, digits: int = 1) -> str:
    """A fraction as a signed percentage; a dash where there is none."""
    return "—" if value is None else _signed(value * 100, digits) + "%"


def _clock(value: datetime | None) -> str:
    return "never" if value is None else value.strftime("%H:%M UTC")


def _stamp(value: datetime | None) -> str:
    return "never" if value is None else value.strftime("%Y-%m-%d %H:%M UTC")


def _hhmm(value: datetime | None, now: datetime) -> str:
    """A UTC clock time, led by the weekday when it is not on ``now``'s date."""
    if value is None:
        return "—"
    return value.strftime("%H:%M" if value.date() == now.date() else "%a %H:%M")


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
    templates.env.filters["hhmm"] = _hhmm
    templates.env.filters["stamp"] = _stamp
    templates.env.filters["percent"] = _percent
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

    async def load_report() -> CandidateReport:
        registry = load_candidates(registry_dir or candidates_dir())
        async with database.session_maker() as session:
            return await candidate_report(session, registry, executor=executor)

    async def report_section() -> Cached[CandidateReport]:
        return await cache.get("report", load_report)

    async def portfolio_section() -> Section[list[PortfolioRow]]:
        report = Section.of(await report_section())
        return report.map(lambda r: portfolio_rows(r.summaries))

    async def candidate_section(name: str, version: int, page: int) -> Section[CandidatePage]:
        report = await report_section()
        if report.value is None:
            return Section(None, report.as_of, report.failure)
        summary = next(
            (s for s in report.value.summaries if (s.name, s.version) == (name, version)), None
        )
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No candidate {name} v{version}")
        running = report.value.running[(name, version)]

        async def load_listings() -> tuple[DecisionHistory | None, LedgerPage]:
            async with database.session_maker() as session:
                decisions = None
                if has_decisions(summary.registered):
                    decisions = await load_decision_history(
                        session, name, version, datetime.now(UTC)
                    )
                ledger = await load_ledger_page(session, name, version, page)
            return decisions, ledger

        # The figures and charts are built from the report entry on every request, so they
        # always describe one snapshot; only the listings are cached per page.
        listings = Section.of(await cache.get(f"candidate:{name}:{version}:{page}", load_listings))
        built = listings.map(lambda found: build_candidate_page(summary, running, *found))
        # The page is as old as the older of its two reads.
        as_of = (
            None if built.as_of is None or report.as_of is None else min(built.as_of, report.as_of)
        )
        return Section(built.value, as_of, built.failure or report.failure)

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

    @app.get("/fragments/status-detail", response_class=HTMLResponse)
    async def status_detail_fragment(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_status_detail.html.j2", {"status": await status_section()}
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
            {
                "coverage": await coverage_section(),
                "closes": await closes_section(),
                "status": await status_section(),
            },
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

    @app.get("/c/{name}/{version}", response_class=HTMLResponse)
    async def candidate(
        request: Request, name: str, version: int, page: PageQuery = 1
    ) -> HTMLResponse:
        section = await candidate_section(name, version, page)
        return templates.TemplateResponse(
            request,
            "candidate.html.j2",
            {"candidate": section, "name": name, "version": version, "page": page},
        )

    @app.get("/fragments/c/{name}/{version}", response_class=HTMLResponse)
    async def candidate_fragment(
        request: Request, name: str, version: int, page: PageQuery = 1
    ) -> HTMLResponse:
        section = await candidate_section(name, version, page)
        return templates.TemplateResponse(
            request,
            "_candidate.html.j2",
            {"candidate": section, "name": name, "version": version, "page": page},
        )

    return app


def create_app_from_settings() -> FastAPI:
    """The app on ``WEB_DATABASE_URL``; raises ``WebDatabaseNotConfigured`` when it is unset."""
    return create_app(web_database_from_settings())
