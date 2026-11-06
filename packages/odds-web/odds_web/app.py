"""The FastAPI app: server-rendered pages over cached, read-only report data."""

from __future__ import annotations

from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from odds_analytics.candidate_report import candidate_report
from odds_core.candidates import candidates_dir, load_candidates

from odds_web.cache import Cached, SingleFlightCache
from odds_web.db import WebDatabase, web_database_from_settings
from odds_web.portfolio import AXIS_PERCENT, FOREST_WIDTH, PortfolioRow, portfolio_rows

PACKAGE_DIR = Path(__file__).parent


@dataclass(frozen=True)
class Section[T]:
    """A page section's data, or why it is unavailable and when it was last good."""

    value: T | None
    as_of: datetime | None
    error: str | None

    @classmethod
    def of(cls, cached: Cached[T]) -> Section[T]:
        return cls(cached.value, cached.as_of, cached.error)


def _signed(value: float, digits: int = 1) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _clock(value: datetime | None) -> str:
    return "never" if value is None else value.strftime("%H:%M:%S UTC")


def create_app(
    database: WebDatabase,
    registry_dir: Path | None = None,
    cache: SingleFlightCache | None = None,
) -> FastAPI:
    """The app over ``database``, whose sessions every reader takes."""
    # One thread, so at most one bootstrap runs at a time.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bootstrap")
    cache = cache or SingleFlightCache()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        executor.shutdown(wait=False, cancel_futures=True)
        await database.engine.dispose()

    app = FastAPI(title="odds-web", lifespan=lifespan, docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=PACKAGE_DIR / "static"), name="static")
    templates = Jinja2Templates(directory=PACKAGE_DIR / "templates")
    templates.env.filters["signed"] = _signed
    templates.env.filters["clock"] = _clock
    templates.env.globals.update(axis_percent=f"{AXIS_PERCENT:g}", forest_width=FOREST_WIDTH)

    async def load_portfolio() -> list[PortfolioRow]:
        registry = load_candidates(registry_dir or candidates_dir())
        async with database.session_maker() as session:
            summaries = await candidate_report(session, registry, executor=executor)
        return portfolio_rows(summaries, registry)

    async def portfolio_section() -> Section[list[PortfolioRow]]:
        return Section.of(await cache.get("portfolio", load_portfolio))

    @app.get("/", response_class=HTMLResponse)
    async def portfolio(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "portfolio.html", {"portfolio": await portfolio_section()}
        )

    @app.get("/fragments/portfolio", response_class=HTMLResponse)
    async def portfolio_fragment(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request, "_portfolio.html", {"portfolio": await portfolio_section()}
        )

    return app


def create_app_from_settings() -> FastAPI:
    """The app on ``WEB_DATABASE_URL``; raises ``WebDatabaseNotConfigured`` when it is unset."""
    return create_app(web_database_from_settings())
