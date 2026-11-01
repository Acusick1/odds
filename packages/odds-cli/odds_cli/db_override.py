"""Shared helper for overriding the application's database from the CLI."""

from __future__ import annotations

import os
from urllib.parse import urlsplit, urlunsplit


def swap_dbname(url: str, new_db: str) -> str:
    """Return ``url`` with its dbname component replaced by ``new_db``.

    Rejects dbnames containing ``/`` or ``?`` so an injected query string or
    path separator can't smuggle extra URL components past the swap.
    """
    if "/" in new_db or "?" in new_db:
        raise ValueError(f"invalid dbname: {new_db!r} (must not contain '/' or '?')")
    parts = urlsplit(url)
    return urlunsplit(parts._replace(path=f"/{new_db}"))


def override_database_url(new_db: str) -> str:
    """Point this process and its subprocesses at database ``new_db``.

    Swaps the dbname in the resolved ``DATABASE_URL`` (honouring ``.env``),
    and in ``AGENT_DATABASE_URL`` when set so the agent subprocess keeps its
    restricted role, writes both to ``os.environ`` so subprocesses inherit
    them, and clears the cached Settings so later ``get_settings()`` calls and
    ``odds_core.database`` sessions use the new database. Returns the new
    ``DATABASE_URL``.
    """
    from odds_core.config import get_settings, reset_settings_cache

    database = get_settings().database
    new_url = swap_dbname(database.url, new_db)
    os.environ["DATABASE_URL"] = new_url
    if database.agent_url:
        os.environ["AGENT_DATABASE_URL"] = swap_dbname(database.agent_url, new_db)
    reset_settings_cache()
    return new_url
