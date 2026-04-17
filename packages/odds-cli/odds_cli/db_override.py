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
    """Swap the dbname in the current DATABASE_URL and reset the settings cache.

    Reads the already-resolved DATABASE_URL via pydantic settings (honouring
    ``.env``), replaces the dbname segment with ``new_db``, writes it back to
    ``os.environ`` so subprocesses inherit it, and clears the cached Settings
    so any later ``get_settings()`` call in this process re-reads the override.
    Returns the new URL.
    """
    from odds_core.config import get_settings, reset_settings_cache

    source = get_settings().database.url
    new_url = swap_dbname(source, new_db)
    os.environ["DATABASE_URL"] = new_url
    reset_settings_cache()
    return new_url
