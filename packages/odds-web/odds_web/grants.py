"""Apply the ``odds_web`` role's grants from ``odds_web.role_check.READ_TABLES``, its one allowlist.

``python -m odds_web.grants`` runs over ``DATABASE_URL`` as a role that can grant (the superuser
on the production host), and is safe to re-run: the role ends up able to ``SELECT`` from the
allowlist and nothing else in ``public``. ``deploy.sh`` runs it after each migration.
"""

from __future__ import annotations

import asyncio

from odds_core.config import get_settings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from odds_web.role_check import READ_TABLES

ROLE = "odds_web"


class GrantsFailed(RuntimeError):
    """The grants cannot be applied, for example because the role does not exist."""


async def apply_grants(connection: AsyncConnection) -> list[str]:
    """Make ``odds_web`` read-only by default and able to ``SELECT`` from ``READ_TABLES`` only.

    Every privilege on a public table is revoked before the allowlist is granted, in the caller's
    transaction, so a table dropped from the list loses its grant and none is readable between
    the two. The caller commits.
    """
    exists = await connection.scalar(
        text("SELECT EXISTS (SELECT FROM pg_roles WHERE rolname = :role)"), {"role": ROLE}
    )
    if not exists:
        raise GrantsFailed(
            f"the role {ROLE} does not exist; create it first (scripts/setup-odds-web-host.sh)"
        )
    quote = connection.dialect.identifier_preparer.quote
    tables = ", ".join(quote(table) for table in READ_TABLES)
    for statement in (
        f"ALTER ROLE {ROLE} SET default_transaction_read_only = on",
        f"GRANT USAGE ON SCHEMA public TO {ROLE}",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {ROLE}",
        f"GRANT SELECT ON {tables} TO {ROLE}",
    ):
        await connection.execute(text(statement))
    return [f"{ROLE} granted SELECT on {len(READ_TABLES)} tables: {', '.join(READ_TABLES)}"]


async def main() -> None:
    engine = create_async_engine(get_settings().database.url)
    try:
        async with engine.begin() as connection:
            for line in await apply_grants(connection):
                print(f"  {line}")
    except GrantsFailed as exc:
        raise SystemExit(f"odds_web grants failed: {exc}") from exc
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
