"""Apply the ``odds_web`` role's grants from ``odds_web.role_check``'s allowlists.

``python -m odds_web.grants`` runs over ``DATABASE_URL`` as a role that can grant (the superuser
on the production host), and is safe to re-run: the role ends up able to ``SELECT`` from
``READ_TABLES``, to execute the ``READ_FUNCTIONS`` this module defines, and nothing else in
``public``. ``deploy.sh`` runs it after each migration.
"""

from __future__ import annotations

import asyncio

from odds_core.config import get_settings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

from odds_web.role_check import READ_FUNCTIONS, READ_TABLES

ROLE = "odds_web"

# The role reads query statistics only through these, which run as their owner (the role applying
# the grants) so they see every role's statements, and return no more than the page shows. They
# are recreated on every run, so a change of their columns needs no migration.
STATISTICS_FUNCTIONS = (
    """
    CREATE FUNCTION public.odds_web_query_stats()
    RETURNS TABLE (
        queryid bigint, query text, role name, calls bigint,
        total_exec_time double precision, mean_exec_time double precision, rows bigint
    )
    LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog
    AS $$
        SELECT s.queryid, s.query, pg_get_userbyid(s.userid), s.calls,
               s.total_exec_time, s.mean_exec_time, s.rows
        FROM public.pg_stat_statements s
        WHERE s.dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
    $$
    """,
    """
    CREATE FUNCTION public.odds_web_query_stats_reset()
    RETURNS timestamptz
    LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog
    AS $$ SELECT stats_reset FROM public.pg_stat_statements_info $$
    """,
)


class GrantsFailed(RuntimeError):
    """The grants cannot be applied, for example because the role does not exist."""


async def apply_grants(connection: AsyncConnection) -> list[str]:
    """Make ``odds_web`` read-only by default, able to ``SELECT`` from ``READ_TABLES`` and to
    execute ``READ_FUNCTIONS`` only.

    Every privilege on a public table or function is revoked before the allowlists are granted,
    in the caller's transaction, so an entry dropped from a list loses its grant and none is
    readable between the two. The statistics functions read ``pg_stat_statements``, so the
    extension must exist. The caller commits.
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
    functions = ", ".join(f"public.{function}" for function in READ_FUNCTIONS)
    for statement in (
        f"ALTER ROLE {ROLE} SET default_transaction_read_only = on",
        f"GRANT USAGE ON SCHEMA public TO {ROLE}",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {ROLE}",
        f"GRANT SELECT ON {tables} TO {ROLE}",
        f"REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM {ROLE}",
        *(f"DROP FUNCTION IF EXISTS public.{function}" for function in READ_FUNCTIONS),
        *STATISTICS_FUNCTIONS,
        f"REVOKE ALL ON FUNCTION {functions} FROM PUBLIC",
        # The extension grants its views to PUBLIC, which would let the role read them directly.
        "REVOKE ALL ON public.pg_stat_statements, public.pg_stat_statements_info FROM PUBLIC",
        f"GRANT EXECUTE ON FUNCTION {functions} TO {ROLE}",
    ):
        await connection.execute(text(statement))
    return [
        f"{ROLE} granted SELECT on {len(READ_TABLES)} tables: {', '.join(READ_TABLES)}",
        f"{ROLE} granted EXECUTE on {', '.join(READ_FUNCTIONS)}",
    ]


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
