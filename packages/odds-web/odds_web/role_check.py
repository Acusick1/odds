"""Check, over a connection made as the ``odds_web`` role, that the role can read and cannot write.

``python -m odds_web.role_check`` is the host wizard's last step before it installs the unit.
"""

from __future__ import annotations

import asyncio

from odds_core.config import get_settings
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine

READ_ONLY_SQL_TRANSACTION = "25006"
INSUFFICIENT_PRIVILEGE = "42501"
# PostgreSQL checks a read-only transaction before privileges, so with the role's
# ``default_transaction_read_only`` the rejection is the first of these, not the second.
WRITE_REJECTIONS = (READ_ONLY_SQL_TRANSACTION, INSUFFICIENT_PRIVILEGE)
WRITE_PRIVILEGES = ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")
PROBE_TABLE = "paper_trades"


class RoleCheckFailed(RuntimeError):
    """The connection's role can write, or cannot read what the UI needs."""


async def check_role(connection: AsyncConnection) -> list[str]:
    """Raise ``RoleCheckFailed`` unless the role reads ``PROBE_TABLE`` and holds no write right.

    The privilege catalogue is the authority; the attempted write confirms the server rejects it.
    """
    report: list[str] = []

    elevated = await connection.scalar(
        text(
            "SELECT rolsuper OR rolcreatedb OR rolcreaterole OR rolbypassrls "
            "FROM pg_roles WHERE rolname = current_user"
        )
    )
    if elevated:
        raise RoleCheckFailed("the role is a superuser or can create databases or roles")
    user = await connection.scalar(text("SELECT current_user"))
    await connection.rollback()

    if not await connection.scalar(
        text("SELECT has_table_privilege(current_user, :table, 'SELECT')"), {"table": PROBE_TABLE}
    ):
        raise RoleCheckFailed(f"{user} cannot SELECT from {PROBE_TABLE}")
    report.append(f"{user} can SELECT from {PROBE_TABLE}")

    # ``information_schema.tables`` lists the tables the role holds any privilege on, so a table
    # with a write privilege is among them.
    privileges = " OR ".join(
        f"has_table_privilege(current_user, format('public.%I', t.table_name), '{privilege}')"
        for privilege in WRITE_PRIVILEGES
    )
    writable = (
        await connection.scalars(
            text(
                "SELECT t.table_name FROM information_schema.tables t "
                "WHERE t.table_schema = 'public' AND t.table_type = 'BASE TABLE' "
                f"AND ({privileges}) ORDER BY t.table_name"
            )
        )
    ).all()
    if writable:
        raise RoleCheckFailed(f"{user} holds write privileges on: {', '.join(writable)}")
    report.append(f"{user} holds no write privilege on any public table")
    await connection.rollback()

    try:
        await connection.execute(text(f"INSERT INTO {PROBE_TABLE} (event_id) VALUES (1)"))
    except DBAPIError as exc:
        sqlstate = getattr(exc.orig, "sqlstate", None)
        if sqlstate not in WRITE_REJECTIONS:
            raise RoleCheckFailed(
                f"the INSERT failed for another reason (SQLSTATE {sqlstate})"
            ) from exc
        report.append(f"INSERT rejected (SQLSTATE {sqlstate})")
    else:
        raise RoleCheckFailed(f"{user} could INSERT into {PROBE_TABLE}")
    finally:
        await connection.rollback()

    return report


async def main() -> None:
    url = get_settings().database.web_url
    if not url:
        raise SystemExit("WEB_DATABASE_URL is not set")
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            for line in await check_role(connection):
                print(f"  {line}")
    except RoleCheckFailed as exc:
        raise SystemExit(f"odds_web check failed, do not continue: {exc}") from exc
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
