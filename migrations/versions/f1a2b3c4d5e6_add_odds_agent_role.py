"""add_odds_agent_role

Revision ID: f1a2b3c4d5e6
Revises: beaf16386050
Create Date: 2026-04-21 18:09:24.197065

Creates a defense-in-depth Postgres role for the betting agent. The agent
process connects as ``odds_agent`` which has:

- ``SELECT`` on every existing table in the public schema (and on future
  tables, via default privileges)
- ``INSERT, UPDATE`` on ``paper_trades``, ``match_briefs``, and
  ``agent_wakeups`` — the only tables the agent legitimately writes to
- Corresponding sequence ``USAGE, SELECT`` so inserts can generate IDs

The role is created WITH LOGIN but without a password. Operators must set a
password out-of-band (``ALTER ROLE odds_agent WITH PASSWORD '...'``) before
handing a connection string to the agent process; committing a password to
migration history would leak it into every environment's git blame and into
every backup of the migrations table. A login role with no password cannot
authenticate via password auth on Neon, which fails closed rather than open.

This migration is intentionally conservative: it does NOT run automatically
against shared/production databases without an explicit ``alembic upgrade``.
Apply it only once per environment, then rotate the password via the
infrastructure secret store.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "beaf16386050"
branch_labels = None
depends_on = None


AGENT_ROLE = "odds_agent"
WRITABLE_TABLES = ("paper_trades", "match_briefs", "agent_wakeups")


def upgrade() -> None:
    """Create the ``odds_agent`` role with read-mostly privileges.

    Uses ``DO`` blocks so re-running against a DB where the role already
    exists is a no-op rather than an error.
    """
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{AGENT_ROLE}') THEN
                CREATE ROLE {AGENT_ROLE} WITH LOGIN;
            END IF;
        END
        $$;
        """
    )

    # Read access across the whole public schema (existing + future tables).
    op.execute(f"GRANT USAGE ON SCHEMA public TO {AGENT_ROLE};")
    op.execute(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {AGENT_ROLE};")
    op.execute(f"ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO {AGENT_ROLE};")

    # Write access only on tables the agent legitimately mutates.
    for table in WRITABLE_TABLES:
        op.execute(f"GRANT INSERT, UPDATE ON TABLE {table} TO {AGENT_ROLE};")

    # Sequence access so INSERTs on SERIAL/identity columns can allocate IDs.
    op.execute(f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {AGENT_ROLE};")
    op.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
        f"GRANT USAGE, SELECT ON SEQUENCES TO {AGENT_ROLE};"
    )


def downgrade() -> None:
    """Revoke privileges and drop the role.

    Guarded with ``IF EXISTS`` so re-running or partial-upgrade teardown
    doesn't crash. ``DROP OWNED BY`` is required before ``DROP ROLE`` to
    clear any default-privilege grants the role owns.

    .. warning::

        ``DROP ROLE`` fails with ``role "odds_agent" cannot be dropped
        because some objects depend on it`` (or, with ``DROP OWNED BY``,
        aborts on any active backend owned by the role) if any sessions
        are still connected as ``odds_agent``. Before running downgrade,
        terminate every live agent session — e.g.::

            SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE usename = 'odds_agent';

        Stop any local ``odds agent run`` / scheduler subprocesses first,
        then run the termination query, then apply the downgrade.
    """
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{AGENT_ROLE}') THEN
                EXECUTE format('REVOKE ALL ON ALL TABLES IN SCHEMA public FROM %I', '{AGENT_ROLE}');
                EXECUTE format(
                    'REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM %I', '{AGENT_ROLE}'
                );
                EXECUTE format('REVOKE USAGE ON SCHEMA public FROM %I', '{AGENT_ROLE}');
                EXECUTE format(
                    'ALTER DEFAULT PRIVILEGES IN SCHEMA public '
                    'REVOKE SELECT ON TABLES FROM %I',
                    '{AGENT_ROLE}'
                );
                EXECUTE format(
                    'ALTER DEFAULT PRIVILEGES IN SCHEMA public '
                    'REVOKE USAGE, SELECT ON SEQUENCES FROM %I',
                    '{AGENT_ROLE}'
                );
                EXECUTE format('DROP OWNED BY %I', '{AGENT_ROLE}');
                EXECUTE format('DROP ROLE %I', '{AGENT_ROLE}');
            END IF;
        END
        $$;
        """
    )
