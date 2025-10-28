"""
Schema validation tests - validates database schema matches SQLModel definitions.

These tests connect to the ACTUAL database (via DATABASE_URL environment variable)
and perform read-only checks to ensure:
1. Alembic migrations have been applied correctly
2. Database schema matches SQLModel table definitions
3. All tables, columns, and indexes from code exist in database

IMPORTANT: These tests are READ-ONLY and safe to run against production.
"""

import os

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

# Import all models to ensure they're registered with SQLModel.metadata
from core.models import DataQualityLog, Event, FetchLog, Odds, OddsSnapshot  # noqa: F401

# Get the actual database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")


@pytest.fixture
async def db_connection_for_inspection():
    """
    Database connection for inspection operations.
    Inspector methods need to be called inside run_sync context.
    """
    if DATABASE_URL is None:
        raise ValueError("DATABASE_URL environment variable is not set.")

    engine = create_async_engine(DATABASE_URL, echo=False)

    async with engine.connect() as conn:
        yield conn

    await engine.dispose()


@pytest.fixture
async def db_connection():
    """Direct database connection for raw SQL queries."""
    if DATABASE_URL is None:
        raise ValueError("DATABASE_URL environment variable is not set.")

    engine = create_async_engine(DATABASE_URL, echo=False)

    async with engine.connect() as conn:
        yield conn

    await engine.dispose()


class TestSchemaValidation:
    """Validates that database schema matches SQLModel definitions (read-only tests)."""

    @pytest.mark.asyncio
    async def test_alembic_at_head(self, db_connection):
        """Verify database is at the latest migration (HEAD)."""
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        # Get HEAD revision from migration files
        alembic_cfg = Config("alembic.ini")
        script = ScriptDirectory.from_config(alembic_cfg)
        head_revision = script.get_current_head()

        # Get current revision from database
        result = await db_connection.execute(text("SELECT version_num FROM alembic_version"))
        db_version = result.scalar()

        assert db_version is not None, "No Alembic version found in database"
        assert db_version == head_revision, (
            f"Database is not at HEAD migration.\n"
            f"  Database version: {db_version}\n"
            f"  HEAD version: {head_revision}\n"
            f"  Run: alembic upgrade head"
        )

    @pytest.mark.asyncio
    async def test_all_sqlmodel_tables_exist(self, db_connection_for_inspection):
        """Verify all SQLModel tables exist in the database."""
        # Get tables defined in SQLModel
        expected_tables = set(SQLModel.metadata.tables.keys())

        # Add alembic_version (not in SQLModel but should exist)
        expected_tables.add("alembic_version")

        # Get actual tables from database
        def get_tables(sync_conn):
            inspector = inspect(sync_conn)
            return set(inspector.get_table_names())

        actual_tables = await db_connection_for_inspection.run_sync(get_tables)

        missing_tables = expected_tables - actual_tables
        extra_tables = actual_tables - expected_tables

        # Build error message if tables are missing
        error_msg = []
        if missing_tables:
            error_msg.append(f"Missing tables in database: {missing_tables}")
        if extra_tables:
            # Extra tables don't fail the test, but warn in error message
            error_msg.append(f"Note: Extra tables in database (not in SQLModel): {extra_tables}")

        assert not missing_tables, "\n".join(error_msg) if error_msg else ""

    @pytest.mark.asyncio
    async def test_table_columns_match_sqlmodel(self, db_connection_for_inspection):
        """Verify each table's columns match SQLModel definitions."""

        def check_columns(sync_conn):
            inspector = inspect(sync_conn)
            mismatches = []
            warnings = []

            for table_name, table in SQLModel.metadata.tables.items():
                # Get columns from database
                db_columns = {col["name"]: col for col in inspector.get_columns(table_name)}

                # Get columns from SQLModel
                model_columns = {col.name: col for col in table.columns}

                # Check for missing columns
                missing_cols = set(model_columns.keys()) - set(db_columns.keys())
                if missing_cols:
                    mismatches.append(f"{table_name}: missing columns {missing_cols}")

                # Check for extra columns (OK, might be from old migrations)
                extra_cols = set(db_columns.keys()) - set(model_columns.keys())
                if extra_cols:
                    warnings.append(f"{table_name}: extra columns in DB {extra_cols}")

            return mismatches, warnings

        mismatches, warnings = await db_connection_for_inspection.run_sync(check_columns)

        # Include warnings in error message if there are mismatches
        error_parts = []
        if mismatches:
            error_parts.append("Column mismatches found:")
            error_parts.extend(mismatches)
        if warnings and mismatches:
            error_parts.append("\nWarnings:")
            error_parts.extend(warnings)

        assert not mismatches, "\n".join(error_parts)

    @pytest.mark.asyncio
    async def test_primary_keys_match(self, db_connection_for_inspection):
        """Verify primary keys match SQLModel definitions."""

        def check_pks(sync_conn):
            inspector = inspect(sync_conn)
            mismatches = []

            for table_name, table in SQLModel.metadata.tables.items():
                # Get primary key from database
                db_pk = inspector.get_pk_constraint(table_name)
                db_pk_cols = set(db_pk["constrained_columns"])

                # Get primary key from SQLModel
                model_pk_cols = {col.name for col in table.primary_key.columns}

                if db_pk_cols != model_pk_cols:
                    mismatches.append(
                        f"{table_name}: DB PK {db_pk_cols} != Model PK {model_pk_cols}"
                    )

            return mismatches

        mismatches = await db_connection_for_inspection.run_sync(check_pks)

        assert not mismatches, "Primary key mismatches:\n" + "\n".join(mismatches)

    @pytest.mark.asyncio
    async def test_foreign_keys_exist(self, db_connection_for_inspection):
        """Verify foreign keys from SQLModel exist in database."""

        def check_fks(sync_conn):
            inspector = inspect(sync_conn)
            mismatches = []

            for table_name, table in SQLModel.metadata.tables.items():
                # Get foreign keys from database
                db_fks = inspector.get_foreign_keys(table_name)
                db_fk_columns = {
                    fk["constrained_columns"][0] for fk in db_fks if fk["constrained_columns"]
                }

                # Get foreign keys from SQLModel
                model_fk_columns = {col.name for col in table.columns if col.foreign_keys}

                # Check if all model FKs exist in database
                missing_fks = model_fk_columns - db_fk_columns
                if missing_fks:
                    mismatches.append(f"{table_name}: missing FK columns {missing_fks}")

            return mismatches

        mismatches = await db_connection_for_inspection.run_sync(check_fks)

        assert not mismatches, "Foreign key mismatches:\n" + "\n".join(mismatches)

    @pytest.mark.asyncio
    async def test_indexes_exist(self, db_connection_for_inspection):
        """Verify indexes from SQLModel exist in database."""

        def check_indexes(sync_conn):
            inspector = inspect(sync_conn)
            mismatches = []

            for table_name, table in SQLModel.metadata.tables.items():
                # Get indexes from database
                db_indexes = inspector.get_indexes(table_name)
                db_index_columns = {tuple(idx["column_names"]) for idx in db_indexes}

                # Get indexes from SQLModel (excluding primary key)
                model_indexes = {tuple(idx.columns.keys()) for idx in table.indexes}

                # Check if model indexes exist in database
                missing_indexes = model_indexes - db_index_columns
                if missing_indexes:
                    mismatches.append(f"{table_name}: missing indexes on columns {missing_indexes}")

            return mismatches

        mismatches = await db_connection_for_inspection.run_sync(check_indexes)

        assert not mismatches, "Index mismatches:\n" + "\n".join(mismatches)
