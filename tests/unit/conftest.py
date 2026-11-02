"""Shared fixtures for unit tests."""

from __future__ import annotations

import asyncio
import base64
import gzip
import json
import os
import socket
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from odds_lambda.oddsportal_client import _key
from py_pglite.config import PGliteConfig
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

OP_FIXTURES = Path(__file__).parent.parent / "fixtures" / "oddsportal"


@pytest.fixture
def oddsportal_fixtures() -> Path:
    return OP_FIXTURES


@pytest.fixture
def encrypt_payload() -> Callable[..., str]:
    """Encrypt JSON the way the OddsPortal feed does, for synthetic responses."""

    def encrypt(obj: Any, *, gzipped: bool = False) -> str:
        data = json.dumps(obj).encode()
        if gzipped:
            data = gzip.compress(data)
        padder = padding.PKCS7(128).padder()
        padded = padder.update(data) + padder.finalize()
        iv = os.urandom(16)
        encryptor = Cipher(algorithms.AES(_key()), modes.CBC(iv)).encryptor()
        ciphertext = encryptor.update(padded) + encryptor.finalize()
        outer = f"{base64.b64encode(ciphertext).decode()}:{iv.hex()}"
        return base64.b64encode(outer.encode("latin1")).decode()

    return encrypt


@pytest.fixture
def make_ml_config() -> Callable[..., Any]:
    """Factory for a small MLTrainingConfig with a tuning section."""
    from datetime import date

    from odds_analytics.training import (
        DataConfig,
        ExperimentConfig,
        FeatureConfig,
        MLTrainingConfig,
        TrainingConfig,
        XGBoostConfig,
    )
    from odds_analytics.training.config import SearchSpace, TuningConfig

    def _make(
        *,
        search_spaces: dict[str, SearchSpace] | None = None,
        metric: str = "val_mse",
        use_kfold: bool = False,
        cv_method: str = "timeseries",
        test_split: float = 0.2,
        strategy_type: str = "xgboost_line_movement",
        tuning: bool = True,
    ) -> MLTrainingConfig:
        data_kwargs: dict[str, Any] = {}
        if cv_method == "walk_forward":
            data_kwargs.update(min_train_events=4, val_step_events=2)
        elif use_kfold:
            data_kwargs.update(n_folds=3)
        return MLTrainingConfig(
            experiment=ExperimentConfig(name="t"),
            training=TrainingConfig(
                strategy_type=strategy_type,
                data=DataConfig(
                    start_date=date(2024, 1, 1),
                    end_date=date(2024, 12, 31),
                    use_kfold=use_kfold,
                    cv_method=cv_method,
                    test_split=test_split,
                    validation_split=0.0 if use_kfold or not tuning else 0.1,
                    **data_kwargs,
                ),
                model=XGBoostConfig(n_estimators=10),
                features=FeatureConfig(target_type="raw"),
            ),
            tuning=(
                TuningConfig(metric=metric, search_spaces=search_spaces or {}) if tuning else None
            ),
        )

    return _make


@pytest_asyncio.fixture(loop_scope="session")
async def legacy_connection(legacy_engine: AsyncEngine) -> AsyncIterator[AsyncConnection]:
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import text

    async with legacy_engine.connect() as connection:
        async with connection.begin():
            await connection.execute(text("CREATE SCHEMA legacy_conversion_test"))
            await connection.execute(text("SET LOCAL search_path TO legacy_conversion_test"))
            for statement in (
                "CREATE TABLE events (id varchar PRIMARY KEY, surrogate_id integer, home_team text, away_team text)",
                "CREATE TABLE snapshot_payloads (id serial PRIMARY KEY, payload jsonb, received_at timestamptz)",
                """CREATE TABLE odds_snapshots (id serial PRIMARY KEY, event_id varchar REFERENCES events(id),
                   snapshot_time timestamptz, created_at timestamptz, source text, market text,
                   time_basis text, payload_id integer REFERENCES snapshot_payloads(id), in_play boolean,
                   market_status text, total_matched numeric, market_start_time timestamptz,
                   CONSTRAINT uq_odds_snapshots_event_source_market_time UNIQUE(event_id,source,market,snapshot_time))""",
                """CREATE TABLE prices (id serial PRIMARY KEY, snapshot_id integer REFERENCES odds_snapshots(id),
                   bookmaker text, outcome text, line numeric, side text, price numeric, size numeric,
                   matched numeric, changed_at timestamptz)""",
                "CREATE TABLE candidate_decisions (input_snapshot_ids integer[])",
                "CREATE TABLE event_external_ids (event_id integer, source text, external_id text)",
                "CREATE TABLE health_check_acknowledgements (event_id integer, check_name text)",
            ):
                await connection.execute(text(statement))
            from importlib.util import module_from_spec, spec_from_file_location

            spec = spec_from_file_location(
                "drop_legacy",
                Path("migrations/versions/b8e2d5f7a1c3_drop_old_odds_representations.py"),
            )
            assert spec and spec.loader
            module = module_from_spec(spec)
            spec.loader.exec_module(module)

            def reconstruct(sync_connection: Connection) -> None:
                with Operations.context(MigrationContext.configure(sync_connection)):
                    module.downgrade()

            await connection.run_sync(reconstruct)
            yield connection
            await connection.rollback()


@pytest.fixture(scope="session")
def legacy_pglite_config() -> PGliteConfig:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port: int = listener.getsockname()[1]
    return PGliteConfig(use_tcp=True, tcp_port=port, cleanup_on_exit=True, timeout=30)


@pytest_asyncio.fixture(loop_scope="session")
async def converted_input(legacy_connection: AsyncConnection) -> AsyncConnection:
    data: dict[str, Any] = json.loads(
        Path("tests/fixtures/legacy_snapshots/representative.json").read_text()
    )
    await seed(legacy_connection, data)
    return legacy_connection


@pytest.fixture(scope="session")
async def legacy_engine(legacy_pglite_config: PGliteConfig) -> AsyncIterator[AsyncEngine]:
    from py_pglite.manager import PGliteManager
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import StaticPool

    # The socket handler must not lock PGlite before database initialization finishes.
    class ReadyPGliteManager(PGliteManager):
        def _generate_tcp_js_content(self, extension_imports: str, extension_config: str) -> str:
            return (
                super()
                ._generate_tcp_js_content(extension_imports, extension_config)
                .replace("const server =", "await db.waitReady; const server =")
            )

    manager = ReadyPGliteManager(legacy_pglite_config)
    import subprocess
    from unittest.mock import patch

    popen = subprocess.Popen

    # Eager WASM optimization exceeds this test host's memory budget.
    def start_node(args: list[str], **kwargs: Any) -> subprocess.Popen[str]:
        command = (
            [args[0], "--liftoff-only", "--wasm-lazy-compilation", *args[1:]]
            if args[0] == "node"
            else args
        )
        return popen(command, **kwargs)

    with patch("py_pglite.manager.subprocess.Popen", side_effect=start_node):
        manager.start()
    engine = create_async_engine(
        legacy_pglite_config.get_asyncpg_uri().replace("postgresql://", "postgresql+asyncpg://"),
        poolclass=StaticPool,
        connect_args={"timeout": 10, "ssl": False},
    )
    try:
        yield engine
    finally:
        with suppress(TimeoutError):
            await asyncio.wait_for(engine.dispose(), timeout=5)
        manager.stop()


async def seed(connection: AsyncConnection, data: dict[str, Any]) -> None:
    for row in data["snapshots"]:
        await connection.execute(
            text(
                "INSERT INTO events VALUES (:event_id, 1, :home_team, :away_team) ON CONFLICT DO NOTHING"
            ),
            row,
        )
        values = {k: row[k] for k in ("id", "event_id", "bookmaker_count")}
        values.update(
            raw=json.dumps(row["raw_data"]),
            created=datetime.fromisoformat(row["created_at"]),
            time=datetime.fromisoformat(row["snapshot_time"]),
        )
        await connection.execute(
            text(
                "INSERT INTO odds_snapshots (id,event_id,bookmaker_count,raw_data,created_at,snapshot_time) VALUES (:id,:event_id,:bookmaker_count,CAST(:raw AS json),:created,:time)"
            ),
            values,
        )
    await connection.execute(
        text("SELECT setval('odds_snapshots_id_seq', (SELECT max(id) FROM odds_snapshots))")
    )
    for row in data["odds"]:
        values = dict(row)
        for key in ("created_at", "last_update", "odds_timestamp"):
            values[key] = datetime.fromisoformat(values[key])
        columns = ",".join(values)
        await connection.execute(
            text(f"INSERT INTO odds ({columns}) VALUES ({','.join(':' + k for k in values)})"),
            values,
        )
