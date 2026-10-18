"""Shared fixtures for unit tests."""

from __future__ import annotations

import base64
import gzip
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from odds_lambda.oddsportal_client import _key

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
