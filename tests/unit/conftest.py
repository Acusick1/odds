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
