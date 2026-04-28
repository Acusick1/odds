"""Resolve Betfair client cert paths, materializing from SSM when configured.

Production deploys keep the cert/key PEM contents in SSM SecureString
parameters (KMS-encrypted, per-parameter IAM, free at <=4 KB) rather than
baking them into the Lambda image or storing them as Lambda env vars
(which share a 4 KB ceiling across all variables on the function).

At cold start the Lambda fetches the parameters once and writes them to
``/tmp``; warm invocations short-circuit on the file existence check.
Local development falls back to the direct ``cert_file`` / ``cert_key``
paths in ``BetfairConfig``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import structlog
from odds_core.config import BetfairConfig

logger = structlog.get_logger()


_CERT_DEST = "/tmp/betfair.crt"
_KEY_DEST = "/tmp/betfair.key"


def _fetch_ssm_secure_string(name: str, ssm_client: Any) -> str:
    response = ssm_client.get_parameter(Name=name, WithDecryption=True)
    return str(response["Parameter"]["Value"])


def _write_pem(path: str, content: str) -> None:
    Path(path).write_text(content)
    os.chmod(path, 0o600)


def resolve_cert_paths(bf: BetfairConfig) -> tuple[str | None, str | None]:
    """Return ``(cert_file, cert_key)`` paths suitable for the Betfair client.

    Precedence:
      1. ``cert_pem_ssm_param`` + ``key_pem_ssm_param`` (production) — fetch
         from SSM SecureString and materialize to ``/tmp`` (idempotent).
      2. ``cert_file`` + ``cert_key`` (local dev) — passed through unchanged.
      3. Both unset — returns ``(None, None)``; the client falls back to
         interactive login (dev probes only).
    """
    if bf.cert_pem_ssm_param and bf.key_pem_ssm_param:
        return _materialize_from_ssm(bf.cert_pem_ssm_param, bf.key_pem_ssm_param)
    return bf.cert_file, bf.cert_key


def _materialize_from_ssm(cert_param: str, key_param: str) -> tuple[str, str]:
    """Fetch cert/key from SSM and write to ``/tmp``. Cached across warm starts."""
    if Path(_CERT_DEST).exists() and Path(_KEY_DEST).exists():
        logger.debug("betfair_cert_cache_hit", cert_path=_CERT_DEST)
        return _CERT_DEST, _KEY_DEST

    import boto3  # lazy import — only Lambda runtime needs it

    ssm = boto3.client("ssm")
    cert_pem = _fetch_ssm_secure_string(cert_param, ssm)
    key_pem = _fetch_ssm_secure_string(key_param, ssm)

    _write_pem(_CERT_DEST, cert_pem)
    _write_pem(_KEY_DEST, key_pem)

    logger.info(
        "betfair_cert_materialized_from_ssm",
        cert_param=cert_param,
        key_param=key_param,
        cert_path=_CERT_DEST,
        key_path=_KEY_DEST,
    )
    return _CERT_DEST, _KEY_DEST
