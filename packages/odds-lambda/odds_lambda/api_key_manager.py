"""API key rotation manager with SSM-backed state persistence."""

import os
from typing import Any

import boto3
import structlog
from botocore.exceptions import ClientError

logger = structlog.get_logger()

DEFAULT_SSM_PARAMETER_NAME = "/odds/active-api-key-index"


class AllKeysExhaustedError(Exception):
    """All API keys have been exhausted."""


class APIKeyManager:
    """Manages multiple API keys with failover on quota exhaustion.

    Stores the active key index in AWS SSM Parameter Store so that
    all Lambda invocations share state without DB schema changes.
    Falls back to in-memory tracking when SSM is unavailable (local dev).
    """

    def __init__(self, keys: list[str], ssm_parameter_name: str | None = None) -> None:
        if not keys:
            raise ValueError("At least one API key is required")
        self._keys = keys
        self._ssm_parameter_name = (
            ssm_parameter_name or os.environ.get("SSM_API_KEY_INDEX") or DEFAULT_SSM_PARAMETER_NAME
        )
        self._active_index: int | None = None
        self._start_index: int | None = None
        self._ssm_available: bool | None = None
        self._ssm_client: Any = None

    @property
    def key_count(self) -> int:
        return len(self._keys)

    def _get_ssm_client(self) -> Any:
        if self._ssm_client is None:
            self._ssm_client = boto3.client("ssm")
        return self._ssm_client

    def _read_index_from_ssm(self) -> int:
        """Read active key index from SSM. Returns 0 if parameter doesn't exist."""
        if self._ssm_available is False:
            return self._active_index or 0

        try:
            client = self._get_ssm_client()
            response = client.get_parameter(Name=self._ssm_parameter_name)
            index = int(response["Parameter"]["Value"])
            self._ssm_available = True
            return index % len(self._keys)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ParameterNotFound":
                self._ssm_available = True
                return 0
            logger.warning("ssm_read_failed", error=str(e))
            self._ssm_available = False
            return self._active_index or 0
        except Exception as e:
            logger.warning("ssm_unavailable", error=str(e))
            self._ssm_available = False
            return self._active_index or 0

    def _write_index_to_ssm(self, index: int) -> None:
        """Write active key index to SSM."""
        if self._ssm_available is False:
            return

        try:
            client = self._get_ssm_client()
            client.put_parameter(
                Name=self._ssm_parameter_name,
                Value=str(index),
                Type="String",
                Overwrite=True,
            )
        except Exception as e:
            logger.warning("ssm_write_failed", error=str(e))

    def get_active_key(self) -> str:
        """Get the currently active API key."""
        if self._active_index is None:
            self._active_index = self._read_index_from_ssm()
            self._start_index = self._active_index

        return self._keys[self._active_index]

    def rotate_key(self) -> str:
        """Rotate to the next API key. Raises AllKeysExhaustedError if we've cycled through all keys."""
        if self._active_index is None:
            self._active_index = self._read_index_from_ssm()
            self._start_index = self._active_index

        next_index = (self._active_index + 1) % len(self._keys)

        if next_index == self._start_index:
            raise AllKeysExhaustedError(f"All {len(self._keys)} API keys exhausted")

        self._active_index = next_index
        self._write_index_to_ssm(next_index)

        logger.info(
            "api_key_rotated",
            new_index=next_index,
            total_keys=len(self._keys),
        )

        return self._keys[next_index]
