"""Unit tests for API key rotation manager."""

from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError
from odds_lambda.api_key_manager import AllKeysExhaustedError, APIKeyManager


class TestAPIKeyManager:
    """Tests for APIKeyManager."""

    def test_single_key(self):
        """Should return the only key available."""
        manager = APIKeyManager(["key1"])
        # Force in-memory mode (no SSM)
        manager._ssm_available = False
        assert manager.get_active_key() == "key1"

    def test_empty_keys_raises(self):
        """Should raise ValueError with no keys."""
        with pytest.raises(ValueError, match="At least one API key"):
            APIKeyManager([])

    def test_get_active_key_reads_ssm_index(self):
        """Should read index from SSM and return corresponding key."""
        manager = APIKeyManager(["key0", "key1", "key2"])

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "1"}}
        manager._ssm_client = mock_ssm

        assert manager.get_active_key() == "key1"
        mock_ssm.get_parameter.assert_called_once()

    def test_get_active_key_defaults_to_zero_on_parameter_not_found(self):
        """Should default to index 0 when SSM parameter doesn't exist."""
        manager = APIKeyManager(["key0", "key1"])

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.side_effect = ClientError(
            {"Error": {"Code": "ParameterNotFound", "Message": "Not found"}},
            "GetParameter",
        )
        manager._ssm_client = mock_ssm

        assert manager.get_active_key() == "key0"

    def test_get_active_key_falls_back_on_ssm_error(self):
        """Should fall back to in-memory on SSM errors."""
        manager = APIKeyManager(["key0", "key1"])

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.side_effect = Exception("connection refused")
        manager._ssm_client = mock_ssm

        assert manager.get_active_key() == "key0"
        assert manager._ssm_available is False

    def test_rotate_key_increments_index(self):
        """Should rotate to the next key and write to SSM."""
        manager = APIKeyManager(["key0", "key1", "key2"])
        manager._ssm_available = False
        manager._active_index = 0

        new_key = manager.rotate_key()
        assert new_key == "key1"
        assert manager._active_index == 1

    def test_rotate_key_raises_when_all_exhausted(self):
        """Should raise AllKeysExhaustedError when cycling back to 0."""
        manager = APIKeyManager(["key0", "key1"])
        manager._ssm_available = False
        manager._active_index = 1

        with pytest.raises(AllKeysExhaustedError, match="2 API keys exhausted"):
            manager.rotate_key()

    def test_rotate_key_writes_to_ssm(self):
        """Should persist the new index to SSM."""
        manager = APIKeyManager(["key0", "key1", "key2"])

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "0"}}
        manager._ssm_client = mock_ssm
        manager._ssm_available = True
        manager._active_index = 0

        manager.rotate_key()

        mock_ssm.put_parameter.assert_called_once_with(
            Name="/odds/active-api-key-index",
            Value="1",
            Type="String",
            Overwrite=True,
        )

    def test_key_count(self):
        """Should return the number of keys."""
        manager = APIKeyManager(["a", "b", "c"])
        assert manager.key_count == 3

    def test_ssm_index_wraps_around(self):
        """Should wrap SSM index modulo key count."""
        manager = APIKeyManager(["key0", "key1"])

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "5"}}
        manager._ssm_client = mock_ssm

        # 5 % 2 = 1
        assert manager.get_active_key() == "key1"

    def test_caches_active_index(self):
        """Should only read SSM once, then cache."""
        manager = APIKeyManager(["key0", "key1"])

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "0"}}
        manager._ssm_client = mock_ssm

        manager.get_active_key()
        manager.get_active_key()

        # SSM should only be called once
        assert mock_ssm.get_parameter.call_count == 1
