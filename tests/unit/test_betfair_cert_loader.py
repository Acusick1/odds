"""Unit tests for Betfair cert loader (SSM materialization + path precedence)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from odds_core.config import BetfairConfig
from odds_lambda.betfair import cert_loader
from odds_lambda.betfair.cert_loader import resolve_cert_paths


class TestResolveCertPaths:
    """Precedence: SSM params > direct file paths > (None, None)."""

    def test_returns_direct_paths_when_ssm_unset(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            bf = BetfairConfig(
                _env_file=None,
                cert_file="/local/client.crt",
                cert_key="/local/client.key",
            )
        assert resolve_cert_paths(bf) == ("/local/client.crt", "/local/client.key")

    def test_returns_none_when_nothing_configured(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            bf = BetfairConfig(_env_file=None)
        assert resolve_cert_paths(bf) == (None, None)

    def test_ssm_params_override_direct_paths(self, tmp_path: Path) -> None:
        cert_dest = tmp_path / "betfair.crt"
        key_dest = tmp_path / "betfair.key"

        with patch.dict(os.environ, {}, clear=True):
            bf = BetfairConfig(
                _env_file=None,
                cert_file="/local/client.crt",
                cert_key="/local/client.key",
                cert_pem_ssm_param="/odds-scheduler/betfair/cert_pem",
                key_pem_ssm_param="/odds-scheduler/betfair/key_pem",
            )

        mock_ssm = MagicMock()
        mock_ssm.get_parameter.side_effect = [
            {"Parameter": {"Value": "CERT_PEM"}},
            {"Parameter": {"Value": "KEY_PEM"}},
        ]

        with (
            patch.object(cert_loader, "_CERT_DEST", str(cert_dest)),
            patch.object(cert_loader, "_KEY_DEST", str(key_dest)),
            patch("boto3.client", return_value=mock_ssm),
        ):
            cf, ck = resolve_cert_paths(bf)

        assert cf == str(cert_dest)
        assert ck == str(key_dest)
        assert cert_dest.read_text() == "CERT_PEM"
        assert key_dest.read_text() == "KEY_PEM"
        # 0o600 permissions to match the production write
        assert (cert_dest.stat().st_mode & 0o777) == 0o600

    def test_warm_invocation_skips_ssm_call(self, tmp_path: Path) -> None:
        cert_dest = tmp_path / "betfair.crt"
        key_dest = tmp_path / "betfair.key"
        cert_dest.write_text("PRE_EXISTING_CERT")
        key_dest.write_text("PRE_EXISTING_KEY")

        with patch.dict(os.environ, {}, clear=True):
            bf = BetfairConfig(
                _env_file=None,
                cert_pem_ssm_param="/odds-scheduler/betfair/cert_pem",
                key_pem_ssm_param="/odds-scheduler/betfair/key_pem",
            )

        mock_boto = MagicMock()
        with (
            patch.object(cert_loader, "_CERT_DEST", str(cert_dest)),
            patch.object(cert_loader, "_KEY_DEST", str(key_dest)),
            patch("boto3.client", mock_boto),
        ):
            cf, ck = resolve_cert_paths(bf)

        assert cf == str(cert_dest)
        assert ck == str(key_dest)
        mock_boto.assert_not_called()
        # Files retain their pre-existing contents (no overwrite)
        assert cert_dest.read_text() == "PRE_EXISTING_CERT"
