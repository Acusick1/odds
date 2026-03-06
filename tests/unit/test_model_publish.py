"""Tests for the model publish CLI command."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml
from odds_cli.commands.model import publish_model
from typer.testing import CliRunner

runner = CliRunner()


def _invoke(args: list[str]) -> object:
    """Invoke the publish command directly (not via sub-app)."""
    import typer

    test_app = typer.Typer()
    test_app.command()(publish_model)
    return runner.invoke(test_app, args)


class TestPublishCommand:
    def test_fails_if_model_file_missing(self, tmp_path: Path) -> None:
        result = _invoke(
            ["--name", "test", "--path", str(tmp_path / "missing.pkl")],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_fails_if_config_yaml_missing(self, tmp_path: Path) -> None:
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"fake model")

        result = _invoke(
            ["--name", "test", "--path", str(model_file)],
        )
        assert result.exit_code == 1
        assert "config not found" in result.output.lower()

    @patch("odds_cli.commands.model.boto3")
    @patch("odds_cli.commands.model._get_git_sha", return_value="abc1234")
    def test_uploads_to_versioned_and_latest(
        self,
        mock_git: MagicMock,
        mock_boto3: MagicMock,
        tmp_path: Path,
    ) -> None:
        model_file = tmp_path / "my_model.pkl"
        model_file.write_bytes(b"fake model data")

        config_file = tmp_path / "my_model.yaml"
        config_data = {
            "training": {"strategy_type": "xgboost_line_movement"},
        }
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        s3 = MagicMock()
        mock_boto3.client.return_value = s3

        result = _invoke(
            [
                "--name",
                "epl-clv-home",
                "--path",
                str(model_file),
                "--bucket",
                "test-bucket",
            ],
        )

        assert result.exit_code == 0
        assert "Published" in result.output

        # 3 files × 2 paths (versioned + latest) = 4 upload_file + 2 put_object
        assert s3.upload_file.call_count == 4
        assert s3.put_object.call_count == 2

        upload_keys = [call.args[2] for call in s3.upload_file.call_args_list]
        assert "epl-clv-home/abc1234/model.pkl" in upload_keys
        assert "epl-clv-home/abc1234/config.yaml" in upload_keys
        assert "epl-clv-home/latest/model.pkl" in upload_keys
        assert "epl-clv-home/latest/config.yaml" in upload_keys

    @patch("odds_cli.commands.model.boto3")
    @patch("odds_cli.commands.model._get_git_sha", return_value=None)
    def test_falls_back_to_timestamp_version(
        self,
        mock_git: MagicMock,
        mock_boto3: MagicMock,
        tmp_path: Path,
    ) -> None:
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"data")
        config_file = tmp_path / "model.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"training": {}}, f)

        s3 = MagicMock()
        mock_boto3.client.return_value = s3

        result = _invoke(
            ["--name", "test", "--path", str(model_file)],
        )

        assert result.exit_code == 0
        upload_keys = [call.args[2] for call in s3.upload_file.call_args_list]
        assert upload_keys[0].startswith("test/2")  # starts with year

    @patch("odds_cli.commands.model.boto3")
    @patch("odds_cli.commands.model._get_git_sha", return_value="def5678")
    def test_explicit_version_override(
        self,
        mock_git: MagicMock,
        mock_boto3: MagicMock,
        tmp_path: Path,
    ) -> None:
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"data")
        config_file = tmp_path / "model.yaml"
        with open(config_file, "w") as f:
            yaml.dump({"training": {}}, f)

        s3 = MagicMock()
        mock_boto3.client.return_value = s3

        result = _invoke(
            [
                "--name",
                "test",
                "--path",
                str(model_file),
                "--version",
                "v1.0",
            ],
        )

        assert result.exit_code == 0
        upload_keys = [call.args[2] for call in s3.upload_file.call_args_list]
        assert "test/v1.0/model.pkl" in upload_keys
