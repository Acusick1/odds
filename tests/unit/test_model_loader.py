"""Tests for S3 model loader with ETag-based caching."""

from unittest.mock import MagicMock, patch

import pytest
import yaml
from botocore.exceptions import ClientError
from odds_analytics.training.config import FeatureConfig
from odds_lambda.model_loader import clear_cache, load_model


def _make_config_yaml(
    sport_key: str = "soccer_epl",
    sharp_bookmakers: list[str] | None = None,
) -> str:
    config = {
        "training": {
            "data": {"sport_key": sport_key},
            "features": {
                "adapter": "xgboost",
                "sharp_bookmakers": sharp_bookmakers or ["bet365"],
                "retail_bookmakers": ["betway", "betfred", "bwin"],
                "markets": ["h2h"],
                "outcome": "home",
                "feature_groups": ["tabular"],
                "target_type": "devigged_bookmaker",
                "target_bookmaker": "bet365",
            },
        }
    }
    return yaml.dump(config)


class TestLoadModel:
    def setup_method(self) -> None:
        clear_cache()

    def test_raises_without_model_name(self) -> None:
        with pytest.raises(ValueError, match="model_name required"):
            load_model(bucket="test-bucket")

    def test_raises_without_bucket(self) -> None:
        with pytest.raises(ValueError, match="bucket required"):
            load_model(model_name="test-model")

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_downloads_on_cold_start(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"abc123"'}
        expected_model = {"model": "xgb", "feature_names": ["f1"], "params": {}}
        mock_joblib.load.return_value = expected_model

        # Config download raises 404 — backward compat
        config_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        s3.exceptions.ClientError = ClientError
        s3.download_file.side_effect = [None, config_error]

        result = load_model(model_name="test-model", bucket="test-bucket")

        assert result["model"] == "xgb"
        assert result["feature_config"] is None
        s3.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-model/latest/model.pkl"
        )

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_cache_hit_skips_download(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"abc123"'}
        expected_model = {"model": "xgb", "feature_names": ["f1"], "params": {}}
        mock_joblib.load.return_value = expected_model

        # Config download raises 404
        config_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        s3.exceptions.ClientError = ClientError
        s3.download_file.side_effect = [None, config_error]

        # First call — downloads
        load_model(model_name="test-model", bucket="test-bucket")
        assert s3.download_file.call_count == 2  # model + config attempt

        # Second call — same ETag, cache hit
        result = load_model(model_name="test-model", bucket="test-bucket")
        assert result["model"] == "xgb"
        assert s3.download_file.call_count == 2  # No additional download
        assert s3.head_object.call_count == 2  # HEAD still called

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_redownloads_on_etag_change(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.side_effect = [
            {"ETag": '"v1"'},
            {"ETag": '"v2"'},
        ]
        mock_joblib.load.side_effect = [
            {"model": "old", "feature_names": [], "params": {}},
            {"model": "new", "feature_names": ["f1"], "params": {}},
        ]

        # Both config downloads raise 404
        config_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        s3.exceptions.ClientError = ClientError
        s3.download_file.side_effect = [None, config_error, None, config_error]

        load_model(model_name="test-model", bucket="test-bucket")
        result = load_model(model_name="test-model", bucket="test-bucket")

        assert result["model"] == "new"
        # 4 download calls: model+config for v1, model+config for v2
        assert s3.download_file.call_count == 4

    @patch("odds_lambda.model_loader._get_s3_client")
    def test_raises_file_not_found_on_404(self, mock_get_client: MagicMock) -> None:
        s3 = MagicMock()
        mock_get_client.return_value = s3
        error_response = {"Error": {"Code": "404", "Message": "Not Found"}}
        s3.head_object.side_effect = ClientError(error_response, "HeadObject")
        s3.exceptions.ClientError = ClientError

        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_model(model_name="missing", bucket="test-bucket")

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_uses_env_vars(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL_NAME", "env-model")
        monkeypatch.setenv("MODEL_BUCKET", "env-bucket")

        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"e1"'}
        mock_joblib.load.return_value = {"model": "x", "feature_names": [], "params": {}}

        config_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        s3.exceptions.ClientError = ClientError
        s3.download_file.side_effect = [None, config_error]

        load_model()

        s3.head_object.assert_called_once_with(
            Bucket="env-bucket", Key="env-model/latest/model.pkl"
        )


class TestLoadModelWithConfig:
    def setup_method(self) -> None:
        clear_cache()

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_loads_feature_config_from_s3(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock, tmp_path: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"cfg1"'}
        mock_joblib.load.return_value = {"model": "xgb", "feature_names": ["f1"], "params": {}}

        config_yaml = _make_config_yaml(sport_key="soccer_epl")

        def fake_download(bucket: str, key: str, path: str) -> None:
            if key.endswith("config.yaml"):
                with open(path, "w") as f:
                    f.write(config_yaml)

        s3.download_file.side_effect = fake_download
        s3.exceptions.ClientError = ClientError

        result = load_model(model_name="epl-model", bucket="test-bucket")

        assert result["feature_config"] is not None
        cfg = result["feature_config"]
        assert isinstance(cfg, FeatureConfig)
        assert cfg.sport_key == "soccer_epl"
        assert cfg.sharp_bookmakers == ["bet365"]
        assert cfg.target_bookmaker == "bet365"
        assert cfg.outcome == "home"

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_config_round_trip(self, mock_joblib: MagicMock, mock_get_client: MagicMock) -> None:
        """Verify that a FeatureConfig survives serialize → S3 → deserialize."""
        original = FeatureConfig(
            adapter="xgboost",
            sharp_bookmakers=["bet365"],
            retail_bookmakers=["betway", "betfred", "bwin"],
            markets=["h2h"],
            outcome="home",
            feature_groups=("tabular",),
            target_type="devigged_bookmaker",
            target_bookmaker="bet365",
            sport_key="soccer_epl",
        )

        # Simulate the config.yaml as published (full training config structure)
        config_data = {
            "training": {
                "data": {"sport_key": "soccer_epl"},
                "features": original.model_dump(mode="json"),
            }
        }
        config_yaml = yaml.dump(config_data)

        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"rt1"'}
        mock_joblib.load.return_value = {"model": "xgb", "feature_names": [], "params": {}}

        def fake_download(bucket: str, key: str, path: str) -> None:
            if key.endswith("config.yaml"):
                with open(path, "w") as f:
                    f.write(config_yaml)

        s3.download_file.side_effect = fake_download
        s3.exceptions.ClientError = ClientError

        result = load_model(model_name="test", bucket="b")
        loaded = result["feature_config"]

        assert loaded == original

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_missing_config_returns_none_with_warning(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"no-cfg"'}
        mock_joblib.load.return_value = {"model": "xgb", "feature_names": [], "params": {}}

        config_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        s3.exceptions.ClientError = ClientError
        s3.download_file.side_effect = [None, config_error]

        result = load_model(model_name="old-model", bucket="b")

        assert result["feature_config"] is None

    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_nosuchkey_error_also_returns_none(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock
    ) -> None:
        """download_file returns NoSuchKey (not 404) for missing objects."""
        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"nsk1"'}
        mock_joblib.load.return_value = {"model": "xgb", "feature_names": [], "params": {}}

        nosuchkey_error = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
            "GetObject",
        )
        s3.exceptions.ClientError = ClientError
        s3.download_file.side_effect = [None, nosuchkey_error]

        result = load_model(model_name="legacy-model", bucket="b")

        assert result["feature_config"] is None


class TestClearCache:
    @patch("odds_lambda.model_loader._get_s3_client")
    @patch("odds_lambda.model_loader.joblib")
    def test_clear_cache_forces_redownload(
        self, mock_joblib: MagicMock, mock_get_client: MagicMock
    ) -> None:
        s3 = MagicMock()
        mock_get_client.return_value = s3
        s3.head_object.return_value = {"ETag": '"same"'}
        mock_joblib.load.return_value = {"model": "x", "feature_names": [], "params": {}}

        config_error = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")
        s3.exceptions.ClientError = ClientError
        s3.download_file.side_effect = [None, config_error, None, config_error]

        load_model(model_name="m", bucket="b")
        clear_cache()
        load_model(model_name="m", bucket="b")

        # 2 model downloads + 2 config attempts
        assert s3.download_file.call_count == 4
