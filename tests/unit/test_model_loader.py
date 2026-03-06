"""Tests for S3 model loader with ETag-based caching."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError
from odds_lambda.model_loader import clear_cache, load_model


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

        result = load_model(model_name="test-model", bucket="test-bucket")

        assert result == expected_model
        s3.head_object.assert_called_once_with(
            Bucket="test-bucket", Key="test-model/latest/model.pkl"
        )
        s3.download_file.assert_called_once()
        mock_joblib.load.assert_called_once()

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

        # First call — downloads
        load_model(model_name="test-model", bucket="test-bucket")
        assert s3.download_file.call_count == 1

        # Second call — same ETag, cache hit
        result = load_model(model_name="test-model", bucket="test-bucket")
        assert result == expected_model
        assert s3.download_file.call_count == 1  # No additional download
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

        load_model(model_name="test-model", bucket="test-bucket")
        result = load_model(model_name="test-model", bucket="test-bucket")

        assert result["model"] == "new"
        assert s3.download_file.call_count == 2

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

        load_model()

        s3.head_object.assert_called_once_with(
            Bucket="env-bucket", Key="env-model/latest/model.pkl"
        )


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

        load_model(model_name="m", bucket="b")
        clear_cache()
        load_model(model_name="m", bucket="b")

        assert s3.download_file.call_count == 2
