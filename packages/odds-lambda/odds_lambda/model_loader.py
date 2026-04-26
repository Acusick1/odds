"""S3 model loader with ETag-based caching for Lambda."""

import os
import tempfile
from typing import Any

import boto3
import joblib
import structlog
import yaml
from odds_analytics.training.config import FeatureConfig
from odds_core.config import get_settings
from odds_core.sports import SportKey

logger = structlog.get_logger(__name__)

# Module-level cache — persists across warm Lambda invocations
_cached_model: dict[str, Any] | None = None
_cached_etag: str | None = None
_cached_model_key: str | None = None


def _get_s3_client() -> Any:
    return boto3.client("s3")


def _model_s3_key(model_name: str) -> str:
    return f"{model_name}/latest/model.pkl"


def _config_s3_key(model_name: str) -> str:
    return f"{model_name}/latest/config.yaml"


def _load_feature_config(s3: Any, bucket: str, model_name: str) -> FeatureConfig | None:
    """Download config.yaml from S3 and parse FeatureConfig from it.

    Returns None if config.yaml is missing.
    """
    key = _config_s3_key(model_name)
    tmp_path = os.path.join(tempfile.gettempdir(), f"{model_name}_config.yaml")

    try:
        s3.download_file(bucket, key, tmp_path)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise

    with open(tmp_path) as f:
        config_data = yaml.safe_load(f)

    training = config_data.get("training", {})
    features_data = training.get("features", {})
    sport_key = training.get("data", {}).get("sport_key")

    if sport_key and "sport_key" not in features_data:
        features_data["sport_key"] = sport_key

    return FeatureConfig(**features_data)


def load_model(
    model_name: str | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Load a model from S3 with ETag-based caching.

    On cold start, downloads the model from S3 to /tmp and loads it.
    On warm invocations, issues a HEAD request to check if the ETag has
    changed — if not, returns the cached in-memory model without downloading.

    Returns:
        Dict with 'model', 'feature_names', 'params', and 'feature_config'
        (FeatureConfig or None if config.yaml missing).

    Raises:
        ValueError: If model_name or bucket is not provided and settings
            (``MODEL_NAME`` / ``MODEL_BUCKET``) are unset.
        FileNotFoundError: If the model does not exist in S3.
    """
    global _cached_model, _cached_etag, _cached_model_key

    settings = get_settings()
    model_name = model_name or settings.model.name
    bucket = bucket or settings.model.bucket

    if not model_name:
        raise ValueError("model_name required (or set MODEL_NAME env var)")
    if not bucket:
        raise ValueError("bucket required (or set MODEL_BUCKET env var)")

    s3 = _get_s3_client()
    key = _model_s3_key(model_name)

    # HEAD request to get current ETag
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except s3.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            raise FileNotFoundError(f"Model not found: s3://{bucket}/{key}") from e
        raise

    etag = head["ETag"]

    # Return cached model if ETag unchanged and same key
    if _cached_model is not None and _cached_etag == etag and _cached_model_key == key:
        logger.debug("model_cache_hit", model_name=model_name, etag=etag)
        return _cached_model

    # Download and load model
    logger.info("model_downloading", model_name=model_name, bucket=bucket, key=key)
    tmp_path = os.path.join(tempfile.gettempdir(), f"{model_name}_model.pkl")
    s3.download_file(bucket, key, tmp_path)

    model_data = joblib.load(tmp_path)

    # Load feature config from config.yaml
    feature_config = _load_feature_config(s3, bucket, model_name)
    if feature_config is None:
        logger.warning(
            "config_yaml_missing",
            model_name=model_name,
            msg="config.yaml not found in S3, feature_config will be None",
        )
    else:
        logger.info(
            "config_loaded",
            model_name=model_name,
            sport_key=feature_config.sport_key,
        )

    model_data["feature_config"] = feature_config

    logger.info(
        "model_loaded",
        model_name=model_name,
        etag=etag,
        features=len(model_data.get("feature_names", [])),
    )

    # Update cache
    _cached_model = model_data
    _cached_etag = etag
    _cached_model_key = key

    return model_data


def model_supports_sport(sport: SportKey | None) -> bool:
    """Whether the configured model serves the given sport.

    Returns ``True`` when ``sport`` is ``None`` (caller defers to the
    model's own ``sport_key``). Returns ``False`` when no model is
    configured, the model artifact has no bundled config, or its
    ``sport_key`` differs from the requested sport.

    Lets pre-sport callers (e.g. ``fetch-oddsportal-mlb``) skip
    speculative scoring without raising the ``sport_mismatch`` error in
    ``score_events``.
    """
    if sport is None:
        return True
    try:
        model_data = load_model()
    except (ValueError, FileNotFoundError):
        return False
    config: FeatureConfig | None = model_data.get("feature_config")
    if config is None or not config.sport_key:
        return False
    return config.sport_key == sport


def get_cached_version() -> str | None:
    """Return the ETag of the currently cached model, or None if no model is cached."""
    return _cached_etag


def clear_cache() -> None:
    """Clear the in-memory model cache. Mainly useful for testing."""
    global _cached_model, _cached_etag, _cached_model_key
    _cached_model = None
    _cached_etag = None
    _cached_model_key = None
