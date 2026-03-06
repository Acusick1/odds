"""S3 model loader with ETag-based caching for Lambda."""

import os
import tempfile
from typing import Any

import boto3
import joblib
import structlog

logger = structlog.get_logger(__name__)

# Module-level cache — persists across warm Lambda invocations
_cached_model: Any = None
_cached_etag: str | None = None
_cached_model_key: str | None = None


def _get_s3_client() -> Any:
    return boto3.client("s3")


def _model_s3_key(model_name: str) -> str:
    return f"{model_name}/latest/model.pkl"


def load_model(
    model_name: str | None = None,
    bucket: str | None = None,
) -> Any:
    """Load a model from S3 with ETag-based caching.

    On cold start, downloads the model from S3 to /tmp and loads it.
    On warm invocations, issues a HEAD request to check if the ETag has
    changed — if not, returns the cached in-memory model without downloading.

    Args:
        model_name: Model name (S3 prefix). Defaults to MODEL_NAME env var.
        bucket: S3 bucket name. Defaults to MODEL_BUCKET env var.

    Returns:
        The loaded model object (dict with 'model', 'feature_names', 'params').

    Raises:
        ValueError: If model_name or bucket is not provided and env vars are unset.
        FileNotFoundError: If the model does not exist in S3.
    """
    global _cached_model, _cached_etag, _cached_model_key

    model_name = model_name or os.environ.get("MODEL_NAME")
    bucket = bucket or os.environ.get("MODEL_BUCKET")

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

    # Download and load
    logger.info("model_downloading", model_name=model_name, bucket=bucket, key=key)
    tmp_path = os.path.join(tempfile.gettempdir(), f"{model_name}_model.pkl")
    s3.download_file(bucket, key, tmp_path)

    model_data = joblib.load(tmp_path)
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


def clear_cache() -> None:
    """Clear the in-memory model cache. Mainly useful for testing."""
    global _cached_model, _cached_etag, _cached_model_key
    _cached_model = None
    _cached_etag = None
    _cached_model_key = None
