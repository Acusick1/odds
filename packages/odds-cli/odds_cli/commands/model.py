"""CLI commands for model management."""

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import typer
import yaml
from rich.console import Console

app = typer.Typer()
console = Console()


def _get_git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_metadata(
    model_name: str,
    config_data: dict,
    version: str,
) -> dict:
    git_sha = _get_git_sha()
    training_cfg = config_data.get("training", {})

    return {
        "model_name": model_name,
        "version": version,
        "git_sha": git_sha,
        "training_date": datetime.now(UTC).isoformat(),
        "strategy_type": training_cfg.get("strategy_type"),
    }


def _upload_to_s3(
    s3_client: Any,
    bucket: str,
    prefix: str,
    model_path: Path,
    config_path: Path,
    metadata: dict,
) -> None:
    s3_client.upload_file(str(model_path), bucket, f"{prefix}/model.pkl")
    s3_client.upload_file(str(config_path), bucket, f"{prefix}/config.yaml")
    s3_client.put_object(
        Bucket=bucket,
        Key=f"{prefix}/metadata.json",
        Body=json.dumps(metadata, indent=2),
        ContentType="application/json",
    )


@app.command("publish")
def publish_model(
    name: str = typer.Option(..., "--name", "-n", help="Model name (S3 prefix)"),
    path: str = typer.Option(..., "--path", "-p", help="Path to model .pkl file"),
    bucket: str = typer.Option("odds-models", "--bucket", "-b", help="S3 bucket name"),
    version: str | None = typer.Option(
        None,
        "--version",
        "-v",
        help="Version label (default: git SHA or timestamp)",
    ),
) -> None:
    """Publish a trained model to S3.

    Uploads model.pkl, config.yaml, and metadata.json to both
    {name}/{version}/ and {name}/latest/ in the S3 bucket.
    """
    model_path = Path(path)
    if not model_path.exists():
        console.print(f"[red]Model file not found: {model_path}[/red]")
        raise typer.Exit(1)

    # Look for config YAML alongside model
    config_path = model_path.with_suffix(".yaml")
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        console.print("Expected YAML config alongside the model file.")
        raise typer.Exit(1)

    # Resolve version
    if version is None:
        version = _get_git_sha() or datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    # Load config for metadata
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    metadata = _build_metadata(name, config_data, version)

    console.print(f"Publishing [bold]{name}[/bold] v{version} to s3://{bucket}/")
    console.print(f"  Model:  {model_path}")
    console.print(f"  Config: {config_path}")

    s3 = boto3.client("s3")

    # Upload to versioned path
    _upload_to_s3(s3, bucket, f"{name}/{version}", model_path, config_path, metadata)
    # Upload to latest path
    _upload_to_s3(s3, bucket, f"{name}/latest", model_path, config_path, metadata)

    console.print(f"[green]Published to s3://{bucket}/{name}/latest/[/green]")
    console.print(f"[green]Versioned at s3://{bucket}/{name}/{version}/[/green]")
