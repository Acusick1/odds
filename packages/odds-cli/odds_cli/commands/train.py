"""CLI commands for ML model training."""

import asyncio
from pathlib import Path

import typer
from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy
from odds_analytics.lstm_strategy import LSTMStrategy
from odds_analytics.training import MLTrainingConfig, prepare_training_data_from_config
from odds_analytics.xgboost_line_movement import XGBoostLineMovementStrategy
from odds_core.database import get_session
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer()
console = Console()

# Strategy registry mapping strategy_type to class
STRATEGY_CLASSES = {
    "xgboost_line_movement": XGBoostLineMovementStrategy,
    "lstm": LSTMStrategy,
    "lstm_line_movement": LSTMLineMovementStrategy,
}

# Default directory for experiment configs
DEFAULT_CONFIG_DIR = Path("experiments")


def load_config(config_path: str) -> MLTrainingConfig:
    """
    Load and validate configuration from file.

    Args:
        config_path: Path to YAML or JSON configuration file

    Returns:
        Validated MLTrainingConfig

    Raises:
        typer.Exit: If file not found or invalid
    """
    path = Path(config_path)

    if not path.exists():
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        raise typer.Exit(1)

    try:
        if path.suffix in (".yaml", ".yml"):
            return MLTrainingConfig.from_yaml(path)
        elif path.suffix == ".json":
            return MLTrainingConfig.from_json(path)
        else:
            console.print(
                f"[red]Error: Unsupported config format '{path.suffix}'. Use .yaml or .json[/red]"
            )
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1) from None


@app.command("run")
def run_training(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to training configuration file (YAML/JSON)"
    ),
    output: str | None = typer.Option(
        None, "--output", "-o", help="Override output path for model"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable detailed output logging"),
):
    """
    Train an ML model using a configuration file.

    Example:
        odds train run --config experiments/xgb_line_movement.yaml
        odds train run --config experiments/lstm_v1.yaml --output models/custom_path.pkl
        odds train run --config experiments/config.yaml --dry-run
    """
    # Load and validate configuration
    ml_config = load_config(config)

    # Override output path if specified
    if output:
        ml_config.training.output_path = str(Path(output).parent)
        ml_config.training.model_name = Path(output).name

    # Display configuration summary
    _display_config_summary(ml_config, verbose)

    if dry_run:
        console.print("\n[yellow]Dry run mode - training will not be executed[/yellow]")
        console.print("\n[bold]Would execute:[/bold]")
        console.print(
            f"  1. Load {ml_config.training.data.start_date} to {ml_config.training.data.end_date} data"
        )
        console.print(f"  2. Prepare training data for {ml_config.training.strategy_type}")
        console.print(f"  3. Train model with {_get_model_params_count(ml_config)} parameters")
        console.print(f"  4. Save model to {_get_output_path(ml_config)}")
        return

    # Run training
    asyncio.run(_run_training_async(ml_config, verbose))


async def _run_training_async(config: MLTrainingConfig, verbose: bool):
    """Execute training workflow asynchronously."""
    strategy_type = config.training.strategy_type

    # Get strategy class
    if strategy_type not in STRATEGY_CLASSES:
        console.print(f"[red]Error: Unknown strategy type '{strategy_type}'[/red]")
        console.print(f"Available types: {', '.join(STRATEGY_CLASSES.keys())}")
        raise typer.Exit(1)

    strategy_class = STRATEGY_CLASSES[strategy_type]

    console.print(f"\n[bold]Training {config.experiment.name}[/bold]")
    console.print(f"Strategy: {strategy_type}")
    console.print(f"Period: {config.training.data.start_date} to {config.training.data.end_date}\n")

    async with get_session() as session:
        # Step 1: Prepare training data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading and preparing training data...", total=None)

            try:
                data_result = await prepare_training_data_from_config(config, session)
            except ValueError as e:
                progress.stop()
                console.print(f"[red]Error preparing data: {e}[/red]")
                raise typer.Exit(1) from None
            except Exception as e:
                progress.stop()
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1) from None

            progress.update(task, description="Data preparation complete")

        # Display data summary
        console.print("\n[green]Data prepared successfully[/green]")
        console.print(f"  Training samples: {data_result.num_train_samples:,}")
        console.print(f"  Test samples: {data_result.num_test_samples:,}")
        console.print(f"  Features: {data_result.num_features}")

        # Step 2: Initialize strategy and train
        strategy = strategy_class()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Training model...", total=None)

            try:
                # Split validation from test if configured
                X_val = None
                y_val = None

                # Use test set as validation for metrics display
                if data_result.num_test_samples > 0:
                    X_val = data_result.X_test
                    y_val = data_result.y_test

                history = strategy.train_from_config(
                    config,
                    data_result.X_train,
                    data_result.y_train,
                    data_result.feature_names,
                    X_val=X_val,
                    y_val=y_val,
                )
            except Exception as e:
                progress.stop()
                console.print(f"[red]Error during training: {e}[/red]")
                raise typer.Exit(1) from None

            progress.update(task, description="Training complete")

        # Display training metrics
        _display_training_metrics(history, verbose)

        # Step 3: Save model
        output_path = _get_output_path(config)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            strategy.save_model(output_path)
            console.print(f"\n[green]Model saved to {output_path}[/green]")
        except Exception as e:
            console.print(f"[red]Error saving model: {e}[/red]")
            raise typer.Exit(1) from None

        # Step 4: Save configuration alongside model
        config_output = Path(output_path).with_suffix(".yaml")
        try:
            config.to_yaml(config_output)
            console.print(f"[green]Configuration saved to {config_output}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not save configuration: {e}[/yellow]")

        console.print("\n[bold green]Training complete![/bold green]")


@app.command("validate")
def validate_config(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to training configuration file (YAML/JSON)"
    ),
):
    """
    Validate a configuration file without executing training.

    Checks for:
    - Valid file format (YAML/JSON)
    - Required fields present
    - Valid date ranges
    - Correct strategy type
    - Model config matches strategy

    Example:
        odds train validate --config experiments/xgb_line_movement.yaml
    """
    console.print(f"\nValidating configuration: {config}\n")

    path = Path(config)
    if not path.exists():
        console.print(f"[red]Error: File not found: {config}[/red]")
        raise typer.Exit(1)

    # Track validation results
    errors = []
    warnings = []

    # Check file extension
    if path.suffix not in (".yaml", ".yml", ".json"):
        errors.append(f"Unsupported file format '{path.suffix}'. Use .yaml or .json")

    # Try to load and validate
    try:
        ml_config = load_config(config)
        console.print("[green]Configuration loaded successfully[/green]\n")
    except typer.Exit:
        raise
    except Exception as e:
        errors.append(f"Failed to parse configuration: {e}")
        ml_config = None

    if ml_config:
        # Validate strategy type
        if ml_config.training.strategy_type not in STRATEGY_CLASSES:
            errors.append(
                f"Unknown strategy_type '{ml_config.training.strategy_type}'. "
                f"Available: {', '.join(STRATEGY_CLASSES.keys())}"
            )

        # Validate date range
        data_config = ml_config.training.data
        if data_config.start_date >= data_config.end_date:
            errors.append(
                f"Invalid date range: start_date ({data_config.start_date}) "
                f"must be before end_date ({data_config.end_date})"
            )

        # Validate splits
        total_split = data_config.test_split + data_config.validation_split
        if total_split >= 1.0:
            errors.append(
                f"Invalid splits: test_split ({data_config.test_split}) + "
                f"validation_split ({data_config.validation_split}) = {total_split} >= 1.0"
            )

        # Check for potential issues (warnings)
        if data_config.test_split < 0.1:
            warnings.append(f"Very small test split ({data_config.test_split})")

        # Display configuration summary
        _display_config_summary(ml_config, verbose=True)

    # Display results
    console.print("\n[bold]Validation Results:[/bold]")

    if errors:
        console.print(f"\n[red]Errors ({len(errors)}):[/red]")
        for error in errors:
            console.print(f"  [red]- {error}[/red]")

    if warnings:
        console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
        for warning in warnings:
            console.print(f"  [yellow]- {warning}[/yellow]")

    if not errors and not warnings:
        console.print("\n[green]Configuration is valid with no issues[/green]")
    elif not errors:
        console.print(f"\n[green]Configuration is valid with {len(warnings)} warning(s)[/green]")
    else:
        console.print(f"\n[red]Configuration has {len(errors)} error(s)[/red]")
        raise typer.Exit(1)


@app.command("list-configs")
def list_configs(
    directory: str = typer.Option(
        "experiments", "--directory", "-d", help="Directory to search for config files"
    ),
):
    """
    List available configuration files in a directory.

    Example:
        odds train list-configs
        odds train list-configs --directory configs/
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        console.print(f"[yellow]Directory not found: {directory}[/yellow]")
        console.print("\nTo create the directory and add a sample config:")
        console.print(f"  mkdir -p {directory}")
        console.print(f"  # Add your YAML config files to {directory}/")
        return

    # Find all config files
    yaml_files = list(dir_path.glob("**/*.yaml")) + list(dir_path.glob("**/*.yml"))
    json_files = list(dir_path.glob("**/*.json"))
    config_files = sorted(yaml_files + json_files)

    if not config_files:
        console.print(f"[yellow]No configuration files found in {directory}[/yellow]")
        console.print("\nSupported formats: .yaml, .yml, .json")
        return

    console.print(f"\n[bold]Configuration files in {directory}:[/bold]\n")

    table = Table()
    table.add_column("File", style="cyan")
    table.add_column("Experiment", style="green")
    table.add_column("Strategy", style="yellow")
    table.add_column("Date Range")

    for config_file in config_files:
        try:
            config = load_config(str(config_file))
            relative_path = (
                config_file.relative_to(dir_path) if dir_path != Path(".") else config_file
            )
            table.add_row(
                str(relative_path),
                config.experiment.name,
                config.training.strategy_type,
                f"{config.training.data.start_date} to {config.training.data.end_date}",
            )
        except Exception as e:
            relative_path = (
                config_file.relative_to(dir_path) if dir_path != Path(".") else config_file
            )
            table.add_row(
                str(relative_path),
                "[red]Error[/red]",
                str(e)[:30],
                "",
            )

    console.print(table)
    console.print(f"\nTotal: {len(config_files)} configuration file(s)")


def _display_config_summary(config: MLTrainingConfig, verbose: bool = False) -> None:
    """Display configuration summary."""
    # Experiment info
    console.print(
        Panel.fit(
            f"[bold]{config.experiment.name}[/bold]\n"
            f"{config.experiment.description or 'No description'}",
            title="Experiment",
        )
    )

    # Training config table
    table = Table(title="Training Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("Strategy Type", config.training.strategy_type)
    table.add_row("Start Date", str(config.training.data.start_date))
    table.add_row("End Date", str(config.training.data.end_date))
    table.add_row("Test Split", f"{config.training.data.test_split:.0%}")
    table.add_row("Validation Split", f"{config.training.data.validation_split:.0%}")
    table.add_row("Output Path", _get_output_path(config))

    if config.experiment.tags:
        table.add_row("Tags", ", ".join(config.experiment.tags))

    console.print(table)

    if verbose:
        # Model parameters
        model_config = config.training.model
        model_table = Table(title="Model Parameters", show_header=False)
        model_table.add_column("Parameter", style="cyan")
        model_table.add_column("Value")

        # Get model params as dict
        model_dict = model_config.model_dump()
        for key, value in model_dict.items():
            model_table.add_row(key, str(value))

        console.print(model_table)

        # Feature configuration
        features = config.training.features
        feature_table = Table(title="Feature Configuration", show_header=False)
        feature_table.add_column("Parameter", style="cyan")
        feature_table.add_column("Value")

        feature_table.add_row("Sharp Bookmakers", ", ".join(features.sharp_bookmakers))
        feature_table.add_row("Retail Bookmakers", ", ".join(features.retail_bookmakers))
        feature_table.add_row("Markets", ", ".join(features.markets))
        feature_table.add_row("Outcome", features.outcome)
        feature_table.add_row("Opening Hours Before", str(features.opening_hours_before))
        feature_table.add_row("Closing Hours Before", str(features.closing_hours_before))

        console.print(feature_table)


def _display_training_metrics(history: dict, verbose: bool) -> None:
    """Display training metrics."""
    console.print("\n[bold]Training Metrics:[/bold]")

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Training", justify="right")
    table.add_column("Validation", justify="right")

    # MSE
    train_mse = history.get("train_mse", 0)
    val_mse = history.get("val_mse", "-")
    table.add_row(
        "MSE", f"{train_mse:.6f}", f"{val_mse:.6f}" if isinstance(val_mse, float) else val_mse
    )

    # MAE
    train_mae = history.get("train_mae", 0)
    val_mae = history.get("val_mae", "-")
    table.add_row(
        "MAE", f"{train_mae:.6f}", f"{val_mae:.6f}" if isinstance(val_mae, float) else val_mae
    )

    # R²
    train_r2 = history.get("train_r2", 0)
    val_r2 = history.get("val_r2", "-")
    table.add_row("R²", f"{train_r2:.4f}", f"{val_r2:.4f}" if isinstance(val_r2, float) else val_r2)

    console.print(table)

    if verbose:
        console.print(f"\nTraining samples: {history.get('n_samples', 'N/A'):,}")
        console.print(f"Features: {history.get('n_features', 'N/A')}")


def _get_output_path(config: MLTrainingConfig) -> str:
    """Get the full output path for the model."""
    output_dir = Path(config.training.output_path)

    if config.training.model_name:
        filename = config.training.model_name
    else:
        # Auto-generate filename
        filename = f"{config.training.strategy_type}_{config.experiment.name}.pkl"

    return str(output_dir / filename)


def _get_model_params_count(config: MLTrainingConfig) -> int:
    """Get count of configurable model parameters."""
    return len(config.training.model.model_dump())


if __name__ == "__main__":
    app()
