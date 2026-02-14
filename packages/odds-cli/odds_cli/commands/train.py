"""CLI commands for ML model training."""

import asyncio
from pathlib import Path
from typing import Any

import typer
from odds_analytics.lstm_line_movement import LSTMLineMovementStrategy
from odds_analytics.lstm_strategy import LSTMStrategy
from odds_analytics.training import (
    CVResult,
    MLTrainingConfig,
    TrackingConfig,
    prepare_training_data_from_config,
)
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
    track: bool = typer.Option(False, "--track", help="Enable MLflow experiment tracking"),
    tracking_uri: str | None = typer.Option(
        None, "--tracking-uri", help="Override MLflow tracking URI (default: mlruns)"
    ),
):
    """
    Train an ML model using a configuration file.

    Example:
        odds train run --config experiments/xgb_line_movement.yaml
        odds train run --config experiments/lstm_v1.yaml --output models/custom_path.pkl
        odds train run --config experiments/config.yaml --dry-run
        odds train run --config experiments/xgb_line_movement.yaml --track
        odds train run --config experiments/lstm_v1.yaml --track --tracking-uri http://localhost:5000
    """
    # Load and validate configuration
    ml_config = load_config(config)

    # Override output path if specified
    if output:
        ml_config.training.output_path = str(Path(output).parent)
        ml_config.training.model_name = Path(output).name

    # Override tracking configuration if flags provided
    if track:
        # Initialize tracking config if not present
        if ml_config.tracking is None:
            ml_config.tracking = TrackingConfig()
        ml_config.tracking.enabled = True
        if tracking_uri:
            ml_config.tracking.tracking_uri = tracking_uri
        # Set experiment name from config if not already set
        if not ml_config.tracking.experiment_name:
            ml_config.tracking.experiment_name = ml_config.experiment.name

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
        if ml_config.tracking and ml_config.tracking.enabled:
            console.print(
                f"  5. Log to MLflow tracking server at {ml_config.tracking.tracking_uri}"
            )
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

    # Initialize experiment tracker if enabled
    tracker = None
    if config.tracking and config.tracking.enabled:
        try:
            from odds_analytics.training import create_tracker

            tracker = create_tracker(config.tracking)
            console.print(f"[cyan]MLflow tracking enabled: {config.tracking.tracking_uri}[/cyan]")
            console.print(f"[cyan]Experiment: {config.tracking.experiment_name}[/cyan]\n")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not initialize tracker: {e}[/yellow]")
            console.print("[yellow]Continuing without tracking...[/yellow]\n")
            tracker = None

    # Start MLflow run if tracker is enabled
    if tracker and config.tracking:
        run_name = config.tracking.run_name or f"{config.experiment.name}_{strategy_type}"
        tracker.start_run(run_name=run_name, tags={"strategy": strategy_type})

    try:
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
                except Exception:
                    progress.stop()
                    console.print_exception()
                    raise typer.Exit(1) from None

                progress.update(task, description="Data preparation complete")

            # Display data summary
            console.print("\n[green]Data prepared successfully[/green]")
            console.print(f"  Training samples: {data_result.num_train_samples:,}")
            console.print(f"  Test samples: {data_result.num_test_samples:,}")
            console.print(f"  Features: {data_result.num_features}")

            # Step 2: Initialize strategy and train
            strategy = strategy_class()
            use_kfold = config.training.data.use_kfold
            cv_result = None

            # Get test data for final evaluation
            X_test = data_result.X_test if data_result.num_test_samples > 0 else None
            y_test = data_result.y_test if data_result.num_test_samples > 0 else None

            if use_kfold:
                # Cross-Validation training
                n_folds = config.training.data.n_folds
                cv_method = config.training.data.cv_method
                cv_method_display = "Time Series CV" if cv_method == "timeseries" else "K-Fold CV"
                console.print(f"\n[bold]Running {n_folds}-Fold {cv_method_display}...[/bold]")

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(
                        f"Cross-validating ({n_folds} folds, {cv_method_display})...", total=None
                    )

                    try:
                        history, cv_result = strategy.train_with_cv(
                            config,
                            data_result.X_train,
                            data_result.y_train,
                            data_result.feature_names,
                            X_test=X_test,
                            y_test=y_test,
                        )
                    except Exception:
                        progress.stop()
                        console.print_exception()
                        raise typer.Exit(1) from None

                    progress.update(task, description="Training complete")

                # Display CV metrics
                _display_cv_metrics(cv_result)

            else:
                # Standard training without CV
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("Training model...", total=None)

                    try:
                        history = strategy.train_from_config(
                            config,
                            data_result.X_train,
                            data_result.y_train,
                            data_result.feature_names,
                            X_val=X_test,
                            y_val=y_test,
                            tracker=tracker,
                        )
                    except Exception:
                        progress.stop()
                        console.print_exception()
                        raise typer.Exit(1) from None

                    progress.update(task, description="Training complete")

            # Display training metrics
            _display_training_metrics(history, verbose, is_cv=use_kfold)

            # Step 3: Save model
            output_path = _get_output_path(config)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            try:
                strategy.save_model(output_path)
                console.print(f"\n[green]Model saved to {output_path}[/green]")
            except Exception:
                console.print_exception()
                raise typer.Exit(1) from None

            # Step 4: Save configuration alongside model
            config_output = Path(output_path).with_suffix(".yaml")
            try:
                config.to_yaml(config_output)
                console.print(f"[green]Configuration saved to {config_output}[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not save configuration: {e}[/yellow]")

            console.print("\n[bold green]Training complete![/bold green]")

            # Display MLflow run info if tracking was enabled
            if tracker and config.tracking:
                console.print(
                    f"\n[cyan]MLflow run completed. View results at: {config.tracking.tracking_uri}[/cyan]"
                )

    finally:
        # End MLflow run
        if tracker:
            tracker.end_run(status="FINISHED")


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
        feature_table.add_row("Opening Tier", features.opening_tier.value)
        feature_table.add_row("Closing Tier", features.closing_tier.value)

        console.print(feature_table)


def _display_cv_metrics(cv_result: CVResult) -> None:
    """Display cross-validation metrics with mean ± std."""
    console.print(f"\n[bold]Cross-Validation Results ({cv_result.n_folds} folds):[/bold]")

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")

    # Validation MSE
    table.add_row(
        "Val MSE",
        f"{cv_result.mean_val_mse:.6f}",
        f"± {cv_result.std_val_mse:.6f}",
    )

    # Validation MAE
    table.add_row(
        "Val MAE",
        f"{cv_result.mean_val_mae:.6f}",
        f"± {cv_result.std_val_mae:.6f}",
    )

    # Validation R²
    table.add_row(
        "Val R²",
        f"{cv_result.mean_val_r2:.4f}",
        f"± {cv_result.std_val_r2:.4f}",
    )

    console.print(table)


def _display_training_metrics(history: dict, verbose: bool, is_cv: bool = False) -> None:
    """Display training metrics."""
    title = "Final Model Metrics" if is_cv else "Training Metrics"
    console.print(f"\n[bold]{title}:[/bold]")

    table = Table(show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Training", justify="right")
    table.add_column("Test" if is_cv else "Validation", justify="right")

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


@app.command("tune")
def run_tuning(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to training configuration file with tuning section"
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Override output path for best config (default: {name}_best.yaml)",
    ),
    train_best: bool = typer.Option(
        False, "--train-best", help="Train final model with best parameters after tuning"
    ),
    n_trials: int | None = typer.Option(
        None, "--n-trials", help="Override number of trials from config"
    ),
    timeout: int | None = typer.Option(
        None, "--timeout", help="Override timeout in seconds from config"
    ),
    study_name: str | None = typer.Option(
        None, "--study-name", help="Optuna study name for persistence/resumption"
    ),
    storage: str | None = typer.Option(
        None, "--storage", help="Optuna storage URL (defaults to DATABASE_URL from config)"
    ),
    track: bool = typer.Option(False, "--track", help="Enable MLflow experiment tracking"),
    tracking_uri: str | None = typer.Option(
        None, "--tracking-uri", help="Override MLflow tracking URI (default: mlruns)"
    ),
):
    """
    Run hyperparameter optimization using Optuna.

    Requires a configuration file with a tuning section that defines search spaces
    for hyperparameters. After optimization, the best parameters are saved to a new
    configuration file.

    When --track is enabled, each Optuna trial is logged as a nested MLflow run
    with the parent run containing study metadata and the best model (if --train-best).

    Example:
        odds train tune --config experiments/xgboost_tuning.yaml
        odds train tune --config experiments/xgboost_tuning.yaml --train-best
        odds train tune --config experiments/xgboost_tuning.yaml --n-trials 50 --timeout 3600
        odds train tune --config experiments/xgboost_tuning.yaml --study-name xgb_opt --storage postgresql://...
        odds train tune --config experiments/xgboost_tuning.yaml --track --tracking-uri http://localhost:5000
    """
    # Load and validate configuration
    ml_config = load_config(config)

    # Validate tuning section exists
    if ml_config.tuning is None:
        console.print("[red]Error: Configuration must include a 'tuning' section[/red]")
        console.print("\nAdd a tuning section to your config:")
        console.print("  tuning:")
        console.print("    n_trials: 100")
        console.print("    direction: minimize")
        console.print("    metric: val_mse")
        console.print("    search_spaces:")
        console.print("      n_estimators:")
        console.print("        type: int")
        console.print("        low: 50")
        console.print("        high: 500")
        raise typer.Exit(1)

    if not ml_config.tuning.search_spaces:
        console.print("[red]Error: tuning section must define search_spaces[/red]")
        console.print("\nDefine search spaces for parameters to tune:")
        console.print("  search_spaces:")
        console.print("    parameter_name:")
        console.print("      type: int|float|categorical")
        console.print("      low: value  # for int/float")
        console.print("      high: value  # for int/float")
        raise typer.Exit(1)

    # Override tuning parameters if specified
    if n_trials is not None:
        ml_config.tuning.n_trials = n_trials
    if timeout is not None:
        ml_config.tuning.timeout = timeout

    # Determine storage URL (default to DATABASE_URL if not specified)
    storage_url = storage
    if storage_url is None:
        try:
            from odds_core.config import get_config

            db_url = get_config().database.url
            # Convert asyncpg URL to psycopg2 URL for Optuna
            if db_url.startswith("postgresql+asyncpg://"):
                storage_url = db_url.replace("postgresql+asyncpg://", "postgresql://")
            else:
                storage_url = db_url
        except Exception:
            # If DATABASE_URL not available, use in-memory storage
            storage_url = None

    # Determine study name
    if study_name is None:
        study_name = f"{ml_config.experiment.name}_tuning"

    # Override tracking configuration if flags provided
    if track:
        # Initialize tracking config if not present
        if ml_config.tracking is None:
            ml_config.tracking = TrackingConfig()
        ml_config.tracking.enabled = True
        if tracking_uri:
            ml_config.tracking.tracking_uri = tracking_uri
        # Set experiment name from config if not already set
        if not ml_config.tracking.experiment_name:
            ml_config.tracking.experiment_name = ml_config.experiment.name

    # Display tuning configuration
    _display_tuning_summary(ml_config, study_name, storage_url)

    # Run tuning
    asyncio.run(_run_tuning_async(ml_config, study_name, storage_url, config, output, train_best))


async def _run_tuning_async(
    ml_config: MLTrainingConfig,
    study_name: str,
    storage_url: str | None,
    config_path: str,
    output_path: str | None,
    train_best: bool,
):
    """Execute tuning workflow asynchronously."""
    from odds_analytics.training.tuner import OptunaTuner, create_objective
    from odds_core.database import get_session

    console.print("\n[bold]Starting Hyperparameter Optimization[/bold]")
    console.print(f"Study: {study_name}")
    console.print(f"Trials: {ml_config.tuning.n_trials}")
    console.print(f"Direction: {ml_config.tuning.direction}")
    console.print(f"Metric: {ml_config.tuning.metric}\n")

    async with get_session() as session:
        # Step 1: Prepare training data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading and preparing training data...", total=None)

            try:
                from odds_analytics.training import prepare_training_data_from_config

                data_result = await prepare_training_data_from_config(ml_config, session)
            except ValueError as e:
                progress.stop()
                console.print(f"[red]Error preparing data: {e}[/red]")
                raise typer.Exit(1) from None
            except Exception:
                progress.stop()
                console.print_exception()
                raise typer.Exit(1) from None

            progress.update(task, description="Data preparation complete")

        # Display data summary
        console.print("\n[green]Data prepared successfully[/green]")
        console.print(f"  Training samples: {data_result.num_train_samples:,}")
        console.print(f"  Validation samples: {data_result.num_val_samples:,}")
        console.print(f"  Test samples: {data_result.num_test_samples:,}")
        console.print(f"  Features: {data_result.num_features}\n")

        # Step 2: Initialize tuner with tracking config
        tuner = OptunaTuner(
            study_name=study_name,
            direction=ml_config.tuning.direction,
            sampler=ml_config.tuning.sampler,
            pruner=ml_config.tuning.pruner,
            storage=storage_url,
            tracking_config=ml_config.tracking,
        )

        # Display tracking info if enabled
        if ml_config.tracking and ml_config.tracking.enabled:
            console.print(
                f"[cyan]MLflow tracking enabled: {ml_config.tracking.tracking_uri}[/cyan]"
            )
            console.print(f"[cyan]Experiment: {ml_config.tracking.experiment_name}[/cyan]\n")

        # Step 3: Create objective function
        # Use validation data for optimization if available, otherwise use test data
        X_val = data_result.X_val if data_result.num_val_samples > 0 else data_result.X_test
        y_val = data_result.y_val if data_result.num_val_samples > 0 else data_result.y_test

        # Pre-compute features for all feature_groups choices if being tuned
        precomputed_features = None
        if "feature_groups" in ml_config.tuning.search_spaces:
            console.print(
                "[cyan]Pre-computing features for all feature group combinations...[/cyan]"
            )
            precomputed_features = {}

            for choice in ml_config.tuning.search_spaces["feature_groups"].choices:
                fg_tuple = tuple(choice) if isinstance(choice, list) else choice

                # Check if this is the same as the default (already computed)
                if fg_tuple == ml_config.training.features.feature_groups:
                    precomputed_features[fg_tuple] = data_result
                    continue

                # Create modified config and extract features
                modified_config = ml_config.model_copy(deep=True)
                modified_config.training.features.feature_groups = fg_tuple

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task(f"Extracting features for {fg_tuple}...", total=None)
                    result = await prepare_training_data_from_config(modified_config, session)
                    progress.update(task, description=f"Extracted {result.num_features} features")

                precomputed_features[fg_tuple] = result

            console.print(
                f"[green]Pre-computed {len(precomputed_features)} feature combinations[/green]\n"
            )

        objective = create_objective(
            config=ml_config,
            X_train=data_result.X_train,
            y_train=data_result.y_train,
            feature_names=data_result.feature_names,
            X_val=X_val,
            y_val=y_val,
            precomputed_features=precomputed_features,
        )

        # Step 4: Run optimization
        console.print("[bold]Running optimization...[/bold]\n")

        try:
            study = tuner.optimize(
                objective=objective,
                n_trials=ml_config.tuning.n_trials,
                timeout=ml_config.tuning.timeout,
            )
        except KeyboardInterrupt:
            console.print("\n[yellow]Optimization interrupted by user[/yellow]")
            if tuner.study is None or len(tuner.study.trials) == 0:
                console.print("[red]No trials completed. Exiting.[/red]")
                raise typer.Exit(1) from None
            console.print(
                f"[yellow]Completed {len(tuner.study.trials)} trials before interruption[/yellow]"
            )
            study = tuner.study
        except Exception as e:
            console.print(f"\n[red]Error during optimization: {e}[/red]")
            raise typer.Exit(1) from None

        # Step 5: Display best parameters
        console.print("\n[bold green]Optimization complete![/bold green]")
        console.print(f"Completed trials: {len(study.trials)}")
        console.print(f"Best trial: #{study.best_trial.number}")
        console.print(f"Best {ml_config.tuning.metric}: {study.best_value:.6f}")

        # Display additional best model metrics from CV if available
        if hasattr(study.best_trial, "user_attrs") and study.best_trial.user_attrs:
            # Group mean and std metrics together
            mean_metrics = {
                k: v
                for k, v in study.best_trial.user_attrs.items()
                if isinstance(v, int | float) and k.startswith("mean_")
            }
            std_metrics = {
                k: v
                for k, v in study.best_trial.user_attrs.items()
                if isinstance(v, int | float) and k.startswith("std_")
            }

            if mean_metrics:
                console.print("\n[bold]Best model cross-validation metrics:[/bold]")
                for mean_key in sorted(mean_metrics.keys()):
                    metric_name = mean_key.replace("mean_", "")
                    std_key = f"std_{metric_name}"
                    mean_val = mean_metrics[mean_key]
                    std_val = std_metrics.get(std_key, 0.0)
                    console.print(f"  {metric_name}: {mean_val:.6f} ± {std_val:.6f}")

        console.print()  # Blank line before parameters
        _display_best_parameters(study.best_params)

        # Step 6: Export best configuration
        best_config = ml_config.model_copy(deep=True)

        # Update model parameters with best values
        for param_name, param_value in study.best_params.items():
            if hasattr(best_config.training.model, param_name):
                setattr(best_config.training.model, param_name, param_value)
            elif hasattr(best_config.training.features, param_name):
                setattr(best_config.training.features, param_name, param_value)

        # Determine output path
        if output_path is None:
            config_file = Path(config_path)
            output_path = str(config_file.parent / f"{config_file.stem}_best.yaml")

        # Save best configuration
        try:
            best_config.to_yaml(Path(output_path))
            console.print(f"\n[green]Best configuration saved to {output_path}[/green]")
        except Exception as e:
            console.print(f"\n[red]Error saving best configuration: {e}[/red]")
            raise typer.Exit(1) from None

        # Step 7: Optionally train model with best parameters
        if train_best:
            console.print("\n[bold]Training model with best parameters...[/bold]")

            # Get strategy class
            strategy_type = best_config.training.strategy_type
            if strategy_type not in STRATEGY_CLASSES:
                console.print(f"[red]Error: Unknown strategy type '{strategy_type}'[/red]")
                raise typer.Exit(1)

            strategy_class = STRATEGY_CLASSES[strategy_type]
            strategy = strategy_class()

            # Train with best config
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Training model...", total=None)

                try:
                    history = strategy.train_from_config(
                        config=best_config,
                        X_train=data_result.X_train,
                        y_train=data_result.y_train,
                        feature_names=data_result.feature_names,
                        X_val=X_val,
                        y_val=y_val,
                    )
                except Exception:
                    progress.stop()
                    console.print_exception()
                    raise typer.Exit(1) from None

                progress.update(task, description="Training complete")

            # Display training metrics
            _display_training_metrics(history, verbose=False)

            # Save model
            output_model_path = _get_output_path(best_config)
            Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)

            try:
                strategy.save_model(output_model_path)
                console.print(f"\n[green]Model saved to {output_model_path}[/green]")

                # Log best model to parent MLflow run if tracking enabled
                if ml_config.tracking and ml_config.tracking.enabled:
                    try:
                        tuner.log_best_model(strategy.model, artifact_path="best_model")
                        console.print(
                            f"[cyan]Best model logged to MLflow (run_id: {tuner._parent_run_id})[/cyan]"
                        )
                    except Exception as e:
                        console.print(
                            f"[yellow]Warning: Could not log model to MLflow: {e}[/yellow]"
                        )

            except Exception:
                console.print_exception()
                raise typer.Exit(1) from None

            console.print("\n[bold green]Training complete![/bold green]")


def _display_tuning_summary(config: MLTrainingConfig, study_name: str, storage_url: str | None):
    """Display tuning configuration summary."""
    console.print(
        Panel.fit(
            f"[bold]{config.experiment.name}[/bold]\n"
            f"{config.experiment.description or 'No description'}",
            title="Experiment",
        )
    )

    # Tuning config table
    table = Table(title="Tuning Configuration", show_header=False)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value")

    table.add_row("Study Name", study_name)
    table.add_row("Number of Trials", str(config.tuning.n_trials))
    table.add_row("Timeout", str(config.tuning.timeout) if config.tuning.timeout else "None")
    table.add_row("Direction", config.tuning.direction)
    table.add_row("Metric", config.tuning.metric)
    table.add_row("Sampler", config.tuning.sampler)
    table.add_row("Pruner", config.tuning.pruner)
    table.add_row("Storage", storage_url if storage_url else "In-memory")
    table.add_row("Search Spaces", str(len(config.tuning.search_spaces)))

    console.print(table)

    # Search spaces table
    if config.tuning.search_spaces:
        search_table = Table(title="Search Spaces", show_header=True)
        search_table.add_column("Parameter", style="cyan")
        search_table.add_column("Type", style="yellow")
        search_table.add_column("Range/Choices")

        for param_name, space in config.tuning.search_spaces.items():
            if space.type == "categorical":
                range_str = f"{space.choices}"
            elif space.type in ("int", "float"):
                log_str = " (log)" if space.log else ""
                step_str = f", step={space.step}" if space.step else ""
                range_str = f"[{space.low}, {space.high}]{step_str}{log_str}"
            else:
                range_str = "unknown"

            search_table.add_row(param_name, space.type, range_str)

        console.print(search_table)


def _display_best_parameters(best_params: dict[str, Any]):
    """Display best parameters in table format."""
    table = Table(title="Best Parameters", show_header=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    for param_name, param_value in best_params.items():
        # Format value based on type
        if isinstance(param_value, float):
            value_str = f"{param_value:.6f}"
        else:
            value_str = str(param_value)

        table.add_row(param_name, value_str)

    console.print(table)


if __name__ == "__main__":
    app()
