"""One-look evaluation of a frozen retail-EV model on a window of fresh events (ADR-0012).

A frozen ``EvaluationSpec`` names the window, the training start, the refit cadence,
the feature groups, the arm, the hyperparameters and the seeds. The evaluation reads
only events that kick off at or before the window end, refits the model on expanding
blocks of window events with the walk-forward gap, scores every window row out of
sample, and decides keep or retire by ADR-0012's two keep-bar conditions.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
import xgboost
import yaml
from odds_core.betfair_close import LIVE_CLOSE_WINDOW
from odds_core.market_prices import THREE_WAY_MARKET, market_outcomes
from odds_core.models import Event, EventStatus
from odds_core.snapshot_sources import SnapshotKind, snapshot_kind
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, PositiveInt, model_validator

from odds_analytics.retail_ev import (
    DECISION_WINDOW,
    HORIZON_BUCKET_LABELS,
    LOSS_WEIGHT_SCALE,
    MAX_EXCHANGE_OVERROUND,
    DropCounts,
    RetailEvDataset,
    RetailEvFeatures,
    fit_retail_ev,
    load_retail_ev_dataset,
)
from odds_analytics.retail_ev_metrics import (
    BootstrapEstimate,
    mse_gain,
    pick_ev_gain,
    weighted_mse,
)
from odds_analytics.training.config import RetailEvArm, RetailEvFeatureGroupName, XGBoostConfig

if TYPE_CHECKING:
    from odds_core.models import OddsSnapshot
    from sqlalchemy.ext.asyncio import AsyncSession

    from odds_analytics.as_of_team_features import TeamHistory

logger = structlog.get_logger()

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
# One EPL matchday; a longer run of events without live collection is reported as an outage.
OUTAGE_MIN_EVENTS = 10
FDUK_OPENING_SOURCE = "fduk_opening"
MANIFEST_FILE = "manifest.json"
REPORT_FILE = "report.json"
SLICES_FILE = "slices.csv"
BLOCKS_FILE = "blocks.csv"
PREDICTIONS_FILE = "predictions.csv.gz"


class WindowEndError(ValueError):
    """An event that kicks off after the evaluation window's end reached the evaluation."""


class IncompleteWindowError(ValueError):
    """A window event is not FINAL or has no FDUK closing snapshot."""


class RepeatedLookError(FileExistsError):
    """An evaluation with the same spec has already been written."""


class Verdict(StrEnum):
    KEEP = "keep"
    RETIRE = "retire"


class EvaluationWindow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: AwareDatetime = Field(description="First kick-off scored (inclusive).")
    end: AwareDatetime = Field(
        description="Last kick-off scored (inclusive); no event after it is read."
    )


class BootstrapSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resamples: PositiveInt = DEFAULT_BOOTSTRAP_RESAMPLES
    seed: int = 0


class SummaryTrial(BaseModel):
    """A tuned trial recorded in an experiment's ``summary.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path = Field(description="summary.json, relative to the spec file.")
    run: str = Field(description="The run label, such as residual_s42.")
    variant: str | None = Field(
        default=None, description="The variant holding the run, for summaries with variants."
    )

    def params(self, base_dir: Path) -> dict[str, Any]:
        summary = json.loads((base_dir / self.path).read_text())
        container = summary if self.variant is None else summary["variants"][self.variant]
        for run in container["runs"]:
            if run["run"] == self.run:
                return dict(run["best_params"])
        raise ValueError(f"no run {self.run!r} in {self.path}")


class EvaluationSpec(BaseModel):
    """Everything the evaluation runs, frozen before the window is scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sport_key: str = "soccer_epl"
    market: str = THREE_WAY_MARKET
    training_start: AwareDatetime = Field(description="First kick-off used for training.")
    window: EvaluationWindow
    refit_every_events: PositiveInt = Field(
        default=50, description="Window events scored by each refit, in kick-off order."
    )
    gap: timedelta = Field(
        default=DECISION_WINDOW,
        description="A block's model trains on events kicking off at least this long before "
        "the block's first event.",
    )
    feature_groups: tuple[RetailEvFeatureGroupName, ...] = ()
    feature_names: tuple[str, ...] = Field(
        description="The expected feature columns, in order; a run that builds others is refused."
    )
    arm: RetailEvArm = RetailEvArm.RESIDUAL
    model: XGBoostConfig = Field(
        default_factory=XGBoostConfig,
        description="XGBoost parameters; a trial's parameters replace the ones it names.",
    )
    trial: SummaryTrial | None = None
    seed: int = Field(default=42, description="XGBoost random_state.")
    bootstrap: BootstrapSpec = BootstrapSpec()

    @model_validator(mode="after")
    def validate_spec(self) -> EvaluationSpec:
        if not self.training_start < self.window.start <= self.window.end:
            raise ValueError("need training_start < window.start <= window.end")
        if self.gap < DECISION_WINDOW:
            raise ValueError(
                f"gap {self.gap} is shorter than the decision window {DECISION_WINDOW}, so a "
                "training close could fall after a scored decision"
            )
        if "random_state" in self.model.model_fields_set:
            raise ValueError("set the XGBoost seed with seed, not model.random_state")
        market_outcomes(self.market)
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> EvaluationSpec:
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def sha256(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()

    def resolved_model(self, base_dir: Path) -> XGBoostConfig:
        """``model`` with the trial's parameters applied and ``seed`` as random_state."""
        params = self.model.model_dump()
        if self.trial is not None:
            trial_params = self.trial.params(base_dir)
            unknown = set(trial_params) - set(params)
            if unknown:
                raise ValueError(f"trial parameters {sorted(unknown)} are not XGBoost fields")
            params |= trial_params
        params["random_state"] = self.seed
        return XGBoostConfig.model_validate(params)


@dataclass(frozen=True)
class EvaluationData:
    """Every row from the training start to the window end, in kick-off order.

    ``window_events`` lists every event in the window, with or without rows, with its
    status and which sources its snapshots hold.
    """

    rows: pd.DataFrame
    X: np.ndarray
    feature_names: list[str]
    window_events: pd.DataFrame
    window_drops: dict[str, Any]

    def fingerprint(self) -> str:
        """SHA-256 over every row's identity, inputs, target and weights."""
        digest = hashlib.sha256()
        digest.update("\n".join(self.rows["event_id"].astype(str)).encode())
        digest.update("\n".join(self.feature_names).encode())
        digest.update(np.ascontiguousarray(self.X, dtype=np.float32).tobytes())
        for col in ("y", "baseline", "w_loss", "w_event"):
            digest.update(np.ascontiguousarray(self.rows[col], dtype=np.float64).tobytes())
        return digest.hexdigest()


def check_window_end(kickoffs: Iterable[datetime], end: datetime) -> None:
    late = [k for k in kickoffs if k > end]
    if late:
        raise WindowEndError(
            f"{len(late)} events kick off after the window end {end.isoformat()}, the first "
            f"at {min(late).isoformat()}"
        )


def _naive_utc(ts: datetime) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC").tz_localize(None)


def _row_frame(dataset: RetailEvDataset) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "event_id": dataset.event_ids,
            "commence_time": pd.to_datetime(dataset.commence_times),
            "decision_time": pd.to_datetime([_naive_utc(r.decision_time) for r in dataset.rows]),
            "horizon": [r.horizon_bucket for r in dataset.rows],
            "retail_source": [r.retail_kind.value for r in dataset.rows],
            "close_source": [r.close.kind.value for r in dataset.rows],
            "outcome": [r.outcome.value for r in dataset.rows],
            "book": [r.book for r in dataset.rows],
            "retail_decimal": [r.retail_decimal for r in dataset.rows],
            "y": dataset.targets,
            "baseline": dataset.baselines,
            "w_event": dataset.event_weights,
            "w_loss": dataset.loss_weights,
        }
    )
    live_events = set(rows.loc[rows["retail_source"] != FDUK_OPENING_SOURCE, "event_id"])
    rows["event_type"] = np.where(rows["event_id"].isin(live_events), "live_collected", "fduk_only")
    return rows


def _window_event_frame(
    events: Sequence[Event], snapshots: dict[str, Sequence[OddsSnapshot]]
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for event in events:
        kickoff = event.commence_time
        kinds = [(snapshot_kind(s.raw_data), s.snapshot_time) for s in snapshots[event.id]]
        records.append(
            {
                "event_id": event.id,
                "commence_time": _naive_utc(kickoff),
                "status": event.status.value,
                "has_fduk_closing": any(k is SnapshotKind.FDUK_CLOSING for k, _ in kinds),
                "has_oddsportal_live": any(k is SnapshotKind.ODDSPORTAL_LIVE for k, _ in kinds),
                "has_live_betfair_near_kickoff": any(
                    k is SnapshotKind.BETFAIR_LIVE and kickoff - LIVE_CLOSE_WINDOW <= t <= kickoff
                    for k, t in kinds
                ),
            }
        )
    return pd.DataFrame(
        records,
        columns=[
            "event_id",
            "commence_time",
            "status",
            "has_fduk_closing",
            "has_oddsportal_live",
            "has_live_betfair_near_kickoff",
        ],
    )


def incomplete_events(window_events: pd.DataFrame) -> list[str]:
    """Window events that are not FINAL or have no FDUK closing snapshot."""
    incomplete = (window_events["status"] != EventStatus.FINAL.value) | ~window_events[
        "has_fduk_closing"
    ].astype(bool)
    return window_events.loc[incomplete, "event_id"].tolist()


async def load_evaluation_data(
    session: AsyncSession,
    spec: EvaluationSpec,
    now: datetime,
    allow_incomplete: Sequence[str] = (),
) -> EvaluationData:
    """Rows of every event kicking off from the training start to the window end.

    Refuses a window that has not ended, and raises ``WindowEndError`` if any event
    read kicks off after the window end. Raises ``IncompleteWindowError`` if a window
    event is not FINAL or has no FDUK closing snapshot, unless ``allow_incomplete`` names it.
    Team history is read only up to the window end. Refuses feature columns other than
    the spec's ``feature_names``.
    """
    from odds_lambda.storage.readers import OddsReader

    if spec.window.end > now:
        raise WindowEndError(
            f"the window ends {spec.window.end.isoformat()}, after now ({now.isoformat()}); "
            "scoring it now would be an interim look"
        )
    reader = OddsReader(session)
    events = await reader.get_events_by_date_range(
        start_date=spec.training_start,
        end_date=spec.window.end,
        sport_key=spec.sport_key,
    )
    check_window_end((e.commence_time for e in events), spec.window.end)
    training_events = [e for e in events if e.commence_time < spec.window.start]
    window_events = [e for e in events if e.commence_time >= spec.window.start]
    window_frame = _window_event_frame(
        window_events, {e.id: await reader.get_snapshots_for_event(e.id) for e in window_events}
    )
    incomplete = incomplete_events(window_frame)
    refused = [e for e in incomplete if e not in set(allow_incomplete)]
    if refused:
        raise IncompleteWindowError(
            f"{len(refused)} window events are not FINAL or have no FDUK closing snapshot, "
            f"such as {refused[:5]}; load the season's data or allow each by event ID"
        )

    train_set = await load_retail_ev_dataset(training_events, session, spec.market, now=now)
    window_set = await load_retail_ev_dataset(window_events, session, spec.market, now=now)
    dataset = RetailEvDataset(rows=train_set.rows + window_set.rows)
    if not dataset.rows:
        raise ValueError("no retail_ev rows from the training start to the window end")
    check_window_end((r.commence_time for r in dataset.rows), spec.window.end)

    teams: TeamHistory | None = None
    if RetailEvFeatures.needs_teams(spec.feature_groups):
        from odds_analytics.as_of_team_features import load_team_history
        from odds_analytics.training.config import FeatureConfig

        teams = await load_team_history(
            session,
            spec.sport_key,
            FeatureConfig.model_fields["form_window"].default,
            with_fixtures="schedule_as_of_t" in spec.feature_groups,
            until=spec.window.end,
        )
    features = RetailEvFeatures(
        market_outcomes(spec.market),
        spec.feature_groups,
        teams=teams,
        events={e.id: e for e in events},
    )
    if features.names != list(spec.feature_names):
        raise ValueError(
            f"the run builds features {features.names}, not the spec's {list(spec.feature_names)}"
        )
    rows = _row_frame(dataset)
    if not rows["commence_time"].is_monotonic_increasing:
        raise ValueError("rows are not in kick-off order")
    return EvaluationData(
        rows=rows,
        X=features.matrix(dataset.rows),
        feature_names=features.names,
        window_events=window_frame,
        window_drops=_drop_counts(window_set.drops),
    )


def _drop_counts(drops: DropCounts) -> dict[str, Any]:
    values: dict[str, Any] = {f.name: getattr(drops, f.name) for f in fields(drops)}
    values["prices_outside_roster"] = dict(drops.prices_outside_roster)
    return values


@dataclass(frozen=True)
class Block:
    index: int
    cutoff: pd.Timestamp
    train_rows: np.ndarray
    eval_rows: np.ndarray


def evaluation_blocks(rows: pd.DataFrame, spec: EvaluationSpec) -> list[Block]:
    """Window events in blocks of ``refit_every_events`` in kick-off order; the last may be partial.

    Each block trains on every row of the events that kick off at least ``gap`` before
    the block's first event.
    """
    first_rows = rows.drop_duplicates("event_id")
    kickoff = dict(zip(first_rows["event_id"], first_rows["commence_time"], strict=True))
    window_start = _naive_utc(spec.window.start)
    window_ids = [e for e in first_rows["event_id"] if kickoff[e] >= window_start]
    if not window_ids:
        raise ValueError("no window event has a row")
    positions = rows.groupby("event_id", sort=False).indices
    blocks: list[Block] = []
    for index, start in enumerate(range(0, len(window_ids), spec.refit_every_events)):
        block_ids = window_ids[start : start + spec.refit_every_events]
        cutoff = kickoff[block_ids[0]] - spec.gap
        train_ids = [e for e in first_rows["event_id"] if kickoff[e] <= cutoff]
        if not train_ids:
            raise ValueError(f"no training event kicks off {spec.gap} before block {index}")
        blocks.append(
            Block(
                index=index,
                cutoff=cutoff,
                train_rows=np.concatenate([positions[e] for e in train_ids]),
                eval_rows=np.concatenate([positions[e] for e in block_ids]),
            )
        )
    return blocks


Fitter = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def xgboost_fitter(feature_names: list[str], model: XGBoostConfig, arm: RetailEvArm) -> Fitter:
    """Fit on (X, y, loss weights) and predict the eval rows on the target's scale."""

    def fit_predict(
        X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, X_eval: np.ndarray
    ) -> np.ndarray:
        strategy = fit_retail_ev(X_train, y_train, w_train, feature_names, model.model_dump(), arm)
        return strategy.predict(X_eval).astype(np.float64)

    return fit_predict


def predict_window(
    data: EvaluationData, blocks: Sequence[Block], fit_predict: Fitter
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-sample predictions for every window row, and one record per block."""
    y = data.rows["y"].to_numpy(dtype=np.float32)
    w = data.rows["w_loss"].to_numpy(dtype=np.float64)
    scored: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for block in blocks:
        pred = fit_predict(
            data.X[block.train_rows],
            y[block.train_rows],
            w[block.train_rows],
            data.X[block.eval_rows],
        )
        part = data.rows.iloc[block.eval_rows].copy()
        part["block"] = block.index
        part["model"] = pred
        scored.append(part)
        train = data.rows.iloc[block.train_rows]
        records.append(
            {
                "block": block.index,
                "cutoff": block.cutoff,
                "train_events": train["event_id"].nunique(),
                "train_rows": len(train),
                "first_train_kickoff": train["commence_time"].min(),
                "last_train_kickoff": train["commence_time"].max(),
                "eval_events": part["event_id"].nunique(),
                "eval_rows": len(part),
                "first_eval_kickoff": part["commence_time"].min(),
                "last_eval_kickoff": part["commence_time"].max(),
                "wmse_baseline": weighted_mse(
                    part["y"].to_numpy(), part["baseline"].to_numpy(), part["w_loss"].to_numpy()
                ),
                "wmse_model": weighted_mse(part["y"].to_numpy(), pred, part["w_loss"].to_numpy()),
            }
        )
    return pd.concat(scored, ignore_index=True), pd.DataFrame(records)


@dataclass(frozen=True)
class KeepBar:
    """ADR-0012's two conditions; the model is kept only if both CIs lie above zero."""

    mse_gain: BootstrapEstimate
    pick_ev_gain: BootstrapEstimate

    @property
    def verdict(self) -> Verdict:
        if self.mse_gain.ci_low > 0 and self.pick_ev_gain.ci_low > 0:
            return Verdict.KEEP
        return Verdict.RETIRE


def keep_bar(scored: pd.DataFrame, bootstrap: BootstrapSpec) -> KeepBar:
    y = scored["y"].to_numpy()
    model = scored["model"].to_numpy()
    base = scored["baseline"].to_numpy()
    events = scored["event_id"].to_numpy()
    return KeepBar(
        mse_gain=mse_gain(
            y, model, base, scored["w_loss"].to_numpy(), events, bootstrap.resamples, bootstrap.seed
        ),
        pick_ev_gain=pick_ev_gain(
            y,
            model,
            base,
            scored["w_event"].to_numpy(),
            events,
            bootstrap.resamples,
            bootstrap.seed,
        ),
    )


def _estimate(prefix: str, e: BootstrapEstimate) -> dict[str, float]:
    return {prefix: e.point, f"{prefix}_ci_low": e.ci_low, f"{prefix}_ci_high": e.ci_high}


def _slice_metrics(sub: pd.DataFrame, bootstrap: BootstrapSpec) -> dict[str, Any]:
    y, w, we = sub["y"].to_numpy(), sub["w_loss"].to_numpy(), sub["w_event"].to_numpy()
    model, base = sub["model"].to_numpy(), sub["baseline"].to_numpy()
    bar = keep_bar(sub, bootstrap)
    wmse_base = weighted_mse(y, base, w)
    return {
        "rows": len(sub),
        "events": sub["event_id"].nunique(),
        "wmse_baseline": wmse_base,
        "wmse_model": weighted_mse(y, model, w),
        **_estimate("mse_gain", bar.mse_gain),
        "mse_gain_share_of_baseline": bar.mse_gain.point / wmse_base,
        **_estimate("pick_ev_gain", bar.pick_ev_gain),
        "picks_model": int((model > 0).sum()),
        "picks_baseline": int((base > 0).sum()),
        "pick_total_ev_model": float(np.sum(we * y * (model > 0))),
        "pick_total_ev_baseline": float(np.sum(we * y * (base > 0))),
    }


SLICE_KEYS: tuple[str, ...] = (
    "horizon",
    "retail_source",
    "close_source",
    "event_type",
    "outcome",
    "block",
)


def slice_table(scored: pd.DataFrame, bootstrap: BootstrapSpec) -> pd.DataFrame:
    """Descriptive keep-bar statistics per slice; they decide nothing."""
    records = [{"slice": "all", "value": "all", **_slice_metrics(scored, bootstrap)}]
    for key in SLICE_KEYS:
        values = scored[key].unique().tolist()
        ordered = (
            [b for b in HORIZON_BUCKET_LABELS if b in values]
            if key == "horizon"
            else sorted(values)
        )
        for value in ordered:
            sub = scored[scored[key] == value]
            records.append({"slice": key, "value": str(value), **_slice_metrics(sub, bootstrap)})
    return pd.DataFrame(records)


def _counts(frame: pd.DataFrame, key: str) -> dict[str, dict[str, int]]:
    grouped = frame.groupby(key)
    return {
        str(k): {"rows": int(n), "events": int(e)}
        for k, n, e in zip(
            grouped.size().index, grouped.size(), grouped["event_id"].nunique(), strict=True
        )
    }


def _runs_without(window_events: pd.DataFrame, flag: str, min_events: int) -> list[dict[str, Any]]:
    """Runs of at least ``min_events`` consecutive window events, in kick-off order, without ``flag``."""
    runs: list[dict[str, Any]] = []
    current: list[int] = []
    ordered = window_events.sort_values(["commence_time", "event_id"]).reset_index(drop=True)
    for i, covered in enumerate([*ordered[flag].astype(bool), True]):
        if not covered:
            current.append(i)
            continue
        if len(current) >= min_events:
            runs.append(
                {
                    "events": len(current),
                    "first_kickoff": str(ordered["commence_time"][current[0]]),
                    "last_kickoff": str(ordered["commence_time"][current[-1]]),
                }
            )
        current = []
    return runs


SOURCE_FLAGS: tuple[str, ...] = (
    "has_fduk_closing",
    "has_oddsportal_live",
    "has_live_betfair_near_kickoff",
)


def coverage(data: EvaluationData, scored: pd.DataFrame, spec: EvaluationSpec) -> dict[str, Any]:
    """Window events by status and by the sources their snapshots hold, the unscored events,
    collection outages, and the scored rows and events by slice."""
    window = data.window_events
    scored_ids = set(scored["event_id"])
    unscored = window[~window["event_id"].isin(scored_ids)]
    pre_window = data.rows[data.rows["commence_time"] < _naive_utc(spec.window.start)]
    return {
        "window_events": len(window),
        "window_events_by_status": {
            str(k): int(v) for k, v in window["status"].value_counts().sort_index().items()
        },
        "window_event_shares": {flag: float(window[flag].mean()) for flag in SOURCE_FLAGS},
        "scored_events": len(scored_ids),
        "scored_rows": len(scored),
        "pre_window_events": int(pre_window["event_id"].nunique()),
        "pre_window_rows": len(pre_window),
        "unscored_events": [
            {**r, "commence_time": str(r["commence_time"])}
            for r in unscored.to_dict(orient="records")
        ],
        "window_drops": data.window_drops,
        "outages": {
            f"without_{flag.removeprefix('has_')}": _runs_without(window, flag, OUTAGE_MIN_EVENTS)
            for flag in ("has_oddsportal_live", "has_live_betfair_near_kickoff")
        },
        "rows_by": {key: _counts(scored, key) for key in SLICE_KEYS if key != "block"},
    }


class OutputExistsError(FileExistsError):
    """The output directory already holds an evaluation."""


def check_output_dir(output_dir: Path, force: bool) -> str | None:
    """Refuse an output directory that is not empty unless ``force``; create nothing.

    Returns the ``predictions_sha256`` of the manifest a forced run would overwrite, if any.
    """
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise OutputExistsError(
            f"{output_dir} is not empty; an evaluation runs once (pass force to overwrite)"
        )
    previous = output_dir / MANIFEST_FILE
    if not previous.exists():
        return None
    return json.loads(previous.read_text()).get("predictions_sha256")


def prior_runs(results_root: Path, spec_sha256: str) -> list[Path]:
    """Manifests under ``results_root/*/`` written for the same spec; unreadable ones are skipped."""
    runs: list[Path] = []
    for path in sorted(results_root.glob(f"*/{MANIFEST_FILE}")):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("unreadable_evaluation_manifest", path=str(path), error=str(error))
            continue
        if isinstance(manifest, dict) and manifest.get("spec_sha256") == spec_sha256:
            runs.append(path)
    return runs


def check_one_look(results_root: Path, spec_sha256: str, force: bool) -> list[Path]:
    """Refuse a spec that already has a manifest under ``results_root`` unless ``force``."""
    runs = prior_runs(results_root, spec_sha256)
    if runs and not force:
        raise RepeatedLookError(
            f"spec {spec_sha256[:12]} was already evaluated in {[str(r) for r in runs]}; "
            "an evaluation runs once (pass force to run it again)"
        )
    return runs


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_tag() -> str | None:
    result = subprocess.run(
        ["git", "describe", "--tags", "--exact-match"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_state() -> dict[str, Any]:
    """HEAD, its exact tag, and whether tracked files differ from it."""
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_tag": _git_tag(),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }


def default_results_root() -> Path:
    return Path(_git("rev-parse", "--show-toplevel")) / "experiments" / "results"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predictions_sha256(scored: pd.DataFrame) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(scored["model"], dtype=np.float64).tobytes()
    ).hexdigest()


@dataclass(frozen=True)
class EvaluationReport:
    verdict: Verdict
    keep_bar: KeepBar
    manifest: dict[str, Any]
    coverage: dict[str, Any]
    slices: pd.DataFrame
    blocks: pd.DataFrame


def evaluate(
    data: EvaluationData, spec: EvaluationSpec, fit_predict: Fitter
) -> tuple[pd.DataFrame, pd.DataFrame, KeepBar]:
    """Predictions for every window row, the block records and the keep-bar statistics."""
    blocks = evaluation_blocks(data.rows, spec)
    scored, block_table = predict_window(data, blocks, fit_predict)
    return scored, block_table, keep_bar(scored, spec.bootstrap)


async def run_evaluation(
    session: AsyncSession,
    spec_path: Path,
    output_dir: Path,
    now: datetime,
    force: bool = False,
    allow_incomplete: Sequence[str] = (),
    results_root: Path | None = None,
) -> EvaluationReport:
    """Load, refit, score and write the report, manifest and predictions to ``output_dir``.

    Refuses a spec already evaluated under ``results_root`` (the repo's
    ``experiments/results`` by default) and a non-empty ``output_dir``, unless ``force``.
    """
    spec = EvaluationSpec.from_yaml(spec_path)
    earlier = check_one_look(results_root or default_results_root(), spec.sha256(), force)
    overwritten = check_output_dir(output_dir, force)
    model = spec.resolved_model(spec_path.parent)
    git = git_state()
    data = await load_evaluation_data(session, spec, now, allow_incomplete)
    scored, block_table, bar = evaluate(
        data, spec, xgboost_fitter(data.feature_names, model, spec.arm)
    )
    slices = slice_table(scored, spec.bootstrap)
    cover = coverage(data, scored, spec)

    manifest: dict[str, Any] = {
        **git,
        "forced": force,
        "earlier_runs_of_spec": [str(r) for r in earlier],
        "overwritten_predictions_sha256": overwritten,
        "allowed_incomplete_events": sorted(allow_incomplete),
        "incomplete_window_events": incomplete_events(data.window_events),
        "spec_path": str(spec_path),
        "spec_file_sha256": _sha256_file(spec_path),
        "spec_sha256": spec.sha256(),
        "spec": spec.model_dump(mode="json"),
        "trial": None
        if spec.trial is None
        else {
            "path": str(spec.trial.path),
            "file_sha256": _sha256_file(spec_path.parent / spec.trial.path),
            "run": spec.trial.run,
            "variant": spec.trial.variant,
        },
        "model": model.model_dump(),
        "code_constants": {
            "loss_weight_scale": LOSS_WEIGHT_SCALE,
            "max_exchange_overround": MAX_EXCHANGE_OVERROUND,
            "decision_window_hours": DECISION_WINDOW.total_seconds() / 3600,
        },
        "feature_names": data.feature_names,
        "data_sha256": data.fingerprint(),
        "predictions_sha256": _predictions_sha256(scored),
        "counts": {
            "rows": len(data.rows),
            "events": int(data.rows["event_id"].nunique()),
            "pre_window_rows": cover["pre_window_rows"],
            "pre_window_events": cover["pre_window_events"],
            "scored_rows": len(scored),
            "scored_events": int(scored["event_id"].nunique()),
            "unscored_window_events": len(cover["unscored_events"]),
            "blocks": len(block_table),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgboost.__version__,
        },
        "run_at": now.isoformat(),
    }
    report: dict[str, Any] = {
        "verdict": bar.verdict.value,
        "keep_bar": {
            "mse_gain": asdict(bar.mse_gain),
            "pick_ev_gain": asdict(bar.pick_ev_gain),
            "bootstrap": spec.bootstrap.model_dump(),
        },
        "coverage": cover,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    (output_dir / REPORT_FILE).write_text(json.dumps(report, indent=2, default=str) + "\n")
    slices.to_csv(output_dir / SLICES_FILE, index=False)
    block_table.to_csv(output_dir / BLOCKS_FILE, index=False)
    scored[
        [
            "event_id",
            "commence_time",
            "block",
            "decision_time",
            "horizon",
            "retail_source",
            "close_source",
            "event_type",
            "outcome",
            "book",
            "retail_decimal",
            "y",
            "baseline",
            "model",
            "w_loss",
            "w_event",
        ]
    ].to_csv(output_dir / PREDICTIONS_FILE, index=False)
    logger.info(
        "retail_ev_evaluation_written",
        output_dir=str(output_dir),
        verdict=bar.verdict.value,
        scored_events=manifest["counts"]["scored_events"],
    )
    return EvaluationReport(
        verdict=bar.verdict,
        keep_bar=bar,
        manifest=manifest,
        coverage=cover,
        slices=slices,
        blocks=block_table,
    )
