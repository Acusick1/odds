"""Round 1's cell selection: replay every cell of a pre-registered grid on the development
blocks and pick the one cell that runs on the sealed holdout, or none, which kills the idea.

The rule is the text at ``ROUND_ONE_RULE``. Each cell is replayed as one pooled draft over the
file's leagues through ``odds_analytics.candidate_backtest.collect``, so the seal applies.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import yaml
from odds_lambda.candidate_rules.registry import resolve_rule
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.candidate_backtest import collect
from odds_analytics.candidate_report import ScoredBet
from odds_analytics.holdout import (
    FIVE_LEAGUES,
    SELECTION_DIR,
    HoldoutRefused,
    Preregistration,
    Selection,
    load_preregistration,
)
from odds_analytics.holdout_run import pooled_draft
from odds_analytics.retail_ev_metrics import event_clustered_ratio

ROUND_ONE_RULE = "https://github.com/Acusick1/odds/issues/821#issuecomment-6769611091"
MIN_BETS = 50
MIN_EVENTS = 30
# A block with fewer bets with a close than this does not need a positive mean EV.
MIN_BLOCK_BETS = 20
INTERVAL_LEVEL = 0.90
RESAMPLES = 10_000
SEED = 821
# An idea that bets above this price also reports its power-devig mean EV (#770).
LONGSHOT_PRICE = 4.0


class SelectionRefused(ValueError):
    """A selection that cannot run; nothing was replayed or written."""


@dataclass(frozen=True)
class Block:
    """Development days of some leagues, inclusive."""

    name: str
    start: date
    end: date
    leagues: tuple[str, ...]


# Seasons run 1 August to 31 July (ADR-0021, ADR-0024).
DEVELOPMENT_BLOCKS = (
    Block("EPL 2022-24", date(2022, 8, 1), date(2024, 7, 31), ("soccer_epl",)),
    Block("EPL 2024-26", date(2024, 8, 1), date(2026, 7, 31), ("soccer_epl",)),
    Block("five leagues 2024-25", date(2024, 8, 1), date(2025, 7, 31), FIVE_LEAGUES),
)


def development_blocks(leagues: Sequence[str]) -> list[Block]:
    """``DEVELOPMENT_BLOCKS`` limited to ``leagues``, leaving out a block with none of them."""
    blocks = []
    for block in DEVELOPMENT_BLOCKS:
        kept = tuple(league for league in block.leagues if league in leagues)
        if kept:
            blocks.append(Block(block.name, block.start, block.end, kept))
    return blocks


@dataclass(frozen=True)
class CellReplay:
    """A cell's placed bets per block, in ``blocks`` order and placement order in each."""

    index: int
    cell: dict[str, object]
    blocks: tuple[tuple[Block, tuple[ScoredBet, ...]], ...]


@dataclass(frozen=True)
class BlockScore:
    name: str
    bets: int
    mean_ev: float | None


@dataclass(frozen=True)
class CellScore:
    """A cell scored over its bets with a close: ``lower_bound`` is the 90% interval's, and
    ``power_mean_ev`` the mean EV against the power-devigged close when the idea bets above
    ``LONGSHOT_PRICE``, else None."""

    index: int
    cell: dict[str, object]
    bets: int
    events: int
    blocks: tuple[BlockScore, ...]
    mean_ev: float | None
    lower_bound: float | None
    power_mean_ev: float | None
    eligible: bool


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _evs(bets: Sequence[ScoredBet]) -> list[float]:
    return [b.ev for b in bets if b.ev is not None]


def lower_bound(bets: Sequence[ScoredBet]) -> float:
    """The lower bound of the 90% event-clustered percentile bootstrap interval of mean EV over
    ``bets``, which all have a close: ``RESAMPLES`` resamples of the events, in id order, with
    ``numpy.random.default_rng(SEED)``, each resample's mean its bets' total EV over their
    count."""
    estimate = event_clustered_ratio(
        np.array([b.event_id for b in bets]),
        np.array([b.ev for b in bets], dtype=float),
        np.ones(len(bets)),
        (INTERVAL_LEVEL,),
        RESAMPLES,
        SEED,
    )
    return estimate.intervals[INTERVAL_LEVEL][0]


def is_eligible(bets: int, events: int, blocks: Sequence[BlockScore]) -> bool:
    """At least ``MIN_BETS`` bets with a close in ``MIN_EVENTS`` events, and a positive mean EV
    in every block with at least ``MIN_BLOCK_BETS`` of them."""
    return (
        bets >= MIN_BETS
        and events >= MIN_EVENTS
        and all(b.mean_ev is not None and b.mean_ev > 0 for b in blocks if b.bets >= MIN_BLOCK_BETS)
    )


def score_cell(replay: CellReplay, longshots: bool) -> CellScore:
    """``replay`` scored over its bets with a close; ``longshots`` reports the power-devig mean
    EV."""
    blocks: list[BlockScore] = []
    scored: list[ScoredBet] = []
    for block, bets in replay.blocks:
        with_close = [b for b in bets if b.ev is not None]
        scored += with_close
        blocks.append(BlockScore(block.name, len(with_close), _mean(_evs(with_close))))
    events = len({b.event_id for b in scored})
    power = [b.ev_power for b in scored if b.ev_power is not None]
    return CellScore(
        index=replay.index,
        cell=replay.cell,
        bets=len(scored),
        events=events,
        blocks=tuple(blocks),
        mean_ev=_mean(_evs(scored)),
        lower_bound=lower_bound(scored) if scored else None,
        power_mean_ev=_mean(power) if longshots and len(power) == len(scored) else None,
        eligible=is_eligible(len(scored), events, blocks),
    )


def bets_longshots(replays: Sequence[CellReplay]) -> bool:
    """Whether any cell placed a bet above ``LONGSHOT_PRICE``."""
    return any(
        bet.price_decimal > LONGSHOT_PRICE for r in replays for _, bets in r.blocks for bet in bets
    )


def pick(scores: Sequence[CellScore]) -> CellScore | None:
    """The eligible cell with the highest lower bound; ties go to the higher mean EV, then more
    events, then the earlier cell in grid order. None when no cell is eligible."""

    def rank(score: CellScore) -> tuple[float, float, int, int]:
        assert score.lower_bound is not None and score.mean_ev is not None  # noqa: S101
        return (score.lower_bound, score.mean_ev, score.events, -score.index)

    return max((s for s in scores if s.eligible), key=rank, default=None)


@dataclass(frozen=True)
class GridSelection:
    """Every cell's replay and score in grid order, and the picked cell, None when the idea is
    killed on development data."""

    prereg: Preregistration
    grid_sha256: str
    blocks: tuple[Block, ...]
    replays: tuple[CellReplay, ...]
    scores: tuple[CellScore, ...]
    picked: CellScore | None
    longshots: bool

    @property
    def selection(self) -> Selection | None:
        if self.picked is None:
            return None
        return Selection(
            issue=self.prereg.issue,
            market=self.prereg.market,
            version=self.prereg.version,
            grid_sha256=self.grid_sha256,
            cell=self.picked.cell,
        )


def select_from(
    prereg: Preregistration, grid_sha256: str, replays: Sequence[CellReplay]
) -> GridSelection:
    """Apply the rule to the cells' replays, which are in grid order."""
    longshots = bets_longshots(replays)
    scores = tuple(score_cell(r, longshots) for r in replays)
    return GridSelection(
        prereg,
        grid_sha256,
        tuple(development_blocks(prereg.leagues)),
        tuple(replays),
        scores,
        pick(scores),
        longshots,
    )


def selection_path(prereg: Preregistration, directory: Path) -> Path:
    return directory / SELECTION_DIR / f"{prereg.stem}.yaml"


def load_selectable(path: Path, directory: Path, today: date) -> Preregistration:
    """The grid file at ``path``, after checking that it follows round 1's rule, has no
    selection yet and that every cell resolves as a pooled draft of a registered rule."""
    try:
        prereg = load_preregistration(path, directory)
    except HoldoutRefused as exc:
        raise SelectionRefused(str(exc)) from exc
    if prereg.selection_rule != ROUND_ONE_RULE:
        raise SelectionRefused(
            f"{path.name} is selected by {prereg.selection_rule}, not round 1's rule "
            f"{ROUND_ONE_RULE}"
        )
    existing = selection_path(prereg, directory)
    if existing.exists():
        raise SelectionRefused(
            f"{existing} already exists, and a selection is never changed: a new cell is a new "
            "version with a new grid"
        )
    for cell in prereg.candidate_cells():
        try:
            resolve_rule(pooled_draft(prereg, cell, today, "Cell selection"))
        except ValueError as exc:
            raise SelectionRefused(f"{path.name}: cell {cell!r}: {exc}") from exc
    return prereg


async def replay_cells(
    session: AsyncSession, prereg: Preregistration, today: date
) -> list[CellReplay]:
    """Every cell of ``prereg`` replayed as one pooled draft over its leagues on each
    development block, in grid order. Reads the database only."""
    blocks = development_blocks(prereg.leagues)
    replays: list[CellReplay] = []
    for index, cell in enumerate(prereg.candidate_cells()):
        draft = pooled_draft(prereg, cell, today, "Cell selection")
        per_block = []
        for block in blocks:
            replayed = await collect(
                session, draft, {league: [(block.start, block.end)] for league in block.leagues}
            )
            per_block.append((block, tuple(replayed.bets)))
        replays.append(CellReplay(index, cell, tuple(per_block)))
    return replays


async def select_cell(
    session: AsyncSession, path: Path, directory: Path, today: date
) -> GridSelection:
    """Round 1's selection over the grid file at ``path`` (in ``directory/preregistrations``).
    Writes nothing; ``write_selection`` writes the picked cell."""
    prereg = load_selectable(path, directory, today)
    grid_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    return select_from(prereg, grid_sha256, await replay_cells(session, prereg, today))


def write_selection(selection: Selection, directory: Path) -> Path:
    """Write ``selection`` to ``directory/selections/<stem>.yaml``, refusing to replace one."""
    path = directory / SELECTION_DIR / f"{selection.stem}.yaml"
    path.parent.mkdir(exist_ok=True)
    with path.open("x") as handle:
        yaml.safe_dump(selection.model_dump(mode="json"), handle, sort_keys=False)
    return path
