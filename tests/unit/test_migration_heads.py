"""The migration history has exactly one head."""

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestMigrationHeads:
    def test_single_head(self) -> None:
        config = Config(str(REPO_ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
        heads = ScriptDirectory.from_config(config).get_heads()
        assert len(heads) == 1, f"alembic has {len(heads)} heads: {heads}; chain them"
