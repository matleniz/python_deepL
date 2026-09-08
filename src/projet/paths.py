from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO_ROOT / "checkpoints"
NOTEBOOKS = REPO_ROOT / "notebooks"


def ensure_checkpoints() -> Path:
    CHECKPOINTS.mkdir(exist_ok=True)
    return CHECKPOINTS
