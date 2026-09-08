"""Bootstrap local + Colab : rend `import projet` possible.

Usage dans un notebook (première cellule) :
    %run ../notebooks/colab_bootstrap.py
ou copie-colle le contenu de `setup()` plus bas.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO = "matleniz/python_deepL"
CLONE_DIR = Path("/content") / "python_deepL"
DEPS = ["pettingzoo", "pygame", "numpy"]


def _find_src() -> Path | None:
    here = Path.cwd().resolve()
    candidates = [
        here / "src",
        here.parent / "src",
        Path.home() / "python_deepL" / "src",
        CLONE_DIR / "src",
    ]
    # notebook dans notebooks/
    if here.name == "notebooks":
        candidates.insert(0, here.parent / "src")
    for p in candidates:
        if (p / "projet").is_dir():
            return p.resolve()
    return None


def _in_colab() -> bool:
    return Path("/content").exists() or "google.colab" in sys.modules


def _clone_or_pull() -> Path:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not CLONE_DIR.exists():
        if not token:
            try:
                import getpass

                token = getpass.getpass("GitHub PAT (repo privé) — scope repo : ").strip()
            except Exception:
                token = ""
        url = (
            f"https://{token}@github.com/{REPO}.git"
            if token
            else f"https://github.com/{REPO}.git"
        )
        subprocess.check_call(["git", "clone", "--depth", "1", url, str(CLONE_DIR)])
    else:
        subprocess.check_call(["git", "-C", str(CLONE_DIR), "pull", "--ff-only"])
    return CLONE_DIR / "src"


def setup() -> Path:
    src = _find_src()
    if src is None and _in_colab():
        src = _clone_or_pull()
    if src is None:
        raise RuntimeError(
            "src/projet introuvable. En local : ouvre le repo. "
            "Sur Colab : pousse le repo GitHub, puis relance (PAT si privé)."
        )
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *DEPS],
        stdout=subprocess.DEVNULL,
    )
    import projet  # noqa: F401

    print(f"projet OK ← {src}")
    return src


if __name__ == "__main__":
    setup()
