"""Bootstrap local + Colab : rend `import projet` possible."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = "matleniz/python_deepL"
CLONE_DIR = Path("/content") / "python_deepL"
DEPS = ["pettingzoo", "pygame", "numpy"]


def _in_colab() -> bool:
    if "google.colab" in sys.modules:
        return True
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    return Path("/content").is_dir() and not (Path.home() / "python_deepL" / "src").is_dir()


def _find_src() -> Path | None:
    here = Path.cwd().resolve()
    candidates = [
        here / "src",
        here.parent / "src",
        Path.home() / "python_deepL" / "src",
        CLONE_DIR / "src",
        Path("/content") / "drive" / "MyDrive" / "python_deepL" / "src",
    ]
    if here.name == "notebooks":
        candidates.insert(0, here.parent / "src")
    try:
        candidates.insert(0, Path(__file__).resolve().parents[1] / "src")
    except NameError:
        pass
    for p in candidates:
        if (p / "projet").is_dir():
            return p.resolve()
    return None


def _pip_install(*packages: str) -> None:
    # 1) uv (venv local)
    try:
        subprocess.check_call(
            ["uv", "pip", "install", "--python", sys.executable, "-q", *packages],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # 2) pip
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    except subprocess.CalledProcessError:
        pass

    # 3) ensurepip puis pip (souvent Colab / venv sans pip)
    subprocess.check_call(
        [sys.executable, "-m", "ensurepip", "--upgrade"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        stdout=subprocess.DEVNULL,
    )


def _github_token() -> str:
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        return tok
    try:
        from google.colab import userdata  # type: ignore

        tok = str(userdata.get("GITHUB_TOKEN")).strip()
        if tok:
            return tok
    except Exception:
        pass
    try:
        import getpass

        return getpass.getpass("GitHub PAT (repo privé, scope repo) : ").strip()
    except Exception:
        return ""


def _clone_or_pull() -> Path:
    token = _github_token()
    root = CLONE_DIR if Path("/content").is_dir() else Path("/tmp") / "python_deepL"
    if not root.exists():
        url = (
            f"https://x-access-token:{token}@github.com/{REPO}.git"
            if token
            else f"https://github.com/{REPO}.git"
        )
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", url, str(root)])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Clone GitHub impossible (repo privé ?). "
                "Définis GITHUB_TOKEN (PAT scope repo) ou un secret Colab du même nom, "
                "ou rends le repo public."
            ) from e
    else:
        try:
            subprocess.check_call(
                ["git", "-C", str(root), "pull", "--ff-only"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            pass
    src = root / "src"
    if not (src / "projet").is_dir():
        raise RuntimeError(f"Pas de src/projet dans {root} — push ton code sur GitHub.")
    return src


def _deps_ok() -> bool:
    try:
        import numpy  # noqa: F401
        import pettingzoo  # noqa: F401

        return True
    except ImportError:
        return False


def setup() -> Path:
    src = _find_src()
    if src is None:
        src = _clone_or_pull()

    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if not _deps_ok():
        _pip_install(*DEPS)

    import projet  # noqa: F401

    print(f"projet OK ← {src}")
    return src


if __name__ == "__main__":
    setup()
