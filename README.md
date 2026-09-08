# python-deepL

Un `.venv` (uv), package `projet` sous `src/`.

```
src/projet/
  play.py
  minimooteur.py
  agents/
  train/
notebooks/           # Colab : cellule bootstrap en premier
tests/
checkpoints/
```

## Setup local

```bash
uv sync --extra dev
uv run pytest
uv run python -m projet.play
```

## Colab

Le runtime Colab **ne voit pas** ton dossier WSL. Dans chaque notebook :

1. Cellule bootstrap (`notebooks/colab_bootstrap.py`) → clone + `sys.path`
2. Puis `from projet.play import ...`

Repo privé : PAT GitHub (`repo`) via `GITHUB_TOKEN` ou prompt.  
**Push** ton code avant Colab, sinon le clone est vieux.
