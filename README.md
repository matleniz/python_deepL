# python-deepL

Un `.venv` (uv), package `projet` sous `src/`.

```
src/projet/
  play.py              # éval PettingZoo
  agents/              # monte_carlo, nn_pytorch, rl_agent (à coder)
  train/
notebooks/
tests/
checkpoints/
```

```bash
uv sync --extra dev
uv run python -m projet.play
uv run pytest
```

Colab : `notebooks/agent_baseline.ipynb` → Kernel → Colab.
