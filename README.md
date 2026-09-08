# python-deepL

Apprentissage Python / agents multi-agents (ml-arena / PettingZoo).

## Setup local (uv)

```bash
uv sync
```

Kernel Jupyter enregistré : **Python (python-deepL)**.

Token API : copie `.env.example` → `.env` et mets `MLARENA_API_TOKEN=...`, ou colle le token dans le notebook.

## Connecter le notebook à Colab (VS Code / Cursor)

1. Extension **Google Colab** installée (`Google.colab`).
2. Ouvre `agent_baseline.ipynb`.
3. En haut à droite : **Select Kernel** → **Colab** → **Auto Connect** (ou New Colab Server).
4. Connecte-toi avec ton compte Google, choisis CPU/GPU.

Ensuite tu peux entraîner / itérer sur l’agent et soumettre via le SDK ml-arena.
