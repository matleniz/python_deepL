# python-deepL

Workspace multi-projets. Chaque dossier a son propre `pyproject.toml`, `.venv` et `uv.lock`.

```
python_deepL/
  puissance4/     # agents Connect Four (ml-arena)
  seance2/        # notebooks AIE séance 2
  <autre>/        # prochain projet : uv init + deps à part
```

## Nouveau projet

```bash
mkdir mon-projet && cd mon-projet
uv init
uv sync
```

## Puissance 4

```bash
cd puissance4
uv sync --extra dev
uv run pytest
```

## Séance 2

```bash
cd seance2
uv sync   # quand tu seras prêt
```
