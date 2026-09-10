"""Helpers internes, partagés par les autres modules du kit.

Rien ici n'est destiné à être appelé directement dans un notebook : ce sont les
petites briques que `explore`, `prep` et `evaluate` réutilisent (conversion en
Series, test « est-ce numérique », devinette du type de tâche, création d'axe,
palette de couleurs).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Palette utilisée par tous les graphes du kit, pour un rendu cohérent.
PALETTE = ["#2f6f9f", "#c1553b", "#5a9367", "#b8860b", "#7a5c9e"]


def _series(y, name: str = "target") -> pd.Series:
    """Accepte une Series, un DataFrame à une colonne, un array ou une liste.

    Sert à ce que toutes les fonctions du kit acceptent aussi bien
    `y_train` (Series) que `y_train[["prediction"]]` (DataFrame) sans râler.
    """
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, pd.DataFrame):
        cols = [c for c in y.columns if c != "id"]
        if len(cols) != 1:
            raise ValueError(f"y a plusieurs colonnes : {list(y.columns)}")
        return y[cols[0]]
    return pd.Series(np.asarray(y), name=name)


def _is_numeric(s: pd.Series) -> bool:
    """True pour une colonne numérique, mais pas pour un booléen (traité en catégorie)."""
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def infer_task(y) -> str:
    """Devine 'classification' ou 'regression' à partir de la cible.

    Heuristique simple : du texte ou <= 2 valeurs -> classification ; des
    entiers avec peu de valeurs distinctes -> classification ; sinon régression.
    Toutes les fonctions qui l'utilisent laissent passer un `task=` explicite
    pour ne pas dépendre de la devinette.
    """
    y = _series(y).dropna()
    if not _is_numeric(y):
        return "classification"
    if y.nunique() <= 2:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return "classification"
    return "regression"


def _ax(ax, figsize):
    """Renvoie l'axe fourni, ou en crée un neuf de la taille demandée."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax
