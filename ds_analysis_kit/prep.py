"""Préparation : transformer un DataFrame brut en matrice prête pour un modèle.

Le point commun de toutes ces fonctions : **tout ce qui est appris l'est sur le
train, puis appliqué au test**. Une médiane, des bornes de déciles ou une
moyenne de scaler calculées sur les lignes qu'on doit prédire sont des
statistiques qui ont vu la réponse.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ._common import _is_numeric

__all__ = [
    "quantile_bins",
    "bin_series",
    "signed_log1p",
    "prepare",
    "Prepared",
]


def quantile_bins(s: pd.Series, q: int = 10) -> np.ndarray:
    """Bornes de `q` déciles, dédoublonnées. À calculer sur le train uniquement.

    C'est ton `make_bins` : on découpe par quantiles (des tranches d'effectif
    égal) plutôt que par largeur égale, pour que chaque tranche compte assez de
    lignes. Renvoie les bornes, à passer ensuite à `bin_series` sur le train
    comme sur le test.

    Compose : `numpy.nanquantile`.
    """
    edges = np.unique(np.nanquantile(np.asarray(s, dtype=float),
                                     np.linspace(0, 1, q + 1)))
    if len(edges) < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges


def bin_series(s: pd.Series, edges) -> pd.Categorical:
    """Applique à une série des bornes apprises ailleurs (via `quantile_bins`).

    Compose : `pandas.cut`.
    """
    return pd.cut(s, bins=edges, include_lowest=True)


def signed_log1p(s):
    """log1p qui accepte le négatif : signe(x) * log(1 + |x|).

    Ta transformation pour `balance`, qui descend à -8019 : un log ordinaire
    refuserait le négatif, celui-ci écrase les grandes valeurs des deux côtés
    de zéro tout en gardant le signe.

    Compose : `numpy`.
    """
    a = np.asarray(s, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


@dataclass
class Prepared:
    """Sortie de `prepare` : les matrices alignées + de quoi rejouer la recette.

    On garde `medians`, `columns` et `scaler` pour pouvoir appliquer plus tard
    exactement la même transformation à de nouvelles données.
    """
    X_train: pd.DataFrame
    X_test: pd.DataFrame | None
    columns: list
    medians: pd.Series
    scaler: object = None
    notes: list = field(default_factory=list)


def prepare(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None,
            drop=("id",), as_category=(), scale: bool = False,
            fill: bool = True) -> Prepared:
    """Trous bouchés, texte encodé, colonnes du test alignées sur le train.

    Rassemble en un appel ce que tu refais à chaque notebook :

    1. on retire les colonnes de `drop` (l'`id`) ;
    2. si `fill`, on bouche les trous — médiane du **train** pour les nombres,
       `"unknown"` pour le texte ;
    3. les colonnes listées dans `as_category` sont converties en texte *avant*
       l'encodage. C'est le correctif qui vaut le plus cher dans le notebook
       vélo : dire à `get_dummies` que `hour` est une catégorie (et non un
       nombre où 23 h vaudrait 23 fois 1 h) fait passer le -MAE de -139 à -100 ;
    4. `get_dummies` encode le texte, puis `reindex` force le test à avoir
       exactement les colonnes du train, dans le même ordre (sinon une
       catégorie absente du test décale tout) ;
    5. si `scale`, un `StandardScaler` ajusté sur le **train** met toutes les
       colonnes à la même échelle — nécessaire pour que le solveur de
       `LogisticRegression` converge quand `balance` et `campaign` n'ont pas du
       tout la même amplitude.

    Renvoie un `Prepared` (voir sa docstring).

    Compose : `pandas` (get_dummies, reindex, fillna, median) et
    `sklearn.preprocessing.StandardScaler`.
    """
    notes = []
    drop = [c for c in drop if c in X_train.columns]
    tr = X_train.drop(columns=drop)
    te = (X_test.drop(columns=[c for c in drop if c in X_test.columns])
          if X_test is not None else None)

    medians = tr.median(numeric_only=True)
    if fill:
        def _fill(f):
            if f is None:
                return None
            f = f.copy()
            f[medians.index] = f[medians.index].fillna(medians)
            for c in f.columns:
                if not _is_numeric(f[c]):
                    f[c] = f[c].astype("object").fillna("unknown")
            return f
        tr, te = _fill(tr), _fill(te)

    for c in as_category:
        if c in tr.columns:
            tr[c] = tr[c].astype(str)
            if te is not None:
                te[c] = te[c].astype(str)
    if as_category:
        notes.append(f"traitées comme catégorielles : {list(as_category)}")

    tr_enc = pd.get_dummies(tr)
    te_enc = (pd.get_dummies(te).reindex(columns=tr_enc.columns, fill_value=0)
              if te is not None else None)

    scaler = None
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(tr_enc)
        tr_enc = pd.DataFrame(scaler.transform(tr_enc), index=tr_enc.index,
                              columns=tr_enc.columns)
        if te_enc is not None:
            te_enc = pd.DataFrame(scaler.transform(te_enc), index=te_enc.index,
                                  columns=te_enc.columns)
        notes.append("colonnes standardisées (moyennes/écarts du train)")

    if te_enc is not None:
        assert list(tr_enc.columns) == list(te_enc.columns)

    print(f"[prepare] train {tr_enc.shape}"
          + (f" | test {te_enc.shape}" if te_enc is not None else "")
          + f" | {tr_enc.shape[1]} colonnes après encodage")
    for note in notes:
        print(f"  - {note}")
    return Prepared(tr_enc, te_enc, list(tr_enc.columns), medians, scaler, notes)
