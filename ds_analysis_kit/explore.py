"""Exploration : auditer un jeu de données et regarder la cible.

Ces fonctions remplacent le copier-coller de l'étape « 2. Read it » et
« 3. Look at it » : un tableau de synthèse par colonne, la distribution de la
cible avec sa baseline, la carte des trous, la heatmap de corrélation, et le
taux de la cible par catégorie ou par tranche.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._common import PALETTE, _ax, _is_numeric, _series, infer_task

__all__ = [
    "overview",
    "target_report",
    "plot_missing",
    "plot_correlation",
    "plot_target_rate",
]


def overview(X: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par colonne : type, trous, cardinalité, et stats des numériques.

    Remplace le trio `X.dtypes` / `X.describe()` / `X.isna().sum()` par un seul
    tableau lisible où chaque colonne se lit d'un coup d'œil. La colonne `flags`
    signale les cas à regarder (trous, colonne constante, identifiant probable).

    Compose : `pandas` (dtypes, isna, nunique, describe).
    """
    n = len(X)
    rows = []
    for col in X.columns:
        s = X[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if _is_numeric(s):
            row.update(kind="num", min=s.min(), median=s.median(), max=s.max(),
                       top=None, top_freq=None)
        else:
            vc = s.value_counts(dropna=True)
            row.update(kind="cat", min=None, median=None, max=None,
                       top=vc.index[0] if len(vc) else None,
                       top_freq=int(vc.iloc[0]) if len(vc) else None)
        rows.append(row)

    out = pd.DataFrame(rows).set_index("column")

    flags = []
    for col, r in out.iterrows():
        f = []
        if r["pct_missing"] > 0:
            f.append("trous")
        if r["n_unique"] <= 1:
            f.append("constante")
        if r["kind"] == "num" and r["n_unique"] == n:
            f.append("identifiant?")
        flags.append(" ".join(f))
    out["flags"] = flags
    return out


def target_report(y, task: str | None = None, plot: bool = True) -> dict:
    """Distribution de la cible et la baseline contre laquelle se juger.

    Classification : la part de chaque classe et l'accuracy qu'on obtient en
    prédisant toujours la classe majoritaire — le nombre que ton modèle doit
    battre (étape 3a du notebook bank).
    Régression : moyenne / médiane / écart-type et la MAE qu'on obtient en
    prédisant la moyenne partout (la baseline du notebook vélo). Prévient aussi
    quand la cible est bornée à 0, car un modèle linéaire prédira du négatif.

    Compose : `pandas` (value_counts, describe) et `sklearn.metrics.mean_absolute_error`.
    """
    y = _series(y)
    task = task or infer_task(y)
    info = {"task": task, "n": len(y), "n_missing": int(y.isna().sum())}

    if task == "classification":
        vc = y.value_counts(normalize=True).sort_index()
        info["class_balance"] = vc.to_dict()
        majority = float(vc.max())
        info["baseline_accuracy_majority"] = majority
        print(f"[cible] {len(y)} lignes, {y.nunique()} classes")
        for k, v in info["class_balance"].items():
            print(f"  classe {k!r:>8} : {100 * v:5.2f} %")
        print(f"  accuracy en prédisant toujours la majorité : {majority:.4f}"
              "   <- le nombre à battre")
        if plot:
            fig, ax = plt.subplots(figsize=(5, 3))
            y.value_counts().sort_index().plot.bar(ax=ax, color=PALETTE[0])
            ax.set_title("Distribution de la cible")
            plt.tight_layout()
    else:
        from sklearn.metrics import mean_absolute_error
        info.update(mean=float(y.mean()), median=float(y.median()),
                    std=float(y.std()), min=float(y.min()), max=float(y.max()))
        info["baseline_mae_mean"] = float(
            mean_absolute_error(y, np.full(len(y), y.mean())))
        print(f"[cible] moyenne {info['mean']:.2f} | médiane {info['median']:.2f} | "
              f"écart-type {info['std']:.2f}")
        print(f"  MAE en prédisant la moyenne : {info['baseline_mae_mean']:.2f}"
              "   <- la baseline à battre")
        if info["min"] >= 0:
            print("  cible bornée à 0 : un modèle non borné prédira du négatif, "
                  "pense à np.clip(pred, 0, None)")
        if plot:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.hist(y.dropna(), bins=60, color=PALETTE[0])
            ax.set_title(f"Distribution de la cible — moyenne {info['mean']:.0f}, "
                         f"médiane {info['median']:.0f}")
            plt.tight_layout()
    return info


def plot_missing(X: pd.DataFrame, ax=None):
    """Carte des valeurs manquantes : une ligne par colonne, sombre = manquant.

    Sert à distinguer des trous *en blocs* (un capteur en panne quelques heures
    d'affilée, qu'on comble mieux par interpolation) de trous *dispersés* (une
    mesure perdue ici et là, qu'on comble par la médiane). C'est la heatmap
    `X.isna().T` du notebook vélo.

    Compose : `matplotlib.imshow`.
    """
    na = X.isna()
    cols = na.columns[na.any()]
    if len(cols) == 0:
        print("aucune valeur manquante")
        return None
    ax = _ax(ax, (11, 0.4 * len(cols) + 1.5))
    cmap = plt.matplotlib.colors.ListedColormap(["#eef3f8", "#c1553b"])
    ax.imshow(na[cols].T.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xticks([])
    ax.set_title("Valeurs manquantes (sombre = manquant) — blocs vs dispersé")
    plt.tight_layout()
    return ax


def plot_correlation(df: pd.DataFrame, ax=None):
    """Heatmap de corrélation des colonnes numériques, pour repérer les redondances.

    Deux colonnes corrélées à ~1 sont la même mesure deux fois (temp et
    feel_temp à 0.99 dans le vélo) : garder les deux rend leurs coefficients
    individuels ininterprétables. Attention, la corrélation ne voit que la part
    *linéaire* d'une relation — c'est le piège de `hour` dans le notebook vélo.

    Compose : `pandas.DataFrame.corr` + `seaborn.heatmap`.
    """
    import seaborn as sns

    num = df.select_dtypes("number")
    ax = _ax(ax, (0.7 * num.shape[1] + 2, 0.6 * num.shape[1] + 1.5))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                ax=ax, annot_kws={"size": 8})
    ax.set_title("Corrélation des colonnes numériques")
    plt.tight_layout()
    return ax


def plot_target_rate(df: pd.DataFrame, col: str, target: str, q: int = 10,
                     max_levels: int = 30, ax=None, show_counts: bool = True):
    """Moyenne de la cible par catégorie ou par tranche de `col`, avec l'effectif.

    Généralise tous tes `sns.barplot(..., estimator="mean")` et tes `pd.cut` +
    moyenne. Une colonne numérique à beaucoup de valeurs est découpée en `q`
    déciles ; une catégorielle est prise telle quelle. La ligne pointillée est
    la moyenne globale, et la courbe grise donne l'effectif de chaque barre —
    de quoi ne pas surinterpréter une catégorie à trois observations (la leçon
    `heavy_rain` du notebook vélo).

    Renvoie le tableau (moyenne, effectif) et trace le graphe.

    Compose : `pandas` (groupby, cut, qcut) + `matplotlib`.
    """
    d = df[[col, target]].dropna(subset=[target])
    s, y = d[col], d[target]

    if _is_numeric(s) and s.nunique() > max_levels:
        groups = pd.qcut(s.fillna(s.median()), q, duplicates="drop")
        xlabel = f"{col} (déciles)"
    else:
        groups = s.astype("object").fillna("__manquant__")
        xlabel = col

    g = y.groupby(groups, observed=True)
    stats = pd.DataFrame({"mean": g.mean(), "count": g.count()}).dropna(subset=["mean"])

    ax = _ax(ax, (max(6, 0.5 * len(stats) + 2), 4))
    x = np.arange(len(stats))
    ax.bar(x, stats["mean"], color=PALETTE[0])
    ax.axhline(y.mean(), color=PALETTE[1], ls="--", lw=1.2,
               label=f"moyenne globale {y.mean():.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in stats.index], rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"moyenne de {target}")
    ax.set_title(f"{target} selon {col}")
    ax.legend(fontsize=8, loc="best")

    if show_counts:
        ax2 = ax.twinx()
        ax2.plot(x, stats["count"], color="#999999", marker=".", lw=1, ls=":")
        ax2.set_ylabel("effectif", color="#999999")
        ax2.tick_params(axis="y", labelcolor="#999999")
        ax2.grid(False)
    plt.tight_layout()
    return stats
