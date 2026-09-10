"""ds_analysis_kit — outils d'exploration, de diagnostic et d'évaluation.

Regroupe ce qui se répète d'un notebook AIE à l'autre (audit des colonnes,
plots de taux par bin, encodage aligné train/test, sweep de seuil, submission)
et ajoute des mesures qui voient au-delà du linéaire : information mutuelle,
rapport de corrélation, V de Cramér, score prédictif par colonne, gain de
binning, détection d'interactions, importance par permutation groupée, drift
train/test.

Usage :

    import sys; sys.path.append("..")      # depuis seanceN/
    import ds_analysis_kit as dsk

    dsk.overview(X_train, y_train)
    dsk.plot_target_rate(df, "age", "prediction")
    dsk.single_feature_score(X_train, y_train)

Dépendances : pandas, numpy, matplotlib, seaborn, scikit-learn (scipy vient
avec sklearn).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    # audit
    "overview", "target_report", "plot_missing", "sentinel_report", "quick_look",
    # associations
    "cramers_v", "correlation_ratio", "association_matrix",
    "plot_association_matrix", "redundant_pairs", "agreement",
    "mutual_info_ranking", "single_feature_score", "linearity_gain",
    # cible vs feature
    "plot_target_rate", "plot_target_rate_grid", "plot_interaction",
    "top_interactions",
    # importance
    "permutation_importance_grouped", "plot_importance", "plot_coefficients",
    "group_encoded_columns",
    # drift
    "psi", "drift_report", "adversarial_validation",
    # préparation
    "quantile_bins", "bin_series", "signed_log1p", "prepare",
    # évaluation
    "regression_report", "classification_report_plus", "threshold_sweep",
    "plot_residuals", "plot_calibration",
    # sortie
    "write_submission",
]

PALETTE = ["#2f6f9f", "#c1553b", "#5a9367", "#b8860b", "#7a5c9e"]


# ---------------------------------------------------------------------------
# helpers internes
# ---------------------------------------------------------------------------

def _series(y, name="target") -> pd.Series:
    """Accepte Series, DataFrame à une colonne, array ou liste."""
    if isinstance(y, pd.Series):
        return y
    if isinstance(y, pd.DataFrame):
        cols = [c for c in y.columns if c != "id"]
        if len(cols) != 1:
            raise ValueError(f"y a plusieurs colonnes : {list(y.columns)}")
        return y[cols[0]]
    return pd.Series(np.asarray(y), name=name)


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def infer_task(y) -> str:
    """'classification' ou 'regression', deviné à partir de la cible."""
    y = _series(y).dropna()
    if not _is_numeric(y):
        return "classification"
    if y.nunique() <= 2:
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 20:
        return "classification"
    return "regression"


def _wilson(k: float, n: float, z: float = 1.96):
    """Intervalle de confiance de Wilson pour une proportion (robuste aux petits n)."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def _ax(ax, figsize):
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


# ---------------------------------------------------------------------------
# 1. Audit
# ---------------------------------------------------------------------------

def overview(X: pd.DataFrame, y=None, sort_by: str | None = None) -> pd.DataFrame:
    """Une ligne par colonne : type, trous, cardinalité, échelle, asymétrie.

    Remplace le trio `dtypes` / `describe` / `isna().sum()` par un seul tableau
    lisible. Si `y` est fourni, ajoute la corrélation (Spearman) avec la cible
    pour les colonnes numériques.
    """
    rows = []
    n = len(X)
    for col in X.columns:
        s = X[col]
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2),
            "n_unique": int(s.nunique(dropna=True)),
            "pct_unique": round(100 * s.nunique(dropna=True) / max(n, 1), 2),
        }
        if _is_numeric(s):
            row.update(
                kind="num",
                min=s.min(), median=s.median(), max=s.max(),
                skew=round(float(s.skew()), 2) if s.nunique() > 1 else 0.0,
                top=None, top_freq=None,
            )
        else:
            vc = s.value_counts(dropna=True)
            row.update(
                kind="cat",
                min=None, median=None, max=None, skew=None,
                top=vc.index[0] if len(vc) else None,
                top_freq=round(100 * vc.iloc[0] / max(n, 1), 1) if len(vc) else None,
            )
        rows.append(row)

    out = pd.DataFrame(rows).set_index("column")

    if y is not None:
        y = _series(y)
        if _is_numeric(y):
            corr = {}
            for col in X.columns:
                if _is_numeric(X[col]) and X[col].nunique() > 1:
                    corr[col] = X[col].corr(y, method="spearman")
            out["spearman_y"] = pd.Series(corr).round(3)

    # signaux qui méritent un coup d'œil
    flags = []
    for col, r in out.iterrows():
        f = []
        if r["pct_missing"] > 0:
            f.append("trous")
        if r["n_unique"] <= 1:
            f.append("constante")
        if r["kind"] == "cat" and r["n_unique"] > 50:
            f.append("haute-cardinalité")
        if r["kind"] == "num" and r["skew"] is not None and abs(r["skew"]) > 2:
            f.append("très-asymétrique")
        if r["pct_unique"] > 99 and r["kind"] == "num":
            f.append("identifiant?")
        flags.append(" ".join(f))
    out["flags"] = flags

    if sort_by:
        out = out.sort_values(sort_by, ascending=False)
    return out


def target_report(y, task: str | None = None, plot: bool = True) -> dict:
    """Distribution de la cible + les baselines contre lesquelles se juger.

    Classification : part de chaque classe, accuracy du "toujours la majorité",
    et le F1 qu'obtient un classifieur qui répond 1 partout (le vrai plancher
    quand on est classé au F1 positif).
    Régression : moyenne/médiane, MAE et R2 du "prédis la moyenne".
    """
    y = _series(y)
    task = task or infer_task(y)
    info = {"task": task, "n": len(y), "n_missing": int(y.isna().sum())}

    if task == "classification":
        vc = y.value_counts(normalize=True).sort_index()
        info["class_balance"] = vc.to_dict()
        majority = vc.max()
        info["baseline_accuracy_majority"] = float(majority)
        if y.nunique() == 2:
            pos = float(y.mean()) if _is_numeric(y) else float(vc.iloc[-1])
            info["positive_rate"] = pos
            # tout prédire à 1 : precision = pos, recall = 1
            info["baseline_f1_all_positive"] = 2 * pos / (1 + pos)
            info["imbalance_ratio"] = round((1 - pos) / pos, 1) if pos else np.inf
        print(f"[cible] {len(y)} lignes, {y.nunique()} classes")
        for k, v in info["class_balance"].items():
            print(f"  classe {k!r:>8} : {100*v:5.2f} %")
        print(f"  accuracy en prédisant toujours la majorité : {majority:.4f}")
        if "baseline_f1_all_positive" in info:
            print(f"  F1 en prédisant toujours 1               : "
                  f"{info['baseline_f1_all_positive']:.4f}")
            print(f"  déséquilibre                             : "
                  f"1 positif pour {info['imbalance_ratio']} négatifs")
        if plot:
            fig, ax = plt.subplots(figsize=(5, 3))
            y.value_counts().sort_index().plot.bar(ax=ax, color=PALETTE[0])
            ax.set_title("Distribution de la cible")
            plt.tight_layout()
    else:
        from sklearn.metrics import mean_absolute_error
        info.update(
            mean=float(y.mean()), median=float(y.median()),
            std=float(y.std()), min=float(y.min()), max=float(y.max()),
            skew=float(y.skew()),
        )
        info["baseline_mae_mean"] = float(mean_absolute_error(y, np.full(len(y), y.mean())))
        info["baseline_mae_median"] = float(mean_absolute_error(y, np.full(len(y), y.median())))
        print(f"[cible] moyenne {info['mean']:.2f} | médiane {info['median']:.2f} | "
              f"écart-type {info['std']:.2f} | asymétrie {info['skew']:.2f}")
        print(f"  MAE en prédisant la moyenne  : {info['baseline_mae_mean']:.2f}")
        print(f"  MAE en prédisant la médiane  : {info['baseline_mae_median']:.2f}"
              "   <- la vraie baseline si le classement est au MAE")
        if info["min"] >= 0:
            print("  cible bornée à 0 : un modèle non borné prédira du négatif, "
                  "pense à np.clip(pred, 0, None)")
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
            axes[0].hist(y.dropna(), bins=60, color=PALETTE[0])
            axes[0].set_title("Distribution de la cible")
            axes[1].hist(np.log1p(y.dropna().clip(lower=0)), bins=60, color=PALETTE[2])
            axes[1].set_title("log1p(cible) — droite = un log aiderait")
            plt.tight_layout()
    return info


def plot_missing(X: pd.DataFrame, ax=None):
    """Carte des trous : une colonne par ligne, sombre = manquant.

    Sert à distinguer des trous *en blocs* (une panne de capteur, à combler par
    interpolation temporelle) de trous *dispersés* (à combler par la médiane).
    """
    na = X.isna()
    cols = na.columns[na.any()]
    if len(cols) == 0:
        print("aucune valeur manquante")
        return None
    ax = _ax(ax, (11, 0.4 * len(cols) + 1.5))
    ax.imshow(na[cols].T.to_numpy(), aspect="auto", interpolation="nearest",
              cmap=plt.matplotlib.colors.ListedColormap(["#eef3f8", "#c1553b"]))
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_xticks([])
    ax.set_title("Valeurs manquantes (sombre = manquant) — blocs vs dispersé")
    plt.tight_layout()
    return ax


def sentinel_report(X: pd.DataFrame, min_share: float = 0.08) -> pd.DataFrame:
    """Repère les colonnes numériques où une valeur unique fait office de drapeau.

    `pdays = -1` ne veut pas dire « il y a moins un jour », ça veut dire
    « jamais contacté ». La colonne mélange une catégorie et une mesure, et le
    modèle, lui, lit -1 comme un nombre et l'aligne avec les 1, 2, 3 jours.

    Le critère : une valeur qui se répète beaucoup *et* qui se trouve à un
    extrême de la distribution, détachée du reste. Les suspects classiques
    sont -1, 0, 999, -999.
    """
    rows = []
    for c in X.columns:
        s = X[c].dropna()
        if not _is_numeric(s) or s.nunique() < 3:
            continue
        vc = s.value_counts()
        val, cnt = vc.index[0], vc.iloc[0]
        share = cnt / len(s)
        if share < min_share:
            continue
        rest = s[s != val]
        if rest.empty:
            continue
        at_edge = val <= rest.min() or val >= rest.max()
        # détaché : l'écart au reste dépasse un écart-type du reste
        gap = min(abs(val - rest.min()), abs(val - rest.max()))
        detached = gap > rest.std() if rest.std() > 0 else False
        if at_edge and (detached or val in (-1, 0, -999, 999, 9999)):
            rows.append({
                "column": c, "valeur": val, "part": round(float(share), 4),
                "reste_min": rest.min(), "reste_max": rest.max(),
                "suggestion": f"{c}_is_{int(val) if float(val).is_integer() else val}"
                              f" = ({c} == {val}), puis remplacer {val} par NaN",
            })
    return pd.DataFrame(rows, columns=["column", "valeur", "part", "reste_min",
                                       "reste_max", "suggestion"])


def _show(obj):
    try:
        from IPython.display import display
        display(obj)
    except Exception:
        print(obj)


def quick_look(X: pd.DataFrame, y=None, task: str | None = None):
    """L'audit complet en un appel : cible, colonnes, trous, sentinelles, redondances."""
    if y is not None:
        target_report(y, task=task)
        print()
    print("[colonnes]")
    _show(overview(X, y))
    print()
    plot_missing(X)

    sent = sentinel_report(X)
    if len(sent):
        print("\n[valeurs-sentinelles] une valeur répétée qui est un drapeau, "
              "pas une quantité")
        _show(sent)

    dup = redundant_pairs(X, threshold=0.95)
    if len(dup):
        print("\n[paires quasi-redondantes] (association >= 0.95)")
        _show(dup)
    return None


# ---------------------------------------------------------------------------
# 2. Associations — y compris non linéaires
# ---------------------------------------------------------------------------

def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association entre deux colonnes catégorielles, dans [0, 1].

    Version corrigée du biais (Bergsma) : 0 = indépendance, 1 = l'une
    détermine l'autre. C'est le pendant de la corrélation pour du texte.
    """
    tab = pd.crosstab(a, b)
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    from scipy.stats import chi2_contingency
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Rapport de corrélation eta : combien une catégorie explique un nombre.

    eta^2 = variance inter-groupes / variance totale. Contrairement à la
    corrélation de Pearson, ça marche pour une colonne texte, et ça ne suppose
    aucune forme (ni linéaire ni même ordonnée).
    """
    df = pd.DataFrame({"c": categories, "v": values}).dropna()
    if df.empty or df["c"].nunique() < 2 or df["v"].var() == 0:
        return 0.0
    grand = df["v"].mean()
    g = df.groupby("c", observed=True)["v"]
    num = (g.count() * (g.mean() - grand) ** 2).sum()
    den = ((df["v"] - grand) ** 2).sum()
    return float(np.sqrt(num / den)) if den > 0 else 0.0


def association_matrix(df: pd.DataFrame, categorical=None, max_levels: int = 30,
                       numeric_method: str = "spearman", nonlinear: bool = False,
                       q: int = 10):
    """Matrice d'association tous types confondus, valeurs dans [0, 1].

    - num / num  : |Spearman| (rang, donc voit le monotone non linéaire)
    - cat / num  : eta (rapport de corrélation)
    - cat / cat  : V de Cramér

    Une seule matrice comparable, là où `df.corr()` ignore purement et
    simplement les neuf colonnes texte du dataset bank.

    `nonlinear=True` découpe les colonnes numériques en déciles et mesure tout
    par eta. Coût : on perd le signe et un peu de finesse. Gain : la matrice
    voit les relations en U. Sur le dataset vélo, `hour` / `count` passe de
    0.24 en Spearman à ~0.6 — la valeur qui correspond à ce que montre le plot.

    Retourne (matrice_de_force, matrice_du_type_de_mesure).
    """
    cols = list(df.columns)
    if categorical is None:
        categorical = [c for c in cols
                       if not _is_numeric(df[c]) or df[c].nunique() <= 2]
    categorical = set(categorical)

    # une catégorielle à 5000 niveaux ne dit rien et coûte cher
    cols = [c for c in cols
            if c not in categorical or df[c].nunique() <= max_levels]

    binned = {}
    if nonlinear:
        for c in cols:
            if c not in categorical and df[c].nunique() > q:
                s = df[c]
                binned[c] = bin_series(s.fillna(s.median()), quantile_bins(s, q))

    m = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)
    kind = pd.DataFrame("", index=cols, columns=cols, dtype=object)
    for a, b in combinations(cols, 2):
        ca, cb = a in categorical, b in categorical
        try:
            if ca and cb:
                v, k = cramers_v(df[a], df[b]), "cramer"
            elif ca and not cb:
                v, k = correlation_ratio(df[a], df[b]), "eta"
            elif cb and not ca:
                v, k = correlation_ratio(df[b], df[a]), "eta"
            elif nonlinear:
                # symétrisé : chacun explique l'autre, on garde le plus fort
                v = max(correlation_ratio(binned.get(a, df[a]), df[b]),
                        correlation_ratio(binned.get(b, df[b]), df[a]))
                k = "eta_binne"
            else:
                v = abs(df[a].corr(df[b], method=numeric_method))
                k = numeric_method
        except Exception:
            v, k = np.nan, "erreur"
        m.loc[a, b] = m.loc[b, a] = v
        kind.loc[a, b] = kind.loc[b, a] = k
    return m, kind


def plot_association_matrix(df: pd.DataFrame, cluster: bool = True,
                            figsize=(9, 7), annot: bool = True, **kwargs):
    """Heatmap de `association_matrix`, réordonnée par clustering hiérarchique.

    Le clustering met côte à côte les colonnes qui disent la même chose : les
    blocs sombres sur la diagonale sont des groupes redondants (temp /
    feel_temp à 0.99, par exemple).
    """
    import seaborn as sns
    m, _ = association_matrix(df, **kwargs)
    order = list(m.columns)
    if cluster and len(order) > 2:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        d = 1 - m.fillna(0).to_numpy()
        np.fill_diagonal(d, 0)
        d = (d + d.T) / 2
        link = linkage(squareform(d, checks=False), method="average")
        order = [m.columns[i] for i in leaves_list(link)]
    m = m.loc[order, order]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(m, annot=annot, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
                ax=ax, annot_kws={"size": 7}, square=True,
                cbar_kws={"label": "force d'association"})
    ax.set_title("Associations (Spearman / eta / Cramér), regroupées")
    plt.tight_layout()
    return m


def redundant_pairs(df: pd.DataFrame, threshold: float = 0.9, **kwargs) -> pd.DataFrame:
    """Les paires de colonnes qui se répètent l'une l'autre, triées.

    Sert avant d'ajouter une colonne à la main : si `pdays == -1` est déjà dans
    la matrice sous le nom `poutcome_unknown`, la fabriquer coûte un
    coefficient et ne rapporte rien.
    """
    m, kind = association_matrix(df, **kwargs)
    rows = []
    for a, b in combinations(m.columns, 2):
        v = m.loc[a, b]
        if pd.notna(v) and v >= threshold:
            rows.append({"a": a, "b": b, "association": round(float(v), 4),
                         "mesure": kind.loc[a, b]})
    out = pd.DataFrame(rows, columns=["a", "b", "association", "mesure"])
    return out.sort_values("association", ascending=False).reset_index(drop=True)


def agreement(a: pd.Series, b: pd.Series, names=("a", "b")) -> pd.DataFrame:
    """Compare deux indicateurs booléens ligne à ligne et compte les désaccords.

    Pour vérifier une intuition du type « pdays == -1 c'est la même chose que
    poutcome == 'unknown' » avant de créer la colonne.
    """
    a = a.astype(bool)
    b = b.astype(bool)
    tab = pd.crosstab(a, b, rownames=[names[0]], colnames=[names[1]])
    same = int((a == b).sum())
    print(f"accord : {same}/{len(a)} lignes ({100*same/len(a):.3f} %) — "
          f"{len(a)-same} désaccords")
    return tab


def mutual_info_ranking(X: pd.DataFrame, y, task: str | None = None,
                        random_state: int = 0, plot: bool = True) -> pd.DataFrame:
    """Information mutuelle entre chaque colonne et la cible.

    L'IM mesure *toute* forme de dépendance, pas seulement linéaire. C'est la
    réponse au piège du notebook vélo : `hour` corrèle à 0.40 avec `count`
    alors que c'est la colonne la plus informative du dataset, parce que sa
    courbe monte, descend, remonte et redescend.

    La colonne `equiv_corr` traduit l'IM en « la corrélation qu'aurait une
    relation gaussienne aussi informative », pour la comparer à un Pearson.
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    y = _series(y)
    task = task or infer_task(y)

    Z = pd.DataFrame(index=X.index)
    discrete = []
    for c in X.columns:
        s = X[c]
        if _is_numeric(s):
            Z[c] = s.fillna(s.median())
            discrete.append(False)
        else:
            Z[c] = pd.factorize(s.astype("object").fillna("__nan__"))[0]
            discrete.append(True)

    mask = y.notna()
    fn = mutual_info_classif if task == "classification" else mutual_info_regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = fn(Z[mask], y[mask], discrete_features=np.array(discrete),
                random_state=random_state)

    out = pd.DataFrame({"feature": X.columns, "mutual_info": mi})
    out["equiv_corr"] = np.sqrt(np.clip(1 - np.exp(-2 * out["mutual_info"]), 0, 1))
    if task == "classification":
        p = y[mask].value_counts(normalize=True)
        entropy = float(-(p * np.log(p)).sum())
        out["pct_entropie_cible"] = (100 * out["mutual_info"] / entropy).round(1)
    out = out.sort_values("mutual_info", ascending=False).reset_index(drop=True)

    if plot:
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(out) + 1))
        ax.barh(out["feature"][::-1], out["mutual_info"][::-1], color=PALETTE[0])
        ax.set_xlabel("information mutuelle (nats)")
        ax.set_title("Information mutuelle avec la cible (voit le non-linéaire)")
        plt.tight_layout()
    return out


def single_feature_score(X: pd.DataFrame, y, task: str | None = None,
                         cv: int = 4, max_depth: int = 4,
                         random_state: int = 0, plot: bool = True) -> pd.DataFrame:
    """Ce que vaut chaque colonne *seule*, mesurée hors échantillon.

    Pour chaque colonne : un petit arbre de décision entraîné dessus et sur
    rien d'autre, évalué en validation croisée, puis normalisé contre le modèle
    bête (médiane / classe majoritaire). Score dans [0, 1] :
    0 = n'apporte rien, 1 = prédit parfaitement à elle seule.

    Deux usages :
    - un classement d'importance *avant* d'avoir un modèle, qui ne suppose
      aucune forme de relation ;
    - un détecteur de fuite. Une colonne seule qui monte très haut est
      suspecte : `duration` dans le dataset bank en est l'exemple, on ne
      connaît la durée de l'appel qu'une fois l'appel passé.

    En classification binaire la métrique est l'aire sous la courbe
    précision-rappel, pas le F1 : sur une cible à 12 % de positifs, un arbre
    utile a le même F1 que la classe majoritaire, et le classement serait plat.
    """
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.metrics import (average_precision_score, f1_score,
                                 mean_absolute_error)
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    y = _series(y)
    task = task or infer_task(y)
    mask = y.notna()
    X, y = X[mask], y[mask]
    binary = task == "classification" and y.nunique() == 2

    predict_method = "predict"
    if task == "classification":
        splitter = StratifiedKFold(cv, shuffle=True, random_state=random_state)
        model = DecisionTreeClassifier(max_depth=max_depth,
                                       min_samples_leaf=20,
                                       random_state=random_state)
        dummy = DummyClassifier(strategy="most_frequent")
        higher_better = True
        if binary:
            predict_method = "predict_proba"
            pos = sorted(y.unique())[-1]
            metric = lambda t, p: average_precision_score((t == pos).astype(int), p)
            base = float((y == pos).mean())   # AP du hasard = taux de positifs
        else:
            metric = lambda t, p: f1_score(t, p, average="weighted")
    else:
        splitter = KFold(cv, shuffle=True, random_state=random_state)
        model = DecisionTreeRegressor(max_depth=max_depth,
                                      min_samples_leaf=20,
                                      random_state=random_state)
        dummy = DummyRegressor(strategy="median")
        metric = mean_absolute_error
        higher_better = False

    if not binary:
        base_pred = cross_val_predict(dummy, np.zeros((len(y), 1)), y, cv=splitter)
        base = metric(y, base_pred)

    rows = []
    for c in X.columns:
        s = X[c]
        if s.nunique(dropna=True) <= 1:
            rows.append({"feature": c, "score": 0.0, "metric": np.nan,
                         "note": "constante"})
            continue
        if _is_numeric(s):
            Z = s.to_frame()
        elif s.nunique() <= 50:
            Z = pd.get_dummies(s.astype("object"), dummy_na=True)
            # dummy_na crée une colonne littéralement nommée NaN, que sklearn refuse
            Z.columns = [f"{c}={v!s}" for v in Z.columns]
        else:
            Z = pd.factorize(s.astype("object").fillna("__nan__"))[0].reshape(-1, 1)
            Z = pd.DataFrame(Z, index=s.index, columns=[c])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred = cross_val_predict(model, Z, y, cv=splitter,
                                         method=predict_method)
            if predict_method == "predict_proba":
                pred = pred[:, 1]
            score = metric(y, pred)
        except Exception as exc:  # colonne pathologique : on n'arrête pas tout
            rows.append({"feature": c, "score": np.nan, "metric": np.nan,
                         "note": type(exc).__name__})
            continue

        if higher_better:
            norm = (score - base) / (1 - base) if base < 1 else 0.0
        else:
            norm = 1 - score / base if base > 0 else 0.0
        rows.append({"feature": c, "score": max(0.0, float(norm)),
                     "metric": float(score), "note": ""})

    out = (pd.DataFrame(rows).sort_values("score", ascending=False)
           .reset_index(drop=True))
    out.attrs["baseline_metric"] = float(base)
    out.attrs["metric_name"] = ("average_precision" if binary else
                                "f1_weighted" if task == "classification" else "mae")

    if plot:
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(out) + 1))
        colors = [PALETTE[1] if v > 0.5 else PALETTE[0] for v in out["score"][::-1]]
        ax.barh(out["feature"][::-1], out["score"][::-1], color=colors)
        ax.set_xlabel("score prédictif seul (0 = rien, 1 = parfait)")
        ax.set_title(f"Pouvoir prédictif colonne par colonne "
                     f"(arbre, {cv}-fold, baseline {out.attrs['metric_name']}"
                     f"={base:.3f})\nen rouge : > 0.5, à regarder de près "
                     f"(fuite ?)")
        plt.tight_layout()
    return out


def linearity_gain(X: pd.DataFrame, y, q: int = 10, cv: int = 4,
                   task: str | None = None, random_state: int = 0,
                   plot: bool = True) -> pd.DataFrame:
    """Combien chaque colonne numérique gagnerait à être découpée en bins.

    Compare, hors échantillon, deux modèles à une variable : un coefficient
    unique (linéaire / logistique) contre la moyenne de la cible par décile.
    Le `gain` est la baisse d'erreur du second sur le premier.

    C'est la réponse chiffrée à la question de l'étape 7 : « un seul
    coefficient peut-il dire ça ? ». Gain élevé = non, découpe la colonne.
    """
    from sklearn.model_selection import KFold

    y = _series(y)
    task = task or infer_task(y)
    mask = y.notna()
    X, y = X[mask], y[mask].astype(float)
    num_cols = [c for c in X.columns if _is_numeric(X[c]) and X[c].nunique() > 2]
    kf = KFold(cv, shuffle=True, random_state=random_state)

    rows = []
    for c in num_cols:
        s = X[c].astype(float)
        s = s.fillna(s.median())
        err_lin, err_bin = [], []
        for tr, va in kf.split(s):
            str_, sva = s.iloc[tr], s.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]

            # 1 coefficient
            if str_.std() > 0:
                beta = np.polyfit(str_, ytr, 1)
                lin = np.polyval(beta, sva)
            else:
                lin = np.full(len(sva), ytr.mean())
            if task == "classification":
                lin = np.clip(lin, 0, 1)

            # 1 nombre libre par décile
            edges = quantile_bins(str_, q)
            btr = pd.cut(str_, edges, include_lowest=True)
            bva = pd.cut(sva, edges, include_lowest=True)
            means = ytr.groupby(btr, observed=False).mean()
            binned = bva.map(means).astype(float).fillna(ytr.mean())

            err_lin.append(float(np.mean((yva - lin) ** 2)))
            err_bin.append(float(np.mean((yva - binned) ** 2)))

        mse_lin, mse_bin = float(np.mean(err_lin)), float(np.mean(err_bin))
        var = float(y.var())
        rows.append({
            "feature": c,
            "mse_lineaire": mse_lin,
            "mse_binne": mse_bin,
            "gain": (mse_lin - mse_bin) / var if var else 0.0,
            "pearson": float(np.corrcoef(s, y)[0, 1]),
        })

    out = (pd.DataFrame(rows).sort_values("gain", ascending=False)
           .reset_index(drop=True))
    if plot and len(out):
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(out) + 1.2))
        ax.barh(out["feature"][::-1], out["gain"][::-1], color=PALETTE[2])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("variance de la cible récupérée en binnant (hors échantillon)")
        ax.set_title(f"Gain du binning en {q} déciles sur un coefficient unique")
        plt.tight_layout()
    return out


# ---------------------------------------------------------------------------
# 3. Cible en fonction d'une feature
# ---------------------------------------------------------------------------

def quantile_bins(s: pd.Series, q: int = 10) -> np.ndarray:
    """Bornes de déciles, dédoublonnées. À calculer sur le train uniquement."""
    edges = np.unique(np.nanquantile(np.asarray(s, dtype=float),
                                     np.linspace(0, 1, q + 1)))
    if len(edges) < 2:
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges


def bin_series(s: pd.Series, edges) -> pd.Categorical:
    """Applique des bornes apprises sur le train à n'importe quelle série."""
    return pd.cut(s, bins=edges, include_lowest=True)


def signed_log1p(s):
    """log1p qui accepte le négatif (pour `balance`, qui descend à -8019)."""
    a = np.asarray(s, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


def plot_target_rate(df: pd.DataFrame, col: str, target: str, q: int = 10,
                     max_levels: int = 30, ax=None, show_counts: bool = True,
                     sort_by_rate: bool = False, min_count: int = 30):
    """Moyenne de la cible par niveau ou par décile, avec intervalle de confiance.

    Généralise tous les plots de taux des notebooks de la séance 2, et ajoute
    ce qui leur manquait : la barre d'erreur et l'effectif. Sans ça,
    `heavy_rain` avec ses trois observations a l'air d'une catégorie comme les
    autres alors qu'elle ne peut soutenir aucun coefficient.

    Ligne pointillée = moyenne globale. Barres hachurées = moins de
    `min_count` observations, donc à ne pas lire.
    """
    d = df[[col, target]].dropna(subset=[target])
    s, y = d[col], d[target]
    binary = y.nunique() <= 2

    if _is_numeric(s) and s.nunique() > max_levels:
        groups = bin_series(s.fillna(s.median()), quantile_bins(s, q))
        xlabel = f"{col} (déciles)"
    else:
        groups = s.astype("object").fillna("__manquant__")
        xlabel = col

    g = y.groupby(groups, observed=True)
    stats = pd.DataFrame({"mean": g.mean(), "count": g.count(), "sum": g.sum()})
    if binary:
        ci = [_wilson(k, n) for k, n in zip(stats["sum"], stats["count"])]
        stats["lo"], stats["hi"] = zip(*ci)
    else:
        sem = g.std() / np.sqrt(g.count())
        stats["lo"], stats["hi"] = stats["mean"] - 1.96 * sem, stats["mean"] + 1.96 * sem
    stats = stats.dropna(subset=["mean"])
    if sort_by_rate:
        stats = stats.sort_values("mean")

    ax = _ax(ax, (max(6, 0.5 * len(stats) + 2), 4))
    x = np.arange(len(stats))
    weak = stats["count"] < min_count
    bars = ax.bar(x, stats["mean"], color=PALETTE[0],
                  yerr=[stats["mean"] - stats["lo"], stats["hi"] - stats["mean"]],
                  capsize=3, ecolor="#444444", error_kw={"lw": 1})
    for b, w in zip(bars, weak):
        if w:
            b.set_hatch("///")
            b.set_alpha(0.45)
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


def plot_target_rate_grid(df: pd.DataFrame, cols, target: str, ncols: int = 2,
                          **kwargs):
    """`plot_target_rate` sur plusieurs colonnes d'un coup, en grille."""
    cols = list(cols)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cols):
        plot_target_rate(df, c, target, ax=ax, show_counts=False, **kwargs)
    for ax in axes[len(cols):]:
        ax.axis("off")
    plt.tight_layout()
    return fig


def plot_interaction(df: pd.DataFrame, col_a: str, col_b: str, target: str,
                     q: int = 6, max_levels: int = 12, annot: bool = True,
                     figsize=(9, 5)):
    """Moyenne de la cible sur la grille (col_a x col_b), plus les profils.

    Si les courbes du panneau de droite sont parallèles, l'effet de `col_a` ne
    dépend pas de `col_b` : un modèle additif suffit. Si elles se croisent, il
    y a une interaction, et aucun coefficient par colonne ne peut la dire —
    c'est exactement `hour` x `workingday` dans le dataset vélo.
    """
    import seaborn as sns

    def _bin(s):
        if _is_numeric(s) and s.nunique() > max_levels:
            return bin_series(s.fillna(s.median()), quantile_bins(s, q))
        return s.astype("object").fillna("__manquant__")

    d = df[[col_a, col_b, target]].dropna(subset=[target])
    a, b = _bin(d[col_a]), _bin(d[col_b])
    piv = d[target].groupby([a, b], observed=True).mean().unstack()
    cnt = d[target].groupby([a, b], observed=True).count().unstack()
    piv = piv.where(cnt >= 5)

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"width_ratios": [1.1, 1]})
    sns.heatmap(piv, annot=annot, fmt=".2f", cmap="rocket_r", ax=axes[0],
                annot_kws={"size": 7}, cbar_kws={"label": f"moyenne {target}"})
    axes[0].set_title(f"{target} : {col_a} x {col_b}")
    axes[0].set_xlabel(col_b)
    axes[0].set_ylabel(col_a)

    for i, lvl in enumerate(piv.columns):
        axes[1].plot(range(len(piv.index)), piv[lvl], marker="o", ms=4,
                     label=f"{col_b}={lvl}", color=PALETTE[i % len(PALETTE)])
    axes[1].set_xticks(range(len(piv.index)))
    axes[1].set_xticklabels([str(i) for i in piv.index], rotation=45, ha="right")
    axes[1].set_xlabel(col_a)
    axes[1].set_ylabel(f"moyenne {target}")
    axes[1].set_title("parallèles = additif, croisées = interaction")
    axes[1].legend(fontsize=7)
    plt.tight_layout()
    return piv


def top_interactions(X: pd.DataFrame, y, features=None, k: int = 10, q: int = 5,
                     cv: int = 4, max_levels: int = 12, shrink: float = 20.0,
                     random_state: int = 0, plot: bool = True) -> pd.DataFrame:
    """Classe les paires de colonnes par force d'interaction, hors échantillon.

    Pour chaque paire, deux prédicteurs sont comparés en validation croisée :
    un modèle **additif** (effet de A + effet de B) et un modèle **par case**
    (une moyenne libre par croisement A x B, régularisée vers l'additif). Ce
    que le second gagne sur le premier ne peut pas s'écrire comme une somme
    d'effets — c'est la définition de l'interaction.

    La colonne `gain_interaction` est en fraction de la variance de la cible.
    Les paires en tête sont les produits croisés qui vaudraient la peine d'être
    ajoutés à un modèle linéaire (un arbre les trouve tout seul).

    Lis la colonne `redondance` avant de conclure : deux colonnes qui sont la
    même mesure (temp / feel_temp à 0.99) sortent en tête sans qu'il y ait
    d'interaction, simplement parce que leur croisement redécoupe l'espace plus
    finement. Une vraie interaction a une redondance faible.
    """
    from sklearn.model_selection import KFold

    y = _series(y).astype(float)
    mask = y.notna()
    X, y = X[mask], y[mask]

    if features is None:
        cand = [c for c in X.columns
                if X[c].nunique() > 1
                and (_is_numeric(X[c]) or X[c].nunique() <= max_levels)]
        if len(cand) > 10:
            sfs = single_feature_score(X[cand], y, plot=False,
                                       random_state=random_state)
            features = list(sfs["feature"].head(10))
        else:
            features = cand
    features = list(features)

    binned = {}
    for c in features:
        s = X[c]
        if _is_numeric(s) and s.nunique() > max_levels:
            binned[c] = bin_series(s.fillna(s.median()), quantile_bins(s, q)).astype(str)
        else:
            binned[c] = s.astype("object").fillna("__manquant__").astype(str)
    B = pd.DataFrame(binned, index=X.index)

    kf = KFold(cv, shuffle=True, random_state=random_state)
    var = float(y.var())
    rows = []
    for a, b in combinations(features, 2):
        e_add, e_full = [], []
        for tr, va in kf.split(B):
            Btr, Bva = B.iloc[tr], B.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            gm = ytr.mean()
            ea = ytr.groupby(Btr[a], observed=True).mean() - gm
            eb = ytr.groupby(Btr[b], observed=True).mean() - gm

            add_tr = gm + Btr[a].map(ea).fillna(0) + Btr[b].map(eb).fillna(0)
            add_va = gm + Bva[a].map(ea).fillna(0) + Bva[b].map(eb).fillna(0)

            cells = pd.DataFrame({"y": ytr, "add": add_tr}).groupby(
                [Btr[a], Btr[b]], observed=True).agg(
                    m=("y", "mean"), n=("y", "size"), a=("add", "mean"))
            # régularisation vers l'additif : une case à 3 lignes ne compte pas
            cells["p"] = (cells["n"] * cells["m"] + shrink * cells["a"]) / \
                         (cells["n"] + shrink)
            key_va = pd.MultiIndex.from_arrays([Bva[a], Bva[b]])
            full_va = pd.Series(cells["p"].reindex(key_va).to_numpy(),
                                index=Bva.index)
            full_va = full_va.fillna(add_va)

            e_add.append(float(np.mean((yva - add_va) ** 2)))
            e_full.append(float(np.mean((yva - full_va) ** 2)))

        mse_add, mse_full = float(np.mean(e_add)), float(np.mean(e_full))
        rows.append({"a": a, "b": b,
                     "mse_additif": mse_add, "mse_croise": mse_full,
                     "gain_interaction": (mse_add - mse_full) / var if var else 0.0,
                     "redondance": round(cramers_v(B[a], B[b]), 3)})

    out = (pd.DataFrame(rows).sort_values("gain_interaction", ascending=False)
           .reset_index(drop=True))
    if plot and len(out):
        top = out.head(k)[::-1]
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1.2))
        ax.barh([f"{r.a} x {r.b}" for r in top.itertuples()],
                top["gain_interaction"],
                color=[PALETTE[3] if r > 0.6 else PALETTE[4]
                       for r in top["redondance"]])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("variance récupérée par le croisement, au-delà de l'additif")
        ax.set_title("Interactions les plus fortes (hors échantillon)\n"
                     "en ocre : les deux colonnes sont redondantes, "
                     "ce n'est pas une interaction")
        plt.tight_layout()
    return out.head(k) if k else out


# ---------------------------------------------------------------------------
# 4. Importance
# ---------------------------------------------------------------------------

def group_encoded_columns(encoded_columns, original_columns) -> dict:
    """Reconstitue quelle colonne d'origine a produit quelles colonnes one-hot.

    `job` -> ['job_admin.', 'job_blue-collar', ...]. Nécessaire pour que
    l'importance d'une catégorielle à douze niveaux ne soit pas diluée en douze
    importances minuscules.
    """
    originals = sorted(original_columns, key=len, reverse=True)
    groups = {}
    for c in encoded_columns:
        parent = c
        for o in originals:
            if c == o or c.startswith(f"{o}_"):
                parent = o
                break
        groups.setdefault(parent, []).append(c)
    return groups


def permutation_importance_grouped(model, X: pd.DataFrame, y, scoring=None,
                                   groups: dict | None = None, n_repeats: int = 5,
                                   random_state: int = 0, plot: bool = True,
                                   top: int = 25) -> pd.DataFrame:
    """Importance par permutation, en mélangeant ensemble les colonnes d'un même groupe.

    On casse le lien entre une feature et la cible en mélangeant ses valeurs,
    puis on regarde de combien le score chute. Contrairement aux coefficients,
    ça marche sur n'importe quel modèle, et contrairement à
    `feature_importances_` d'un arbre, ça se mesure sur des lignes que le
    modèle n'a pas vues (passe-lui ton X_test).

    `groups` : dict {nom -> [colonnes]}, typiquement `group_encoded_columns(...)`.
    Sans lui, chaque colonne one-hot est permutée séparément et une
    catégorielle importante passe inaperçue.
    """
    from sklearn.metrics import get_scorer

    y = _series(y)
    if scoring is None:
        scoring = "f1" if infer_task(y) == "classification" else "neg_mean_absolute_error"
    scorer = get_scorer(scoring) if isinstance(scoring, str) else scoring
    if groups is None:
        groups = {c: [c] for c in X.columns}

    rng = np.random.default_rng(random_state)
    base = scorer(model, X, y)

    rows = []
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(Xp))
            Xp[cols] = Xp[cols].to_numpy()[perm]   # même permutation pour tout le groupe
            drops.append(base - scorer(model, Xp, y))
        rows.append({"feature": name, "importance": float(np.mean(drops)),
                     "std": float(np.std(drops)), "n_cols": len(cols)})

    out = (pd.DataFrame(rows).sort_values("importance", ascending=False)
           .reset_index(drop=True))
    out.attrs["baseline_score"] = float(base)
    out.attrs["scoring"] = scoring if isinstance(scoring, str) else "custom"
    if plot:
        plot_importance(out, top=top,
                        title=f"Importance par permutation ({out.attrs['scoring']}, "
                              f"score de départ {base:.4f})")
    return out


def plot_importance(imp: pd.DataFrame, top: int = 25, title: str = "Importance",
                    ax=None):
    """Barres horizontales à partir d'un DataFrame feature / importance / std."""
    d = imp.head(top)[::-1]
    ax = _ax(ax, (7, 0.32 * len(d) + 1.2))
    err = d["std"] if "std" in d else None
    ax.barh(d["feature"], d["importance"], xerr=err, color=PALETTE[0],
            ecolor="#888888", capsize=2)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_coefficients(model, feature_names, top: int = 25, ax=None):
    """Coefficients d'un modèle linéaire, triés par valeur absolue.

    Ne se lit que si les colonnes ont été mises à la même échelle : sinon un
    gros coefficient veut juste dire que la colonne est en petites unités.
    """
    coef = np.ravel(getattr(model, "coef_", None))
    if coef is None:
        raise AttributeError("ce modèle n'a pas de coef_")
    d = (pd.DataFrame({"feature": list(feature_names), "coef": coef})
         .assign(abs_coef=lambda t: t["coef"].abs())
         .sort_values("abs_coef", ascending=False).head(top)[::-1])
    ax = _ax(ax, (7, 0.32 * len(d) + 1.2))
    ax.barh(d["feature"], d["coef"],
            color=[PALETTE[1] if v < 0 else PALETTE[0] for v in d["coef"]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Coefficients (valables seulement sur des colonnes scalées)")
    plt.tight_layout()
    return d


# ---------------------------------------------------------------------------
# 5. Drift train / test
# ---------------------------------------------------------------------------

def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index entre deux échantillons d'une même colonne.

    Repère < 0.1 stable, 0.1-0.25 à surveiller, > 0.25 la colonne ne raconte
    pas la même histoire dans le test que dans le train.
    """
    e, a = pd.Series(expected).dropna(), pd.Series(actual).dropna()
    if len(e) == 0 or len(a) == 0:
        return np.nan
    if _is_numeric(e):
        edges = quantile_bins(e, bins)
        edges[0], edges[-1] = -np.inf, np.inf
        pe = pd.cut(e, edges).value_counts(normalize=True, sort=False)
        pa = pd.cut(a, edges).value_counts(normalize=True, sort=False)
    else:
        levels = sorted(set(e.astype(str)) | set(a.astype(str)))
        pe = e.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0)
        pa = a.astype(str).value_counts(normalize=True).reindex(levels, fill_value=0)
    eps = 1e-6
    pe, pa = pe.to_numpy() + eps, pa.to_numpy() + eps
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def drift_report(X_train: pd.DataFrame, X_test: pd.DataFrame,
                 bins: int = 10) -> pd.DataFrame:
    """PSI colonne par colonne entre le train et le jeu à prédire.

    Le test du challenge vélo est les cinq derniers mois d'un système en
    croissance : les colonnes de calendrier dérivent massivement et le score
    hors échantillon en pâtit. Autant le voir avant de soumettre.
    """
    rows = []
    for c in X_train.columns.intersection(X_test.columns):
        v = psi(X_train[c], X_test[c], bins=bins)
        note = ("stable" if v < 0.1 else "à surveiller" if v < 0.25 else "DÉRIVE")
        row = {"feature": c, "psi": round(float(v), 4), "verdict": note}
        if _is_numeric(X_train[c]):
            row["mean_train"] = X_train[c].mean()
            row["mean_test"] = X_test[c].mean()
        rows.append(row)
    return (pd.DataFrame(rows).sort_values("psi", ascending=False)
            .reset_index(drop=True))


def adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame,
                           n_estimators: int = 200, cv: int = 3,
                           random_state: int = 0, plot: bool = True):
    """Entraîne un modèle à distinguer le train du test, et renvoie son AUC.

    AUC ~ 0.5 : les deux jeux sont interchangeables, ton score de validation
    est une bonne estimation du leaderboard.
    AUC proche de 1 : ils sont distinguables, la validation aléatoire ment, et
    les features en tête du classement retourné sont *celles par lesquelles*
    ils diffèrent (souvent une date déguisée).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    cols = list(X_train.columns.intersection(X_test.columns))
    both = pd.concat([X_train[cols], X_test[cols]], ignore_index=True)
    both = both.drop(columns=[c for c in ("id",) if c in both.columns])
    lab = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))]

    Z = pd.get_dummies(both, dummy_na=True)
    Z.columns = [str(c) for c in Z.columns]
    Z = Z.fillna(Z.median(numeric_only=True)).fillna(0)

    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                 min_samples_leaf=5, random_state=random_state)
    proba = cross_val_predict(
        clf, Z, lab, cv=StratifiedKFold(cv, shuffle=True, random_state=random_state),
        method="predict_proba")[:, 1]
    auc = float(roc_auc_score(lab, proba))

    clf.fit(Z, lab)
    imp = (pd.DataFrame({"feature": Z.columns, "importance": clf.feature_importances_})
           .sort_values("importance", ascending=False).reset_index(drop=True))

    verdict = ("train et test sont indiscernables — validation aléatoire fiable"
               if auc < 0.6 else
               "train et test diffèrent nettement — méfie-toi de la validation aléatoire"
               if auc > 0.75 else "différence modérée")
    print(f"[adversarial validation] AUC = {auc:.3f} — {verdict}")
    if plot:
        plot_importance(imp, top=15,
                        title="Par quoi le test se distingue du train")
    return auc, imp


# ---------------------------------------------------------------------------
# 6. Préparation
# ---------------------------------------------------------------------------

@dataclass
class Prepared:
    """Sortie de `prepare` : matrices alignées + tout ce qu'il faut pour rejouer."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame | None
    columns: list
    medians: pd.Series
    scaler: object = None
    as_category: tuple = ()
    dropped: tuple = ()
    notes: list = field(default_factory=list)


def prepare(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None,
            drop=("id",), as_category=(), scale: bool = False,
            missing_flags: bool = False, fill: str = "median") -> Prepared:
    """Trous bouchés, texte encodé, colonnes du test alignées sur le train.

    Toutes les statistiques (médianes, colonnes, moyennes du scaler) sont
    apprises sur le train et *appliquées* au test : une médiane calculée sur les
    lignes qu'on doit prédire est une médiane qui a vu la réponse.

    `as_category` : colonnes numériques à traiter comme du texte. C'est le
    correctif qui vaut le plus cher dans le notebook vélo — dire à
    `get_dummies` que `hour` est une catégorie fait passer le -MAE de -139 à
    -100 sans changer de modèle.
    `missing_flags` : ajoute une colonne `<col>_was_missing`. Le fait qu'une
    valeur manque est parfois lui-même un signal.
    """
    notes = []
    drop = [c for c in drop if c in X_train.columns]
    tr = X_train.drop(columns=drop)
    te = X_test.drop(columns=[c for c in drop if c in X_test.columns]) \
        if X_test is not None else None

    if missing_flags:
        for c in tr.columns[tr.isna().any()]:
            tr[f"{c}_was_missing"] = tr[c].isna().astype(int)
            if te is not None:
                te[f"{c}_was_missing"] = te[c].isna().astype(int)
        notes.append("indicateurs de valeur manquante ajoutés")

    medians = tr.median(numeric_only=True)
    def _fill(f):
        if f is None:
            return None
        f = f.copy()
        if fill == "median":
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
    for n in notes:
        print(f"  - {n}")
    return Prepared(tr_enc, te_enc, list(tr_enc.columns), medians, scaler,
                    tuple(as_category), tuple(drop), notes)


# ---------------------------------------------------------------------------
# 7. Évaluation
# ---------------------------------------------------------------------------

def regression_report(y_true, y_pred, baseline: str = "median",
                      plot: bool = True) -> dict:
    """R2 / RMSE / MAE, la baseline en face, et le compte de prédictions négatives."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true = np.asarray(_series(y_true), dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ref = np.median(y_true) if baseline == "median" else np.mean(y_true)
    base = np.full(len(y_true), ref)

    out = {
        "r2": r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
        "mae_baseline": mean_absolute_error(y_true, base),
        "n_negatives": int((y_pred < 0).sum()),
    }
    out["mae_gain_pct"] = 100 * (1 - out["mae"] / out["mae_baseline"])
    print(f"R2   {out['r2']:7.4f}")
    print(f"RMSE {out['rmse']:7.2f}")
    print(f"MAE  {out['mae']:7.2f}   (baseline {baseline} : {out['mae_baseline']:.2f}"
          f"  ->  {out['mae_gain_pct']:.1f} % mieux)")
    if out["n_negatives"]:
        print(f"/!\\ {out['n_negatives']} prédictions négatives sur {len(y_pred)} — "
              "np.clip(pred, 0, None) est gratuit")
    if plot:
        plot_residuals(y_true, y_pred)
    return out


def plot_residuals(y_true, y_pred, bins: int = 40):
    """Résidus contre prédiction, et distribution. Une forme = du signal non capté."""
    y_true = np.asarray(_series(y_true), dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    res = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    axes[0].scatter(y_pred, res, s=5, alpha=0.15, color=PALETTE[0])
    axes[0].axhline(0, color=PALETTE[1], lw=1)
    axes[0].set_xlabel("prédiction")
    axes[0].set_ylabel("résidu")
    axes[0].set_title("Résidus — une forme ici = du signal laissé sur la table")
    axes[1].hist(res, bins=bins, color=PALETTE[0])
    axes[1].axvline(0, color=PALETTE[1], lw=1)
    axes[1].set_title(f"Distribution des résidus (biais moyen {res.mean():.2f})")
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    axes[2].scatter(y_true, y_pred, s=5, alpha=0.15, color=PALETTE[0])
    axes[2].plot(lim, lim, color=PALETTE[1], lw=1)
    axes[2].set_xlabel("vrai")
    axes[2].set_ylabel("prédit")
    axes[2].set_title("Prédit contre vrai")
    plt.tight_layout()
    return fig


def classification_report_plus(y_true, proba, threshold: float = 0.5,
                               plot: bool = True) -> dict:
    """Accuracy / précision / rappel / F1 au seuil donné, + matrice, ROC et PR.

    Sur une cible à 11.7 % de positifs, la courbe précision-rappel est plus
    lisible que la ROC : la ROC reste flatteuse quand les négatifs écrasent
    tout, l'aire sous la PR se compare directement au taux de base.
    """
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 confusion_matrix, f1_score, precision_score,
                                 precision_recall_curve, recall_score,
                                 roc_auc_score, roc_curve)
    y_true = np.asarray(_series(y_true)).astype(int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, 1]
    pred = (proba >= threshold).astype(int)

    out = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(pred.mean()),
    }
    out["accuracy_majority"] = max(out["positive_rate"], 1 - out["positive_rate"])

    print(f"seuil {threshold:.2f}")
    print(f"  accuracy  {out['accuracy']:.4f}   (majorité : {out['accuracy_majority']:.4f})")
    print(f"  précision {out['precision']:.4f}")
    print(f"  rappel    {out['recall']:.4f}")
    print(f"  F1        {out['f1']:.4f}")
    print(f"  ROC AUC   {out['roc_auc']:.4f}   |   PR AUC {out['pr_auc']:.4f} "
          f"(hasard : {out['positive_rate']:.4f})")
    if abs(out["accuracy"] - out["accuracy_majority"]) < 0.01:
        print("  /!\\ accuracy ~ celle de la majorité : elle ne dit rien ici")

    if plot:
        import seaborn as sns
        cm = confusion_matrix(y_true, pred)
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                    xticklabels=["prédit 0", "prédit 1"],
                    yticklabels=["vrai 0", "vrai 1"], cbar=False)
        axes[0].set_title("Matrice de confusion")

        fpr, tpr, _ = roc_curve(y_true, proba)
        axes[1].plot(fpr, tpr, color=PALETTE[0], label=f"AUC {out['roc_auc']:.3f}")
        axes[1].plot([0, 1], [0, 1], ls="--", color="#999999")
        axes[1].set_xlabel("faux positifs")
        axes[1].set_ylabel("vrais positifs")
        axes[1].set_title("ROC")
        axes[1].legend(fontsize=8)

        prec, rec, _ = precision_recall_curve(y_true, proba)
        axes[2].plot(rec, prec, color=PALETTE[1], label=f"AP {out['pr_auc']:.3f}")
        axes[2].axhline(out["positive_rate"], ls="--", color="#999999",
                        label=f"hasard {out['positive_rate']:.3f}")
        axes[2].set_xlabel("rappel")
        axes[2].set_ylabel("précision")
        axes[2].set_title("Précision-rappel (la bonne courbe quand c'est déséquilibré)")
        axes[2].legend(fontsize=8)
        plt.tight_layout()
    return out


def threshold_sweep(y_true, proba, metric: str = "f1", lo: float = 0.01,
                    hi: float = 0.99, step: float = 0.01, plot: bool = True,
                    cost_fp: float | None = None, cost_fn: float | None = None):
    """Balaye le seuil de décision et retourne celui qui maximise la métrique.

    0.5 est une convention, pas une propriété du problème. Sur le dataset bank
    (11.7 % de positifs) déplacer le seuil vaut ~0.13 de F1, plus que tout
    changement de famille de modèle.

    `metric` : "f1", "f2" (le rappel compte quatre fois plus que la précision),
    "precision", "recall", "youden" (TPR - FPR), "balanced_accuracy", ou "cost"
    si tu renseignes `cost_fp` et `cost_fn` — c'est la seule version qui
    corresponde à une vraie décision métier : rater un client qui aurait
    souscrit et déranger un client qui n'aurait pas souscrit ne coûtent pas la
    même chose, et c'est ce rapport qui fixe le seuil, pas le F1.
    """
    from sklearn.metrics import (balanced_accuracy_score, confusion_matrix,
                                 fbeta_score, precision_score, recall_score)
    y_true = np.asarray(_series(y_true)).astype(int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, 1]

    def _rates(t, p):
        tn, fp, fn_, tp = confusion_matrix(t, p, labels=[0, 1]).ravel()
        return tn, fp, fn_, tp

    fns = {
        "f1": lambda t, p: fbeta_score(t, p, beta=1, zero_division=0),
        "f2": lambda t, p: fbeta_score(t, p, beta=2, zero_division=0),
        "precision": lambda t, p: precision_score(t, p, zero_division=0),
        "recall": lambda t, p: recall_score(t, p, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
        "youden": lambda t, p: (lambda r: r[3] / max(r[3] + r[2], 1)
                                - r[1] / max(r[1] + r[0], 1))(_rates(t, p)),
        "cost": lambda t, p: -(lambda r: cost_fp * r[1] + cost_fn * r[2])(_rates(t, p)),
    }
    if metric == "cost" and (cost_fp is None or cost_fn is None):
        raise ValueError("metric='cost' demande cost_fp et cost_fn")
    fn = fns[metric]
    ts = np.arange(lo, hi + 1e-9, step)
    rows = []
    for t in ts:
        p = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            metric: fn(y_true, p),
            "precision": precision_score(y_true, p, zero_division=0),
            "recall": recall_score(y_true, p, zero_division=0),
            "predicted_positive_rate": float(p.mean()),
        })
    curve = pd.DataFrame(rows)
    best = curve.loc[curve[metric].idxmax()]
    best_t, best_v = float(best["threshold"]), float(best[metric])
    at_half = fn(y_true, (proba >= 0.5).astype(int))
    print(f"meilleur seuil {best_t:.2f} -> {metric} = {best_v:.4f}   "
          f"(au seuil 0.5 : {at_half:.4f}, gain {best_v - at_half:+.4f})")
    print(f"  précision {best['precision']:.4f} | rappel {best['recall']:.4f} | "
          f"{100*best['predicted_positive_rate']:.1f} % de positifs prédits")

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(curve["threshold"], curve[metric], color=PALETTE[0], lw=2, label=metric)
        ax.plot(curve["threshold"], curve["precision"], color=PALETTE[1], lw=1,
                ls="--", label="précision")
        ax.plot(curve["threshold"], curve["recall"], color=PALETTE[2], lw=1,
                ls="--", label="rappel")
        ax.axvline(best_t, color="black", ls=":", lw=1,
                   label=f"meilleur seuil {best_t:.2f}")
        ax.axvline(0.5, color="#bbbbbb", lw=1, label="défaut 0.5")
        ax.set_xlabel("seuil")
        ax.set_ylabel("score")
        ax.set_title(f"{metric} en fonction du seuil — le compromis est ton choix")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
    return best_t, best_v, curve


def plot_calibration(y_true, proba, bins: int = 10, ax=None):
    """Probabilité prédite contre fréquence observée.

    Un modèle calibré suit la diagonale : quand il dit 0.3, 30 % de ces clients
    souscrivent. Utile avant de choisir un seuil sur des probabilités, et pour
    voir qu'un RandomForest, lui, ne les donne pas calibrées.
    """
    y_true = np.asarray(_series(y_true)).astype(int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, 1]
    edges = np.quantile(proba, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, len(edges) - 2)
    d = pd.DataFrame({"p": proba, "y": y_true, "b": idx}).groupby("b").agg(
        p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    ax = _ax(ax, (5, 4.5))
    ax.plot([0, d["p"].max()], [0, d["p"].max()], ls="--", color="#999999")
    ax.plot(d["p"], d["y"], marker="o", color=PALETTE[0])
    for _, r in d.iterrows():
        ax.annotate(int(r["n"]), (r["p"], r["y"]), fontsize=6,
                    textcoords="offset points", xytext=(3, 3))
    ax.set_xlabel("probabilité prédite (moyenne du bin)")
    ax.set_ylabel("fréquence observée")
    ax.set_title("Calibration")
    plt.tight_layout()
    return d


# ---------------------------------------------------------------------------
# 8. Sortie
# ---------------------------------------------------------------------------

def write_submission(ids, predictions, path: str = "submission.csv",
                     binary: bool = False, threshold: float = 0.5,
                     clip=None) -> pd.DataFrame:
    """Écrit `submission.csv` et repasse les vérifications avant l'upload.

    `binary=True` seuille des probabilités en 0/1 (le challenge bank rejette
    une probabilité, il ne la seuille pas pour toi).
    `clip=(0, None)` borne les prédictions (le challenge vélo n'attend pas un
    nombre négatif de vélos).
    """
    ids = np.asarray(_series(ids))
    p = np.asarray(predictions, dtype=float)
    if p.ndim == 2:
        p = p[:, 1]
    if binary:
        p = (p >= threshold).astype(int)
    elif clip is not None:
        p = np.clip(p, clip[0], clip[1])

    sub = pd.DataFrame({"id": ids, "prediction": p})
    assert len(sub) == len(ids), "une ligne par id"
    assert sub["id"].is_unique, "ids dupliqués"
    assert sub["prediction"].notna().all(), "prédictions manquantes"
    assert np.isfinite(sub["prediction"]).all(), "inf dans les prédictions"
    if binary:
        assert set(sub["prediction"].unique()) <= {0, 1}, "valeurs hors {0, 1}"
    sub.to_csv(path, index=False)
    print(f"{path} écrit — {sub.shape[0]} lignes, "
          + (f"{100*sub['prediction'].mean():.1f} % de positifs"
             if binary else
             f"prédiction moyenne {sub['prediction'].mean():.2f}"))
    return sub
