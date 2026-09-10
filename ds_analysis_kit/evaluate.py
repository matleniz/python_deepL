"""Évaluation : mesurer un modèle, choisir un seuil, écrire la soumission.

Ces fonctions rassemblent les blocs de métriques que tu réécris à chaque
notebook (les quatre scores de classification, R2/RMSE/MAE en régression), le
balayage de seuil de l'étape 6, et les vérifications de `submission.csv`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._common import PALETTE, _series

__all__ = [
    "regression_summary",
    "classification_summary",
    "plot_roc",
    "threshold_sweep",
    "write_submission",
]


def regression_summary(y_true, y_pred, baseline: str = "mean",
                       plot: bool = False) -> dict:
    """R2 / RMSE / MAE, la baseline en face, et le compte de prédictions négatives.

    Reprend le bloc du notebook vélo. La baseline est le modèle bête « prédis la
    moyenne partout » : c'est contre elle que ton score veut être lu, pas contre
    zéro. Compte aussi les prédictions négatives, que le notebook vélo signale
    (un modèle linéaire non borné prédit un nombre de vélos négatif).

    Compose : `sklearn.metrics` (r2_score, mean_squared_error, mean_absolute_error).
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(_series(y_true), dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ref = np.median(y_true) if baseline == "median" else np.mean(y_true)
    base = np.full(len(y_true), ref)

    out = {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mae_baseline": float(mean_absolute_error(y_true, base)),
        "n_negatives": int((y_pred < 0).sum()),
    }
    print(f"R2   {out['r2']:7.4f}")
    print(f"RMSE {out['rmse']:7.2f}")
    print(f"MAE  {out['mae']:7.2f}   (baseline {baseline} : {out['mae_baseline']:.2f})")
    if out["n_negatives"]:
        print(f"/!\\ {out['n_negatives']} prédictions négatives sur {len(y_pred)} — "
              "np.clip(pred, 0, None) est gratuit")
    if plot:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.scatter(y_true, y_pred, s=5, alpha=0.15, color=PALETTE[0])
        ax.plot(lim, lim, color=PALETTE[1], lw=1)
        ax.set_xlabel("vrai")
        ax.set_ylabel("prédit")
        ax.set_title("Prédit contre vrai")
        plt.tight_layout()
    return out


def classification_summary(y_true, y_pred, plot: bool = True) -> dict:
    """Accuracy / précision / rappel / F1, avec l'accuracy majoritaire en face.

    Le bloc de quatre métriques du notebook bank. L'accuracy seule ment sur une
    cible déséquilibrée : on l'affiche donc à côté de l'accuracy qu'on aurait en
    prédisant toujours la classe majoritaire, et on prévient si les deux sont
    trop proches. La matrice de confusion rend les quatre nombres lisibles d'un
    coup.

    Ce n'est pas une copie de `sklearn.classification_report` : il y ajoute la
    baseline majoritaire et la matrice.

    Compose : `sklearn.metrics` (accuracy/precision/recall/f1_score,
    confusion_matrix) + `seaborn`.
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)

    y_true = np.asarray(_series(y_true)).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    pos_rate = float(y_true.mean())

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy_majority": max(pos_rate, 1 - pos_rate),
    }
    print(f"  accuracy  {out['accuracy']:.4f}   (majorité : {out['accuracy_majority']:.4f})")
    print(f"  précision {out['precision']:.4f}")
    print(f"  rappel    {out['recall']:.4f}")
    print(f"  F1        {out['f1']:.4f}")
    if abs(out["accuracy"] - out["accuracy_majority"]) < 0.01:
        print("  /!\\ accuracy ~ celle de la majorité : elle ne dit rien ici, "
              "regarde le F1")
    if plot:
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["prédit 0", "prédit 1"],
                    yticklabels=["vrai 0", "vrai 1"])
        ax.set_title("Matrice de confusion")
        plt.tight_layout()
    return out


def plot_roc(y_true, proba, ax=None):
    """Courbe ROC et son aire (AUC), comme dans le notebook bank.

    Trace le taux de vrais positifs contre le taux de faux positifs quand on
    fait varier le seuil. L'aire résume le pouvoir de séparation du modèle,
    indépendamment du seuil choisi.

    Compose : `sklearn.metrics` (roc_curve, auc).
    """
    from sklearn.metrics import auc, roc_curve

    y_true = np.asarray(_series(y_true)).astype(int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, 1]
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    ax.set_xlabel("taux de faux positifs")
    ax.set_ylabel("taux de vrais positifs")
    ax.set_title("Courbe ROC")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    return roc_auc


def threshold_sweep(y_true, proba, lo: float = 0.01, hi: float = 0.99,
                    step: float = 0.01, plot: bool = True):
    """Balaye le seuil de décision et renvoie celui qui maximise le F1.

    `predict()` décide à 0.5, mais 0.5 est une convention, pas une propriété du
    problème : sur une cible à 11.7 % de positifs, déplacer le seuil vaut plus
    de F1 que changer de famille de modèle (étape 6 du notebook bank). On teste
    tous les seuils de `lo` à `hi`, on trace le F1 (et la précision / le rappel
    pour voir le compromis), et on renvoie `(meilleur_seuil, meilleur_f1, courbe)`.

    Compose : `sklearn.metrics` (f1_score, precision_score, recall_score).
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(_series(y_true)).astype(int)
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 2:
        proba = proba[:, 1]

    ts = np.arange(lo, hi + 1e-9, step)
    rows = []
    for t in ts:
        p = (proba >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "f1": f1_score(y_true, p, zero_division=0),
            "precision": precision_score(y_true, p, zero_division=0),
            "recall": recall_score(y_true, p, zero_division=0),
        })
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["f1"].idxmax()]
    best_t, best_f1 = float(best["threshold"]), float(best["f1"])
    at_half = f1_score(y_true, (proba >= 0.5).astype(int), zero_division=0)
    print(f"meilleur seuil {best_t:.2f} -> F1 = {best_f1:.4f}   "
          f"(au seuil 0.5 : {at_half:.4f}, gain {best_f1 - at_half:+.4f})")
    print(f"  précision {best['precision']:.4f} | rappel {best['recall']:.4f}")

    if plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(curve["threshold"], curve["f1"], color=PALETTE[0], lw=2, label="F1")
        ax.plot(curve["threshold"], curve["precision"], color=PALETTE[1], lw=1,
                ls="--", label="précision")
        ax.plot(curve["threshold"], curve["recall"], color=PALETTE[2], lw=1,
                ls="--", label="rappel")
        ax.axvline(best_t, color="black", ls=":", lw=1,
                   label=f"meilleur seuil {best_t:.2f}")
        ax.axvline(0.5, color="#bbbbbb", lw=1, label="défaut 0.5")
        ax.set_xlabel("seuil")
        ax.set_ylabel("score")
        ax.set_title("F1 selon le seuil — le compromis précision/rappel est ton choix")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
    return best_t, best_f1, curve


def write_submission(ids, predictions, path: str = "submission.csv",
                     binary: bool = False, threshold: float = 0.5,
                     clip=None) -> pd.DataFrame:
    """Écrit `submission.csv` et repasse les vérifications avant l'upload.

    Rassemble les asserts que tu remets à la fin de chaque notebook : une ligne
    par id, ids uniques, aucune valeur manquante. Deux options selon le
    challenge :

    - `binary=True` seuille des probabilités en 0/1 (le challenge bank rejette
      une probabilité, il ne la seuille pas pour toi) ;
    - `clip=(0, None)` borne les prédictions (le challenge vélo n'attend pas un
      nombre de vélos négatif).

    Compose : `pandas` + `numpy`.
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
          + (f"{100 * sub['prediction'].mean():.1f} % de positifs" if binary
             else f"prédiction moyenne {sub['prediction'].mean():.2f}"))
    return sub
