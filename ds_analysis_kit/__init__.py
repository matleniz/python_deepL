"""ds_analysis_kit — boîte à outils personnelle d'analyse de données.

Regroupe le code d'exploration, de préparation et d'évaluation que tu réutilises
d'un notebook à l'autre. Tout est **général** (aucune logique propre à un
challenge) et **compose** des fonctions de pandas / seaborn / scikit-learn
plutôt que de les réécrire.

Le module grandit avec toi : on n'y ajoute que des fonctions dont tu maîtrises
déjà l'idée. La feuille de route des ajouts futurs est dans le README.

Usage typique (depuis un dossier de séance) :

    import sys; sys.path.append("..")
    import ds_analysis_kit as dsk

    dsk.overview(X_train)
    dsk.target_report(y_train)
    dsk.plot_target_rate(df, "age", "prediction")

    prep = dsk.prepare(X_train, X_test, as_category=["hour"], scale=True)
    ...
    dsk.classification_summary(y_val, preds)
    best_t, best_f1, curve = dsk.threshold_sweep(y_val, proba)
    dsk.write_submission(X_test["id"], proba, binary=True, threshold=best_t)
"""

from ._common import infer_task
from .evaluate import (classification_summary, plot_roc, regression_summary,
                       threshold_sweep, write_submission)
from .explore import (overview, plot_correlation, plot_missing,
                      plot_target_rate, target_report)
from .prep import Prepared, bin_series, prepare, quantile_bins, signed_log1p

__all__ = [
    # exploration
    "overview", "target_report", "plot_missing", "plot_correlation",
    "plot_target_rate",
    # préparation
    "quantile_bins", "bin_series", "signed_log1p", "prepare", "Prepared",
    # évaluation
    "regression_summary", "classification_summary", "plot_roc",
    "threshold_sweep", "write_submission",
    # utilitaire
    "infer_task",
]
