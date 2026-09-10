# ds_analysis_kit

Ma boîte à outils personnelle d'analyse de données : le code d'exploration, de
préparation et d'évaluation que je réutilise d'un notebook à l'autre, rassemblé
en un seul module importable.

## Principe

Trois règles, pour que ce kit reste un outil que je maîtrise de bout en bout :

1. **Uniquement du général.** Aucune fonction ne connaît le nom d'une colonne ni
   la logique d'un challenge. Ce qui est spécifique (un `calculate_day_of_week`,
   un taux de souscription) reste dans le notebook.
2. **On ne réécrit pas l'existant.** Chaque fonction *compose* pandas / seaborn /
   scikit-learn. Je ne reprogramme jamais un `r2_score`, un `StandardScaler` ou
   une `roc_curve` : je les enchaîne pour supprimer le copier-coller.
3. **Rien que je n'aie pas encore vu.** Le module ne contient que des fonctions
   dont je maîtrise déjà l'idée. On l'agrandit **ensemble, pas à pas** — voir la
   feuille de route en bas.

## Installation / import

En local (workflow `uv` par séance), depuis un dossier `seanceN/` :

```python
import sys; sys.path.append("..")   # remonte à la racine du repo
import ds_analysis_kit as dsk
```

Sur Google Colab, récupérer le package depuis GitHub en tête de notebook :

```python
!wget -q https://raw.githubusercontent.com/matleniz/python_deepL/main/ds_analysis_kit/__init__.py -P ds_analysis_kit
!wget -q https://raw.githubusercontent.com/matleniz/python_deepL/main/ds_analysis_kit/_common.py -P ds_analysis_kit
!wget -q https://raw.githubusercontent.com/matleniz/python_deepL/main/ds_analysis_kit/explore.py -P ds_analysis_kit
!wget -q https://raw.githubusercontent.com/matleniz/python_deepL/main/ds_analysis_kit/prep.py -P ds_analysis_kit
!wget -q https://raw.githubusercontent.com/matleniz/python_deepL/main/ds_analysis_kit/evaluate.py -P ds_analysis_kit
import ds_analysis_kit as dsk
```

## Organisation

| Fichier | Rôle |
| --- | --- |
| `_common.py` | briques internes (non exposées) : `_series`, `_is_numeric`, `infer_task`, palette |
| `explore.py` | auditer un jeu de données et regarder la cible |
| `prep.py` | transformer le DataFrame brut en matrice prête pour un modèle |
| `evaluate.py` | mesurer un modèle, choisir un seuil, écrire la soumission |

## Les fonctions

### Exploration (`explore.py`)

- **`overview(X)`** — un tableau, une ligne par colonne : dtype, trous,
  cardinalité, stats des numériques, et une colonne `flags` pour les cas à
  regarder. Remplace `X.dtypes` / `X.describe()` / `X.isna().sum()`.
  *Compose : pandas.* — vient de l'étape « 2. Read it ».
- **`target_report(y, task=None)`** — distribution de la cible et sa baseline :
  accuracy de la classe majoritaire (classif) ou MAE de « prédis la moyenne »
  (régression), plus l'alerte cible bornée à 0.
  *Compose : pandas, sklearn.metrics.* — vient de l'étape 3a.
- **`plot_missing(X)`** — la heatmap des `NaN` (sombre = manquant), pour voir si
  les trous sont en blocs ou dispersés.
  *Compose : matplotlib.* — vient de l'étape 3f du notebook vélo.
- **`plot_correlation(df)`** — heatmap de corrélation des colonnes numériques,
  pour repérer deux colonnes qui sont la même mesure.
  *Compose : pandas.corr, seaborn.* — présente dans les deux notebooks.
- **`plot_target_rate(df, col, target, q=10)`** — moyenne de la cible par
  catégorie ou par décile, avec l'effectif de chaque barre. Généralise tous mes
  `barplot(estimator="mean")` et `pd.cut` + moyenne.
  *Compose : pandas, matplotlib.* — vient des étapes 3b à 3e.

### Préparation (`prep.py`)

- **`quantile_bins(s, q=10)`** / **`bin_series(s, edges)`** — mon `make_bins` :
  des bornes de déciles apprises sur le train, appliquées ailleurs.
  *Compose : numpy.quantile, pandas.cut.*
- **`signed_log1p(s)`** — `signe(x) * log(1+|x|)`, un log qui accepte le négatif
  (pour `balance`). *Compose : numpy.*
- **`prepare(X_train, X_test=None, drop=("id",), as_category=(), scale=False)`** —
  bouche les trous (médiane du train + `"unknown"`), encode le texte avec
  `get_dummies`, aligne le test sur le train avec `reindex`, et met à l'échelle
  au besoin. `as_category=["hour"]` traite une colonne numérique comme du texte
  (le correctif qui vaut -139 → -100 dans le vélo). Renvoie un objet `Prepared`.
  *Compose : pandas, sklearn.preprocessing.StandardScaler.* — c'est mon `encode`
  de séance 3 + le scaling de séance 2.

### Évaluation (`evaluate.py`)

- **`regression_summary(y_true, y_pred, baseline="mean")`** — R2 / RMSE / MAE, la
  baseline « prédis la moyenne » en face, et le compte de prédictions négatives.
  *Compose : sklearn.metrics.* — vient du notebook vélo.
- **`classification_summary(y_true, y_pred)`** — accuracy / précision / rappel /
  F1, avec l'accuracy majoritaire en face et la matrice de confusion. N'imite
  pas `sklearn.classification_report` : il ajoute la baseline et la matrice.
  *Compose : sklearn.metrics, seaborn.* — vient du notebook bank.
- **`plot_roc(y_true, proba)`** — la courbe ROC et son AUC.
  *Compose : sklearn.metrics.* — vient de l'étape 6 du notebook bank.
- **`threshold_sweep(y_true, proba)`** — balaie le seuil, trace le F1 (et
  précision / rappel), renvoie le meilleur seuil. Le mouvement le plus rentable
  sur une cible déséquilibrée. *Compose : sklearn.metrics.* — étape 6 du bank.
- **`write_submission(ids, predictions, binary=False, threshold=0.5, clip=None)`**
  — écrit `submission.csv` et repasse les asserts (une ligne par id, ids
  uniques, pas de `NaN`). `binary` seuille (bank), `clip` borne à 0 (vélo).
  *Compose : pandas, numpy.*

## Exemple de bout en bout

```python
import sys; sys.path.append("..")
import ds_analysis_kit as dsk

# 2-3. lire et regarder
dsk.overview(X_train)
dsk.target_report(y_train)
df = X_train.assign(target=y_train)
dsk.plot_target_rate(df, "age", "target")
dsk.plot_correlation(df)

# 4. mettre en nombres
prep = dsk.prepare(X_train, X_test, as_category=["hour"], scale=True)

# 5-6. modèle, mesure, seuil
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000).fit(prep.X_train, y_train)
proba = model.predict_proba(prep.X_train)[:, 1]
dsk.plot_roc(y_train, proba)
best_t, best_f1, curve = dsk.threshold_sweep(y_train, proba)

# 7. soumission
proba_test = model.predict_proba(prep.X_test)[:, 1]
dsk.write_submission(X_test["id"], proba_test, binary=True, threshold=best_t)
```

## Feuille de route — à ajouter ensemble, pas à pas

Ces idées ne sont **pas** encore dans le kit : on les codera quand j'aurai vu le
concept, pour que je comprenne chaque ligne.

- **Associations non linéaires** : rapport de corrélation (eta) pour cat↔num,
  V de Cramér pour cat↔cat, et une matrice d'association tous types confondus —
  là où `corr()` ignore le texte et ne voit que le linéaire.
- **Importance des features** : information mutuelle, pouvoir prédictif d'une
  colonne seule (détecteur de fuite), importance par permutation (tout modèle),
  coefficients d'un modèle linéaire.
- **Au-delà du linéaire** : gain du binning sur un coefficient unique, détection
  et visualisation des interactions entre colonnes (heatmap + profils).
- **Qualité des colonnes** : paires redondantes automatiques, détection des
  valeurs-sentinelles (`-1` = « jamais contacté » et non une quantité).
- **Robustesse train/test** : PSI de dérive par colonne, validation
  adversariale (le train et le test sont-ils discernables ?).
- **Diagnostics de modèle** : résidus, calibration des probabilités, courbe
  précision-rappel, seuils par coût métier (FP vs FN).
