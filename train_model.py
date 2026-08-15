"""
Entrainement du modele de risque de credit.
Reproduit en Python (scikit-learn) le pipeline fait en R/sparklyr :
  imputation -> encodage -> standardisation -> RandomForest

Usage : python train_model.py
Genere : model.pkl (le pipeline complet, pret a l'emploi dans Streamlit)
"""

import pandas as pd
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------
# 1. Chargement des donnees
# ---------------------------------------------------------------
df = pd.read_csv("credit_risk_dataset.csv")

# Nettoyage rapide : quelques ages/anciennetes aberrants (ex: age=144)
# equivalent d'un controle de qualite qu'on ferait aussi en R
df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 60]

# ---------------------------------------------------------------
# 2. Separation features / cible (equivalent de sdf_random_split)
# ---------------------------------------------------------------
target = "loan_status"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

# ---------------------------------------------------------------
# 3. Definition des colonnes
# ---------------------------------------------------------------
num_cols = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
cat_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

# ---------------------------------------------------------------
# 4. Pipeline de pretraitement
#    - imputation mediane pour les numeriques (comme ft_imputer)
#    - standardisation (comme ft_standard_scaler)
#    - one-hot encoding pour les categorielles (comme ft_one_hot_encoder)
# ---------------------------------------------------------------
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", num_transformer, num_cols),
    ("cat", cat_transformer, cat_cols),
])

# ---------------------------------------------------------------
# 5. Pipeline complet : pretraitement + modele
#    (equivalent de votre ml_pipeline() sparklyr)
# ---------------------------------------------------------------
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=123)),
])

# ---------------------------------------------------------------
# 6. Optimisation des hyperparametres (equivalent de ml_cross_validator)
# ---------------------------------------------------------------
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__criterion": ["gini", "entropy"],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)

print("Entrainement en cours (validation croisee, 5 folds)...")
grid_search.fit(X_train, y_train)

print("\nMeilleurs parametres :", grid_search.best_params_)
print("Meilleur AUC (validation croisee) :", grid_search.best_score_)

best_model = grid_search.best_estimator_

# ---------------------------------------------------------------
# 7. Evaluation sur le jeu de test
#    (equivalent de ml_binary_classification_evaluator)
# ---------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Matrice de confusion ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Rapport de classification ---")
print(classification_report(y_test, y_pred))

print("AUC sur le test set :", roc_auc_score(y_test, y_proba))

# ---------------------------------------------------------------
# 8. Sauvegarde du modele pour Streamlit
# ---------------------------------------------------------------
joblib.dump(best_model, "model.pkl")
print("\nModele sauvegarde dans model.pkl")

# ---------------------------------------------------------------
# 9. Sauvegarde des metriques + importance des variables
#    (utilise par l'onglet "Vue d'ensemble" du tableau de bord)
# ---------------------------------------------------------------
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances = best_model.named_steps["classifier"].feature_importances_

# on nettoie les noms (num__person_age -> person_age)
clean_names = [f.split("__", 1)[-1] for f in feature_names]

feat_imp = (
    pd.DataFrame({"variable": clean_names, "importance": importances})
    .sort_values("importance", ascending=False)
    .head(12)
    .to_dict(orient="records")
)

metrics = {
    "best_params": grid_search.best_params_,
    "cv_auc": grid_search.best_score_,
    "test_auc": roc_auc_score(y_test, y_proba),
    "accuracy": (y_pred == y_test).mean(),
    "n_train": len(X_train),
    "n_test": len(X_test),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "feature_importance": feat_imp,
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Metriques sauvegardees dans metrics.json")
