"""
DataMind AI - Machine Learning Tools

End-to-end Auto-ML training, evaluation, and feature importance utilities
powered by scikit-learn:
- Regression (Random Forest, Gradient Boosting, Ridge, Linear Regression)
- Classification (Random Forest, Gradient Boosting, Logistic Regression)
- Unsupervised Clustering (K-Means with Silhouette scoring)
- Automated train/test splits, encoding, and scaling pipelines
- Metric extraction (R2, RMSE, MAE, Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance and model comparison
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.tools.data_loader import get_dataset

logger = logging.getLogger(__name__)


def auto_detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Determine if problem is classification or regression.
    """
    series = df[target_col].dropna()
    n_unique = series.nunique()

    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 10 or (n_unique / len(series) < 0.05 and n_unique <= 20):
            return "classification"
        return "regression"
    else:
        return "classification"


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Create a scikit-learn ColumnTransformer for automatic imputation and encoding.
    """
    numeric_features = []
    categorical_features = []

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


def train_and_evaluate_model(
    dataset_id: str,
    target_column: str,
    feature_columns: list[str] | None = None,
    task_type: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and benchmark models for regression or classification.
    """
    df = get_dataset(dataset_id)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    clean_df = df.dropna(subset=[target_column]).copy()
    if len(clean_df) < 20:
        raise ValueError(f"Need at least 20 samples to train a model. Got {len(clean_df)}.")

    detected_type = task_type or auto_detect_problem_type(clean_df, target_column)

    features = feature_columns or [c for c in clean_df.columns if c != target_column]
    features = [f for f in features if f in clean_df.columns and f != target_column]

    if not features:
        raise ValueError("No feature columns available to train.")

    X = clean_df[features]
    y = clean_df[target_column]

    # Preprocessor
    preprocessor, num_feats, cat_feats = build_preprocessor(clean_df, features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if detected_type == "regression":
        return _train_regression(preprocessor, X_train, X_test, y_train, y_test, features, target_column)
    else:
        return _train_classification(preprocessor, X_train, X_test, y_train, y_test, features, target_column)


def _train_regression(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    target_name: str,
) -> dict[str, Any]:
    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "RidgeRegression": Ridge(alpha=1.0),
    }

    results = []
    best_pipe = None
    best_r2 = -float("inf")
    best_model_name = ""

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        r2 = float(r2_score(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        results.append({
            "model": name,
            "r2_score": round(r2, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
        })

        if r2 > best_r2:
            best_r2 = r2
            best_pipe = pipe
            best_model_name = name

    # Feature Importances from best model if tree-based
    feature_importances = []
    if best_pipe and hasattr(best_pipe.named_steps["model"], "feature_importances_"):
        raw_importances = best_pipe.named_steps["model"].feature_importances_
        try:
            ohe = best_pipe.named_steps["prep"].named_transformers_.get("cat", {}).named_steps.get("encoder")
            feature_names_out = best_pipe.named_steps["prep"].get_feature_names_out()
            feature_names_clean = [f.split("__")[-1] for f in feature_names_out]
        except Exception:
            feature_names_clean = [f"feat_{i}" for i in range(len(raw_importances))]

        for name, imp in zip(feature_names_clean, raw_importances):
            feature_importances.append({"feature": name, "importance": round(float(imp), 4)})
        feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "task_type": "regression",
        "target_column": target_name,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_benchmarks": results,
        "best_model": best_model_name,
        "best_metrics": {
            "r2": round(best_r2, 4),
            "explained_variance_pct": round(max(0.0, best_r2 * 100), 2),
        },
        "top_features": feature_importances[:10],
    }


def _train_classification(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
    target_name: str,
) -> dict[str, Any]:
    models = {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }

    # Ensure discrete labels for classification algorithms
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)

    results = []
    best_pipe = None
    best_f1 = -float("inf")
    best_model_name = ""

    for name, model in models.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = float(accuracy_score(y_test, preds))
        f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))
        prec = float(precision_score(y_test, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y_test, preds, average="weighted", zero_division=0))

        results.append({
            "model": name,
            "accuracy": round(acc, 4),
            "f1_score": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_pipe = pipe
            best_model_name = name

    feature_importances = []
    if best_pipe and hasattr(best_pipe.named_steps["model"], "feature_importances_"):
        raw_importances = best_pipe.named_steps["model"].feature_importances_
        try:
            feature_names_out = best_pipe.named_steps["prep"].get_feature_names_out()
            feature_names_clean = [f.split("__")[-1] for f in feature_names_out]
        except Exception:
            feature_names_clean = [f"feat_{i}" for i in range(len(raw_importances))]

        for name, imp in zip(feature_names_clean, raw_importances):
            feature_importances.append({"feature": name, "importance": round(float(imp), 4)})
        feature_importances.sort(key=lambda x: x["importance"], reverse=True)

    return {
        "task_type": "classification",
        "target_column": target_name,
        "classes": [str(c) for c in np.unique(y_train)],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_benchmarks": results,
        "best_model": best_model_name,
        "best_metrics": {
            "f1_score": round(best_f1, 4),
            "accuracy": results[0]["accuracy"] if results else 0.0,
        },
        "top_features": feature_importances[:10],
    }


def train_clustering(
    dataset_id: str,
    feature_columns: list[str],
    n_clusters: int = 3,
) -> dict[str, Any]:
    """
    Run K-Means clustering on specified numeric features.
    """
    df = get_dataset(dataset_id)
    features = [f for f in feature_columns if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    if len(features) < 2:
        raise ValueError("Clustering requires at least 2 numeric feature columns.")

    clean_df = df[features].dropna()
    if len(clean_df) < n_clusters * 3:
        raise ValueError("Insufficient data rows for clustering.")

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clean_df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_data)

    sil_score = float(silhouette_score(scaled_data, labels)) if len(clean_df) < 10000 else 0.0

    cluster_counts = pd.Series(labels).value_counts().to_dict()
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    centers_formatted = []
    for idx, center in enumerate(cluster_centers):
        centers_formatted.append({
            "cluster_id": idx,
            "size": int(cluster_counts.get(idx, 0)),
            "pct": round(int(cluster_counts.get(idx, 0)) / len(clean_df) * 100, 1),
            "centroid": {features[i]: round(float(center[i]), 3) for i in range(len(features))},
        })

    return {
        "task_type": "clustering",
        "algorithm": "KMeans",
        "features": features,
        "n_clusters": n_clusters,
        "silhouette_score": round(sil_score, 3),
        "clusters": centers_formatted,
    }
