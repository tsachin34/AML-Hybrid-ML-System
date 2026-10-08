"""Train the full AML pipeline and save the fitted models and metrics.

Usage:
    python train.py --data data/creditcard.csv --out models
"""
import argparse
import json
from pathlib import Path

import joblib

from src.clustering import add_cluster_feature, fit_clusters
from src.data import (balance_with_smote, load_transactions, preprocess,
                      split_stratified)
from src.models import (evaluate, train_base_random_forest,
                        train_enhanced_random_forest, train_xgboost)


def run(data_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing data...")
    df = load_transactions(data_path)
    X, y, scaler = preprocess(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_stratified(X, y)

    # Baseline: imbalanced training data, no clustering.
    print("Training base Random Forest...")
    rf_base = train_base_random_forest(X_train, y_train)
    metrics = {"base_random_forest": evaluate(rf_base, X_val, y_val)}

    # Hybrid: SMOTE-balanced data plus K-means cluster feature.
    print("Balancing with SMOTE and clustering...")
    X_train_bal, y_train_bal = balance_with_smote(X_train, y_train)
    X_val_bal, y_val_bal = balance_with_smote(X_val, y_val)
    kmeans, cluster_labels = fit_clusters(X_train_bal)
    X_train_bal = X_train_bal.copy()
    X_train_bal["Cluster"] = cluster_labels
    X_val_bal = add_cluster_feature(X_val_bal, kmeans)

    print("Training enhanced Random Forest...")
    rf_enhanced = train_enhanced_random_forest(X_train_bal, y_train_bal)
    metrics["enhanced_random_forest"] = evaluate(rf_enhanced, X_val_bal, y_val_bal)

    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train_bal, y_train_bal)
    metrics["xgboost"] = evaluate(xgb_model, X_val_bal, y_val_bal)

    print("Saving models and metrics...")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(kmeans, out_dir / "kmeans.joblib")
    joblib.dump(rf_base, out_dir / "rf_base.joblib")
    joblib.dump(rf_enhanced, out_dir / "rf_enhanced.joblib")
    joblib.dump(xgb_model, out_dir / "xgb.joblib")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for name, m in metrics.items():
        print(f"{name}: AUC-ROC={m['auc_roc']:.4f} accuracy={m['accuracy']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/creditcard.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    run(args.data, args.out)
