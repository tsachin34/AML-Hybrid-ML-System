"""Model definitions and evaluation, with the hyperparameters used in the notebook."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from xgboost import XGBClassifier

RANDOM_STATE = 42


def train_base_random_forest(X, y):
    """Baseline Random Forest on the imbalanced data, without clustering."""
    model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced")
    return model.fit(X, y)


def train_enhanced_random_forest(X, y):
    """Random Forest on balanced data with the cluster feature."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model.fit(X, y)


def train_xgboost(X, y):
    """XGBoost on balanced data with the cluster feature."""
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        scale_pos_weight=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model.fit(X, y)


def evaluate(model, X, y):
    """Accuracy, AUC-ROC, confusion matrix and per-class report for a fitted model."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "auc_roc": float(roc_auc_score(y, y_proba)),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, output_dict=True),
    }
