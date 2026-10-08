"""Loading, scaling, splitting and balancing the credit card transaction data."""
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TARGET = "Class"
RAW_SCALED_COLS = ["Amount", "Time"]
SCALED_COLS = ["Scaled_Amount", "Scaled_Time"]


def load_transactions(path):
    """Read the Kaggle credit card fraud CSV."""
    return pd.read_csv(path)


def preprocess(df):
    """Scale Amount and Time, then split features from the target.

    Returns the feature matrix X, target y, and the fitted scaler (needed to
    preprocess new transactions at prediction time).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[RAW_SCALED_COLS])
    scaled_df = pd.DataFrame(scaled, columns=SCALED_COLS, index=df.index)

    prepared = pd.concat([df.drop(columns=RAW_SCALED_COLS), scaled_df], axis=1)
    X = prepared.drop(columns=[TARGET])
    y = prepared[TARGET]
    return X, y, scaler


def preprocess_new(df, scaler):
    """Apply an already-fitted scaler to new transactions (no refitting)."""
    scaled = scaler.transform(df[RAW_SCALED_COLS])
    scaled_df = pd.DataFrame(scaled, columns=SCALED_COLS, index=df.index)
    return pd.concat([df.drop(columns=RAW_SCALED_COLS + [TARGET], errors="ignore"),
                      scaled_df], axis=1)


def split_stratified(X, y):
    """60% train, 20% validation, 20% test, stratified on the class label."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def balance_with_smote(X, y):
    """Oversample the minority (fraud) class with SMOTE."""
    return SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
