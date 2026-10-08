"""K-means clustering used as an extra feature for the supervised models."""
from sklearn.cluster import KMeans

RANDOM_STATE = 42
# Features chosen from the base Random Forest importance ranking (notebook).
CLUSTER_FEATURES = ["Scaled_Amount", "Scaled_Time", "V4", "V10", "V17", "V14"]
N_CLUSTERS = 2  # chosen from the elbow curve in the notebook


def fit_clusters(X_train):
    """Fit K-means on the (balanced) training data.

    Returns the fitted model and the training cluster labels.
    """
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X_train[CLUSTER_FEATURES])
    return kmeans, labels


def add_cluster_feature(X, kmeans):
    """Return a copy of X with a 'Cluster' column assigned by the fitted K-means."""
    X = X.copy()
    X["Cluster"] = kmeans.predict(X[CLUSTER_FEATURES])
    return X
