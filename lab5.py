import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_percentage_error,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
# Load & Preprocess Data
def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.drop(columns=['filename'])  # remove filename column
    # Encode categorical columns
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    df = df.dropna()
    return df, label_encoders
# Regression Evaluation
def evaluate_regression(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAPE": mape, "R2": r2}

# Clustering Evaluation
def evaluate_clustering(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    return {"Silhouette": sil, "CH Score": ch, "DB Index": db}
# Main Execution (A1–A7)
if __name__ == "__main__":
    file_path = "combined_elephant_dataset_10000.csv"
    df, encoders = load_and_preprocess(file_path)

    # Choose target column (numeric)
    target_col = 'low_freq'
    feature_cols = [col for col in df.columns if col != target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], df[target_col], test_size=0.2, random_state=42
    )

    # A1: Regression with single feature
    single_feature = [feature_cols[0]]
    reg_single = LinearRegression().fit(X_train[single_feature], y_train)
    train_metrics_single = evaluate_regression(reg_single, X_train[single_feature], y_train)
    test_metrics_single = evaluate_regression(reg_single, X_test[single_feature], y_test)

    print("\nA1: Regression with Single Feature")
    print("Train:", train_metrics_single)
    print("Test:", test_metrics_single)

    # A3: Regression with all features
    reg_all = LinearRegression().fit(X_train, y_train)
    train_metrics_all = evaluate_regression(reg_all, X_train, y_train)
    test_metrics_all = evaluate_regression(reg_all, X_test, y_test)

    print("\nA3: Regression with All Features")
    print("Train:", train_metrics_all)
    print("Test:", test_metrics_all)

    # A4: KMeans Clustering (k=2)
    X_cluster = df[feature_cols]
    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_cluster)
    print("\nA4: KMeans Labels:", kmeans.labels_)
    print("Cluster Centers:\n", kmeans.cluster_centers_)

    # A5: Clustering Scores for k=2
    clustering_scores = evaluate_clustering(X_cluster, kmeans.labels_)
    print("\nA5: Clustering Scores (k=2):", clustering_scores)

    # A6: Clustering for different k values
    print("\nA6: Scores for different k values")
    for k in range(2, 6):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_cluster)
        scores = evaluate_clustering(X_cluster, km.labels_)
        print(f"k={k} -> {scores}")

    # A7: Elbow Plot
    distortions = []
    K_range = range(2, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_cluster)
        distortions.append(km.inertia_)
    plt.plot(K_range, distortions, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    plt.show()
