import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load elephant data and prepare binary classification dataset
def load_elephant_data(filepath):
    # Read CSV and filter required columns
    df = pd.read_csv(filepath)
    df = df[['Behaviour', 'Age', 'Sex', 'Distress']].dropna()

    # Convert target to binary
    df = df[df['Distress'].isin(['No distress', 'Distress'])]
    df['Distress'] = df['Distress'].map({'No distress': 0, 'Distress': 1})

    # Return features and target
    X = df[['Behaviour', 'Age', 'Sex']]
    y = df['Distress']
    return X, y

# Evaluate classifier with pipeline
def evaluate_classification_model(X, y, k_list=[5], test_size=0.3, random_state=42):
    # One-hot encode categorical features
    cat_features = ['Behaviour', 'Age', 'Sex']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    results = {}
    for k in k_list:
        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        pipeline.fit(X_train, y_train)

        # Evaluate
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        metrics = {}
        for split, true, pred in [('train', y_train, y_train_pred), ('test', y_test, y_test_pred)]:
            metrics[split] = {
                'confusion_matrix': confusion_matrix(true, pred),
                'precision': precision_score(true, pred),
                'recall': recall_score(true, pred),
                'f1_score': f1_score(true, pred)
            }
        results[k] = metrics
    return results

# Generate synthetic 2D dataset
def generate_synthetic_train(n_samples=20, random_state=42):
    rng = np.random.RandomState(random_state)
    X = rng.uniform(1, 10, size=(n_samples, 2))
    y = np.where(X[:, 0] + X[:, 1] > 11, 1, 0)
    return X, y

# kNN decision boundary plot
def classify_and_plot(X_train, y_train, k=3):
    xx, yy = np.meshgrid(np.arange(0, 10.1, 0.1), np.arange(0, 10.1, 0.1))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    Z = knn.predict(grid_points).reshape(xx.shape)

    # Plot decision boundaries
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], label='Class 0', edgecolor='k')
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], label='Class 1', edgecolor='k')
    plt.title(f'kNN Decision Boundary (k={k})')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.legend()
    plt.show()

# Q6 - Analyze overfitting/generalization
def analyze_knn_behavior_on_synthetic(X, y):
    """
    Q6: Analyze decision boundary complexity by plotting for different k values.
    Higher k smooths decision boundary; lower k may overfit.
    """
    for k in [1, 3, 5, 11]:
        classify_and_plot(X, y, k)
        print(f"Q6 - Decision boundary visualized for k = {k}\n")

# Q7 - Find best k with GridSearchCV
def find_best_k(X, y, k_range=range(1, 21), cv=5):
    cat_features = ['Behaviour', 'Age', 'Sex']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])
    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('knn', KNeighborsClassifier())
    ])
    param_grid = {'knn__n_neighbors': list(k_range)}
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1')
    grid.fit(X, y)
    return grid.best_params_['knn__n_neighbors'], grid.best_score_

# === MAIN EXECUTION ===
if __name__ == '__main__':
    filepath = 'combined_elephant_dataset_10000.csv'
    X, y = load_elephant_data(filepath)

    # Q1
    print("\n--- Q1: Evaluation with k = 5 ---\n")
    result_q1 = evaluate_classification_model(X, y, k_list=[5])
    for split in ['train', 'test']:
        print(f"Q1 - {split.title()} Confusion Matrix:\n{result_q1[5][split]['confusion_matrix']}")
        print(f"Q1 - {split.title()} Precision: {result_q1[5][split]['precision']:.3f}")
        print(f"Q1 - {split.title()} Recall:    {result_q1[5][split]['recall']:.3f}")
        print(f"Q1 - {split.title()} F1-Score:  {result_q1[5][split]['f1_score']:.3f}\n")

    # Q2
    print("\n--- Q2: Evaluation with k = 1 and 11 ---\n")
    result_q2 = evaluate_classification_model(X, y, k_list=[1, 11])
    for k in result_q2:
        for split in ['train', 'test']:
            print(f"Q2 - k={k} - {split.title()} Confusion Matrix:\n{result_q2[k][split]['confusion_matrix']}")
            print(f"Q2 - k={k} - {split.title()} Precision: {result_q2[k][split]['precision']:.3f}")
            print(f"Q2 - k={k} - {split.title()} Recall:    {result_q2[k][split]['recall']:.3f}")
            print(f"Q2 - k={k} - {split.title()} F1-Score:  {result_q2[k][split]['f1_score']:.3f}\n")

    # Q3
    print("\n--- Q3: Synthetic Data Scatter Plot ---\n")
    X_syn, y_syn = generate_synthetic_train()
    plt.figure()
    plt.scatter(X_syn[y_syn==0,0], X_syn[y_syn==0,1], label='Class 0', edgecolor='k')
    plt.scatter(X_syn[y_syn==1,0], X_syn[y_syn==1,1], label='Class 1', edgecolor='k')
    plt.title('Q3: Synthetic Data')
    plt.xlabel('Feature X')
    plt.ylabel('Feature Y')
    plt.legend()
    plt.show()

    # Q4 & Q5
    print("\n--- Q4 & Q5: kNN Decision Boundaries ---\n")
    classify_and_plot(X_syn, y_syn, k=3)
    for k in [1, 5, 11]:
        classify_and_plot(X_syn, y_syn, k)

    # Q6
    print("\n--- Q6: Analysis of Decision Boundary Complexity ---\n")
    analyze_knn_behavior_on_synthetic(X_syn, y_syn)

    # Q7
    print("\n--- Q7: Optimal k using GridSearchCV ---\n")
    best_k, best_score = find_best_k(X, y)
    print(f'Q7 - Best k = {best_k} with F1-score = {best_score:.3f}')
