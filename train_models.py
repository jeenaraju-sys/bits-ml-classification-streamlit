"""Train and save the five required classification pipelines."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
dataset = load_breast_cancer()
X = pd.DataFrame(dataset.data, columns=[c.replace(" ", "_").replace("/", "_") for c in dataset.feature_names])
y = (dataset.target == 0).astype(int)  # 1 = Malignant, 0 = Benign
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

models = {
    "logistic_regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=3000, random_state=42))]),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "knn": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "naive_bayes": GaussianNB(),
    "svm": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, random_state=42))]),
}
(ROOT / "models").mkdir(exist_ok=True)
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, ROOT / "models" / f"{name}.joblib")
print("Saved all five models in models/.")
