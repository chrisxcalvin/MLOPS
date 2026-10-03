"""Train KNN / SVM / RandomForest on a given dataset version and track everything in MLflow."""
import sys, hashlib, mlflow, mlflow.sklearn, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, ConfusionMatrixDisplay

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("dry-bean-classification")

MODELS = {
    "KNN": (lambda p: make_pipeline(StandardScaler(), KNeighborsClassifier(**p)),
            [{"n_neighbors": 5}, {"n_neighbors": 15, "weights": "distance"}]),
    "SVM": (lambda p: make_pipeline(StandardScaler(), SVC(**p)),
            [{"C": 1.0, "kernel": "rbf"}, {"C": 10.0, "kernel": "rbf"}]),
    "RandomForest": (lambda p: RandomForestClassifier(random_state=42, n_jobs=-1, **p),
                     [{"n_estimators": 100, "max_depth": 10}, {"n_estimators": 300, "max_depth": None}]),
}

def run(path, version):
    df = pd.read_csv(path)
    X, y = df.drop(columns="Class"), df["Class"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    for name, (build, grids) in MODELS.items():
        for params in grids:
            with mlflow.start_run(run_name=f"{name}-{version}"):
                mlflow.set_tags({"data_version": version, "model_family": name})
                mlflow.log_params({"model": name, **params, "data_version": version, "data_md5": md5,
                                   "n_rows": len(df), "n_features": X.shape[1], "test_size": .2})
                m = build(params).fit(Xtr, ytr)
                ptr, pte = m.predict(Xtr), m.predict(Xte)
                p, r, f, _ = precision_recall_fscore_support(yte, pte, average="weighted", zero_division=0)
                mlflow.log_metrics({"train_accuracy": accuracy_score(ytr, ptr), "test_accuracy": accuracy_score(yte, pte),
                                    "precision": p, "recall": r, "f1_score": f})
                fig, ax = plt.subplots(figsize=(7, 6))
                ConfusionMatrixDisplay.from_predictions(yte, pte, ax=ax, xticks_rotation=45, colorbar=False)
                fig.tight_layout(); mlflow.log_figure(fig, "confusion_matrix.png"); plt.close(fig)
                mlflow.sklearn.log_model(m, "model", serialization_format="cloudpickle")
                print(f"{version:3} {name:13} {params} test_acc={accuracy_score(yte, pte):.4f} f1={f:.4f}")

if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
