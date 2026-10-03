"""Simulated workflow: executes the same 5 stages locally in order (what the KFP DAG does on a cluster)."""
import sys, tempfile, os, time
sys.path.insert(0, ".")
from types import SimpleNamespace as NS
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

path = sys.argv[1] if len(sys.argv) > 1 else "versions/dry_bean_v2.csv"
def stage(n, name): print(f"[{time.strftime('%H:%M:%S')}] stage {n}/5 {name}")
stage(1, "data_collection"); df = pd.read_csv(path); print("  rows:", len(df))
stage(2, "data_validation")
assert df.isna().sum().sum() == 0 and df.Class.nunique() == 7 and len(df) > 10000; print("  checks passed")
stage(3, "train")
Xtr, Xte, ytr, yte = train_test_split(df.drop(columns="Class"), df.Class, test_size=.2, stratify=df.Class, random_state=42)
m = make_pipeline(StandardScaler(), SVC(C=10)).fit(Xtr, ytr); os.makedirs("artifacts", exist_ok=True); joblib.dump(m, "artifacts/model.joblib")
stage(4, "evaluate"); p = m.predict(Xte); acc = accuracy_score(yte, p); print(f"  accuracy={acc:.4f} f1={f1_score(yte,p,average='weighted'):.4f}")
stage(5, "deploy")
print("  gate passed -> model promoted (artifacts/model.joblib); on GCP: cloud/deploy_vertex.py" if acc >= .90 else "  gate FAILED -> deployment skipped")
