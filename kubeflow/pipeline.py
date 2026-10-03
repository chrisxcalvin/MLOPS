"""Kubeflow Pipelines (KFP v2) definition: collect -> validate -> train -> evaluate -> deploy.
Compile:  python kubeflow/pipeline.py   -> kubeflow/dry_bean_pipeline.yaml (uploadable to any KFP / Vertex AI Pipelines)
Simulate: python kubeflow/run_local.py  -> runs the same components locally."""
from kfp import dsl, compiler

BASE = "python:3.11-slim"
PKGS = ["pandas", "scikit-learn", "mlflow", "joblib"]

@dsl.component(base_image=BASE, packages_to_install=PKGS)
def data_collection(dvc_rev: str, data_url: str, dataset: dsl.Output[dsl.Dataset]):
    import pandas as pd
    # in a real cluster: `dvc get <repo> data/dry_bean.csv --rev <dvc_rev>`; here we read from the URL/remote path
    pd.read_csv(data_url).to_csv(dataset.path, index=False)

@dsl.component(base_image=BASE, packages_to_install=PKGS)
def data_validation(dataset: dsl.Input[dsl.Dataset], validated: dsl.Output[dsl.Dataset]) -> bool:
    import pandas as pd
    df = pd.read_csv(dataset.path)
    assert df.isna().sum().sum() == 0, "missing values found"
    assert "Class" in df.columns and df.Class.nunique() == 7, "unexpected label set"
    assert len(df) > 10000, "too few rows"
    df.to_csv(validated.path, index=False)
    return True

@dsl.component(base_image=BASE, packages_to_install=PKGS)
def train(validated: dsl.Input[dsl.Dataset], model: dsl.Output[dsl.Model], test_set: dsl.Output[dsl.Dataset], C: float = 10.0):
    import pandas as pd, joblib
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    df = pd.read_csv(validated.path)
    Xtr, Xte, ytr, yte = train_test_split(df.drop(columns="Class"), df.Class, test_size=.2, stratify=df.Class, random_state=42)
    joblib.dump(make_pipeline(StandardScaler(), SVC(C=C)).fit(Xtr, ytr), model.path)
    pd.concat([Xte, yte], axis=1).to_csv(test_set.path, index=False)

@dsl.component(base_image=BASE, packages_to_install=PKGS)
def evaluate(model: dsl.Input[dsl.Model], test_set: dsl.Input[dsl.Dataset], metrics: dsl.Output[dsl.Metrics], threshold: float = 0.90) -> bool:
    import pandas as pd, joblib
    from sklearn.metrics import accuracy_score, f1_score
    df = pd.read_csv(test_set.path); m = joblib.load(model.path)
    pred = m.predict(df.drop(columns="Class"))
    acc, f1 = accuracy_score(df.Class, pred), f1_score(df.Class, pred, average="weighted")
    metrics.log_metric("accuracy", acc); metrics.log_metric("f1_weighted", f1)
    return acc >= threshold

@dsl.component(base_image=BASE, packages_to_install=PKGS + ["google-cloud-aiplatform"])
def deploy(model: dsl.Input[dsl.Model], project: str, region: str):
    from google.cloud import aiplatform
    aiplatform.init(project=project, location=region)
    # upload the artifact and expose it behind a managed endpoint (see cloud/deploy_vertex.py)
    m = aiplatform.Model.upload(display_name="dry-bean-svm", artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest")
    m.deploy(machine_type="n1-standard-2", min_replica_count=1, max_replica_count=3)

@dsl.pipeline(name="dry-bean-mlops", description="Collect -> Validate -> Train -> Evaluate -> (gate) Deploy")
def dry_bean_pipeline(data_url: str, dvc_rev: str = "data-v2", project: str = "my-gcp-project", region: str = "us-central1"):
    c = data_collection(dvc_rev=dvc_rev, data_url=data_url)
    v = data_validation(dataset=c.outputs["dataset"])
    t = train(validated=v.outputs["validated"]); t.after(v)
    e = evaluate(model=t.outputs["model"], test_set=t.outputs["test_set"])
    with dsl.If(e.outputs["Output"] == True):  # quality gate: only deploy a model that passes the threshold
        deploy(model=t.outputs["model"], project=project, region=region)

if __name__ == "__main__":
    compiler.Compiler().compile(dry_bean_pipeline, "kubeflow/dry_bean_pipeline.yaml")
    print("compiled -> kubeflow/dry_bean_pipeline.yaml")
