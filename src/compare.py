import mlflow, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
mlflow.set_tracking_uri("sqlite:///mlflow.db")
df = mlflow.search_runs(experiment_names=["dry-bean-classification"])
cols = ["tags.model_family","tags.data_version","params.n_neighbors","params.C","params.n_estimators","params.max_depth",
        "metrics.train_accuracy","metrics.test_accuracy","metrics.precision","metrics.recall","metrics.f1_score"]
t = df[cols].rename(columns=lambda c: c.split(".")[-1]).sort_values(["data_version","model_family","test_accuracy"])
t.to_csv("report/mlflow_comparison.csv", index=False); print(t.round(4).to_string(index=False))
best = t.loc[t.groupby(["model_family","data_version"]).test_accuracy.idxmax()]
fig, ax = plt.subplots(figsize=(8,4.5)); w=.35
fams = ["KNN","SVM","RandomForest"]
for i,v in enumerate(["v1","v2"]):
    vals=[best[(best.model_family==f)&(best.data_version==v)].test_accuracy.iloc[0] for f in fams]
    b=ax.bar([x+i*w for x in range(3)], vals, w, label=f"data {v}")
    ax.bar_label(b, fmt="%.3f", fontsize=8)
ax.set_xticks([x+w/2 for x in range(3)], fams); ax.set_ylim(.88,.94); ax.set_ylabel("Best test accuracy"); ax.legend()
ax.set_title("Dry Bean: best test accuracy per model and dataset version"); fig.tight_layout(); fig.savefig("report/img/model_comparison.png", dpi=150)
