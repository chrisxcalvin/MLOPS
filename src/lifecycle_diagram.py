import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
st=[("1 Data\nIngestion","UCI Dry Bean CSV\n13,611 x 17\nDVC data-v1"),("2 Data\nPreprocessing","dedupe, outlier clip,\ndrop collinear\nDVC data-v2"),
("3 Model\nTraining","KNN / SVM / RF\nMLflow runs"),("4 Model\nEvaluation","acc, P/R/F1,\nconfusion matrix\nMLflow compare"),
("5 Deployment","MLflow model ->\nVertex AI endpoint\n(Kubeflow pipeline)"),("6 Monitoring","drift, accuracy,\nlatency; retrain\ntrigger")]
fig,ax=plt.subplots(figsize=(15,4.6)); ax.axis("off"); ax.set_xlim(0,15); ax.set_ylim(0,5)
cols=["#4C78A8","#59A14F","#F28E2B","#E15759","#76B7B2","#B07AA1"]
for i,(a,b) in enumerate(st):
    x=.3+i*2.45
    ax.add_patch(FancyBboxPatch((x,2.4),2.0,1.5,boxstyle="round,pad=.05",fc=cols[i],ec="none"))
    ax.text(x+1,3.15,a,ha="center",va="center",color="white",fontsize=12,fontweight="bold")
    ax.text(x+1,1.6,b,ha="center",va="center",fontsize=9)
    if i<5: ax.annotate("",(x+2.4,3.15),(x+2.07,3.15),arrowprops=dict(arrowstyle="->",lw=2))
ax.annotate("",(1.3,2.25),(13.2,2.25),arrowprops=dict(arrowstyle="->",lw=1.5,ls="--",color="gray",connectionstyle="arc3,rad=-0.0"))
ax.text(7.5,.55,"Feedback loop: monitoring detects drift / decay -> new data version -> retrain (automated by Kubeflow Pipeline)",ha="center",fontsize=10,style="italic")
ax.set_title("End-to-end MLOps lifecycle - Dry Bean classification",fontsize=14,fontweight="bold")
fig.savefig("report/img/lifecycle.png",dpi=150,bbox_inches="tight")
