"""Deploy the trained model to a Google Vertex AI endpoint.
NOT executed in this submission (needs a GCP project + billing). Usage:
  gcloud auth application-default login
  python cloud/deploy_vertex.py --project <PROJECT> --bucket gs://<BUCKET>
"""
import argparse, joblib, subprocess
from google.cloud import aiplatform

ap = argparse.ArgumentParser(); ap.add_argument("--project", required=True); ap.add_argument("--bucket", required=True)
ap.add_argument("--region", default="us-central1"); a = ap.parse_args()

# Vertex's prebuilt sklearn container expects a file named model.joblib in the artifact dir
subprocess.run(["gsutil", "cp", "artifacts/model.joblib", f"{a.bucket}/dry-bean/model.joblib"], check=True)
aiplatform.init(project=a.project, location=a.region, staging_bucket=a.bucket)
model = aiplatform.Model.upload(display_name="dry-bean-svm", artifact_uri=f"{a.bucket}/dry-bean/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest")
endpoint = model.deploy(machine_type="n1-standard-2", min_replica_count=1, max_replica_count=3)
print(endpoint.predict(instances=[[28395, 610.291, 208.178, 173.889, 1.197, 0.549, 0.763, 0.9889, 0.9582, 0.0073, 0.0031]]))
# monitoring: enable model-monitoring jobs on the endpoint for feature-skew / drift alerts
