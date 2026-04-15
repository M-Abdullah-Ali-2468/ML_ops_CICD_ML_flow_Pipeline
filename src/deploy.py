import os
import mlflow
from huggingface_hub import login, upload_file
from mlflow.tracking import MlflowClient

# FIX: set tracking URI
mlflow.set_tracking_uri("mlruns")

# login
token = os.environ["HF_TOKEN"]
login(token=token)

client = MlflowClient()
model_name = "Best_Iris_Model"

# get staging model
versions = client.get_latest_versions(model_name, stages=["Staging"])

if len(versions) == 0:
    raise Exception("No Staging model found")

model_version = versions[0]

# promote to production
for mv in client.search_model_versions(f"name='{model_name}'"):
    if mv.current_stage == "Production":
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Archived"
        )

client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

print("Model promoted to Production")

# upload .pkl
model_path = "models/best_model.pkl"

upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.pkl",
    repo_id="mabdullahali/iris-model",
    repo_type="model"
)

print("Model uploaded")