import os
import mlflow
from huggingface_hub import login, upload_file
from mlflow.tracking import MlflowClient


 
# MLflow Setup
 
mlflow.set_tracking_uri("file:./mlruns")


 
# Hugging Face Login
 
token = os.environ.get("HF_TOKEN")

if token is None:
    raise Exception("HF_TOKEN not found in environment variables")

login(token=token)


 
# MLflow Client
 
client = MlflowClient()
model_name = "Best_Iris_Model"


 
# Get Staging Model
 
versions = client.get_latest_versions(model_name, stages=["Staging"])

if len(versions) == 0:
    print("No new model in Staging → skipping deployment")
    exit()

model_version = versions[0]
print(f"Using Staging model version: {model_version.version}")


 
# Move old Production → Archived
 
for mv in client.search_model_versions(f"name='{model_name}'"):
    if mv.current_stage == "Production":
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Archived"
        )


 
# Promote Staging → Production
 
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

print(f"Model version {model_version.version} moved to Production")


 
# Upload Best Model to Hugging Face
 
model_path = "models/best_model.pkl"

if not os.path.exists(model_path):
    raise Exception("best_model.pkl not found. Run training first.")

upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.pkl",
    repo_id="mabdullahali/iris-model",
    repo_type="model"
)

print("Best model uploaded to Hugging Face")