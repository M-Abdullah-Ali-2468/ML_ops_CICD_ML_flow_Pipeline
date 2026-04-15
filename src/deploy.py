import os
import mlflow
from huggingface_hub import login, upload_file
from mlflow.tracking import MlflowClient
import joblib


 
# MLflow Setup (FIXED)
 
mlflow.set_tracking_uri("sqlite:///mlflow.db")


 
# Hugging Face Login
 
token = os.environ.get("HF_TOKEN")

if token is None:
    raise Exception("HF_TOKEN not found")

login(token=token)


 
# MLflow Client
 
client = MlflowClient()
model_name = "Best_Iris_Model"


 
# Safe Staging Fetch
 
try:
    versions = client.get_latest_versions(model_name, stages=["Staging"])
except Exception:
    print("No model found → skipping deployment")
    exit()

if len(versions) == 0:
    print("No model in Staging → skipping")
    exit()

model_version = versions[0]
print(f"Using Staging model version: {model_version.version}")


 
# Archive old Production
 
for mv in client.search_model_versions(f"name='{model_name}'"):
    if mv.current_stage == "Production":
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Archived"
        )


 
# Promote to Production
 
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production"
)

print("Model promoted to Production")


 
# Load model from MLflow
 
model_uri = f"models:/{model_name}/{model_version.version}"
model = mlflow.sklearn.load_model(model_uri)

joblib.dump(model, "best_model.pkl")


 
# Upload to Hugging Face
 
upload_file(
    path_or_fileobj="best_model.pkl",
    path_in_repo="best_model.pkl",
    repo_id="mabdullahali/iris-model",
    repo_type="model"
)

print("Model uploaded to Hugging Face")