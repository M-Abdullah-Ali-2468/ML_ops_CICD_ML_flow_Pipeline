import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import os
import joblib


# MLflow Setup

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris_Classification")


# Load Data
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



# Preprocessing

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Best Model Tracking
best_score = 0
best_run_id = None
best_model = None



# Logistic Regression Runs
for C in [0.1, 1, 10]:
    with mlflow.start_run(run_name=f"Logistic_C_{C}") as run:

        model = LogisticRegression(C=C, max_iter=200)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"Logistic C={C} → F1: {f1}")

        if f1 > best_score:
            best_score = f1
            best_run_id = run.info.run_id
            best_model = model


# Random Forest Runs

for n in [50, 100, 200]:
    with mlflow.start_run(run_name=f"RF_{n}_trees") as run:

        model = RandomForestClassifier(n_estimators=n, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        f1 = f1_score(y_test, y_pred, average="macro")

        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", n)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, "model")

        print(f"RF n={n} → F1: {f1}")

        if f1 > best_score:
            best_score = f1
            best_run_id = run.info.run_id
            best_model = model


# Save Best Model (.pkl)
save_path = r"C:\Users\M Abdullah Ali\Documents\Learning Material\MlOps\ML_ops_CICD_ML_flow_Pipeline\models"

os.makedirs(save_path, exist_ok=True)

model_file = os.path.join(save_path, "best_model.pkl")

joblib.dump(best_model, model_file)

print(f"Best model saved at: {model_file}")

# log artifact
mlflow.log_artifact(model_file)



# Register Model
client = MlflowClient()

model_name = "Best_Iris_Model"
model_uri = f"runs:/{best_run_id}/model"

result = mlflow.register_model(model_uri=model_uri, name=model_name)
new_version = result.version


# Maintain ONLY ONE STAGING MODEL
for mv in client.search_model_versions(f"name='{model_name}'"):
    if mv.current_stage == "Staging":
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Archived"
        )


# move new model to Staging
client.transition_model_version_stage(
    name=model_name,
    version=new_version,
    stage="Staging"
)

print(f"Model version {new_version} is now in Staging")