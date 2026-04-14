import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Iris_Classification")


X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


best_score = 0
best_run_id = None
best_model_name = ""
best_params = {}


# Logistic Regression runs
for C in [0.1, 1, 10]:
    with mlflow.start_run(run_name=f"Logistic_C_{C}") as run:

        print(f"Training Logistic Regression with C={C}")

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

        print(f"Results -> Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}\n")

        if f1 > best_score:
            best_score = f1
            best_run_id = run.info.run_id
            best_model_name = "LogisticRegression"
            best_params = {"C": C}


# Random Forest runs
for n in [50, 100, 200]:
    with mlflow.start_run(run_name=f"RF_{n}_trees") as run:

        print(f"Training Random Forest with n_estimators={n}")

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

        print(f"Results -> Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}\n")

        if f1 > best_score:
            best_score = f1
            best_run_id = run.info.run_id
            best_model_name = "RandomForest"
            best_params = {"n_estimators": n}


# Best model summary
print("Best Model Summary:")
print(f"Model: {best_model_name}")
print(f"Parameters: {best_params}")
print(f"Best F1 Score: {best_score}")


# Register best model
model_uri = f"runs:/{best_run_id}/model"

mlflow.register_model(
    model_uri=model_uri,
    name="Best_Iris_Model"
)


# Assign stage
client = MlflowClient()

latest_version = client.get_latest_versions("Best_Iris_Model")[0].version

client.transition_model_version_stage(
    name="Best_Iris_Model",
    version=latest_version,
    stage="Staging"
)


print("Model registered and moved to Staging")