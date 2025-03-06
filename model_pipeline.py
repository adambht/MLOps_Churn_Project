import mlflow.sklearn
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import mlflow.pyfunc
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5003")


def prepare_data(data_path="churn_combined.csv"):
    mlflow.set_tag("stage", "Data Preparation")
    df = pd.read_csv(data_path)
    df = df.drop(["Area code", "State"], axis=1)
    for col in ["International plan", "Voice mail plan"]:
        df[col] = df[col].map({"Yes": 1, "No": 0})
    df["Churn"] = df["Churn"].astype(int)

    imputer = SimpleImputer(strategy="mean")
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    y = df_imputed["Churn"]
    X = df_imputed.drop(["Churn"], axis=1)
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, "feature_names.joblib")
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=1
    )

    smote_enn = SMOTEENN(random_state=42)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(x_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "scaler.joblib")

    return X_train_scaled, X_test_scaled, y_train_resampled, y_test, scaler


def train_model(x_train, y_train):
    with mlflow.start_run():
        mlflow.set_tag("stage", "Data Model Training")

        model = RandomForestClassifier(n_estimators=100, random_state=1)
        model.fit(x_train, y_train)

        # Log model parameters
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)

        # Save model to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/random_forest_model"
        mlflow.register_model(model_uri, "random_forest_model")
        mlflow.log_artifact("requirements.txt")
        return model


def evaluate_model(model, x_test, y_test):
    # Ensure the run is active
    with mlflow.start_run():
        mlflow.set_tag("stage", "Model Evaluation")

        # Make predictions
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1]  # For ROC AUC

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Log all metrics in MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual Label")
        plt.xlabel("Predicted Label")

        # Save confusion matrix as figure
        conf_matrix_path = "confusion_matrix.png"
        plt.tight_layout()
        plt.savefig(conf_matrix_path)
        plt.close()

        # Log the figure in MLflow
        mlflow.log_artifact(conf_matrix_path)

        # Generate classification report and log as text
        report = classification_report(y_test, y_pred)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")

        # Print summary
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)

        return accuracy


def save_model(model, model_path="models/model.pkl"):
    os.makedirs(
        os.path.dirname(model_path), exist_ok=True
    )  # Crée le dossier s'il n'existe pas
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)
    print("Model saved successfully.")


def load_model(file_path):
    return joblib.load(file_path)


def predict_with_mlflow(model_uri, input_data):
    mlflow.set_tag("stage", "Model Predict")
    model = mlflow.pyfunc.load_model(model_uri)
    input_df = pd.DataFrame(input_data)
    predictions = model.predict(input_df)
    return predictions
