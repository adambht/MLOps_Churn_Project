from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_with_mlflow,
)
import argparse
import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--predict", action="store_true")

    args = parser.parse_args()

    if args.prepare:
        x_train, x_test, y_train, y_test, scaler = prepare_data()
        joblib.dump((x_train, x_test, y_train, y_test, scaler), "prepared_data.joblib")
        print("✅ Data prepared.")

    if args.prepare:
        x_train, x_test, y_train, y_test, scaler = prepare_data()
        joblib.dump((x_train, x_test, y_train, y_test, scaler), "prepared_data.joblib")
        print("✅ Data prepared.")

    if args.train:
        X_train, X_test, y_train, y_test, scaler = joblib.load("prepared_data.joblib")
        model = train_model(X_train, y_train)
        save_model(model)
    if args.evaluate:
        X_train, X_test, y_train, y_test, scaler = joblib.load("prepared_data.joblib")
        model = load_model("models/model.pkl")
        evaluate_model(model, X_test, y_test)

    if args.deploy:
        # Load the trained model
        model = load_model("models/model.pkl")

    if args.predict:  # ✅ Ajout de la prédiction via MLflow
        model_uri = "models:/random_forest_model/latest"
        sample_input = np.array([joblib.load("prepared_data.joblib")[1][36]])
        prediction = predict_with_mlflow(model_uri, sample_input)
        print(f"Sample Prediction: {prediction}")
        print(sample_input)


if __name__ == "__main__":
    main()
