import pandas as pd
import joblib
import json
import os
import argparse
from sklearn.metrics import mean_squared_error, r2_score

def main(model_path, X_test_path, y_test_path, metrics_path, prediction_path):
    # Load data and model
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save predictions
    preds_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
    preds_df.to_csv(prediction_path, index=False)

    # Save metrics
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump({"mse": mse, "r2": r2}, f, indent=4)

    print(f"Evaluation done. MSE: {mse:.4f}, R²: {r2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/trained_model.joblib")
    parser.add_argument("--X", type=str, default="data/processed_data/X_test_scaled.csv")
    parser.add_argument("--y", type=str, default="data/processed_data/y_test.csv")
    parser.add_argument("--metrics", type=str, default="metrics/scores.json")
    parser.add_argument("--predictions", type=str, default="data/predictions.csv")
    args = parser.parse_args()

    main(args.model, args.X, args.y, args.metrics, args.predictions)
