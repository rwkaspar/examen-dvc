import pandas as pd
import joblib
import argparse
import os
from sklearn.ensemble import RandomForestRegressor

def main(X_path, y_path, param_path, output_model_path):
    # Load training data
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()

    # Load best hyperparameters
    best_params = joblib.load(param_path)

    # Train final model
    model = RandomForestRegressor(**best_params)
    model.fit(X, y)

    # Save model
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(model, output_model_path)
    print(f"Model trained and saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", type=str, default="data/processed_data/X_train_scaled.csv")
    parser.add_argument("--y", type=str, default="data/processed_data/y_train.csv")
    parser.add_argument("--params", type=str, default="models/best_params.pkl")
    parser.add_argument("--output", type=str, default="models/trained_model.joblib")
    args = parser.parse_args()

    main(args.X, args.y, args.params, args.output)
