import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler
import joblib

def normalize_data(input_dir, output_dir, scaler_path=None):
    """Normalizes the feature data using StandardScaler and saves the scaled 
    data."""
    # Load train/test feature data
    X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))

    # Keep only numeric columns
    X_train = X_train.select_dtypes(include=["number"])
    X_test = X_test.select_dtypes(include=["number"])

    # Fit scaler on training data, transform both
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaled data
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        os.path.join(output_dir, "X_train_scaled.csv"),
        index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
        os.path.join(output_dir, "X_test_scaled.csv"),
        index=False
    )

    # # Save scaler for inference
    # joblib.dump(scaler, scaler_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed_data")
    parser.add_argument("--output", type=str, default="data/processed_data")
    # parser.add_argument("--scaler", type=str, default="models/scaler.pkl")
    args = parser.parse_args()

    normalize_data(args.input, args.output)#, args.scaler)
