import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def split_data(input_path, output_dir, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets and saves them 
    to CSV files."""
    # Load the data
    df = pd.read_csv(input_path)

    # Features and target
    X = df.drop(columns=['silica_concentrate', 'date'])
    y = df['silica_concentrate']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/raw_data/raw.csv", help="Path to raw data CSV")
    parser.add_argument("--output", type=str, default="data/processed_data", help="Output directory for split data")
    # parser.add_argument("--test_size", type=float, default=0.2, help="Test set size ratio")
    # parser.add_argument("--random_state", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    split_data(args.input, args.output)#, args.test_size, args.random_state)
