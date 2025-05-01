import pandas as pd
import os
import argparse
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def load_data(X_path, y_path):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    return X, y

def main(X_path, y_path, output_path):
    X, y = load_data(X_path, y_path)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [None, 5, 10]
    }

    grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, scoring="neg_mean_squared_error", cv=5)
    grid.fit(X, y)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_score = -grid.best_score_

    os.makedirs(output_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(output_path, "best_model.pkl"))
    joblib.dump(best_params, os.path.join(output_path, "best_params.pkl"))

    with open(os.path.join(output_path, "best_model.txt"), "w") as f:
        f.write(f"RandomForest: {best_params}, MSE={best_score:.4f}")

    print(f"Best RF model: {best_params}, MSE={best_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", type=str, default="data/processed_data/X_train_scaled.csv")
    parser.add_argument("--y", type=str, default="data/processed_data/y_train.csv")
    parser.add_argument("--output", type=str, default="models/models")
    args = parser.parse_args()

    main(args.X, args.y, args.output)
