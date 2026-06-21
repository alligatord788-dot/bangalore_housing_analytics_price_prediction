import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from data_cleaning import PROCESSED_DATA_PATH, clean_data


PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pkl"
COLUMNS_PATH = ARTIFACTS_DIR / "columns.json"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"


def load_clean_data():
    if not PROCESSED_DATA_PATH.exists():
        clean_data()
    return pd.read_csv(PROCESSED_DATA_PATH)


def prepare_features(df):
    location_dummies = pd.get_dummies(df["location"], dtype=int)
    X = pd.concat([df.drop(["location", "price"], axis=1), location_dummies], axis=1)
    y = df["price"]
    return X, y


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
    }


def train_models():
    df = load_clean_data()
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)
        trained_models[name] = model

    best_model_name = max(results, key=lambda name: results[name]["r2_score"])
    best_model = trained_models[best_model_name]

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(best_model, file)

    with open(COLUMNS_PATH, "w") as file:
        json.dump({"columns": X.columns.tolist()}, file, indent=2)

    with open(METRICS_PATH, "w") as file:
        json.dump(
            {"best_model": best_model_name, "model_results": results},
            file,
            indent=2,
        )

    print("Model comparison:")
    for name, metrics in results.items():
        print(f"{name}: {metrics}")

    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Columns saved to: {COLUMNS_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    train_models()

