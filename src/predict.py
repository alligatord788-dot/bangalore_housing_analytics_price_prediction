import json
import pickle
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "artifacts" / "model.pkl"
COLUMNS_PATH = PROJECT_DIR / "artifacts" / "columns.json"


def load_artifacts():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(COLUMNS_PATH, "r") as file:
        columns = json.load(file)["columns"]

    return model, columns


def predict_price(location, sqft, bath, bhk):
    model, columns = load_artifacts()

    input_data = {column: 0 for column in columns}
    input_data["total_sqft"] = sqft
    input_data["bath"] = bath
    input_data["bhk"] = bhk

    if location in input_data:
        input_data[location] = 1
    elif "other" in input_data:
        input_data["other"] = 1

    input_df = pd.DataFrame([input_data], columns=columns)
    predicted_price = model.predict(input_df)[0]
    return round(float(predicted_price), 2)


if __name__ == "__main__":
    price = predict_price("1st Phase JP Nagar", sqft=1000, bath=2, bhk=2)
    print(f"Predicted price: {price} lakhs")

