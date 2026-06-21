from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "Bengaluru_House_Data.csv"


def explore_data():
    df = pd.read_csv(RAW_DATA_PATH)

    print("First 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nBasic information:")
    print(df.info())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nNumerical summary:")
    print(df.describe())


if __name__ == "__main__":
    explore_data()

