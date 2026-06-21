from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "Bengaluru_House_Data.csv"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed" / "cleaned_bengaluru_house_data.csv"


def convert_sqft_to_number(value):
    """Convert values like '1200', '1200 - 1500' into one numeric sqft value."""
    value = str(value).strip()
    parts = value.split("-")

    if len(parts) == 2:
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return None

    try:
        return float(value)
    except ValueError:
        return None


def remove_price_per_sqft_outliers(df):
    """Remove very cheap or very costly homes inside each location group."""
    cleaned_parts = []

    for _, location_df in df.groupby("location"):
        mean_price = location_df.price_per_sqft.mean()
        std_price = location_df.price_per_sqft.std()
        filtered = location_df[
            (location_df.price_per_sqft > mean_price - std_price)
            & (location_df.price_per_sqft <= mean_price + std_price)
        ]
        cleaned_parts.append(filtered)

    return pd.concat(cleaned_parts, ignore_index=True)


def remove_bhk_outliers(df):
    """Remove cases where a larger BHK is cheaper per sqft than a smaller BHK nearby."""
    rows_to_remove = []

    for _, location_df in df.groupby("location"):
        bhk_stats = {}

        for bhk, bhk_df in location_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": bhk_df.price_per_sqft.mean(),
                "count": bhk_df.shape[0],
            }

        for bhk, bhk_df in location_df.groupby("bhk"):
            previous_bhk = bhk_stats.get(bhk - 1)

            if previous_bhk and previous_bhk["count"] > 5:
                smaller_than_previous = bhk_df[
                    bhk_df.price_per_sqft < previous_bhk["mean"]
                ]
                rows_to_remove.extend(smaller_than_previous.index.tolist())

    return df.drop(rows_to_remove)


def clean_data(raw_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    """Clean the raw Bangalore housing dataset and save a model-ready CSV."""
    df = pd.read_csv(raw_path)
    print(f"Raw data shape: {df.shape}")

    df = df.drop(["area_type", "availability", "society", "balcony"], axis=1)
    df = df.dropna()

    df["bhk"] = df["size"].apply(lambda x: int(str(x).split()[0]))
    df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_number)
    df = df.dropna(subset=["total_sqft"])

    df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft"]
    df["location"] = df["location"].apply(lambda x: str(x).strip())

    location_counts = df["location"].value_counts()
    rare_locations = location_counts[location_counts <= 10].index
    df["location"] = df["location"].apply(
        lambda location: "other" if location in rare_locations else location
    )

    df = df[df["total_sqft"] / df["bhk"] >= 300]
    df = remove_price_per_sqft_outliers(df)
    df = remove_bhk_outliers(df)

    df = df.drop(["size", "price_per_sqft"], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Cleaned data shape: {df.shape}")
    print(f"Cleaned data saved to: {output_path}")
    return df


if __name__ == "__main__":
    clean_data()
