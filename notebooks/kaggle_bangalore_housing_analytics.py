# %% [markdown]
# # Bangalore Housing Analytics & Price Prediction
#
# This notebook builds a simple CampusX-style machine learning project:
#
# - Load Bangalore housing data
# - Clean messy columns
# - Create useful features
# - Remove outliers
# - Train regression models
# - Compare model performance
# - Save the best model
# - Make price predictions from user input

# %%
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# %% [markdown]
# ## 1. Find And Load Dataset
#
# In Kaggle, datasets usually appear inside `/kaggle/input/`.
# This cell searches for the CSV automatically.
# If it fails, manually set `DATA_PATH` to your dataset path.

# %%
possible_paths = list(Path("/kaggle/input").rglob("*.csv"))

print("CSV files found:")
for path in possible_paths:
    print(path)

DATA_PATH = None
for path in possible_paths:
    if "bengaluru" in path.name.lower() or "bangalore" in path.name.lower():
        DATA_PATH = path
        break

if DATA_PATH is None and possible_paths:
    DATA_PATH = possible_paths[0]

print("\nUsing dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
df.head()


# %% [markdown]
# ## 2. Basic Data Exploration
#
# These commands help us understand the dataset before cleaning.

# %%
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()


# %%
print("Missing values:")
print(df.isna().sum())

print("\nNumerical summary:")
df.describe()


# %% [markdown]
# ## 3. Drop Less Useful Columns
#
# For this beginner project, we keep only the columns directly useful for price prediction.
#
# We drop:
# - `area_type`
# - `availability`
# - `society`
# - `balcony`
#
# This keeps the project simple and interview-friendly.

# %%
df1 = df.drop(["area_type", "availability", "society", "balcony"], axis=1)
df1.head()


# %% [markdown]
# ## 4. Remove Missing Values

# %%
print("Before dropping missing values:", df1.shape)
df2 = df1.dropna()
print("After dropping missing values:", df2.shape)
print(df2.isna().sum())


# %% [markdown]
# ## 5. Convert Size Column Into BHK
#
# The `size` column contains values like:
#
# - `2 BHK`
# - `4 Bedroom`
#
# We extract only the first number and store it in a new column called `bhk`.

# %%
df2 = df2.copy()
df2["bhk"] = df2["size"].apply(lambda x: int(str(x).split()[0]))
df2[["size", "bhk"]].head()


# %% [markdown]
# ## 6. Clean Total Square Feet
#
# Some values in `total_sqft` are simple numbers:
#
# - `1200`
#
# Some are ranges:
#
# - `1200 - 1500`
#
# Some contain units and cannot be directly converted.
#
# We convert ranges into their average and remove invalid values.

# %%
def convert_sqft_to_number(value):
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


# %%
df3 = df2.copy()
df3["total_sqft"] = df3["total_sqft"].apply(convert_sqft_to_number)
df3 = df3.dropna(subset=["total_sqft"])

print("Shape after sqft cleaning:", df3.shape)
df3.head()


# %% [markdown]
# ## 7. Create Price Per Square Feet
#
# The dataset price is in lakhs.
#
# So:
#
# `price_per_sqft = price * 100000 / total_sqft`
#
# This feature is mainly used to detect outliers.

# %%
df4 = df3.copy()
df4["price_per_sqft"] = (df4["price"] * 100000) / df4["total_sqft"]
df4.head()


# %% [markdown]
# ## 8. Handle Location Column
#
# There are many locations.
# If we one-hot encode all rare locations, the model becomes unnecessarily large.
#
# So locations appearing 10 or fewer times are grouped as `other`.

# %%
df4["location"] = df4["location"].apply(lambda x: str(x).strip())

location_counts = df4["location"].value_counts()
print("Number of unique locations before grouping:", len(location_counts))

rare_locations = location_counts[location_counts <= 10].index
df4["location"] = df4["location"].apply(
    lambda location: "other" if location in rare_locations else location
)

print("Number of unique locations after grouping:", df4["location"].nunique())


# %% [markdown]
# ## 9. Remove Unrealistic Sqft Per BHK Values
#
# A very small square-foot value for a high BHK count is usually unrealistic.
#
# We remove rows where:
#
# `total_sqft / bhk < 300`

# %%
print("Before removing sqft/BHK outliers:", df4.shape)

df5 = df4[df4["total_sqft"] / df4["bhk"] >= 300]

print("After removing sqft/BHK outliers:", df5.shape)


# %% [markdown]
# ## 10. Remove Price Per Sqft Outliers
#
# For each location, we remove houses whose price per sqft is far from that location's average.
#
# We keep rows within:
#
# `mean - std` to `mean + std`

# %%
def remove_price_per_sqft_outliers(data):
    cleaned_parts = []

    for _, location_df in data.groupby("location"):
        mean_price = location_df.price_per_sqft.mean()
        std_price = location_df.price_per_sqft.std()

        filtered = location_df[
            (location_df.price_per_sqft > mean_price - std_price)
            & (location_df.price_per_sqft <= mean_price + std_price)
        ]
        cleaned_parts.append(filtered)

    return pd.concat(cleaned_parts, ignore_index=True)


df6 = remove_price_per_sqft_outliers(df5)
print("After price_per_sqft outlier removal:", df6.shape)


# %% [markdown]
# ## 11. Remove BHK Outliers
#
# Sometimes in the same location, a 3 BHK has lower price per sqft than a 2 BHK.
# Some cases may be valid, but many are noisy records.
#
# We remove a BHK group if its price per sqft is lower than the previous smaller BHK average,
# but only when the previous BHK group has enough examples.

# %%
def remove_bhk_outliers(data):
    rows_to_remove = []

    for _, location_df in data.groupby("location"):
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

    return data.drop(rows_to_remove)


df7 = remove_bhk_outliers(df6)
print("After BHK outlier removal:", df7.shape)


# %% [markdown]
# ## 12. Create Final Clean Dataset
#
# We remove columns that were useful for cleaning but are not needed by the final model.

# %%
df8 = df7.drop(["size", "price_per_sqft"], axis=1)
df8.head()


# %%
print("Final cleaned shape:", df8.shape)
print(df8.columns.tolist())


# %% [markdown]
# ## 13. One-Hot Encode Location
#
# ML models cannot directly understand text locations.
# We convert location into numeric columns using one-hot encoding.

# %%
location_dummies = pd.get_dummies(df8["location"], dtype=int)

X = pd.concat([df8.drop(["location", "price"], axis=1), location_dummies], axis=1)
y = df8["price"]

print("X shape:", X.shape)
print("y shape:", y.shape)
X.head()


# %% [markdown]
# ## 14. Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# %% [markdown]
# ## 15. Train And Compare Models
#
# We compare:
#
# - Linear Regression
# - Lasso Regression
# - Decision Tree Regressor

# %%
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return {
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
    }


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

results


# %%
results_df = pd.DataFrame(results).T.sort_values("r2_score", ascending=False)
results_df


# %% [markdown]
# ## 16. Select Best Model

# %%
best_model_name = max(results, key=lambda name: results[name]["r2_score"])
best_model = trained_models[best_model_name]

print("Best model:", best_model_name)
print("Best model metrics:", results[best_model_name])


# %% [markdown]
# ## 17. Save Model And Columns
#
# In Kaggle, `/kaggle/working/` is the output folder.
# Files saved here can be downloaded after notebook execution.

# %%
OUTPUT_DIR = Path("/kaggle/working/house_price_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "model.pkl"
COLUMNS_PATH = OUTPUT_DIR / "columns.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CLEANED_DATA_PATH = OUTPUT_DIR / "cleaned_bengaluru_house_data.csv"

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

df8.to_csv(CLEANED_DATA_PATH, index=False)

print("Saved files:")
print(MODEL_PATH)
print(COLUMNS_PATH)
print(METRICS_PATH)
print(CLEANED_DATA_PATH)


# %% [markdown]
# ## 18. Create Prediction Function
#
# This function takes user input and returns predicted price in lakhs.

# %%
def predict_price(location, sqft, bath, bhk):
    input_data = {column: 0 for column in X.columns}
    input_data["total_sqft"] = sqft
    input_data["bath"] = bath
    input_data["bhk"] = bhk

    if location in input_data:
        input_data[location] = 1
    elif "other" in input_data:
        input_data["other"] = 1

    input_df = pd.DataFrame([input_data], columns=X.columns)
    predicted_price = best_model.predict(input_df)[0]
    return round(float(predicted_price), 2)


# %% [markdown]
# ## 19. Test Prediction
#
# Change these values to check different properties.

# %%
location = "1st Phase JP Nagar"
sqft = 1000
bath = 2
bhk = 2

predicted_price = predict_price(location, sqft, bath, bhk)
print(f"Predicted price: {predicted_price} lakhs")


# %% [markdown]
# ## 20. Check Available Locations
#
# Use this cell if you want to know valid location names for prediction.

# %%
locations = sorted([column for column in X.columns if column not in ["total_sqft", "bath", "bhk"]])
locations[:30]


# %% [markdown]
# ## 21. Interview Summary
#
# You can explain this project like this:
#
# I built a Bangalore housing analytics and price prediction model on Kaggle using a public housing dataset.
# I cleaned missing values, converted BHK and total square feet into numeric features,
# created price per square feet for outlier detection, grouped rare locations as `other`,
# one-hot encoded location, trained multiple regression models, compared them using R2,
# MAE, and RMSE, saved the best model, and created a prediction function for new inputs.
