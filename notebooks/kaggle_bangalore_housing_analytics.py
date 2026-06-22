# %% [markdown]
# # Bangalore Housing Analytics & Price Prediction
# ### End-to-end Kaggle workflow: load data, clean features, remove outliers, train regression models, compare metrics, save artifacts, create predictions, and build dashboard-style analytics.

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
# # 1. Find And Load Dataset
# ### Load the uploaded Kaggle CSV directly so we can inspect the raw housing records with `df.head()`.

# %%
DATA_PATH = "/kaggle/input/datasets/alligatordx/bengaluru-house-data/Bengaluru_House_Data.csv"
df = pd.read_csv(DATA_PATH)
df.head()


# %% [markdown]
# # 2. Basic Data Exploration
# ### Check shape, columns, data types, missing values, and summary statistics before cleaning. This tells us which columns need conversion, deletion, or imputation.

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
# # 3. Drop Less Useful Columns
# ### Drop `area_type`, `availability`, `society`, and `balcony` because this beginner model focuses on core numeric/location features. `society` has many missing values, while availability and area type add extra categorical complexity.

# %%
df1 = df.drop(["area_type", "availability", "society", "balcony"], axis=1)
df1.head()


# %% [markdown]
# # 4. Remove Missing Values
# ### Remove rows with missing important values because the remaining missing count is small after dropping weak columns, and regression models cannot train directly on NaN values.

# %%
print("Before dropping missing values:", df1.shape)
df2 = df1.dropna()
print("After dropping missing values:", df2.shape)
print(df2.isna().sum())


# %% [markdown]
# # 5. Convert Size Column Into BHK
# ### Convert text values like `2 BHK` and `4 Bedroom` into a numeric `bhk` column because ML models need numerical inputs.

# %%
df2 = df2.copy()
df2["bhk"] = df2["size"].apply(lambda x: int(str(x).split()[0]))
df2[["size", "bhk"]].head()


# %% [markdown]
# # 6. Clean Total Square Feet
# ### Convert `total_sqft` into a number. Simple values stay as floats, ranges like `1200 - 1500` become their average, and invalid unit-based values are removed because regression needs one numeric sqft value.

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
# # 7. Create Price Per Square Feet
# ### Create `price_per_sqft = price * 100000 / total_sqft`. We use it mainly for outlier detection because prices are easier to compare after normalizing by property size.

# %%
df4 = df3.copy()
df4["price_per_sqft"] = (df4["price"] * 100000) / df4["total_sqft"]
df4.head()


# %% [markdown]
# # 8. Handle Location Column
# ### Strip extra spaces and group rare locations as `other`. This reduces one-hot encoded columns and avoids overfitting to locations with very few examples.

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
# # 9. Remove Unrealistic Sqft Per BHK Values
# ### Remove records where `total_sqft / bhk < 300` because very small area per bedroom is usually unrealistic or noisy for house-price modeling.

# %%
print("Before removing sqft/BHK outliers:", df4.shape)

df5 = df4[df4["total_sqft"] / df4["bhk"] >= 300]

print("After removing sqft/BHK outliers:", df5.shape)


# %% [markdown]
# # 10. Remove Price Per Sqft Outliers
# ### For each location, keep homes within one standard deviation of that location's average `price_per_sqft`. This removes unusually cheap/costly entries that can distort the regression line.

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
# # 11. Remove BHK Outliers
# ### Within the same location, remove cases where a larger BHK has lower `price_per_sqft` than the smaller BHK average. This catches noisy listings and makes location-wise pricing more consistent.

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
# # 12. Create Final Clean Dataset
# ### Drop helper columns used only for cleaning (`size`, `price_per_sqft`) so the final model trains on clean prediction features.

# %%
df8 = df7.drop(["size", "price_per_sqft"], axis=1)
df8.head()


# %%
print("Final cleaned shape:", df8.shape)
print(df8.columns.tolist())


# %% [markdown]
# # 13. One-Hot Encode Location
# ### Convert text locations into numeric one-hot columns because Scikit-learn regression models cannot use raw strings directly.

# %%
location_dummies = pd.get_dummies(df8["location"], dtype=int)

X = pd.concat([df8.drop(["location", "price"], axis=1), location_dummies], axis=1)
y = df8["price"]

print("X shape:", X.shape)
print("y shape:", y.shape)
X.head()


# %% [markdown]
# # 14. Train-Test Split
# ### Split data into training and testing sets so the model is evaluated on unseen records instead of the same data used for fitting.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])


# %% [markdown]
# # 15. Train And Compare Models
# ### Train and compare Linear Regression, Lasso Regression, and Decision Tree Regressor. Comparing models shows whether a simple linear approach is enough or a non-linear tree helps.

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
# # 16. Select Best Model
# ### Select the model with the highest R2 score on the test set and use it for prediction/artifact saving.

# %%
best_model_name = max(results, key=lambda name: results[name]["r2_score"])
best_model = trained_models[best_model_name]

print("Best model:", best_model_name)
print("Best model metrics:", results[best_model_name])


# %% [markdown]
# # 17. Save Model And Columns
# ### Save the trained model, column order, metrics, and cleaned CSV to `/kaggle/working/`. We need columns because prediction input must match the exact training feature order.

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
# # 18. Create Prediction Function
# ### Build a reusable prediction function that converts user input into the same one-hot encoded format used during training and returns price in lakhs.

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
# # 19. Test Prediction
# ### Test the prediction function with one sample property. Change these values to simulate different user inputs.

# %%
location = "1st Phase JP Nagar"
sqft = 1000
bath = 2
bhk = 2

predicted_price = predict_price(location, sqft, bath, bhk)
print(f"Predicted price: {predicted_price} lakhs")


# %% [markdown]
# # 20. Check Available Locations
# ### List valid location columns so we know what location names can be passed to `predict_price()`.

# %%
locations = sorted([column for column in X.columns if column not in ["total_sqft", "bath", "bhk"]])
locations[:30]


# %% [markdown]
# # 21. Interview Summary
# ### Interview summary: I built a Bangalore housing analytics and price prediction model on Kaggle by cleaning raw housing data, engineering BHK/sqft/location features, removing outliers, comparing regression models with R2/MAE/RMSE, saving artifacts, and creating a prediction function.


# %% [markdown]
# # 22. Dashboard KPIs
# ### Create dashboard KPIs inside Kaggle. These summarize dataset size, average price, average price per sqft, and best model performance.

# %%
dashboard_df = df8.copy()
dashboard_df["price_per_sqft"] = (dashboard_df["price"] * 100000) / dashboard_df["total_sqft"]

total_raw_records = df.shape[0]
total_clean_records = dashboard_df.shape[0]
average_price = dashboard_df["price"].mean()
average_price_per_sqft = dashboard_df["price_per_sqft"].mean()

dashboard_kpis = pd.DataFrame(
    {
        "Metric": [
            "Raw Records",
            "Cleaned Records",
            "Average Price",
            "Average Price / Sqft",
            "Best Model",
            "Best R2 Score",
        ],
        "Value": [
            f"{total_raw_records:,}",
            f"{total_clean_records:,}",
            f"{average_price:.2f} lakhs",
            f"Rs. {average_price_per_sqft:.0f}",
            best_model_name,
            results[best_model_name]["r2_score"],
        ],
    }
)

dashboard_kpis


# %% [markdown]
# # 23. Plotly Dashboard Charts
# ### Build inline Plotly charts for model comparison, location pricing, BHK trends, sqft-price relation, and affordability segments. This recreates the dashboard inside the notebook.

# %%
import plotly.express as px
import plotly.io as pio
from IPython.display import HTML, display


def display_plotly(fig):
    """Display Plotly charts inside Kaggle when fig.show() gives blank output."""
    html = fig.to_html(full_html=False, include_plotlyjs=True)
    display(HTML(html))


# %%
model_results_df = pd.DataFrame(results).T.reset_index()
model_results_df = model_results_df.rename(columns={"index": "model"})

fig_model = px.bar(
    model_results_df,
    x="model",
    y="r2_score",
    color="model",
    text="r2_score",
    title="Model Comparison by R2 Score",
)
display_plotly(fig_model)


# %%
location_summary = (
    dashboard_df.groupby("location")
    .agg(
        avg_price=("price", "mean"),
        avg_price_per_sqft=("price_per_sqft", "mean"),
        listing_count=("price", "count"),
    )
    .reset_index()
    .sort_values("avg_price", ascending=False)
    .head(15)
)

fig_location = px.bar(
    location_summary,
    x="location",
    y="avg_price",
    color="avg_price_per_sqft",
    title="Top 15 Locations by Average Price",
    labels={
        "location": "Location",
        "avg_price": "Average Price (lakhs)",
        "avg_price_per_sqft": "Average Price / Sqft",
    },
)
display_plotly(fig_location)


# %%
bhk_summary = (
    dashboard_df.groupby("bhk")
    .agg(avg_price=("price", "mean"), listing_count=("price", "count"))
    .reset_index()
    .sort_values("bhk")
)

fig_bhk = px.bar(
    bhk_summary,
    x="bhk",
    y="avg_price",
    text="listing_count",
    title="Average Price by BHK",
    labels={"bhk": "BHK", "avg_price": "Average Price (lakhs)"},
)
display_plotly(fig_bhk)


# %%
fig_scatter = px.scatter(
    dashboard_df.sample(min(3000, dashboard_df.shape[0]), random_state=42),
    x="total_sqft",
    y="price",
    color="bhk",
    hover_data=["location", "bath"],
    title="Price vs Total Square Feet",
    labels={"total_sqft": "Total Sqft", "price": "Price (lakhs)", "bhk": "BHK"},
)
display_plotly(fig_scatter)


# %%
def get_price_segment(price):
    if price < 50:
        return "Affordable"
    if price < 120:
        return "Mid-range"
    return "Premium"


dashboard_df["price_segment"] = dashboard_df["price"].apply(get_price_segment)
segment_counts = dashboard_df["price_segment"].value_counts().reset_index()
segment_counts.columns = ["segment", "count"]

fig_segment = px.pie(
    segment_counts,
    names="segment",
    values="count",
    title="Affordability Segments",
)
display_plotly(fig_segment)


# %% [markdown]
# # 24. Business Insights From The Dashboard
# ### Extract simple business insights from the dashboard, such as premium locations, common BHK type, and affordability distribution.

# %%
top_location = location_summary.iloc[0]
most_common_bhk = dashboard_df["bhk"].mode().iloc[0]
premium_share = (
    dashboard_df[dashboard_df["price_segment"] == "Premium"].shape[0]
    / dashboard_df.shape[0]
) * 100

print("Business Insights")
print("-----------------")
print(f"1. {top_location['location']} has the highest average price among the top locations shown.")
print(f"2. {most_common_bhk} BHK is the most common property configuration in the cleaned dataset.")
print(f"3. Premium properties form {premium_share:.2f}% of the cleaned dataset.")
print(f"4. Price per sqft helps compare value across locations with different property sizes.")


# %% [markdown]
# # 25. Real-Time Prediction Cell
# ### Simulate the web-app prediction form inside Kaggle. Edit the input values and rerun the cell to get a new predicted price.

# %%
input_location = "1st Phase JP Nagar"
input_sqft = 1000
input_bath = 2
input_bhk = 2

predicted_price = predict_price(input_location, input_sqft, input_bath, input_bhk)
print("Input")
print("-----")
print("Location:", input_location)
print("Total Sqft:", input_sqft)
print("Bathrooms:", input_bath)
print("BHK:", input_bhk)
print("\nPrediction")
print("----------")
print(f"Predicted Price: {predicted_price} lakhs")


# %% [markdown]
# # 26. Save Dashboard As HTML
# ### Save all Plotly charts into one downloadable HTML dashboard file in `/kaggle/working/`.

# %%
dashboard_html_path = Path("/kaggle/working/bangalore_housing_dashboard.html")

html_parts = [
    "<html><head><title>Bangalore Housing Analytics Dashboard</title></head><body>",
    "<h1>Bangalore Housing Analytics & Price Prediction</h1>",
    "<h2>KPIs</h2>",
    dashboard_kpis.to_html(index=False),
    "<h2>Model Comparison</h2>",
    pio.to_html(fig_model, full_html=False, include_plotlyjs="cdn"),
    "<h2>Top Locations by Average Price</h2>",
    pio.to_html(fig_location, full_html=False, include_plotlyjs=False),
    "<h2>Average Price by BHK</h2>",
    pio.to_html(fig_bhk, full_html=False, include_plotlyjs=False),
    "<h2>Price vs Total Square Feet</h2>",
    pio.to_html(fig_scatter, full_html=False, include_plotlyjs=False),
    "<h2>Affordability Segments</h2>",
    pio.to_html(fig_segment, full_html=False, include_plotlyjs=False),
    "</body></html>",
]

dashboard_html_path.write_text("\n".join(html_parts), encoding="utf-8")
print("Saved dashboard HTML to:", dashboard_html_path)


# %% [markdown]
# # 27. Save Streamlit App File For Download
# ### Save a Streamlit app file for download. Kaggle shows notebook outputs, while the saved app can be run locally with `python -m streamlit run dashboard_app.py`.

# %%
streamlit_app_code = r'''
import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
CLEAN_DATA_PATH = PROJECT_DIR / "cleaned_bengaluru_house_data.csv"
MODEL_PATH = PROJECT_DIR / "model.pkl"
COLUMNS_PATH = PROJECT_DIR / "columns.json"
METRICS_PATH = PROJECT_DIR / "metrics.json"


st.set_page_config(page_title="Bangalore Housing Analytics", layout="wide")


@st.cache_data
def load_data():
    data = pd.read_csv(CLEAN_DATA_PATH)
    data["price_per_sqft"] = (data["price"] * 100000) / data["total_sqft"]
    return data


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    with open(COLUMNS_PATH, "r") as file:
        columns = json.load(file)["columns"]
    return model, columns


def predict_price(location, sqft, bath, bhk, model, columns):
    input_data = {column: 0 for column in columns}
    input_data["total_sqft"] = sqft
    input_data["bath"] = bath
    input_data["bhk"] = bhk
    if location in input_data:
        input_data[location] = 1
    elif "other" in input_data:
        input_data["other"] = 1
    input_df = pd.DataFrame([input_data], columns=columns)
    return round(float(model.predict(input_df)[0]), 2)


df = load_data()
model, columns = load_model()

st.title("Bangalore Housing Analytics & Price Prediction")
st.write("Dashboard generated from the Kaggle-trained model artifacts.")

col1, col2, col3 = st.columns(3)
col1.metric("Cleaned Records", f"{df.shape[0]:,}")
col2.metric("Average Price", f"{df['price'].mean():.2f} lakhs")
col3.metric("Avg Price / Sqft", f"Rs. {df['price_per_sqft'].mean():.0f}")

location_summary = (
    df.groupby("location")
    .agg(avg_price=("price", "mean"), avg_price_per_sqft=("price_per_sqft", "mean"))
    .reset_index()
    .sort_values("avg_price", ascending=False)
    .head(15)
)

st.plotly_chart(
    px.bar(location_summary, x="location", y="avg_price", color="avg_price_per_sqft",
           title="Top Locations by Average Price"),
    use_container_width=True,
)

bhk_summary = df.groupby("bhk")["price"].mean().reset_index()
st.plotly_chart(
    px.bar(bhk_summary, x="bhk", y="price", title="Average Price by BHK"),
    use_container_width=True,
)

st.subheader("Predict Price")
location = st.selectbox("Location", sorted(df["location"].unique()))
sqft = st.number_input("Total Sqft", min_value=100.0, value=1000.0)
bath = st.number_input("Bathrooms", min_value=1, value=2)
bhk = st.number_input("BHK", min_value=1, value=2)

if st.button("Predict"):
    prediction = predict_price(location, sqft, bath, bhk, model, columns)
    st.success(f"Predicted Price: {prediction} lakhs")
'''

streamlit_app_path = Path("/kaggle/working/dashboard_app.py")
streamlit_app_path.write_text(streamlit_app_code, encoding="utf-8")
print("Saved Streamlit app file to:", streamlit_app_path)


# %% [markdown]
# # 28. Copy Required App Artifacts To Kaggle Working Folder
# ### Copy model artifacts beside the generated Streamlit app so the downloaded app has the files it needs for prediction.

# %%
for source_path in [MODEL_PATH, COLUMNS_PATH, METRICS_PATH, CLEANED_DATA_PATH]:
    target_path = Path("/kaggle/working") / source_path.name
    target_path.write_bytes(source_path.read_bytes())
    print("Copied:", target_path)


# %% [markdown]
# # 29. Final Kaggle Output Files
# ### Print the final output paths so they are easy to find and download from Kaggle.

# %%
print("Kaggle outputs created:")
print("- /kaggle/working/house_price_artifacts/model.pkl")
print("- /kaggle/working/house_price_artifacts/columns.json")
print("- /kaggle/working/house_price_artifacts/metrics.json")
print("- /kaggle/working/house_price_artifacts/cleaned_bengaluru_house_data.csv")
print("- /kaggle/working/bangalore_housing_dashboard.html")
print("- /kaggle/working/dashboard_app.py")
print("- /kaggle/working/model.pkl")
print("- /kaggle/working/columns.json")
print("- /kaggle/working/metrics.json")
print("- /kaggle/working/cleaned_bengaluru_house_data.csv")
