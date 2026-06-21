import json
import pickle
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "Bengaluru_House_Data.csv"
CLEAN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "cleaned_bengaluru_house_data.csv"
MODEL_PATH = PROJECT_DIR / "artifacts" / "model.pkl"
COLUMNS_PATH = PROJECT_DIR / "artifacts" / "columns.json"
METRICS_PATH = PROJECT_DIR / "artifacts" / "metrics.json"


st.set_page_config(
    page_title="Bangalore Housing Analytics",
    page_icon="🏠",
    layout="wide",
)


@st.cache_data
def load_data():
    raw_df = pd.read_csv(RAW_DATA_PATH)
    clean_df = pd.read_csv(CLEAN_DATA_PATH)
    clean_df["price_per_sqft"] = (clean_df["price"] * 100000) / clean_df["total_sqft"]
    return raw_df, clean_df


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(COLUMNS_PATH, "r") as file:
        columns = json.load(file)["columns"]

    return model, columns


@st.cache_data
def load_metrics():
    with open(METRICS_PATH, "r") as file:
        return json.load(file)


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
    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)


def get_price_segment(price):
    if price < 50:
        return "Affordable"
    if price < 120:
        return "Mid-range"
    return "Premium"


raw_df, df = load_data()
model, columns = load_model()
metrics = load_metrics()

st.title("Bangalore Housing Analytics & Price Prediction")
st.write(
    "Interactive dashboard for exploring Bangalore housing trends and predicting "
    "property price using the trained regression model."
)

st.sidebar.header("Dashboard Filters")
all_locations = sorted(df["location"].unique())
selected_locations = st.sidebar.multiselect(
    "Select locations",
    options=all_locations,
    default=["other"] if "other" in all_locations else all_locations[:5],
)

all_bhk = sorted(df["bhk"].unique())
selected_bhk = st.sidebar.multiselect(
    "Select BHK",
    options=all_bhk,
    default=all_bhk,
)

filtered_df = df[
    df["location"].isin(selected_locations)
    & df["bhk"].isin(selected_bhk)
].copy()

if filtered_df.empty:
    st.warning("No records found for the selected filters. Please change filters.")
    st.stop()

st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw Records", f"{raw_df.shape[0]:,}")
col2.metric("Cleaned Records", f"{df.shape[0]:,}")
col3.metric("Average Price", f"{filtered_df['price'].mean():.2f} lakhs")
col4.metric("Average Price / Sqft", f"₹{filtered_df['price_per_sqft'].mean():.0f}")

st.subheader("Model Performance")
best_model = metrics["best_model"]
best_score = metrics["model_results"][best_model]["r2_score"]
mae = metrics["model_results"][best_model]["mae"]
rmse = metrics["model_results"][best_model]["rmse"]

perf1, perf2, perf3 = st.columns(3)
perf1.metric("Best Model", best_model)
perf2.metric("R2 Score", best_score)
perf3.metric("MAE", f"{mae:.2f} lakhs")

model_df = pd.DataFrame(metrics["model_results"]).T.reset_index()
model_df = model_df.rename(columns={"index": "model"})
fig_model = px.bar(
    model_df,
    x="model",
    y="r2_score",
    color="model",
    title="Model Comparison by R2 Score",
    text="r2_score",
)
st.plotly_chart(fig_model, use_container_width=True)

st.subheader("Business Analytics")

location_summary = (
    filtered_df.groupby("location")
    .agg(avg_price=("price", "mean"), avg_price_per_sqft=("price_per_sqft", "mean"))
    .reset_index()
    .sort_values("avg_price", ascending=False)
    .head(15)
)

fig_location = px.bar(
    location_summary,
    x="location",
    y="avg_price",
    color="avg_price_per_sqft",
    title="Top Locations by Average Price",
    labels={
        "location": "Location",
        "avg_price": "Average Price (lakhs)",
        "avg_price_per_sqft": "Avg Price / Sqft",
    },
)
st.plotly_chart(fig_location, use_container_width=True)

chart1, chart2 = st.columns(2)

bhk_summary = (
    filtered_df.groupby("bhk")["price"]
    .mean()
    .reset_index()
    .sort_values("bhk")
)
fig_bhk = px.bar(
    bhk_summary,
    x="bhk",
    y="price",
    title="Average Price by BHK",
    labels={"bhk": "BHK", "price": "Average Price (lakhs)"},
)
chart1.plotly_chart(fig_bhk, use_container_width=True)

filtered_df["price_segment"] = filtered_df["price"].apply(get_price_segment)
segment_counts = filtered_df["price_segment"].value_counts().reset_index()
segment_counts.columns = ["segment", "count"]
fig_segment = px.pie(
    segment_counts,
    names="segment",
    values="count",
    title="Affordability Segments",
)
chart2.plotly_chart(fig_segment, use_container_width=True)

fig_scatter = px.scatter(
    filtered_df,
    x="total_sqft",
    y="price",
    color="bhk",
    hover_data=["location", "bath"],
    title="Price vs Total Square Feet",
    labels={"total_sqft": "Total Sqft", "price": "Price (lakhs)", "bhk": "BHK"},
)
st.plotly_chart(fig_scatter, use_container_width=True)

st.subheader("Business Insights")
top_location = location_summary.iloc[0]
common_bhk = filtered_df["bhk"].mode().iloc[0]
avg_price = filtered_df["price"].mean()

st.write(
    f"- **{top_location['location']}** has the highest average price among selected locations."
)
st.write(
    f"- **{common_bhk} BHK** is the most common property type in the selected data."
)
st.write(
    f"- Average selected-market price is **{avg_price:.2f} lakhs**, useful as a quick benchmark for customer affordability."
)

st.subheader("Real-Time Price Prediction")

pred_col1, pred_col2 = st.columns(2)
with pred_col1:
    input_location = st.selectbox("Location", options=all_locations)
    input_sqft = st.number_input("Total Sqft", min_value=100.0, value=1000.0, step=50.0)

with pred_col2:
    input_bath = st.number_input("Bathrooms", min_value=1, value=2, step=1)
    input_bhk = st.number_input("BHK", min_value=1, value=2, step=1)

if st.button("Predict Price"):
    predicted_price = predict_price(
        input_location,
        input_sqft,
        input_bath,
        input_bhk,
        model,
        columns,
    )
    st.success(f"Predicted Price: {predicted_price} lakhs")

    location_average = df[df["location"] == input_location]["price"].mean()
    if pd.notna(location_average):
        difference = predicted_price - location_average
        st.write(
            f"Location average price is {location_average:.2f} lakhs. "
            f"This prediction is {difference:.2f} lakhs from the location average."
        )
