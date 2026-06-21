# Bangalore Housing Analytics & Price Prediction

## Why Bangalore?

Use Bangalore/Bengaluru because the CampusX reference project is based on Bangalore housing data. This version extends the regression workflow into a simple analytics dashboard plus real-time price prediction app.

## Data Collection Plan

Preferred option:

- Use an existing public Bangalore housing CSV dataset.
- Do not scrape real-estate websites at the beginning.
- A CSV dataset is enough for interview preparation because the main goal is to learn data cleaning, feature engineering, model training, and evaluation.

Expected columns:

- `location`
- `size`
- `total_sqft`
- `bath`
- `balcony`
- `price`

Later optional improvement:

- Collect a small extra sample manually from public listings and compare model predictions with current listings.

## Project Goal

Analyze Bangalore housing trends and predict house price using property features like location, BHK, total square feet, and bathrooms.

## Current Status

The first working version is complete.

Raw dataset:

```text
data/raw/Bengaluru_House_Data.csv
```

Generated files:

```text
data/processed/cleaned_bengaluru_house_data.csv
artifacts/model.pkl
artifacts/columns.json
artifacts/metrics.json
```

Model comparison from the first run:

| Model | R2 Score | MAE | RMSE |
|---|---:|---:|---:|
| Linear Regression | 0.8646 | 19.2420 | 40.0168 |
| Lasso Regression | 0.8646 | 19.2171 | 40.0212 |
| Decision Tree | 0.7876 | 18.1770 | 50.1223 |

Best saved model:

```text
Linear Regression
```

Sample prediction:

```text
Input: location = 1st Phase JP Nagar, sqft = 1000, bath = 2, bhk = 2
Output: 94.19 lakhs
```

## How To Run

From this folder:

```bash
cd bangalore_housing_analytics_price_prediction
python src/explore_data.py
python src/data_cleaning.py
python src/train_model.py
python src/predict.py
```

To open the local website:

```bash
python web_app.py
```

Then visit:

```text
http://127.0.0.1:8000
```

Website link notes are saved in:

```text
WEBSITE_LINK.md
```

To open the analytics dashboard:

```bash
python -m streamlit run dashboard_app.py
```

The dashboard includes:

- KPI cards for raw records, cleaned records, average price, and average price per sqft
- Model comparison using R2 score
- Location-wise average price analysis
- BHK-wise price analysis
- Price vs square-foot scatter plot
- Affordability segment chart
- Business insights from selected filters
- Real-time model prediction form

## Kaggle Notebook Version

For running this project on Kaggle, use:

```text
notebooks/kaggle_bangalore_housing_analytics.py
```

It contains cell-wise code with `# %%` sections. Copy the cells into a Kaggle notebook and run them from top to bottom.

Kaggle instructions are in:

```text
notebooks/KAGGLE_CELLS_README.md
```

## CampusX-Style Build Order

### Step 1: Understand The Problem

Question:

Given property details, can we estimate the house price?

ML type:

- Supervised learning
- Regression problem

Target variable:

- `price`

### Step 2: Load And Inspect Data

Learn:

- `pandas.read_csv`
- `df.head()`
- `df.shape`
- `df.info()`
- `df.describe()`
- Missing values

### Step 3: Clean Data

Tasks:

- Drop columns that are not useful at beginner level, if needed.
- Convert `size` like `2 BHK` into numeric `bhk`.
- Convert `total_sqft` ranges like `1200 - 1500` into average value.
- Remove rows where important values are missing.

### Step 4: Feature Engineering

Tasks:

- Create `price_per_sqft`.
- Group rare locations into `other`.
- Remove unrealistic values, such as very tiny sqft per BHK.
- Remove extreme outliers location-wise.

### Step 5: Prepare ML Features

Tasks:

- One-hot encode `location`.
- Create `X` and `y`.
- Split data into train and test sets.

### Step 6: Train Models

Start simple:

- Linear Regression

Then compare:

- Lasso Regression
- Decision Tree Regressor

Optional later:

- Random Forest Regressor

### Step 7: Evaluate

Metrics:

- R2 score
- Mean Absolute Error
- Root Mean Squared Error

### Step 8: Prediction Function

Create a simple function:

```python
predict_price(location, sqft, bath, bhk)
```

This function should convert input into the same feature format used during training and return predicted price.

### Step 9: Save Model

Save:

- Trained model using `pickle`
- Feature column names using JSON

### Step 10: Optional Simple UI

Only after the model works:

- Streamlit app or Flask app

## Interview Explanation

This project analyzes Bangalore housing trends and predicts house prices using regression. I cleaned raw real-estate data, converted text fields like BHK and square-foot ranges into numeric features, handled rare locations, removed outliers, trained regression models, compared metrics, and built prediction interfaces.

Dashboard explanation:

I extended the ML project into a data analytics dashboard using Streamlit and Plotly. The dashboard helps analyze location-wise prices, BHK trends, sqft-price relation, affordability segments, model performance, and real-time price prediction.

## Interview Questions To Prepare

1. **Why is this a regression problem?**

Because the target variable `price` is a continuous numeric value.

2. **Why did you create `bhk` from `size`?**

The original `size` column contains text like `2 BHK`. ML models need numeric features, so I extracted the number.

3. **How did you handle `total_sqft` ranges?**

Some values are written like `1200 - 1500`, so I converted them to the average value.

4. **Why did you create `price_per_sqft`?**

It helps detect outliers. A house with an extremely high or low price per square foot compared to the same location may be unrealistic.

5. **Why group rare locations as `other`?**

One-hot encoding too many rare locations creates too many columns. Grouping rare locations keeps the model simpler.

6. **Why did Linear Regression work well?**

After cleaning and one-hot encoding location, the relation between size, BHK, bathrooms, location, and price becomes reasonably learnable by a linear model.

## Folder Plan

```text
bangalore_housing_analytics_price_prediction/
  data/
    raw/
    processed/
  notebooks/
    kaggle_bangalore_housing_analytics.py
    KAGGLE_CELLS_README.md
  src/
    explore_data.py
    data_cleaning.py
    train_model.py
    predict.py
  web_app.py
  dashboard_app.py
  artifacts/
    model.pkl
    columns.json
    metrics.json
  README.md
```
