# Kaggle Notebook Cells For Bangalore Housing Analytics & Price Prediction

Use this file:

```text
notebooks/kaggle_bangalore_housing_analytics.py
```

It is written with `# %%` notebook cells. You can either:

1. Copy each cell into a Kaggle notebook manually, or
2. Upload the `.py` file as a notebook-style script and run it section by section.

## Kaggle Setup

1. Create a new Kaggle notebook.
2. Add the Bangalore/Bengaluru housing dataset from Kaggle's right sidebar.
3. Copy the cells from `kaggle_bangalore_housing_analytics.py`.
4. Run from top to bottom.

## What The Notebook Does

- Loads the CSV from `/kaggle/input`.
- Explores the dataset.
- Cleans missing and messy values.
- Converts `size` into `bhk`.
- Converts `total_sqft` ranges into numbers.
- Creates `price_per_sqft`.
- Groups rare locations as `other`.
- Removes simple outliers.
- One-hot encodes location.
- Trains Linear Regression, Lasso Regression, and Decision Tree.
- Saves model artifacts to `/kaggle/working/house_price_artifacts`.
- Provides a `predict_price(location, sqft, bath, bhk)` function.

## Kaggle Output Files

After running, these files are saved:

```text
/kaggle/working/house_price_artifacts/model.pkl
/kaggle/working/house_price_artifacts/columns.json
/kaggle/working/house_price_artifacts/metrics.json
/kaggle/working/house_price_artifacts/cleaned_bengaluru_house_data.csv
```

## What To Tell Interviewers

You can say:

> I built and ran the complete ML pipeline on Kaggle. I used Kaggle for dataset management, notebook execution, model training, metric comparison, and saving model artifacts. The project includes data cleaning, feature engineering, outlier removal, one-hot encoding, regression model comparison, and a final prediction function.
