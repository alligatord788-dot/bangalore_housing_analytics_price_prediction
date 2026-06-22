# Kaggle Full Run Guide

Use this notebook script:

```text
notebooks/kaggle_bangalore_housing_analytics.py
```

It now includes the full project flow:

1. Load the Bangalore housing CSV from `/kaggle/input`.
2. Clean and preprocess the data.
3. Train Linear Regression, Lasso, and Decision Tree models.
4. Reproduce the same model-comparison metrics using `random_state=42`.
5. Save model artifacts in `/kaggle/working/house_price_artifacts`.
6. Show Plotly dashboard charts inline inside the Kaggle notebook.
7. Save a dashboard HTML file:

```text
/kaggle/working/bangalore_housing_dashboard.html
```

8. Save a Streamlit app file:

```text
/kaggle/working/dashboard_app.py
```

## Important Kaggle Limitation

Your laptop can run:

```bash
python -m streamlit run dashboard_app.py
```

and open:

```text
http://localhost:8501
```

Kaggle notebooks generally show notebook outputs rather than hosting a persistent public localhost web app. So the Kaggle-friendly version shows the same dashboard through inline Plotly cells and saves an HTML dashboard file.

## How To Explain This In Interview

Say:

> I ran the complete ML pipeline on Kaggle, including data cleaning, feature engineering, model comparison, artifact saving, prediction testing, and dashboard-style analytics. Since Kaggle notebooks are notebook-based rather than persistent app servers, I recreated the dashboard through inline Plotly charts and saved an HTML dashboard output, while keeping the Streamlit app file available for local deployment.

