# Diabetes Risk Dashboard

A simple academic project that analyzes patient health data, predicts diabetes risk, and presents the results in an interactive Dash dashboard.

## Project Structure

```text
ISM Dashboard/
├── diabetes_risk.csv
├── diabetes_risk_analysis.ipynb
├── app.py
├── README.md
└── saved_data/
    ├── diabetes_model.joblib
    ├── processed_data.csv
    ├── model_metrics.json
    └── feature_metadata.json
```

## Dataset

The dataset contains 15,000 patient records. The prediction target is `diabetes_risk`, with three classes:

- `High`
- `Moderate`
- `Low`

`patient_id` is used only as an identifier and is excluded from model features.

## Algorithm Used

The project uses a **Random Forest Classifier**.

A Random Forest is an ensemble classification algorithm that trains multiple decision trees and combines their predictions. It was selected because:

- It works well for classification problems with mixed numerical and categorical data.
- It can model non-linear relationships between health measurements and diabetes risk.
- It usually requires little manual feature engineering.
- It provides `predict_proba`, which allows the dashboard to display class probabilities and confidence.
- It provides feature importances, which are used in the dashboard's feature importance chart and prediction explanation.
- It is appropriate for a small academic dataset of approximately 15,000 records and is easy to explain.

## Preprocessing

The notebook automatically detects feature types and uses a scikit-learn `ColumnTransformer`:

- Numerical features: median imputation followed by standard scaling.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- Unknown categories are safely ignored during prediction.

The preprocessing and classifier are combined in one scikit-learn `Pipeline` and saved as:

```text
saved_data/diabetes_model.joblib
```

## Model Results

The saved evaluation results are based on a stratified 80/20 train-test split with `random_state=42`:

| Metric | Score |
|---|---:|
| Accuracy | 0.7877 |
| Weighted precision | 0.7808 |
| Weighted recall | 0.7877 |
| Weighted F1 score | 0.7826 |
| Test records | 3,000 |

These results are intended for academic demonstration and should not be treated as clinical validation.

## Notebook Workflow

Run `diabetes_risk_analysis.ipynb` once to:

1. Load and inspect the raw CSV.
2. Check missing values, duplicates, and data types.
3. Clean the data and create exploratory visualizations.
4. Train and evaluate the Random Forest pipeline.
5. Save the processed data and model artifacts in `saved_data/`.

The dashboard does not retrain the model and does not load the raw CSV.

## Dashboard Features

`app.py` provides:

- Overview KPI cards for total patients, risk counts, and average age.
- Filters for city, gender, age, BMI, activity level, and risk class.
- Interactive risk distribution, gender, age-group, and BMI charts.
- Filtered patient data table.
- AI Diabetes Risk Prediction form using the saved model.
- Prediction probability and confidence display.
- Feature importance visualization.
- Basic prediction explanation based on important model features.
- What-if analysis by changing patient inputs.
- Patient-versus-dataset-average comparison.
- Rule-based Smart Health Insight for the filtered dataset.

## Installation

From the project folder, install the dependencies:

```powershell
py -3 -m pip install dash pandas plotly joblib scikit-learn==1.7.2
```

The model was serialized with scikit-learn `1.7.2`, so using that version avoids model-loading compatibility errors.

## Run the Dashboard

From `G:\ISM Dashboard`:

```powershell
py -3 app.py
```

Open the address shown by Dash, normally:

```text
http://127.0.0.1:8050/
```

## Important Note

This is an educational predictive analytics project. The predictions and insights are not medical diagnoses and should not replace advice from a qualified healthcare professional.
