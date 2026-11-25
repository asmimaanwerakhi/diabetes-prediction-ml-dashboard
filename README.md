# Diabetes Risk Dashboard

An interactive dashboard for exploring patient diabetes-risk patterns and trying model-based predictions. The project was created for academic demonstration using a saved Random Forest model and a processed dataset.

> **Educational use only:** predictions and insights are not medical diagnoses and should not replace advice from a qualified healthcare professional.

## Dashboard at a glance

The responsive dashboard is organized around a few key tasks:

- **Overview:** KPI cards summarize the filtered patient count, High/Moderate/Low risk counts, and average age.
- **Patient filters:** narrow the dataset by city, gender, age range, BMI range, activity level, and diabetes-risk class. Charts, insights, and table rows update with the selection.
- **Risk analysis:** explore risk distribution, risk by gender and age group, and BMI versus HbA1c.
- **Model explainability:** review the most important model features and a short explanation for each what-if prediction.
- **What-if prediction:** adjust a patient profile to see the predicted risk class, model confidence, class probabilities, and numeric measurements compared with dataset averages.
- **Patient records:** browse and sort the filtered records in a paginated table.

## Project files

```text
ISM Dashboard/
├── app.py
├── diabetes_risk.csv
├── diabetes_risk_analysis.ipynb
├── README.md
└── saved_data/
    ├── diabetes_model.joblib
    ├── feature_metadata.json
    ├── model_metrics.json
    └── processed_data.csv
```

The dashboard loads the processed dataset and saved model from `saved_data/`. It does not retrain the model when it starts.

## Requirements and setup

Use Python 3 and install the project dependencies from the project folder:

```powershell
py -3 -m pip install dash pandas plotly joblib scikit-learn==1.7.2
```

The saved model was serialized with scikit-learn `1.7.2`; using the same version helps avoid model-loading compatibility issues.

## Run the dashboard

From the project folder, start the Dash app:

```powershell
py -3 app.py
```

Open the local address printed in the terminal (normally [http://127.0.0.1:8050/](http://127.0.0.1:8050/)).

## Dataset and target

The processed dataset contains 15,000 patient records. The target column is `diabetes_risk`, with three classes:

| Class |
|---|
| High |
| Moderate |
| Low |

`patient_id` is an identifier and is excluded from the model features.

## Model and preprocessing

The notebook trains a **Random Forest Classifier** using a scikit-learn pipeline. The pipeline applies:

- Numerical features: median imputation followed by standard scaling.
- Categorical features: most-frequent imputation followed by one-hot encoding.
- Unknown categorical values: ignored by the encoder during prediction.

The trained pipeline is saved as `saved_data/diabetes_model.joblib`. The dashboard uses its `predict` and `predict_proba` outputs and its feature importances.

## Evaluation results

The saved evaluation results use a stratified 80/20 train-test split with `random_state=42`:

| Metric | Score |
|---|---:|
| Accuracy | 0.7877 |
| Weighted precision | 0.7808 |
| Weighted recall | 0.7877 |
| Weighted F1 score | 0.7826 |
| Test records | 3,000 |

These are academic model results, not evidence of clinical validation.

## Rebuild the saved model

Run `diabetes_risk_analysis.ipynb` to:

1. Load and inspect the raw CSV.
2. Check missing values, duplicates, and data types.
3. Clean the data and create exploratory visualizations.
4. Train and evaluate the Random Forest pipeline.
5. Save the processed dataset, model, metrics, and feature metadata in `saved_data/`.

Restart the dashboard after regenerating the saved artifacts.
