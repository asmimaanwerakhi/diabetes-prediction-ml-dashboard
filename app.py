from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, dash_table


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "saved_data"
DATA = pd.read_csv(DATA_DIR / "processed_data.csv")
MODEL = joblib.load(DATA_DIR / "diabetes_model.joblib")
METRICS = __import__("json").loads((DATA_DIR / "model_metrics.json").read_text())
METADATA = __import__("json").loads((DATA_DIR / "feature_metadata.json").read_text())

TARGET = METADATA["target_column"]
FEATURES = METADATA["feature_columns"]
NUMERIC_FEATURES = METADATA["numerical_features"]
CATEGORICAL_FEATURES = METADATA["categorical_features"]
CLASSES = METADATA["target_classes"]

RISK_COLORS = {"High": "#d95f59", "Moderate": "#e3a64b", "Low": "#2e9d83"}
PAGE_STYLE = {
    "fontFamily": "Arial, sans-serif",
    "backgroundColor": "#f4f7f6",
    "color": "#18333a",
    "minHeight": "100vh",
    "padding": "28px 5vw 44px",
}
CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #dbe7e4",
    "borderRadius": "10px",
    "padding": "18px",
    "boxShadow": "0 3px 12px rgba(25, 61, 67, 0.06)",
}

AGE_MIN = int(DATA["age"].min())
AGE_MAX = int(DATA["age"].max())
BMI_MIN = float(DATA["bmi"].min())
BMI_MAX = float(DATA["bmi"].max())


def options(column):
    return [{"label": str(value), "value": value} for value in sorted(DATA[column].dropna().unique())]


def field_label(name):
    return name.replace("_", " ").title()


def make_filter(label, component):
    return html.Div([html.Label(label, className="filter-label"), component], className="filter-item")


def kpi(title, value, accent):
    return html.Div([html.Div(title, className="kpi-title"), html.Div(value, className="kpi-value", style={"color": accent})], style=CARD_STYLE)


def filter_data(city, gender, age_range, bmi_range, activity, risk):
    filtered = DATA.copy()
    if city:
        filtered = filtered[filtered["city"].isin(city)]
    if gender:
        filtered = filtered[filtered["gender"].isin(gender)]
    if age_range:
        filtered = filtered[filtered["age"].between(age_range[0], age_range[1])]
    if bmi_range:
        filtered = filtered[filtered["bmi"].between(bmi_range[0], bmi_range[1])]
    if activity:
        filtered = filtered[filtered["physical_activity_level"].isin(activity)]
    if risk:
        filtered = filtered[filtered[TARGET].isin(risk)]
    return filtered


def smart_insight(filtered):
    if filtered.empty:
        return "No patients match the current filters. Broaden the selection to view an insight."
    risk_share = filtered[TARGET].value_counts(normalize=True)
    dominant_risk = risk_share.index[0]
    messages = [f"{dominant_risk} risk is the largest group in this selection ({risk_share.iloc[0]:.0%})."]
    if "hba1c_level" in filtered and filtered["hba1c_level"].mean() >= 6.5:
        messages.append("Average HbA1c is at or above 6.5, so glucose monitoring may deserve attention.")
    if "physical_activity_level" in filtered:
        sedentary_share = (filtered["physical_activity_level"] == "Sedentary").mean()
        if sedentary_share >= 0.5:
            messages.append("At least half of this group reports sedentary activity.")
    return " ".join(messages)


def prediction_explanation(row, predicted_class):
    classifier = MODEL.named_steps["classifier"]
    importances = feature_importance_table()
    top = importances.head(4)
    factors = []
    for _, item in top.iterrows():
        feature = item["feature"]
        value = row.get(feature)
        if pd.notna(value):
            factors.append(f"{field_label(feature)} ({value})")
    factor_text = ", ".join(factors) if factors else "the available patient features"
    return f"The model classified this patient as {predicted_class}. The strongest model factors for this estimate are {factor_text}."


def feature_importance_table():
    preprocessor = MODEL.named_steps["preprocessor"]
    classifier = MODEL.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_
    grouped = {}
    for name, importance in zip(names, importances):
        clean_name = name.split("__", 1)[-1]
        source = next((feature for feature in FEATURES if clean_name == feature or clean_name.startswith(feature + "_")), clean_name)
        grouped[source] = grouped.get(source, 0) + float(importance)
    return pd.DataFrame({"feature": list(grouped), "importance": list(grouped.values())}).sort_values("importance", ascending=False)


def prediction_card(values):
    row = pd.DataFrame([{feature: values.get(feature) for feature in FEATURES}])
    prediction = MODEL.predict(row)[0]
    probabilities = MODEL.predict_proba(row)[0]
    confidence = float(max(probabilities))
    probability_text = " | ".join(f"{label}: {probability:.1%}" for label, probability in zip(MODEL.classes_, probabilities))
    return html.Div([
        html.Div("Predicted risk", className="kpi-title"),
        html.Div(prediction, className="prediction-result", style={"color": RISK_COLORS.get(prediction, "#18333a")}),
        html.Div(f"Confidence: {confidence:.1%}", className="confidence"),
        html.Div(probability_text, className="probabilities"),
        html.P(prediction_explanation(row.iloc[0], prediction), className="explanation"),
    ], style=CARD_STYLE)


app = Dash(__name__)
app.title = "Diabetes Risk Dashboard"
app.layout = html.Div([
    html.H1("Diabetes Risk Dashboard"),
    html.P("A focused view of patient risk patterns and model-based what-if predictions.", className="subtitle"),
    html.Div([
        make_filter("City", dcc.Dropdown(id="filter-city", options=options("city"), multi=True, placeholder="All cities")),
        make_filter("Gender", dcc.Dropdown(id="filter-gender", options=options("gender"), multi=True, placeholder="All genders")),
        make_filter("Age range", dcc.RangeSlider(id="filter-age", min=AGE_MIN, max=AGE_MAX, value=[AGE_MIN, AGE_MAX], marks=None, tooltip={"placement": "bottom", "always_visible": True})),
        make_filter("BMI range", dcc.RangeSlider(id="filter-bmi", min=BMI_MIN, max=BMI_MAX, step=0.1, value=[BMI_MIN, BMI_MAX], marks=None, tooltip={"placement": "bottom", "always_visible": True})),
        make_filter("Activity level", dcc.Dropdown(id="filter-activity", options=options("physical_activity_level"), multi=True, placeholder="All activity levels")),
        make_filter("Diabetes risk", dcc.Dropdown(id="filter-risk", options=[{"label": risk, "value": risk} for risk in CLASSES], multi=True, placeholder="All risk classes")),
    ], className="filter-panel", style=CARD_STYLE),
    html.Div(id="kpis", className="kpi-grid"),
    html.Div([
        html.Div(dcc.Graph(id="risk-distribution"), style=CARD_STYLE),
        html.Div(dcc.Graph(id="gender-risk"), style=CARD_STYLE),
        html.Div(dcc.Graph(id="age-risk"), style=CARD_STYLE),
        html.Div(dcc.Graph(id="bmi-risk"), style=CARD_STYLE),
    ], className="chart-grid"),
    html.Div([
        html.Div([
            html.H2("Smart Health Insight"),
            html.Div(id="smart-insight", className="insight-box"),
        ], style=CARD_STYLE),
        html.Div([
            html.H2("Feature importance"),
            dcc.Graph(id="feature-importance", config={"displayModeBar": False}),
        ], style=CARD_STYLE),
    ], className="two-column"),
    html.Div([
        html.H2("AI Diabetes Risk Prediction"),
        html.P("Adjust the patient profile to perform a what-if prediction using the saved Random Forest pipeline."),
        html.Div([html.Div([html.Label(field_label(feature), className="form-label"),
                            dcc.Input(id=f"input-{feature}", type="number", value=float(DATA[feature].median()), step="any") if feature in NUMERIC_FEATURES
                            else dcc.Dropdown(id=f"input-{feature}", options=options(feature), value=DATA[feature].mode().iloc[0])], className="form-item") for feature in FEATURES], className="form-grid"),
        html.Div(id="prediction-output", style={"marginTop": "20px"}),
    ], style=CARD_STYLE),
    html.Div([
        html.H2("Patient vs dataset average"),
        dcc.Graph(id="patient-average"),
    ], style=CARD_STYLE),
    html.Div([
        html.H2("Filtered patient data"),
        dash_table.DataTable(id="patient-table", page_size=10, sort_action="native", style_table={"overflowX": "auto"}, style_cell={"padding": "8px", "textAlign": "left", "fontFamily": "Arial", "fontSize": "13px"}, style_header={"backgroundColor": "#e6f0ed", "fontWeight": "bold"}),
    ], style={**CARD_STYLE, "marginTop": "22px"}),
], style=PAGE_STYLE)

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { margin: 0; background: #f4f7f6; } h1 { color: #124e5a; margin-bottom: 4px; } h2 { color: #124e5a; font-size: 20px; margin-top: 0; }
            .subtitle { color: #587177; margin-top: 0; } .filter-panel { display: grid; grid-template-columns: repeat(6, 1fr); gap: 14px; margin: 24px 0; }
            .filter-label, .form-label { display: block; font-size: 12px; font-weight: bold; color: #587177; margin-bottom: 7px; text-transform: uppercase; letter-spacing: .04em; }
            .filter-item { min-width: 0; } .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 22px; }
            .kpi-title { color: #587177; font-size: 13px; font-weight: bold; text-transform: uppercase; letter-spacing: .03em; } .kpi-value { font-size: 28px; font-weight: bold; margin-top: 8px; }
            .chart-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px; } .two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 18px; }
            .insight-box { background: #e7f3ef; border-left: 4px solid #2e9d83; border-radius: 6px; padding: 18px; line-height: 1.55; } .form-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
            .form-item input, .form-item .Select { width: 100%; } .prediction-result { font-size: 36px; font-weight: bold; margin: 8px 0; } .confidence { font-size: 18px; font-weight: bold; } .probabilities { color: #587177; margin-top: 5px; } .explanation { color: #385960; line-height: 1.5; }
            @media (max-width: 1000px) { .filter-panel, .kpi-grid { grid-template-columns: repeat(2, 1fr); } .chart-grid, .two-column { grid-template-columns: 1fr; } .form-grid { grid-template-columns: repeat(2, 1fr); } }
            @media (max-width: 600px) { .filter-panel, .kpi-grid, .form-grid { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
"""


@app.callback(
    Output("kpis", "children"), Output("risk-distribution", "figure"), Output("gender-risk", "figure"),
    Output("age-risk", "figure"), Output("bmi-risk", "figure"), Output("patient-table", "data"),
    Output("patient-table", "columns"), Output("smart-insight", "children"), Output("feature-importance", "figure"),
    Input("filter-city", "value"), Input("filter-gender", "value"), Input("filter-age", "value"),
    Input("filter-bmi", "value"), Input("filter-activity", "value"), Input("filter-risk", "value"),
)
def update_dashboard(city, gender, age_range, bmi_range, activity, risk):
    filtered = filter_data(city, gender, age_range, bmi_range, activity, risk)
    counts = filtered[TARGET].value_counts()
    kpis = [kpi("Total patients", f"{len(filtered):,}", "#124e5a"), kpi("High risk", f"{counts.get('High', 0):,}", RISK_COLORS["High"]), kpi("Moderate risk", f"{counts.get('Moderate', 0):,}", RISK_COLORS["Moderate"]), kpi("Low risk", f"{counts.get('Low', 0):,}", RISK_COLORS["Low"]), kpi("Average age", f"{filtered['age'].mean():.1f}" if len(filtered) else "-", "#287b8a")]
    empty = px.scatter(title="No matching patients")
    risk_fig = px.histogram(filtered, x=TARGET, color=TARGET, category_orders={TARGET: CLASSES}, color_discrete_map=RISK_COLORS, title="Risk distribution") if len(filtered) else empty
    gender_fig = px.histogram(filtered, x="gender", color=TARGET, barmode="group", color_discrete_map=RISK_COLORS, title="Risk by gender") if len(filtered) else empty
    age_data = filtered.copy()
    age_data["age_group"] = pd.cut(age_data["age"], bins=[0, 30, 45, 60, 200], labels=["18-30", "31-45", "46-60", "61+"])
    age_fig = px.histogram(age_data, x="age_group", color=TARGET, barmode="group", category_orders={"age_group": ["18-30", "31-45", "46-60", "61+"]}, color_discrete_map=RISK_COLORS, title="Risk by age group") if len(filtered) else empty
    bmi_fig = px.scatter(filtered, x="bmi", y="hba1c_level", color=TARGET, color_discrete_map=RISK_COLORS, opacity=0.55, hover_data=["age", "gender", "city"], title="BMI vs HbA1c by risk") if len(filtered) else empty
    importance_fig = px.bar(feature_importance_table().head(10).sort_values("importance"), x="importance", y="feature", orientation="h", title="Top model features", color_discrete_sequence=["#287b8a"])
    columns = [{"name": column.replace("_", " ").title(), "id": column} for column in DATA.columns]
    return kpis, risk_fig, gender_fig, age_fig, bmi_fig, filtered.to_dict("records"), columns, smart_insight(filtered), importance_fig


@app.callback(
    Output("prediction-output", "children"), Output("patient-average", "figure"),
    [Input(f"input-{feature}", "value") for feature in FEATURES],
)
def update_prediction(*values):
    value_map = dict(zip(FEATURES, values))
    result = prediction_card(value_map)
    numeric_values = pd.DataFrame({
        "Measure": NUMERIC_FEATURES,
        "Patient value": pd.to_numeric(
            pd.Series([value_map.get(feature) for feature in NUMERIC_FEATURES]),
            errors="coerce",
        ),
        "Dataset average": DATA[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").mean().to_numpy(),
    })
    comparison = px.bar(numeric_values.melt(id_vars="Measure", var_name="Source", value_name="Value"), x="Value", y="Measure", color="Source", barmode="group", title="Numeric patient values vs dataset averages", color_discrete_sequence=["#287b8a", "#b8c9c5"])
    return result, comparison


if __name__ == "__main__":
    app.run(debug=True)