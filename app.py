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
    "fontFamily": "Inter, Segoe UI, sans-serif",
    "backgroundColor": "#f3f6f8",
    "color": "#172b3a",
    "minHeight": "100vh",
    "padding": "32px clamp(18px, 4vw, 64px) 48px",
}
CARD_STYLE = {
    "backgroundColor": "#ffffff",
    "border": "1px solid #e3eaf0",
    "borderRadius": "18px",
    "padding": "22px",
    "boxShadow": "0 8px 26px rgba(25, 48, 69, 0.045)",
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
    return html.Div([
        html.Div([html.Span(className="kpi-dot", style={"backgroundColor": accent}), html.Div(title, className="kpi-title")], className="kpi-heading"),
        html.Div(value, className="kpi-value", style={"color": accent}),
    ], className="kpi-card", style=CARD_STYLE)


def style_figure(figure):
    figure.update_layout(
        autosize=True,
        height=340,
        margin={"l": 18, "r": 18, "t": 62, "b": 22},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, Segoe UI, sans-serif", "color": "#526779", "size": 12},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 16, "color": "#172b3a"}},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 1, "xanchor": "right"},
        hoverlabel={"bgcolor": "#172b3a", "font": {"color": "#ffffff"}},
    )
    figure.update_xaxes(showgrid=False, zeroline=False, linecolor="#e8eef2")
    figure.update_yaxes(showgrid=True, gridcolor="#edf1f4", zeroline=False)
    return figure


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
    html.Div([
        html.Div([
            html.Div("V", className="brand-mark"),
            html.Div([
                html.Div("VITAL  /  HEALTH INTELLIGENCE", className="eyebrow"),
                html.H1("Diabetes Risk Dashboard"),
                html.P("A clearer view of patient risk patterns and model-powered what-if predictions.", className="subtitle"),
            ]),
        ], className="hero-copy"),
        html.Div([
            html.Span(className="live-dot"),
            html.Div([html.Div("DATASET READY", className="status-label"), html.Div(f"{len(DATA):,} patient records", className="status-value")]),
        ], className="dataset-status"),
    ], className="hero"),
    html.Div([
        html.Div([
            html.Div("REFINE YOUR VIEW", className="eyebrow"),
            html.H2("Patient filters", className="panel-title"),
        ], className="filter-heading"),
        html.Div([
            make_filter("City", dcc.Dropdown(id="filter-city", options=options("city"), multi=True, placeholder="All cities")),
            make_filter("Gender", dcc.Dropdown(id="filter-gender", options=options("gender"), multi=True, placeholder="All genders")),
            make_filter("Age range", dcc.RangeSlider(id="filter-age", min=AGE_MIN, max=AGE_MAX, value=[AGE_MIN, AGE_MAX], marks=None, tooltip={"placement": "bottom", "always_visible": True})),
            make_filter("BMI range", dcc.RangeSlider(id="filter-bmi", min=BMI_MIN, max=BMI_MAX, step=0.1, value=[BMI_MIN, BMI_MAX], marks=None, tooltip={"placement": "bottom", "always_visible": True})),
            make_filter("Activity level", dcc.Dropdown(id="filter-activity", options=options("physical_activity_level"), multi=True, placeholder="All activity levels")),
            make_filter("Diabetes risk", dcc.Dropdown(id="filter-risk", options=[{"label": risk, "value": risk} for risk in CLASSES], multi=True, placeholder="All risk classes")),
        ], className="filter-grid"),
    ], className="filter-panel", style=CARD_STYLE),
    html.Div(id="kpis", className="kpi-grid"),
    html.Div([
        html.Div(dcc.Graph(id="risk-distribution", config={"displayModeBar": False, "responsive": True}), className="chart-card", style=CARD_STYLE),
        html.Div(dcc.Graph(id="gender-risk", config={"displayModeBar": False, "responsive": True}), className="chart-card", style=CARD_STYLE),
        html.Div(dcc.Graph(id="age-risk", config={"displayModeBar": False, "responsive": True}), className="chart-card", style=CARD_STYLE),
        html.Div(dcc.Graph(id="bmi-risk", config={"displayModeBar": False, "responsive": True}), className="chart-card", style=CARD_STYLE),
    ], className="chart-grid"),
    html.Div([
        html.Div([
            html.Div("AT A GLANCE", className="eyebrow"),
            html.H2("Smart Health Insight"),
            html.Div(id="smart-insight", className="insight-box"),
        ], className="section-card", style=CARD_STYLE),
        html.Div([
            html.Div("MODEL EXPLAINABILITY", className="eyebrow"),
            html.H2("Feature importance"),
            dcc.Graph(id="feature-importance", config={"displayModeBar": False, "responsive": True}),
        ], className="section-card", style=CARD_STYLE),
    ], className="two-column"),
    html.Div([
        html.Div([
            html.Div([
                html.Div("INTERACTIVE MODEL", className="eyebrow"),
                html.H2("AI Diabetes Risk Prediction"),
                html.P("Adjust the profile below to explore a what-if estimate from the saved Random Forest model.", className="section-description"),
            ]),
            html.Div("RANDOM FOREST", className="model-badge"),
        ], className="section-heading"),
        html.Div([html.Div([html.Label(field_label(feature), className="form-label"),
                            dcc.Input(id=f"input-{feature}", type="number", value=float(DATA[feature].median()), step="any") if feature in NUMERIC_FEATURES
                            else dcc.Dropdown(id=f"input-{feature}", options=options(feature), value=DATA[feature].mode().iloc[0])], className="form-item") for feature in FEATURES], className="form-grid"),
        html.Div(id="prediction-output", style={"marginTop": "20px"}),
    ], className="section-card prediction-panel", style=CARD_STYLE),
    html.Div([
        html.Div("PROFILE COMPARISON", className="eyebrow"),
        html.H2("Patient vs dataset average"),
        dcc.Graph(id="patient-average", config={"displayModeBar": False, "responsive": True}),
    ], className="section-card", style=CARD_STYLE),
    html.Div([
        html.Div("RECORDS", className="eyebrow"),
        html.H2("Filtered patient data"),
        dash_table.DataTable(id="patient-table", page_size=10, sort_action="native", style_table={"overflowX": "auto"}, style_cell={"padding": "12px 14px", "textAlign": "left", "fontFamily": "Inter, Segoe UI, sans-serif", "fontSize": "13px", "border": "none", "borderBottom": "1px solid #edf1f4"}, style_header={"backgroundColor": "#f5f8fa", "fontWeight": "600", "color": "#526779", "border": "none", "borderBottom": "1px solid #e3eaf0"}),
    ], className="section-card table-panel", style=CARD_STYLE),
    html.P("For educational use only. Risk estimates are not a medical diagnosis.", className="disclaimer"),
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
            * { box-sizing: border-box; }
            body { margin: 0; background: #f3f6f8; color: #172b3a; -webkit-font-smoothing: antialiased; }
            h1, h2, p { margin-top: 0; }
            h1 { color: #172b3a; font-size: clamp(28px, 3vw, 38px); letter-spacing: -.04em; margin-bottom: 8px; }
            h2 { color: #172b3a; font-size: 19px; letter-spacing: -.02em; margin-bottom: 16px; }
            .hero { display: flex; justify-content: space-between; align-items: center; gap: 24px; margin-bottom: 26px; }
            .hero-copy { display: flex; align-items: center; gap: 18px; }
            .brand-mark { width: 54px; height: 54px; display: grid; place-items: center; flex: 0 0 auto; border-radius: 17px; color: white; font-size: 25px; font-weight: 800; background: linear-gradient(145deg, #1d8e86, #176a78); box-shadow: 0 8px 18px rgba(29, 142, 134, .2); }
            .eyebrow { color: #17847e; font-size: 10px; font-weight: 750; letter-spacing: .14em; margin-bottom: 8px; }
            .subtitle, .section-description { color: #708191; font-size: 14px; margin-bottom: 0; }
            .dataset-status { display: flex; align-items: center; gap: 11px; padding: 13px 17px; background: #fff; border: 1px solid #e3eaf0; border-radius: 14px; box-shadow: 0 6px 20px rgba(25, 48, 69, .035); }
            .live-dot, .kpi-dot { width: 9px; height: 9px; display: inline-block; flex: 0 0 auto; border-radius: 50%; background: #31a77b; }
            .status-label { color: #718291; font-size: 9px; font-weight: 750; letter-spacing: .12em; }
            .status-value { color: #20394a; font-size: 13px; font-weight: 650; margin-top: 3px; }
            .filter-panel { margin-bottom: 18px; }
            .filter-heading { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 18px; }
            .panel-title { margin: 0; }
            .filter-grid { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 18px; align-items: end; }
            .filter-label, .form-label { display: block; color: #718291; font-size: 10px; font-weight: 750; margin-bottom: 9px; text-transform: uppercase; letter-spacing: .09em; }
            .filter-item { min-width: 0; }
            .filter-item .Select-control, .form-item .Select-control { border-color: #e1e8ed; border-radius: 10px; min-height: 40px; box-shadow: none; }
            .filter-item .Select-control:hover, .form-item .Select-control:hover { border-color: #9dcac4; }
            .filter-item .Select-placeholder, .form-item .Select-placeholder { color: #91a0ac; }
            .filter-item .rc-slider-track { background-color: #238d83; }
            .filter-item .rc-slider-handle { border-color: #238d83; box-shadow: 0 0 0 4px rgba(35, 141, 131, .12); }
            .kpi-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 14px; margin-bottom: 18px; }
            .kpi-card { transition: transform .18s ease, box-shadow .18s ease; }
            .kpi-card:hover, .chart-card:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(25, 48, 69, .075) !important; }
            .kpi-heading { display: flex; align-items: center; gap: 9px; }
            .kpi-dot { width: 8px; height: 8px; }
            .kpi-title { color: #718291; font-size: 10px; font-weight: 750; text-transform: uppercase; letter-spacing: .09em; }
            .kpi-value { font-size: clamp(24px, 2.2vw, 31px); font-weight: 720; letter-spacing: -.04em; margin-top: 14px; }
            .chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
            .chart-card, .section-card { min-width: 0; }
            .chart-card { padding: 10px 12px 4px !important; transition: transform .18s ease, box-shadow .18s ease; }
            .chart-card .dash-graph, .section-card .dash-graph { width: 100%; }
            .two-column { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }
            .insight-box { background: linear-gradient(120deg, #f0faf7, #f7fbfa); border: 1px solid #dcefe9; border-left: 4px solid #2e9d83; border-radius: 12px; color: #3c5a5d; padding: 17px 18px; line-height: 1.65; }
            .section-heading { display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 22px; }
            .section-heading h2 { margin-bottom: 7px; }
            .model-badge { flex: 0 0 auto; padding: 8px 11px; color: #167a74; background: #eaf6f3; border: 1px solid #d8ede7; border-radius: 999px; font-size: 9px; font-weight: 750; letter-spacing: .1em; }
            .prediction-panel, .section-card { margin-top: 16px; }
            .form-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; }
            .form-item { min-width: 0; }
            .form-item input { width: 100%; min-height: 40px; padding: 9px 11px; border: 1px solid #e1e8ed; border-radius: 10px; outline: none; color: #263d4c; font: inherit; }
            .form-item input:focus { border-color: #55a79e; box-shadow: 0 0 0 3px rgba(35, 141, 131, .12); }
            .prediction-result { font-size: 34px; font-weight: 750; letter-spacing: -.04em; margin: 8px 0; }
            .confidence { color: #20394a; font-size: 17px; font-weight: 700; }
            .probabilities { color: #718291; margin-top: 7px; font-size: 13px; }
            .explanation { color: #526779; line-height: 1.6; margin-bottom: 0; }
            .table-panel { margin-top: 16px; }
            .disclaimer { color: #82909b; font-size: 11px; margin: 20px 2px 0; text-align: center; }
            @media (max-width: 1100px) { .filter-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
            @media (max-width: 760px) { .hero { align-items: flex-start; flex-direction: column; } .dataset-status { align-self: stretch; } .chart-grid, .two-column { grid-template-columns: 1fr; } .form-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
            @media (max-width: 520px) { .filter-grid, .kpi-grid, .form-grid { grid-template-columns: 1fr; } .filter-heading { align-items: flex-start; } .section-heading { align-items: flex-start; flex-direction: column; } .section-card, .filter-panel, .kpi-card { padding: 17px !important; } .chart-card { padding: 7px !important; } }
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
    return (
        kpis, style_figure(risk_fig), style_figure(gender_fig), style_figure(age_fig), style_figure(bmi_fig),
        filtered.to_dict("records"), columns, smart_insight(filtered), style_figure(importance_fig),
    )


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
    comparison = px.bar(numeric_values.melt(id_vars="Measure", var_name="Source", value_name="Value"), x="Value", y="Measure", color="Source", barmode="group", title="Numeric patient values vs dataset averages", color_discrete_sequence=["#238d83", "#b9c9d2"])
    return result, style_figure(comparison)


if __name__ == "__main__":
    app.run(debug=True)