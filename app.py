from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report, roc_curve
)

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
FEATURES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area", "mean_smoothness",
    "mean_compactness", "mean_concavity", "mean_concave_points", "mean_symmetry",
    "mean_fractal_dimension", "radius_error", "texture_error", "perimeter_error",
    "area_error", "smoothness_error", "compactness_error", "concavity_error",
    "concave_points_error", "symmetry_error", "fractal_dimension_error",
    "worst_radius", "worst_texture", "worst_perimeter", "worst_area",
    "worst_smoothness", "worst_compactness", "worst_concavity",
    "worst_concave_points", "worst_symmetry", "worst_fractal_dimension"
]
METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(page_title="ClassifyLab | ML Model Studio", page_icon="🧠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: radial-gradient(circle at 80% 0%, #1e2a55 0%, #10172b 38%, #0b1020 100%); color: #eef2ff; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #151e38 0%, #11182d 100%); border-right: 1px solid #283454; }
section[data-testid="stSidebar"] * { color: #e8edff; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f4f6ff !important; }
p, label, .stMarkdown { color: #cbd5ee; }
div[data-testid="stMetric"] { background: linear-gradient(135deg, rgba(39,54,93,.9), rgba(25,35,64,.9)); border: 1px solid #35466f; padding: 16px 18px; border-radius: 16px; box-shadow: 0 8px 24px rgba(0,0,0,.16); }
div[data-testid="stMetric"] label { color: #aebce0 !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-family: 'Space Grotesk'; }
div[data-testid="stTabs"] button { color: #cbd5ee; }
div[data-testid="stDataFrame"] { border: 1px solid #35466f; border-radius: 14px; overflow: hidden; }
div.stButton > button { background: linear-gradient(90deg,#7467f0,#4f8cff); color: white; border: 0; border-radius: 10px; font-weight: 700; }
.hero { padding: 26px 30px; border: 1px solid #35466f; border-radius: 22px; background: linear-gradient(115deg, rgba(93,82,205,.27), rgba(37,116,184,.15)); margin-bottom: 22px; }
.hero-kicker { color: #a9b6ff; letter-spacing: 2px; font-size: 11px; font-weight: 700; }
.hero-title { font-family: 'Space Grotesk'; font-size: 34px; font-weight: 700; color: #fff; margin: 6px 0; }
.hero-sub { color: #c5d0eb; font-size: 14px; }
.panel { background: rgba(22,32,58,.78); border: 1px solid #2d3d63; border-radius: 18px; padding: 18px 20px; }
.small-muted { color: #98a8ce; font-size: 12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    return {name: joblib.load(ROOT / "models" / file) for name, file in MODEL_FILES.items()}

models = load_models()

def score_model(model, frame):
    X = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError("Input features contain missing or non-numeric values.")
    y_true = pd.to_numeric(frame["target"], errors="raise").astype(int)
    if not set(y_true.unique()).issubset({0, 1}):
        raise ValueError("The target column must contain only 0 (Benign) and 1 (Malignant).")
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X)[:, 1]
    else:
        y_score = model.decision_function(X)
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return result, y_true, y_pred, y_score

def styled_chart(fig, height=330):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=height,
        margin=dict(l=18, r=18, t=48, b=18),
        font=dict(family="DM Sans", color="#dbe5ff"),
        title_font=dict(size=16, color="#f4f6ff"),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# Sidebar
with st.sidebar:
    st.markdown("## 🧠 ClassifyLab")
    st.caption("MACHINE LEARNING MODEL STUDIO")
    st.markdown("---")
    page = st.radio("WORKSPACE", ["Model Evaluation", "Model Comparison", "About Dataset"], index=0)
    st.markdown("---")
    selected_model = st.selectbox("SELECT CLASSIFIER", list(MODEL_FILES.keys()), index=0)
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("UCI Wisconsin Diagnostic Breast Cancer")
    st.caption("569 rows  ·  30 features  ·  Binary")
    st.markdown("---")
    st.markdown("<span class='small-muted'>Built with Streamlit · scikit-learn</span>", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-kicker">INTELLIGENT CLASSIFICATION WORKSPACE</div>
  <div class="hero-title">Model performance, made clear.</div>
  <div class="hero-sub">Evaluate five machine learning classifiers, compare metrics, and inspect predictions on unseen test data.</div>
</div>
""", unsafe_allow_html=True)

default_path = ROOT / "test_data.csv"
uploaded = st.file_uploader("Upload test dataset (CSV)", type=["csv"], help="CSV must include all 30 feature columns and a target column (0 = Benign, 1 = Malignant).")
try:
    data = pd.read_csv(uploaded) if uploaded is not None else pd.read_csv(default_path)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

missing = [c for c in FEATURES + ["target"] if c not in data.columns]
if missing:
    st.error("Your CSV is missing required columns: " + ", ".join(missing))
    st.stop()
if data.empty:
    st.error("The uploaded CSV has no rows.")
    st.stop()
if data["target"].nunique() < 2:
    st.warning("AUC requires both target classes in the uploaded test data.")

st.caption(f"Evaluation data: **{len(data):,} rows** · {'Uploaded CSV' if uploaded is not None else 'Bundled hold-out test_data.csv'} · Models are loaded from disk; no retraining is performed.")

if page == "About Dataset":
    c1, c2, c3 = st.columns(3)
    c1.metric("Instances", "569")
    c2.metric("Features", "30")
    c3.metric("Classes", "2")
    st.markdown("### Dataset overview")
    st.write("The Wisconsin Diagnostic Breast Cancer dataset contains numeric measurements computed from digitized images of fine needle aspirates of breast masses.")
    st.markdown("- **Target:** `target` (0 = Benign, 1 = Malignant)")
    st.markdown("- **Input:** 30 numeric measurements describing cell nuclei")
    st.markdown("- **Split:** stratified 80% training / 20% hold-out test, random_state=42")
    st.markdown("- **Source:** UCI Machine Learning Repository — Breast Cancer Wisconsin (Diagnostic)")
    st.dataframe(data.head(10), use_container_width=True)
elif page == "Model Comparison":
    results = []
    outputs = {}
    try:
        for name, model in models.items():
            metrics, yt, yp, ys = score_model(model, data)
            results.append({"Model": name, **metrics})
            outputs[name] = (yt, yp, ys)
    except Exception as exc:
        st.error(f"Evaluation error: {exc}")
        st.stop()
    comparison = pd.DataFrame(results)
    st.markdown("### 🏆 All-model leaderboard")
    st.dataframe(comparison.style.format({m: "{:.4f}" for m in METRIC_ORDER}).highlight_max(subset=METRIC_ORDER, color="#294b63"), use_container_width=True, hide_index=True)
    long_df = comparison.melt(id_vars="Model", value_vars=METRIC_ORDER, var_name="Metric", value_name="Score")
    fig = px.bar(long_df, x="Metric", y="Score", color="Model", barmode="group", title="Metric comparison across all classifiers", range_y=[0, 1.05])
    st.plotly_chart(styled_chart(fig, 420), use_container_width=True)
    rank = comparison.sort_values(["F1", "AUC", "MCC"], ascending=False).iloc[0]
    st.success(f"Top model on this uploaded test set (ranked by F1, then AUC and MCC): **{rank['Model']}**")
    st.markdown("### Model observations")
    for _, row in comparison.iterrows():
        st.write(f"**{row['Model']}** — Accuracy {row['Accuracy']:.3f}, F1 {row['F1']:.3f}, AUC {row['AUC']:.3f}, MCC {row['MCC']:.3f}.")
else:
    model = models[selected_model]
    try:
        metrics, y_true, y_pred, y_score = score_model(model, data)
    except Exception as exc:
        st.error(f"Evaluation error: {exc}")
        st.stop()
    st.markdown(f"### 📊 {selected_model}")
    st.caption("Hold-out evaluation · positive class: Malignant")
    metric_cols = st.columns(6)
    for col, metric in zip(metric_cols, METRIC_ORDER):
        value = metrics[metric]
        col.metric(metric, "N/A" if pd.isna(value) else f"{value:.3f}")
    tab1, tab2, tab3 = st.tabs(["Performance", "Confusion Matrix", "Classification Report"])
    with tab1:
        left, right = st.columns([1.2, 1])
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(1, len(y_score)+1)), y=np.sort(y_score), mode="lines", name="Sorted model score", line=dict(width=3, color="#8b83ff")))
            fig.update_layout(title="Sorted prediction scores", xaxis_title="Test observations (sorted)", yaxis_title="Model score")
            st.plotly_chart(styled_chart(fig), use_container_width=True)
        with right:
            roc = go.Figure()
            if len(np.unique(y_true)) == 2:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {metrics['AUC']:.3f})", line=dict(width=3, color="#49c6a8"), fill="tozeroy"))
                roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random baseline", line=dict(dash="dash", color="#8492b0")))
            else:
                roc.add_annotation(text="ROC curve needs both target classes in the uploaded CSV.", x=0.5, y=0.5, showarrow=False)
            roc.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", xaxis=dict(range=[0,1]), yaxis=dict(range=[0,1]))
            st.plotly_chart(styled_chart(roc), use_container_width=True)
    with tab2:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Purples", x=["Predicted Benign", "Predicted Malignant"], y=["Actual Benign", "Actual Malignant"], title="Confusion matrix")
        st.plotly_chart(styled_chart(fig, 420), use_container_width=True)
        st.caption("Rows = actual class · Columns = predicted class")
    with tab3:
        report = classification_report(y_true, y_pred, target_names=["Benign", "Malignant"], output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df, use_container_width=True)
    with st.expander("Preview uploaded test data"):
        st.dataframe(data.head(20), use_container_width=True)

st.markdown("---")
st.markdown("<div class='small-muted'>Educational demonstration only. This model is not a medical diagnostic tool. Predictions must not be used for clinical decisions.</div>", unsafe_allow_html=True)
