import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global style ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark sidebar */
  section[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
  }
  section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem; }

  /* Main background */
  .main { background: #f8fafc; }

  /* Metric cards */
  .metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }
  .metric-card .val {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    color: #0f172a;
    line-height: 1;
  }
  .metric-card .lbl {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-top: 6px;
  }
  .metric-card .sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 4px;
  }

  /* Section headers */
  .section-head {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94a3b8;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 8px;
    margin-bottom: 20px;
  }

  /* Prediction result */
  .pred-box {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 12px 0;
  }
  .pred-box.positive { background: #fef2f2; border: 2px solid #fca5a5; }
  .pred-box.negative { background: #f0fdf4; border: 2px solid #86efac; }
  .pred-box .headline { font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }
  .pred-box.positive .headline { color: #dc2626; }
  .pred-box.negative .headline { color: #16a34a; }
  .pred-box .sub-text { color: #64748b; font-size: 0.9rem; }

  /* Tab styling */
  .stTabs [data-baseweb="tab-list"] { gap: 16px; background: #f1f5f9; border-radius: 10px; padding: 4px; }
  .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 500; color: #64748b; }
  .stTabs [aria-selected="true"] { background: white !important; color: #0f172a !important; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }

  /* Hide streamlit branding */
  #MainMenu, footer { visibility: hidden; }

  /* Chart containers */
  .chart-container {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
  }
</style>
""", unsafe_allow_html=True)

# ── Palette for charts ───────────────────────────────────────
PALETTE = {
    "primary":   "#e11d48",   # rose-600
    "secondary": "#0ea5e9",   # sky-500
    "accent":    "#f59e0b",   # amber-500
    "neutral":   "#64748b",   # slate-500
    "bg":        "#f8fafc",
    "surface":   "#ffffff",
    "border":    "#e2e8f0",
    "text":      "#0f172a",
    "muted":     "#94a3b8",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["surface"],
    "axes.facecolor":   PALETTE["surface"],
    "axes.edgecolor":   PALETTE["border"],
    "axes.labelcolor":  PALETTE["neutral"],
    "xtick.color":      PALETTE["muted"],
    "ytick.color":      PALETTE["muted"],
    "text.color":       PALETTE["text"],
    "grid.color":       PALETTE["border"],
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

# ── Dataset ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load Heart Disease Cleveland dataset directly from UCI."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    cols = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv(url, header=None, names=cols, na_values="?")
    df.dropna(inplace=True)
    df["target"] = (df["target"] > 0).astype(int)
    df = df.astype({
        "sex": int, "cp": int, "fbs": int, "restecg": int,
        "exang": int, "slope": int, "ca": int, "thal": int
    })
    return df

@st.cache_data
def get_sample_data():
    """Fallback: generate realistic synthetic data if network unavailable."""
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        "age":      np.random.randint(29, 77, n),
        "sex":      np.random.randint(0, 2, n),
        "cp":       np.random.randint(0, 4, n),
        "trestbps": np.random.randint(94, 200, n),
        "chol":     np.random.randint(126, 565, n),
        "fbs":      np.random.randint(0, 2, n),
        "restecg":  np.random.randint(0, 3, n),
        "thalach":  np.random.randint(71, 202, n),
        "exang":    np.random.randint(0, 2, n),
        "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
        "slope":    np.random.randint(0, 3, n),
        "ca":       np.random.randint(0, 4, n),
        "thal":     np.random.choice([3, 6, 7], n),
        "target":   np.random.randint(0, 2, n),
    })
    return df

try:
    df = load_data()
    data_source = "UCI Repository (live)"
except Exception:
    df = get_sample_data()
    data_source = "Synthetic fallback (network unavailable)"

# ── Feature metadata ─────────────────────────────────────────
FEATURE_META = {
    "age":      {"label": "Age", "unit": "years", "type": "num"},
    "sex":      {"label": "Sex", "unit": "", "type": "cat", "options": {0: "Female", 1: "Male"}},
    "cp":       {"label": "Chest Pain Type", "unit": "", "type": "cat",
                 "options": {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-Anginal", 3: "Asymptomatic"}},
    "trestbps": {"label": "Resting Blood Pressure", "unit": "mm Hg", "type": "num"},
    "chol":     {"label": "Serum Cholesterol", "unit": "mg/dl", "type": "num"},
    "fbs":      {"label": "Fasting Blood Sugar > 120 mg/dl", "unit": "", "type": "cat",
                 "options": {0: "No", 1: "Yes"}},
    "restecg":  {"label": "Resting ECG", "unit": "", "type": "cat",
                 "options": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}},
    "thalach":  {"label": "Max Heart Rate Achieved", "unit": "bpm", "type": "num"},
    "exang":    {"label": "Exercise-Induced Angina", "unit": "", "type": "cat",
                 "options": {0: "No", 1: "Yes"}},
    "oldpeak":  {"label": "ST Depression (Exercise vs Rest)", "unit": "", "type": "num"},
    "slope":    {"label": "Peak Exercise ST Slope", "unit": "", "type": "cat",
                 "options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}},
    "ca":       {"label": "Major Vessels Colored by Fluoroscopy", "unit": "", "type": "num"},
    "thal":     {"label": "Thalassemia", "unit": "", "type": "cat",
                 "options": {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}},
}

FEATURES = list(FEATURE_META.keys())

# ── ML pipeline ──────────────────────────────────────────────
@st.cache_data
def prepare_ml(df, test_size, random_state):
    X = df[FEATURES]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler

@st.cache_data
def train_all_models(test_size, random_state):
    X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = prepare_ml(
        df, test_size, random_state
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=random_state),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        "SVM":                 SVC(probability=True, random_state=random_state),
        "KNN":                 KNeighborsClassifier(),
    }
    SCALED = {"Logistic Regression", "SVM", "KNN"}
    results = {}
    for name, model in models.items():
        Xtr = X_train_s if name in SCALED else X_train.values
        Xte = X_test_s  if name in SCALED else X_test.values
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "model":     model,
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall":    recall_score(y_test, y_pred),
            "f1":        f1_score(y_test, y_pred),
            "auc":       auc(fpr, tpr),
            "fpr":       fpr,
            "tpr":       tpr,
            "y_pred":    y_pred,
            "y_prob":    y_prob,
            "cm":        confusion_matrix(y_test, y_pred),
            "scaled":    name in SCALED,
        }
    return results, X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫀 Heart Disease ML")
    st.markdown(f"<small style='color:#64748b'>Data: {data_source}</small>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### ⚙️ Model Settings")
    test_size     = st.slider("Test split", 0.10, 0.40, 0.20, 0.05)
    random_state  = st.slider("Random seed", 0, 100, 42)
    selected_model_name = st.selectbox(
        "Active model",
        ["Random Forest", "Logistic Regression", "Gradient Boosting", "SVM", "KNN"]
    )

    st.markdown("---")
    st.markdown("#### 📊 Dataset")
    st.markdown(f"**{len(df)}** patients · **{len(FEATURES)}** features")
    pos = df["target"].sum()
    neg = len(df) - pos
    st.markdown(f"❤️ With disease: **{pos}** ({pos/len(df)*100:.0f}%)")
    st.markdown(f"💚 Healthy: **{neg}** ({neg/len(df)*100:.0f}%)")

    st.markdown("---")
    st.caption("Cleveland Heart Disease Dataset · UCI ML Repository")

# ── Train ────────────────────────────────────────────────────
with st.spinner("Training models…"):
    results, X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler = train_all_models(
        test_size, random_state
    )

sel = results[selected_model_name]

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 8px 0;'>
  <h1 style='font-size:2rem; font-weight:700; color:#ffffff; margin:0; line-height:1.1;'>
    Heart Disease Risk Predictor
  </h1>
  <p style='color:#64748b; margin-top:8px; font-size:0.95rem;'>
    Cleveland Clinic dataset · 303 patients · 13 clinical features · Binary classification
  </p>
</div>
""", unsafe_allow_html=True)

# ── Top metrics ──────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (c1, f"{sel['accuracy']*100:.1f}%", "Accuracy",  selected_model_name),
    (c2, f"{sel['precision']*100:.1f}%", "Precision", "of positive predictions correct"),
    (c3, f"{sel['recall']*100:.1f}%",    "Recall",    "of cases caught"),
    (c4, f"{sel['f1']*100:.1f}%",        "F1 Score",  "harmonic mean"),
    (c5, f"{sel['auc']:.3f}",            "AUC-ROC",   "area under curve"),
]
for col, val, lbl, sub in metrics:
    with col:
        st.markdown(f"""
        <div class='metric-card'>
          <div class='val'>{val}</div>
          <div class='lbl'>{lbl}</div>
          <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab2, tab4, tab3, tab1 = st.tabs([
    "📊  Explore Data", "📈  Feature Importance", "🤖  Model Comparison", "🔍  Predict"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — Predict
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-head'>Patient Input</div>", unsafe_allow_html=True)
    st.markdown("Adjust the sliders and dropdowns to enter patient data, then hit **Run Prediction**.")

    with st.form("predict_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Demographics**")
            age  = st.slider("Age (years)", 20, 80, 54)
            sex  = st.selectbox("Sex", [0, 1], format_func=lambda x: FEATURE_META["sex"]["options"][x])
            fbs  = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                [0, 1], format_func=lambda x: FEATURE_META["fbs"]["options"][x])

            st.markdown("**Blood Work**")
            chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 245)
            trestbps = st.slider("Resting BP (mm Hg)", 80, 210, 130)

        with col_b:
            st.markdown("**Cardiac Symptoms**")
            cp    = st.selectbox("Chest Pain Type", [0, 1, 2, 3],
                                 format_func=lambda x: FEATURE_META["cp"]["options"][x])
            thalach = st.slider("Max Heart Rate (bpm)", 60, 210, 150)
            exang   = st.selectbox("Exercise-Induced Angina",
                                   [0, 1], format_func=lambda x: FEATURE_META["exang"]["options"][x])
            oldpeak = st.slider("ST Depression", 0.0, 7.0, 1.0, 0.1)

        with col_c:
            st.markdown("**Clinical Tests**")
            restecg = st.selectbox("Resting ECG", [0, 1, 2],
                                   format_func=lambda x: FEATURE_META["restecg"]["options"][x])
            slope   = st.selectbox("ST Slope", [0, 1, 2],
                                   format_func=lambda x: FEATURE_META["slope"]["options"][x])
            ca      = st.slider("Major Vessels (Fluoroscopy)", 0, 4, 0)
            thal    = st.selectbox("Thalassemia", [3, 6, 7],
                                   format_func=lambda x: FEATURE_META["thal"]["options"][x])

        submitted = st.form_submit_button("🫀 Run Prediction", use_container_width=True, type="primary")

    if submitted:
        patient = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                             thalach, exang, oldpeak, slope, ca, thal]])

        model = sel["model"]
        if sel["scaled"]:
            patient_input = scaler.transform(patient)
        else:
            patient_input = patient

        prediction = model.predict(patient_input)[0]
        probability = model.predict_proba(patient_input)[0]
        risk_pct = probability[1] * 100

        r1, r2 = st.columns([1, 1])
        with r1:
            if prediction == 1:
                st.markdown(f"""
                <div class='pred-box positive'>
                  <div class='headline'>⚠️ Heart Disease Detected</div>
                  <div style='font-size:3rem; font-weight:800; color:#dc2626; font-family:monospace;'>
                    {risk_pct:.1f}%
                  </div>
                  <div class='sub-text'>estimated disease probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-box negative'>
                  <div class='headline'>✅ Low Heart Disease Risk</div>
                  <div style='font-size:3rem; font-weight:800; color:#16a34a; font-family:monospace;'>
                    {risk_pct:.1f}%
                  </div>
                  <div class='sub-text'>estimated disease probability</div>
                </div>""", unsafe_allow_html=True)

        with r2:
            st.markdown("**Probability breakdown**")
            fig_pred, ax_pred = plt.subplots(figsize=(5, 2.5))
            bars = ax_pred.barh(
                ["No Disease", "Disease"],
                [probability[0]*100, probability[1]*100],
                color=[PALETTE["secondary"], PALETTE["primary"]],
                height=0.5, edgecolor="none"
            )
            for bar, pct in zip(bars, [probability[0]*100, probability[1]*100]):
                ax_pred.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                            f"{pct:.1f}%", va="center", fontweight="600", color=PALETTE["text"])
            ax_pred.set_xlim(0, 115)
            ax_pred.set_xlabel("Probability (%)")
            ax_pred.axvline(50, color=PALETTE["muted"], linestyle="--", linewidth=0.8, alpha=0.6)
            fig_pred.tight_layout()
            st.pyplot(fig_pred, use_container_width=True)
            plt.close()

        st.markdown("> ⚠️ *This tool is for educational purposes only and does not constitute medical advice.*")

# ═══════════════════════════════════════════════════════════
# TAB 2 — Explore Data
# ═══════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-head'>Dataset Overview</div>", unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        # Age distribution by target
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        for tgt, color, label in [(0, PALETTE["secondary"], "No Disease"), (1, PALETTE["primary"], "Disease")]:
            subset = df[df["target"] == tgt]["age"]
            ax1.hist(subset, bins=15, alpha=0.75, color=color, label=label, edgecolor="white", linewidth=0.5)
        ax1.set_xlabel("Age")
        ax1.set_ylabel("Count")
        ax1.set_title("Age Distribution by Outcome", fontweight="600", pad=12)
        ax1.legend(frameon=False)
        ax1.grid(axis="y", alpha=0.4)
        fig1.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with d2:
        # Chest pain vs target
        cp_labels = {0: "Typical\nAngina", 1: "Atypical\nAngina", 2: "Non-\nAnginal", 3: "Asympto-\nmatic"}
        cp_counts = df.groupby(["cp", "target"]).size().unstack(fill_value=0)
        # Ensure we have a Series for each target value aligned to the cp index
        no_disease = cp_counts[0] if 0 in cp_counts.columns else pd.Series(0, index=cp_counts.index)
        disease = cp_counts[1] if 1 in cp_counts.columns else pd.Series(0, index=cp_counts.index)
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        x = np.arange(len(cp_counts))
        w = 0.35
        ax2.bar(x - w/2, no_disease.values, w, color=PALETTE["secondary"],
            label="No Disease", edgecolor="white")
        ax2.bar(x + w/2, disease.values, w, color=PALETTE["primary"],
            label="Disease", edgecolor="white")
        ax2.set_xticks(x)
        # Use safe lookup for labels in case unexpected cp values appear
        ax2.set_xticklabels([cp_labels.get(i, str(i)) for i in cp_counts.index], fontsize=9)
        ax2.set_ylabel("Count")
        ax2.set_title("Chest Pain Type vs Outcome", fontweight="600", pad=12)
        ax2.legend(frameon=False)
        ax2.grid(axis="y", alpha=0.4)
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("<br>", unsafe_allow_html=True)

    d3, d4 = st.columns(2)
    with d3:
        # Max HR vs age scatter
        fig3, ax3 = plt.subplots(figsize=(6, 3.5))
        for tgt, color, label in [(0, PALETTE["secondary"], "No Disease"), (1, PALETTE["primary"], "Disease")]:
            sub = df[df["target"] == tgt]
            ax3.scatter(sub["age"], sub["thalach"], alpha=0.55, color=color,
                        label=label, s=30, edgecolors="none")
        ax3.set_xlabel("Age (years)")
        ax3.set_ylabel("Max Heart Rate (bpm)")
        ax3.set_title("Age vs Max Heart Rate", fontweight="600", pad=12)
        ax3.legend(frameon=False)
        ax3.grid(alpha=0.3)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with d4:
        # Correlation heatmap (numeric only)
        num_cols = ["age", "trestbps", "chol", "thalach", "oldpeak", "target"]
        corr = df[num_cols].corr()
        fig4, ax4 = plt.subplots(figsize=(6, 3.5))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt=".2f", linewidths=0.5, ax=ax4,
                    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
        ax4.set_title("Correlation Matrix (Numeric Features)", fontweight="600", pad=12)
        fig4.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    # Raw data
    with st.expander("📋 View raw dataset"):
        st.dataframe(
            df.style.background_gradient(cmap="RdYlGn_r", subset=["target"]),
            use_container_width=True, height=320
        )
        st.caption(f"{len(df)} rows × {len(df.columns)} columns — downloaded from UCI ML Repository")

# ═══════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-head'>All Models vs Metrics</div>", unsafe_allow_html=True)

    # Summary table
    summary_rows = []
    for name, r in results.items():
        summary_rows.append({
            "Model": name,
            "Accuracy":  f"{r['accuracy']*100:.1f}%",
            "Precision": f"{r['precision']*100:.1f}%",
            "Recall":    f"{r['recall']*100:.1f}%",
            "F1":        f"{r['f1']*100:.1f}%",
            "AUC-ROC":   f"{r['auc']:.3f}",
        })
    summary_df = pd.DataFrame(summary_rows).set_index("Model")
    st.dataframe(summary_df, use_container_width=True)

    m1, m2 = st.columns(2)

    with m1:
        # ROC curves
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        colors_roc = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
                      "#8b5cf6", "#10b981"]
        for (name, r), color in zip(results.items(), colors_roc):
            ax_roc.plot(r["fpr"], r["tpr"], color=color, linewidth=1.8,
                        label=f"{name} ({r['auc']:.3f})")
        ax_roc.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curves — All Models", fontweight="600", pad=12)
        ax_roc.legend(fontsize=7.5, frameon=False, loc="lower right")
        ax_roc.grid(alpha=0.3)
        fig_roc.tight_layout()
        st.pyplot(fig_roc, use_container_width=True)
        plt.close()

    with m2:
        # Confusion matrix for selected model
        cm = sel["cm"]
        fig_cm, ax_cm = plt.subplots(figsize=(4.5, 3.8))
        im = ax_cm.imshow(cm, cmap="RdYlGn", vmin=0)
        for i in range(2):
            for j in range(2):
                ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center",
                           fontsize=20, fontweight="700",
                           color="white" if cm[i, j] > cm.max()*0.6 else PALETTE["text"])
        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Predicted\nNo Disease", "Predicted\nDisease"])
        ax_cm.set_yticklabels(["Actual\nNo Disease", "Actual\nDisease"])
        ax_cm.set_title(f"Confusion Matrix — {selected_model_name}", fontweight="600", pad=12)
        fig_cm.colorbar(im, ax=ax_cm, shrink=0.8)
        fig_cm.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)
        plt.close()

    # Bar chart: metric comparison
    metric_names = ["accuracy", "precision", "recall", "f1", "auc"]
    fig_bar, ax_bar = plt.subplots(figsize=(10, 3.5))
    x_bar = np.arange(len(results))
    bar_w = 0.15
    bar_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], "#8b5cf6", "#10b981"]
    for i, (metric, color) in enumerate(zip(metric_names, bar_colors)):
        vals = [results[n][metric] for n in results]
        offset = (i - 2) * bar_w
        bars = ax_bar.bar(x_bar + offset, vals, bar_w, color=color,
                          label=metric.capitalize(), edgecolor="white", linewidth=0.3)
    ax_bar.set_xticks(x_bar)
    ax_bar.set_xticklabels(list(results.keys()), fontsize=9)
    ax_bar.set_ylim(0.5, 1.05)
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Performance Metrics Across All Models", fontweight="600", pad=12)
    ax_bar.legend(frameon=False, fontsize=8, ncol=5)
    ax_bar.grid(axis="y", alpha=0.3)
    fig_bar.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

# ═══════════════════════════════════════════════════════════
# TAB 4 — Feature Importance
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-head'>What Drives Predictions?</div>", unsafe_allow_html=True)

    f1c, f2c = st.columns(2)

    with f1c:
        # Random Forest built-in importance
        rf = results["Random Forest"]["model"]
        importances = rf.feature_importances_
        feat_names = [FEATURE_META[f]["label"] for f in FEATURES]
        sorted_idx = np.argsort(importances)

        fig_fi, ax_fi = plt.subplots(figsize=(6, 5))
        colors_fi = [PALETTE["primary"] if importances[i] > np.median(importances) else PALETTE["secondary"]
                     for i in sorted_idx]
        bars = ax_fi.barh(
            [feat_names[i] for i in sorted_idx],
            importances[sorted_idx],
            color=colors_fi, edgecolor="none", height=0.65
        )
        ax_fi.set_xlabel("Gini Importance")
        ax_fi.set_title("Random Forest Feature Importance", fontweight="600", pad=12)
        ax_fi.grid(axis="x", alpha=0.3)
        legend_items = [
            mpatches.Patch(color=PALETTE["primary"], label="Above median"),
            mpatches.Patch(color=PALETTE["secondary"], label="Below median"),
        ]
        ax_fi.legend(handles=legend_items, frameon=False, fontsize=8)
        fig_fi.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)
        plt.close()

    with f2c:
        # Permutation importance
        rf_model = results["Random Forest"]["model"]
        perm = permutation_importance(
            rf_model, X_test.values, y_test.values, n_repeats=10, random_state=42
        )
        perm_means = perm.importances_mean
        perm_stds  = perm.importances_std
        perm_idx   = np.argsort(perm_means)

        fig_perm, ax_perm = plt.subplots(figsize=(6, 5))
        ax_perm.barh(
            [feat_names[i] for i in perm_idx],
            perm_means[perm_idx],
            xerr=perm_stds[perm_idx],
            color=PALETTE["accent"], edgecolor="none", height=0.65,
            error_kw={"ecolor": PALETTE["muted"], "linewidth": 1}
        )
        ax_perm.set_xlabel("Mean Accuracy Decrease")
        ax_perm.set_title("Permutation Importance (Random Forest)", fontweight="600", pad=12)
        ax_perm.grid(axis="x", alpha=0.3)
        ax_perm.axvline(0, color=PALETTE["muted"], linewidth=0.8)
        fig_perm.tight_layout()
        st.pyplot(fig_perm, use_container_width=True)
        plt.close()

    # Logistic Regression coefficients
    lr = results["Logistic Regression"]["model"]
    coefs = lr.coef_[0]
    coef_idx = np.argsort(np.abs(coefs))

    fig_lr, ax_lr = plt.subplots(figsize=(10, 3.5))
    bar_colors_lr = [PALETTE["primary"] if c > 0 else PALETTE["secondary"] for c in coefs[coef_idx]]
    ax_lr.barh(
        [feat_names[i] for i in coef_idx],
        coefs[coef_idx],
        color=bar_colors_lr, edgecolor="none", height=0.6
    )
    ax_lr.axvline(0, color=PALETTE["text"], linewidth=1)
    ax_lr.set_xlabel("Coefficient Value")
    ax_lr.set_title("Logistic Regression Coefficients (positive = increases disease risk)", fontweight="600", pad=12)
    ax_lr.grid(axis="x", alpha=0.3)
    legend_lr = [
        mpatches.Patch(color=PALETTE["primary"],   label="Increases risk"),
        mpatches.Patch(color=PALETTE["secondary"], label="Decreases risk"),
    ]
    ax_lr.legend(handles=legend_lr, frameon=False, fontsize=8)
    fig_lr.tight_layout()
    st.pyplot(fig_lr, use_container_width=True)
    plt.close()
