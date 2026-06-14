import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page Config ──
st.set_page_config(
    page_title="Health Risk Classifier",
    page_icon="🏥",
    layout="wide"
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .header-container {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 32px;
        text-align: center;
    }
    .header-title { font-size: 2.4rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; margin: 0; }
    .header-subtitle { font-size: 1rem; color: #7a9cc4; margin-top: 8px; font-weight: 400; }
    .badge {
        display: inline-block; background: #0f3460; color: #4da6ff;
        border: 1px solid #1e5a9c; border-radius: 20px; padding: 4px 14px;
        font-size: 0.75rem; font-weight: 500; margin-bottom: 16px;
        letter-spacing: 1px; text-transform: uppercase;
    }
    .section-card { background: #1a1f2e; border: 1px solid #1e3a5f; border-radius: 12px; padding: 24px; margin-bottom: 16px; }
    .section-title {
        font-size: 0.75rem; font-weight: 600; color: #4da6ff;
        letter-spacing: 1.5px; text-transform: uppercase;
        margin-bottom: 16px; padding-bottom: 10px; border-bottom: 1px solid #1e3a5f;
    }
    .result-healthy {
        background: linear-gradient(135deg, #0d2b1a, #0a3d1f);
        border: 1px solid #1a6b35; border-radius: 12px; padding: 28px 32px; text-align: center;
    }
    .result-unhealthy {
        background: linear-gradient(135deg, #2b0d0d, #3d0a0a);
        border: 1px solid #6b1a1a; border-radius: 12px; padding: 28px 32px; text-align: center;
    }
    .result-label { font-size: 1.8rem; font-weight: 700; margin: 0; }
    .result-score { font-size: 1rem; margin-top: 8px; opacity: 0.8; }
    .metric-row { display: flex; gap: 12px; margin-top: 20px; justify-content: center; }
    .metric-box { background: rgba(255,255,255,0.05); border-radius: 8px; padding: 12px 20px; text-align: center; min-width: 120px; }
    .metric-value { font-size: 1.4rem; font-weight: 700; color: #ffffff; }
    .metric-label { font-size: 0.7rem; color: #7a9cc4; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .shap-container {
        background: #1a1f2e; border: 1px solid #1e3a5f;
        border-radius: 12px; padding: 24px; margin-top: 24px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #0f3460, #1a5fa8) !important;
        color: white !important; border: none !important; border-radius: 10px !important;
        padding: 14px 0 !important; font-size: 1rem !important; font-weight: 600 !important;
        letter-spacing: 0.5px !important; width: 100% !important;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #1a5fa8, #2176d4) !important; }
    label { color: #a8bdd4 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
@st.cache_resource
def load_model():
    model  = joblib.load('health_classifier.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_resource
def load_xgb_model():
    # Load XGBoost from inside stacking for SHAP
    import joblib
    stk = joblib.load('health_classifier.pkl')
    # Extract XGBoost base learner
    for name, estimator in stk.estimators_:
        if name == 'xgb':
            return estimator
    return None

try:
    model, scaler = load_model()
    xgb_model = load_xgb_model()
    model_loaded = True
except Exception as e:
    model_loaded = False

# ── Header ──
st.markdown("""
<div class="header-container">
    <div class="badge">🔬 Powered by Stacking Ensemble · 95.88% Recall</div>
    <h1 class="header-title">🏥 Health Risk Classifier</h1>
    <p class="header-subtitle">Enter patient health parameters to predict risk classification · RF + XGBoost + SVM → Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Make sure `health_classifier.pkl` and `scaler.pkl` are in the same folder as app.py")
    st.stop()

# ── Input Form ──
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-card"><div class="section-title">📊 Physiological Measurements</div>', unsafe_allow_html=True)
    age            = st.slider("Age (years)", 18, 90, 30)
    bmi            = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.0, step=0.1)
    blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=60, max_value=200, value=120)
    cholesterol    = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
    glucose        = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100)
    heart_rate     = st.number_input("Heart Rate (bpm)", min_value=40, max_value=150, value=72)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card"><div class="section-title">🌿 Lifestyle Factors</div>', unsafe_allow_html=True)
    sleep_hours    = st.slider("Sleep Hours / day", 1, 12, 7)
    exercise_hours = st.slider("Exercise Hours / day", 0, 5, 1)
    water_intake   = st.slider("Water Intake (litres)", 0, 5, 2)
    stress_level   = st.slider("Stress Level (1–10)", 1, 10, 5)
    smoking        = st.selectbox("Smoking", [0, 1], format_func=lambda x: "🚬 Smoker" if x else "✅ Non-Smoker")
    alcohol        = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "🍺 Yes" if x else "✅ No")
    diet_type      = st.selectbox("Diet Type", ["Non-Vegetarian", "Vegetarian", "Vegan"])
    diet_vegan      = 1 if diet_type == "Vegan"      else 0
    diet_vegetarian = 1 if diet_type == "Vegetarian" else 0
    diet            = 0 if diet_type == "Non-Vegetarian" else 1 if diet_type == "Vegetarian" else 2
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-card"><div class="section-title">🩺 Medical History & Profile</div>', unsafe_allow_html=True)
    mental_health     = st.slider("Mental Health Score (1–10)", 1, 10, 5)
    physical_activity = st.slider("Physical Activity Level (1–10)", 1, 10, 5)
    medical_history   = st.selectbox("Prior Medical Conditions", [0, 1], format_func=lambda x: "⚠️ Yes" if x else "✅ No")
    allergies         = st.selectbox("Known Allergies", [0, 1], format_func=lambda x: "⚠️ Yes" if x else "✅ No")
    blood_group    = st.selectbox("Blood Group", ["A", "AB", "B", "O"])
    blood_group_ab = 1 if blood_group == "AB" else 0
    blood_group_b  = 1 if blood_group == "B"  else 0
    blood_group_o  = 1 if blood_group == "O"  else 0
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Predict Button ──
predict_btn = st.button("🔍 Predict Health Risk", use_container_width=True)

FEATURE_COLS = [
    'Age', 'BMI', 'Blood_Pressure', 'Cholesterol', 'Glucose_Level',
    'Heart_Rate', 'Sleep_Hours', 'Exercise_Hours', 'Water_Intake',
    'Stress_Level', 'Smoking', 'Alcohol', 'Diet', 'MentalHealth',
    'PhysicalActivity', 'MedicalHistory', 'Allergies',
    'Diet_Type__Vegan', 'Diet_Type__Vegetarian', 'Blood_Group_AB',
    'Blood_Group_B', 'Blood_Group_O'
]

if predict_btn:
    input_data = pd.DataFrame([[
        age, bmi, blood_pressure, cholesterol, glucose,
        heart_rate, sleep_hours, exercise_hours, water_intake,
        stress_level, smoking, alcohol, diet, mental_health,
        physical_activity, medical_history, allergies,
        diet_vegan, diet_vegetarian, blood_group_ab,
        blood_group_b, blood_group_o
    ]], columns=FEATURE_COLS)

    prediction  = model.predict(input_data)
    probability = model.predict_proba(input_data)

    healthy_pct   = probability[0][0] * 100
    unhealthy_pct = probability[0][1] * 100

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Result Card ──
    if prediction[0] == 1:
        st.markdown(f"""
        <div class="result-unhealthy">
            <p class="result-label" style="color:#ff6b6b;">⚠️ UNHEALTHY — At Risk</p>
            <p class="result-score" style="color:#ffaaaa;">This individual is classified as <b>at higher health risk</b></p>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-value" style="color:#ff6b6b;">{unhealthy_pct:.1f}%</div>
                    <div class="metric-label">Risk Score</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color:#7a9cc4;">{healthy_pct:.1f}%</div>
                    <div class="metric-label">Healthy Probability</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color:#4da6ff;">95.88%</div>
                    <div class="metric-label">Model Recall</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-healthy">
            <p class="result-label" style="color:#51cf66;">✅ HEALTHY</p>
            <p class="result-score" style="color:#a9e6b8;">This individual is classified as <b>generally healthy</b></p>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-value" style="color:#51cf66;">{healthy_pct:.1f}%</div>
                    <div class="metric-label">Confidence</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color:#7a9cc4;">{unhealthy_pct:.1f}%</div>
                    <div class="metric-label">Risk Probability</div>
                </div>
                <div class="metric-box">
                    <div class="metric-value" style="color:#4da6ff;">95.88%</div>
                    <div class="metric-label">Model Recall</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SHAP Explanation ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="shap-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔍 WHY THIS PREDICTION? — SHAP Explanation</div>', unsafe_allow_html=True)

    try:
        with st.spinner("Calculating SHAP values..."):
            # Use XGBoost from stacking for SHAP (tree explainer works best)
            explainer  = shap.TreeExplainer(xgb_model)
            shap_vals  = explainer.shap_values(input_data)

            # Get top 10 features by absolute SHAP value
            shap_series = pd.Series(shap_vals[0], index=FEATURE_COLS)
            top10       = shap_series.abs().nlargest(10).index
            top10_vals  = shap_series[top10]

            # ── Bar Chart ──
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#1a1f2e')
            ax.set_facecolor('#1a1f2e')

            colors = ['#ff6b6b' if v > 0 else '#51cf66' for v in top10_vals]
            bars   = ax.barh(top10[::-1], top10_vals[::-1], color=colors[::-1],
                             edgecolor='none', height=0.6)

            ax.set_xlabel('SHAP Value  (+ = pushes toward Unhealthy,  − = pushes toward Healthy)',
                          color='#7a9cc4', fontsize=9)
            ax.tick_params(colors='#a8bdd4', labelsize=9)
            ax.spines['bottom'].set_color('#1e3a5f')
            ax.spines['left'].set_color('#1e3a5f')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.axvline(0, color='#2a3f5f', linewidth=1)
            ax.set_title('Top 10 Features Driving This Prediction',
                         color='#ffffff', fontsize=11, fontweight='bold', pad=12)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # ── Text Summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        risk_factors    = top10_vals[top10_vals > 0].sort_values(ascending=False)
        protect_factors = top10_vals[top10_vals < 0].sort_values()

        col_r, col_p = st.columns(2)

        with col_r:
            st.markdown("**⚠️ Risk Factors (pushing toward Unhealthy)**")
            if len(risk_factors) > 0:
                for feat, val in risk_factors.head(5).items():
                    st.markdown(f"- `{feat}` → +{val:.3f}")
            else:
                st.markdown("- None significant")

        with col_p:
            st.markdown("**✅ Protective Factors (pushing toward Healthy)**")
            if len(protect_factors) > 0:
                for feat, val in protect_factors.head(5).items():
                    st.markdown(f"- `{feat}` → {val:.3f}")
            else:
                st.markdown("- None significant")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {str(e)}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="color:#7a9cc4; font-size:0.8rem; text-align:center;">⚕️ Disclaimer: This tool is for research purposes only and does not constitute medical advice.</p>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<p style="text-align:center; color:#2a3f5f; font-size:0.75rem;">
    NovaGen Research Labs · Stacking Ensemble (RF + XGBoost + SVM) · Train Recall: 98.17% · Test Recall: 95.88%
</p>
""", unsafe_allow_html=True)