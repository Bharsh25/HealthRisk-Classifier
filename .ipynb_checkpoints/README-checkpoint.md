# 🏥 NovaGen Health Risk Classifier

> A machine learning web application that classifies individuals as **Healthy** or **Unhealthy** based on physiological measurements, lifestyle factors, and medical history.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-1.6+-green)
![Recall](https://img.shields.io/badge/Test%20Recall-95.88%25-brightgreen)

---

## 📌 Problem Statement

**NovaGen Research Labs** conducts large-scale population health studies to understand how underlying health conditions influence disease risk and long-term outcomes. Researchers needed a reliable way to distinguish between individuals with generally healthy profiles and those at higher health risk — to support:

- Selecting eligible participants for clinical trials
- Stratifying populations for risk-based analysis and outcome comparison

---

## 📊 Dataset

| Property | Details |
|---|---|
| Records | 9,800 individuals |
| Features | 22 (numerical + categorical) |
| Target | Binary — Healthy (0) / Unhealthy (1) |
| Class Balance | 52.14% Unhealthy / 47.86% Healthy |
| Source | Multi-observational health study |

### Features Used

| Feature | Description |
|---|---|
| Age | Age of individual (years) |
| BMI | Body Mass Index |
| Blood_Pressure | Systolic blood pressure (mmHg) |
| Cholesterol | Cholesterol level (mg/dL) |
| Glucose_Level | Blood glucose level (mg/dL) |
| Heart_Rate | Resting heart rate (bpm) |
| Sleep_Hours | Average sleep hours/day |
| Exercise_Hours | Average exercise hours/day |
| Water_Intake | Daily water intake (litres) |
| Stress_Level | Stress level (1–10) |
| Smoking | Smoking habit (0/1) |
| Alcohol | Alcohol consumption (0/1) |
| Diet | Diet category (encoded) |
| MentalHealth | Mental health score |
| PhysicalActivity | Physical activity level |
| MedicalHistory | Prior medical conditions (0/1) |
| Allergies | Known allergies (0/1) |
| Diet_Type__Vegan | One-hot: Vegan diet |
| Diet_Type__Vegetarian | One-hot: Vegetarian diet |
| Blood_Group_AB | One-hot: Blood group AB |
| Blood_Group_B | One-hot: Blood group B |
| Blood_Group_O | One-hot: Blood group O |

---

## 🤖 Model Architecture

### Stacking Ensemble (Best Model)

```
Base Learners:
├── Random Forest Classifier
├── XGBoost Classifier  
└── SVM (inside Pipeline with StandardScaler)
         ↓
Meta Learner:
└── Logistic Regression
```

### Why Stacking?
- Medical data has complex non-linear relationships
- Stacking captures patterns no single algorithm does alone
- Diverse base learners (tree-based + distance-based) complement each other
- Best accuracy + recall on tabular health data

---

## 📈 Results

### Model Comparison

| Model | Accuracy | Recall |
|---|---|---|
| Random Forest | ~88-90% | ~87-89% |
| XGBoost | ~90-92% | ~90-92% |
| SVM | ~85-88% | ~84-87% |
| **Stacking (Final)** | **~94%** | **95.88%** |

### Overfitting Check

```
Train Recall : 98.17%
Test  Recall : 95.88%
Gap          : 2.29%  ✅ No significant overfitting
```

> Generalisation gap under 3% — model performs consistently on unseen data.

---

## 🔍 SHAP Explainability

Top features driving predictions (in order of importance):

```
1. BMI              ████████████████  Most Important
2. Blood_Pressure   █████████████
3. Cholesterol      █████████
4. Glucose_Level    ████████
5. Sleep_Hours      ███████
6. Stress_Level     ██████
7. Water_Intake     █████
8. Heart_Rate       ████
9. Age              ███
10. Exercise_Hours  ███
```

### Key Insights
- 🔴 High BMI, Blood Pressure, Cholesterol → pushes toward **Unhealthy**
- 🔵 More Sleep, Lower Stress → pushes toward **Healthy**
- ✅ All SHAP findings align with real medical knowledge

---

## 🚀 Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/health-classifier.git
cd health-classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

Opens at → `http://localhost:8501`

---

## 📁 Project Structure

```
health-classifier/
│
├── app.py                  # Streamlit web application
├── health_classifier.pkl   # Trained stacking model
├── scaler.pkl              # Fitted StandardScaler
├── requirements.txt        # Python dependencies
├── notebook.ipynb          # Full ML pipeline notebook
└── README.md               # Project documentation
```

---

## 🧪 Test Cases

| Case | Profile | Expected |
|---|---|---|
| Young healthy lifestyle | Age 25, BMI 21.5, No smoking | ✅ Healthy |
| Older unhealthy lifestyle | Age 55, BMI 38, Smoker, High BP | ⚠️ Unhealthy |
| Borderline mixed | Age 42, BMI 27, Moderate stress | 🟡 Borderline |
| Young but unhealthy | Age 22, BMI 32, Smoker | ⚠️ Unhealthy |
| Old but healthy | Age 68, BMI 23, Active lifestyle | ✅ Healthy |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.8+ | Core language |
| Scikit-learn | ML models + preprocessing |
| XGBoost | Gradient boosting base learner |
| SHAP | Model explainability |
| Streamlit | Web application |
| Joblib | Model serialization |
| Pandas / NumPy | Data processing |
| Matplotlib / Seaborn | Visualization |

---

## ⚕️ Disclaimer

> This tool is developed for **research and educational purposes only**. It does not constitute medical advice and should not be used as a substitute for professional medical diagnosis or treatment.

---

## 👨‍💻 Author

**Harsh Bhendarkar**  
📧 bhendarkarharsh92@gmail.com

---

## 📄 License

This project is licensed under the MIT License.
