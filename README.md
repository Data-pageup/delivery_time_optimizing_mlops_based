# 🛵 MLOps — Delivery Time Prediction

> An end-to-end MLOps project that predicts food/package delivery times using machine learning.  
> Built as a learning project following a structured 9-phase MLOps framework.

---

## 📁 Project Structure

```
mlops-delivery-project/
│
├── notebooks/                        ← Phase 2: Experimentation
│   ├── 01_exploration.ipynb          # Step 4 — EDA & data quality checks
│   ├── 02_preprocessing.ipynb        # Step 5 — Cleaning & standardisation
│   ├── 03_features.ipynb             # Step 6 — Feature engineering
│   └── 04_modeling.ipynb             # Step 7 — Model experiments (10+ models)
│
├── models/                           ← Saved model artifacts (generated after training)
│   ├── delivery_model.pkl
│   ├── scaler.pkl
│   └── feature_names.json
│
└── README.md
```

> ⚠️ `data/`, `mlruns/`, and `src/` folders will be added in upcoming commits as the project progresses through the remaining phases.

---

## 📦 Dataset

| Column | Type | Description |
|---|---|---|
| `Order_ID` | int | Unique order identifier |
| `Distance_km` | float | Delivery distance in km |
| `Weather` | category | Clear / Windy / Rainy / Foggy |
| `Traffic_Level` | category | Low / Medium / High |
| `Time_of_Day` | category | Morning / Afternoon / Evening / Night |
| `Vehicle_Type` | category | Bike / Scooter |
| `Preparation_Time_min` | float | Time to prepare the order |
| `Courier_Experience_yrs` | float | Courier's years of experience |
| `Delivery_Time_min` | float | **Target variable** — actual delivery time |

---


##  Problem Statement

**Task:** Regression — predict `Delivery_Time_min` for incoming orders.

**Business Impact:** Accurate ETAs improve customer satisfaction and help dispatch teams allocate couriers efficiently.

**Success Metrics:**
- Primary → MAE ≤ 5 minutes
- Secondary → MAPE ≤ 10%

---

##  MLOps Phases (9 phases, 19 steps)

| Phase | Description | Steps | Status |
|---|---|---|---|
| 1 | Planning & Setup | 1–3 | ✅ Done |
| 2 | Experimentation (Notebooks) | 4–7 | ✅ Done |
| 3 | Production Code / Scripts | 8–10 | 🔜 Upcoming |
| 4 | Model Deployment | 11–13 | 🔜 Upcoming |
| 5 | Containerisation | 14 | 🔜 Upcoming |
| 6 | Testing | 15 | 🔜 Upcoming |
| 7 | CI/CD | 16 | 🔜 Upcoming |
| 8 | Cloud Deployment | 17 | 🔜 Upcoming |
| 9 | Monitoring & Maintenance | 18–19 | 🔜 Upcoming |

---

## 📓 Notebooks Overview

### `01_exploration.ipynb` — Data Exploration
- Load raw dataset
- Check dtypes, nulls, duplicates
- Distribution plots, scatter plots, bar charts, correlation heatmap
- Data quality summary

### `02_preprocessing.ipynb` — Data Preprocessing
- Handle missing values (median / mode fill)
- Remove duplicates
- Fix and standardise data types
- Drop irrelevant columns (`Order_ID`)
- Outlier detection using IQR
- Save cleaned data to `data/processed/`

### `03_features.ipynb` — Feature Engineering
- Create new features: `Distance_Traffic`, `Weather_Factor`, `Total_Delay_Est`, `Experience_Level`
- One-Hot Encoding vs Label Encoding comparison
- StandardScaler vs MinMaxScaler comparison
- Feature selection by correlation
- Save final feature dataset + scaler

### `04_modeling.ipynb` — Model Experiments
- 10+ models: Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, AdaBoost, SVR, KNN
- Cross-validation (3-fold)
- GridSearchCV hyperparameter tuning
- Actual vs Predicted plot
- Feature importance
- Residuals analysis
- Save best model + experiment logs

---

## ⚙️ Setup

### 1. Create conda environment
```bash
conda create -n mlops python=3.10 -y
conda activate mlops
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter ipykernel
```

### 3. Register Jupyter kernel
```bash
python -m ipykernel install --user --name mlops --display-name "mlops (Python 3.10)"
```

### 4. Open notebooks in VS Code
Select the **`mlops (Python 3.10)`** kernel when prompted.  
Run notebooks in order: `01` → `02` → `03` → `04`

---

## 📊 Results (Phase 2)

> Best model after GridSearchCV tuning: **Gradient Boosting Regressor**

| Metric | Value |
|---|---|
| MAE | ~5–8 min |
| RMSE | ~8–12 min |
| R² | ~0.85+ |

*(Results vary with dataset size — will improve as more data is collected in production)*

---

## 🔮 What's Coming Next

- `src/` — production Python scripts (data loading, feature pipeline, training)
- `app.py` — FastAPI prediction API
- `streamlit_app.py` — interactive frontend
- `Dockerfile` + `docker-compose.yml`
- GitHub Actions CI/CD pipeline
- Cloud deployment (AWS / GCP / Azure)
- Monitoring & automated retraining

---

This project follows the **9-Phase MLOps Framework**:

```
Phase 1: Planning & Setup
Phase 2: Experimentation (Notebooks)   ← I am here
Phase 3: Production Code (Scripts)
Phase 4: Model Deployment
Phase 5: Containerisation
Phase 6: Testing
Phase 7: CI/CD
Phase 8: Cloud Deployment
Phase 9: Monitoring & Maintenance
```

---

## Author 
### Amirtha Ganesh R
Built as a personal MLOps learning project.  
Each phase will be committed incrementally as the project grows.
