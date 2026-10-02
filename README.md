# 🌊 FloodImpact-LK

**FloodImpact-LK** is a multi-source spatio-temporal disaster-impact decision-support system for Sri Lanka. It integrates official information from the **Disaster Management Centre (DMC), Department of Meteorology, Irrigation Department, and NBRO** to forecast national human impact and safety-centre demand while providing hazard context and explainable AI.

## 🚀 Key Features

* **Multi-source data integration** from official Sri Lankan disaster agencies.
* **Timestamp-safe processing** to reduce temporal data leakage.
* **Human-impact forecasting** for the next eligible DMC Situation Report.
* **Two-stage safety-centre forecasting** for usage and expected population.
* **90% model-based prediction interval** for human-impact forecasts.
* **SHAP explainability** for understanding model predictions.
* **DRPI and SCPI research indicators** for response priority and safety-centre pressure.
* **Trend and rapid-growth monitoring** of affected populations.
* **Similar-event retrieval** from historical disaster situations.
* **District and GN-level hazard-context visualization**.
* **Source freshness monitoring** for current, stale, or unavailable information.
* **Frozen-model live inference** using a locked **46-feature schema**.
* **Safe-stop protection**, which preserves the previous valid dashboard output if an update fails.
* **Three audience views:** Public, Disaster Management/Relief, and Technical Evaluation.

## 📊 Dataset

* **1,281 labelled historical records**
* **46 locked model features**
* Main target: **affected population in the next eligible national DMC Situation Report**
* Maximum eligible report gap: **48 hours**
* District, DSD, and GN information is used for **hazard context**, not local impact forecasting.

## 🧠 Main Models

| Task                     | Model                                     |
| ------------------------ | ----------------------------------------- |
| Human Impact             | GradientBoostingRegressor with Huber loss |
| Safety-Centre Usage      | RandomForestClassifier                    |
| Safety-Centre Population | GradientBoostingRegressor with Huber loss |
| Escalation Risk          | HistGradientBoostingClassifier            |
| Prediction Interval      | Quantile Gradient Boosting models         |

The main human-impact model predicts the change between the **current and next affected population**.

## 📈 Held-Out Performance

* **Human Impact:** MAE 21,306.60, RMSE 65,900.17, R² 0.6481
* **Safety-Centre Usage:** Accuracy 88.67%, Balanced Accuracy 88.68%, F1 88.44%
* **Safety-Centre Population:** MAE 439.27, RMSE 947.27, R² 0.9724
* **Escalation Risk:** Accuracy 48.00%, Macro-F1 33.73%
* **90% Prediction Interval:** 80.67% empirical coverage

The human-impact model **did not outperform the persistence baseline on MAE**, and the escalation-risk model is retained only as an **experimental research component**.

## ⚙️ Live Workflow

**Official Sources → New Report Detection → PDF Processing → Cleaning → Scope Validation → Timestamp-Safe Integration → Feature Engineering → 46-Feature Validation → Frozen Models → Predictions & Analytics → Output Validation → Dashboard**

If a critical stage fails, the system performs a **safe stop** and retains the last valid dashboard output.

## 🗺️ Hazard Context

The dashboard combines:

* Flood and river conditions
* Weather and heavy-rain information
* Landslide warnings
* DMC Situation Report hazard information
* District and GN-level contextual maps

Human-impact and safety-centre predictions remain **national-level**. District and GN maps provide **context only**.

## 🛠️ Technology Stack

**Python, Streamlit, Pandas, NumPy, Scikit-learn, SHAP, PyMuPDF, Folium, GeoJSON, Altair, Joblib, Requests, Git/GitHub**

## ⚠️ Key Limitations

* Human-impact prediction does not currently beat persistence on held-out MAE.
* Report intervals are irregular.
* The 90% prediction interval achieved **80.67%** empirical coverage.
* Escalation-risk classification has weak temporal generalisation.
* District/GN-level impact targets are unavailable for validated local forecasting.
* SCPI is not actual shelter-capacity utilisation.
* Missing or stale hazard information **does not mean an area is safe**.
* SHAP explanations show model behaviour, not causality.

## 🎯 Intended Use

FloodImpact-LK is an **academic disaster-impact research and decision-support system** designed to support:

* National human-impact forecasting
* Safety-centre demand assessment
* Hazard-context interpretation
* Uncertainty-aware decision support
* Explainable machine-learning analysis
* Historical disaster comparison

It is designed to **support human decision-making, not replace official disaster agencies or warnings**.

## ▶️ Run the Application

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

**Live Dashboard:**
https://floodimpact-lk-mxfflfgacnn2mdvdp3pvkm.streamlit.app/

## 👤 Author

**N. Yathursan**
BSc (Hons) in Data Science
Sabaragamuwa University of Sri Lanka
