🌊 FloodImpact-LK

A multi-source spatio-temporal disaster-impact decision-support system for national human-impact forecasting, safety-centre demand estimation, explainability and hazard-context visualization in Sri Lanka.

FloodImpact-LK combines publicly available disaster information from the Disaster Management Centre (DMC), Department of Meteorology, Irrigation Department, and National Building Research Organisation (NBRO). It processes official reports, performs timestamp-safe multi-source integration, generates national-level forecasts using frozen machine-learning models, and presents the results through an audience-aware Streamlit dashboard.

Current application behavior: Live updates use a locked 46-feature schema and frozen deployment models. Routine updates do not retrain the models. If a critical processing, validation or inference step fails, the system performs a safe stop and keeps the previous valid dashboard outputs.

🚀 Key Features

📄 Multi-Source Official Data Integration: Combines DMC Situation Reports, weather/advisory information, river/flood information, and rainfall-induced landslide warnings.

🕒 Timestamp-Safe Integration: Uses only source information that was available at or before each prediction timestamp, reducing temporal leakage.

👥 National Human-Impact Forecasting: Estimates the affected-person total expected in the next eligible general DMC Situation Report.

🏥 Two-Stage Safety-Centre Forecasting: Predicts whether positive safety-centre use is expected and, if positive, estimates the next safety-centre population.

📉 Uncertainty Estimation: Uses lower and upper quantile models to produce a nominal 90% model-based prediction interval for human impact.

🧠 SHAP Explainability: Shows which features influenced the latest human-impact forecast and provides global model influence information.

🚦 Experimental Escalation Risk: Produces project-defined Low, Moderate, High and Critical research classes while clearly showing that this component has weak held-out generalisation.

📊 DRPI and SCPI Research Indicators: Provides project-defined 0–100 decision-support indices for response priority and relative safety-centre pressure.

📈 Trend & Rapid-Growth Monitoring: Tracks recent affected-population changes and identifies unusually rapid growth relative to historical patterns.

🕰️ Similar Event Retrieval: Retrieves historical situations that are nearest to the current engineered feature state.

🌧️ Multi-Source Hazard Context: Summarizes flood, weather/heavy-rain, river and landslide context together with source freshness.

🗺️ District Hazard Context Map: Visualizes available local hazard context while keeping supervised impact predictions at national level.

📍 GN Division Detail View: Uses official Grama Niladhari boundary geometry to localize available hazard context without downscaling the national model.

🧭 Source Freshness Monitoring: Explicitly distinguishes current, stale and unavailable supporting hazard information.

🔒 Frozen-Model Live Inference: Routine updates load saved .joblib models and do not automatically retrain the system.

🛡️ Safe-Stop Output Protection: Invalid or incomplete updates do not replace the previous valid dashboard state.

👥 Three Audience Views: Provides different information depth for the public, disaster-management users and technical evaluators.

🖥️ Interactive Streamlit Interface: Provides practical access to forecasts, explanations, maps, trends, historical context and evaluation results.

📊 Dataset

FloodImpact-LK is built from publicly available Sri Lankan disaster-report streams.

Main Official Sources

Source

Main Information

Role in FloodImpact-LK

Disaster Management Centre (DMC)

Situation Reports, affected people, safety-centre information, report timestamps

Primary national human-impact target source

Department of Meteorology

Weather forecasts, rainfall information and advisories

Meteorological context

Irrigation Department

River levels, station information, flood/alert status

River and flood context

NBRO

Rainfall-induced landslide warnings and affected administrative areas

Landslide context

Modelling Dataset

📑 1,281 labelled historical records used for the final deployment refit

🔢 46 locked model features

🎯 Main human-impact target: next eligible national DMC Situation Report

⏱️ Maximum eligible report gap: 48 hours

🗺️ District / DSD / GN information is used for hazard context, not local human-impact prediction

⚙️ End-to-End Workflow

Official Data Sources
        │
        ▼
New Report Detection
        │
        ▼
Source-Specific PDF Processing
        │
        ▼
Cleaning & Standardization
        │
        ▼
DMC Scope Classification
        │
        ▼
Timestamp-Safe Multi-Source Integration
        │
        ▼
Feature Engineering
        │
        ▼
46-Feature Schema Validation
        │
        ▼
Frozen Machine-Learning Models
        │
        ├──────────────────────────────┐
        ▼                              ▼
Predictions & Uncertainty        Research Analytics
        │                         SHAP / DRPI / SCPI
        │                         Trends / Similar Events
        └──────────────┬───────────────┘
                       ▼
                Output Validation
                       │
                 ┌─────┴─────┐
                 ▼           ▼
             Success       Failure
                 │           │
                 ▼           ▼
        Update Dashboard   Safe Stop
                              │
                              ▼
                    Keep Last Valid Output

🧠 Predictive Models

The final deployment configuration uses task-specific machine-learning models.

Task

Final Model

Main Purpose

👥 Human Impact

GradientBoostingRegressor with Huber loss

Predict next-report national affected-population change

🏥 Safety-Centre Usage

RandomForestClassifier

Predict probability of positive safety-centre use

🏥 Safety-Centre Population

GradientBoostingRegressor with Huber loss

Estimate positive-demand amount

🚦 Escalation Risk

HistGradientBoostingClassifier

Experimental Low / Moderate / High / Critical classification

📉 Human-Impact Interval

Two Quantile GradientBoostingRegressors

Estimate 5th and 95th percentile bounds

👥 Human-Impact Forecasting

The central human-impact model predicts the signed-log transformed change:

next affected population - current affected population

The predicted change is inverse-transformed and added to the current reported affected population.

The deployed model uses:

GradientBoostingRegressor
loss = huber
n_estimators = 250
learning_rate = 0.03
max_depth = 2
min_samples_leaf = 10

Why this design?

handles nonlinear relationships;

reduces the influence of extreme residuals through Huber loss;

uses shallow trees and a low learning rate to discourage overly complex fitting to individual event spikes.

🏥 Safety-Centre Forecasting

Safety-centre demand is modeled as a two-stage process.

Stage 1 — Usage Classification

RandomForestClassifier
n_estimators = 400
min_samples_leaf = 3
max_features = 0.7
class_weight = balanced
threshold = 0.5

This stage predicts whether the next eligible report is expected to contain positive safety-centre use.

Stage 2 — Positive-Demand Amount

GradientBoostingRegressor
loss = huber
n_estimators = 250
learning_rate = 0.03
max_depth = 2
min_samples_leaf = 8

If Stage 1 predicts positive usage, Stage 2 estimates the next safety-centre population.

The two-stage design separates whether demand occurs from how large the demand is.

📉 Human-Impact Uncertainty

Two quantile Gradient Boosting models estimate:

Lower bound: 5th percentile  (α = 0.05)
Upper bound: 95th percentile (α = 0.95)

Together they produce a nominal:

90% model-based prediction interval

This is not a guaranteed 90% confidence interval. Held-out empirical coverage was lower than the nominal level.

📈 Final Held-Out Performance

Output

Main Held-Out Result

Interpretation

👥 Human Impact

MAE 21,306.60, RMSE 65,900.17, R² 0.6481

Persistence MAE 21,283.91 was slightly better

🏥 Safety-Centre Usage

Accuracy 88.67%, Balanced Accuracy 88.68%, F1 88.44%

Comparatively strong temporal generalisation

🏥 Safety-Centre Population

MAE 439.27, RMSE 947.27, R² 0.9724

Persistence comparison is metric-dependent

🚦 Escalation Risk

Accuracy 48.00%, Balanced Accuracy 33.84%, Macro-F1 33.73%

Weak temporal generalisation; experimental only

📉 90% Human-Impact Interval

Empirical coverage 80.67%

Undercoverage relative to the nominal 90% level

Important Interpretation

The human-impact ML model is not claimed to outperform persistence on MAE.

Safety-centre population performance is metric-dependent relative to persistence.

The escalation-risk model is retained only as an experimental research component.

The 90% interval is not guaranteed to contain the true next value 90% of the time.

🚦 DRPI and SCPI

DRPI — Disaster Response Priority Index

A project-defined 0–100 research indicator combining:

current human impact;

predicted impact growth;

predicted safety demand;

hazard context;

rapid-growth behaviour.

SCPI — Safety-Centre Pressure Index

A project-defined 0–100 relative demand-pressure indicator combining:

current safety-centre population;

forecast safety-centre population;

safety-use probability;

predicted safety growth;

hazard context.

Research Bands

Band

Score

Low

< 25

Moderate

25 – < 50

High

50 – < 75

Very High

≥ 75

DRPI and SCPI are research-derived interpretation aids, not official Sri Lankan government warning categories.

SCPI is not shelter occupancy or capacity utilisation because reliable facility-capacity data were not available.

📈 Trend and Rapid-Growth Monitoring

The dashboard separately monitors observed historical affected-population changes.

The refined rapid-growth detector uses:

Minimum material increase = 142 people
Extreme percentile = 95th percentile

A rapid-growth anomaly requires:

valid comparable previous context;

a meaningful absolute increase;

unusually high absolute growth;

unusually high relative growth.

This helps avoid false alarms caused by:

very small absolute increases with large percentage changes;

large absolute changes that are historically normal for a large event.

Trend monitoring is based on observed history and is separate from the next-report forecast.

🕰️ Similar Event Retrieval

The system retrieves historical situations that are nearest to the current transformed feature state.

The similarity output is intended to answer:

“Have we seen a historical situation that looked somewhat like this one?”

Similarity is:

a feature-space resemblance measure;

useful for historical reference and briefing;

not a probability;

not forecast confidence;

not evidence that history will repeat.

🧠 SHAP Explainability

FloodImpact-LK uses Tree SHAP to explain the human-impact model.

Latest Prediction Explanation

Shows which transformed features pushed the modelled change:

upward;

downward;

relative to the model baseline.

Global Explainability

Mean absolute SHAP values are used to summarize which features are generally most influential across the evaluated data.

SHAP explains the model's signed-log change prediction, not the final affected-person total directly.

SHAP values describe model behaviour, not causal relationships.

🌧️ Hazard Context

The dashboard combines supporting information from:

flood / river conditions;

weather and heavy-rain context;

landslide warnings;

latest Situation Report hazard mentions.

It also displays source freshness so that stale or missing information is not mistaken for current evidence.

Unavailable or stale information does not mean that an area is safe.

🗺️ District and GN Hazard Context

District Overview

Displays available district-level hazard context.

GN Division Detail

Uses official Grama Niladhari boundary polygons to localize available warning context.

Important interpretation rules:

human-impact forecasts remain national;

safety-centre forecasts remain national;

district and GN maps are contextual visualization only;

a GN without a matched current warning is not automatically safe;

DSD-level landslide warnings may be displayed across GN polygons inside the parent DSD;

DSD-derived shading remains DSD-level information, not an independent GN warning;

a river-station location does not mean the whole surrounding GN is flooded.

👤 Audience-Specific Dashboard Views

FloodImpact-LK provides three audience views.

👥 General / Public

Designed for simple interpretation.

Pages include:

🏠 Overview

🌧️ Hazard Context

🗺️ District Map

📈 Trend & Anomalies

📘 How to Read the System

🛟 Disaster Management / Relief Team

Designed for planning-oriented information.

Pages include:

🏠 Overview

👥 Human Impact

🏥 Safety Centres

🌧️ Hazard Context

🗺️ District Map

🚦 Risk & Priority

📈 Trend & Anomalies

🕰️ Similar Events

📘 How to Read the System

🧪 Evaluation Panel / Technical

Provides the complete technical view.

Pages include:

🏠 Overview

👥 Human Impact

🏥 Safety Centres

🌧️ Hazard Context

🗺️ District Map

🚦 Risk & Priority

📈 Trend & Anomalies

🕰️ Similar Events

🧠 Explainability

📊 Model Performance

📘 How to Read the System

Changing the audience view changes the presentation and page access, not the underlying predictions.

🛡️ Safe Latest-Data Inference

Routine live updates do not retrain the models.

The live workflow performs:

Check official sources
        ↓
Download new reports
        ↓
Incremental PDF processing
        ↓
Scope validation
        ↓
Timestamp standardization
        ↓
Timestamp-safe source integration
        ↓
Rebuild 46-feature input
        ↓
Validate feature schema
        ↓
Load frozen models
        ↓
Generate predictions and analytics
        ↓
Validate candidate outputs
        ↓
Replace dashboard outputs only if valid

If a critical step fails:

SAFE STOP
    ↓
Keep previous valid dashboard outputs

🛠️ Technology Stack

Final Streamlit Application

🐍 Python

🖥️ Streamlit

🐼 Pandas

🔢 NumPy

🤖 Scikit-learn

🧠 SHAP

📄 PyMuPDF

🗺️ Folium / GeoJSON

📊 Altair

💾 Joblib

Project / Data Pipeline

🌐 Requests

📓 Jupyter Notebook

🌿 Git & GitHub

📦 Python Dependencies

The main dependencies include:

streamlit
pandas
numpy
scikit-learn
shap
PyMuPDF
folium
joblib
altair
requests

Install the repository dependencies using:

python -m pip install -r requirements.txt

📁 Project Structure

FloodImpact-LK/
│
├── app.py
├── update_latest.py
├── latest_inference.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── scraping/
│   ├── scrape_dmc.py
│   ├── scrape_weather.py
│   ├── scrape_river.py
│   └── scrape_landslide.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── core_model_features.joblib
│   ├── deployment_human_impact_model.joblib
│   ├── deployment_safety_usage_model.joblib
│   ├── deployment_safety_demand_model.joblib
│   ├── deployment_escalation_risk_model.joblib
│   ├── deployment_lower_90_model.joblib
│   ├── deployment_upper_90_model.joblib
│   └── model_metadata.json
│
└── outputs/
    └── figures/

⚡ Installation

1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd FloodImpact-LK

2. Create a Virtual Environment

python -m venv .venv

3. Activate the Virtual Environment

Windows PowerShell

.\.venv\Scripts\Activate.ps1

Windows Command Prompt

.venv\Scripts\activate

Linux / macOS

source .venv/bin/activate

4. Install Dependencies

python -m pip install -r requirements.txt

▶️ Run the Application

From the project root directory:

streamlit run app.py

or:

python -m streamlit run app.py

The application normally opens at:

http://localhost:8501

🔄 Check Latest Official Data

From the Dashboard

Click:

Check latest official data

From the Terminal

python update_latest.py --once

This checks the supported official sources and invokes the safe latest-inference workflow when appropriate.

Routine latest-data checks do not automatically retrain the machine-learning models.

📋 Using the Application

Start the Streamlit dashboard.

Select an Audience View.

Check the latest snapshot timestamp.

Start with Overview.

Review Hazard Context and source freshness.

Use District Map / GN Division Detail for local hazard context.

For operational detail, inspect:

Human Impact;

Safety Centres;

Risk & Priority;

Trends & Anomalies;

Similar Events.

For technical evaluation, inspect:

SHAP Explainability;

Model Performance.

Use Check latest official data when a new-source check is required.

Confirm that the latest update completed successfully before interpreting newly generated outputs.

🧪 Model Evaluation Design

FloodImpact-LK uses a chronological and event-aware evaluation strategy rather than random splitting.

The development process separates:

Training
   ↓
Validation / Model Selection
   ↓
Locked Model Choice
   ↓
Final Held-Out Chronological Test

This is important because adjacent DMC Situation Reports are:

temporally related;

cumulative;

often highly similar.

Randomly mixing past and future observations could therefore overstate model generalisation.

⚠️ Limitations

DMC Situation Report targets are cumulative, making persistence a strong short-horizon baseline.

The human-impact ML model did not outperform persistence on held-out MAE.

Report timing is irregular, so the forecast horizon is defined by the next eligible report rather than a fixed number of hours.

The escalation-risk classifier has weak temporal generalisation and is retained only as an experimental research signal.

The nominal 90% prediction interval achieved only 80.67% empirical held-out coverage.

Some source PDFs may contain layout changes, extraction problems or missing fields.

Missing or stale source information must not be interpreted as absence of hazard.

The current supervised human-impact and safety-centre targets are national-level.

Reliable district/GN human-impact targets were not available for validated local forecasting.

Actual safety-centre facility-capacity data were not available, so SCPI is not a true occupancy indicator.

DRPI and SCPI are project-defined research indicators rather than official government classifications.

Similar-event scores are not probabilities.

SHAP values are model attributions and do not establish causal relationships.

🎯 Intended Use

FloodImpact-LK is designed as a:

Multi-source disaster-impact research and decision-support system

It aims to support:

integrated interpretation of fragmented official reports;

national human-impact forecasting research;

safety-centre demand assessment;

uncertainty-aware decision support;

transparent model inspection;

historical context exploration;

hazard-context visualization.

The system is intended to support human interpretation, not replace official agencies.

FloodImpact-LK is not intended to:

replace DMC warnings;

replace NBRO landslide warnings;

replace Department of Meteorology advisories;

replace Irrigation Department flood information;

automatically issue evacuation instructions;

provide validated district/GN human-impact forecasts;

provide guaranteed emergency outcomes.

🔮 Future Work

Possible future improvements include:

larger and more diverse historical disaster datasets;

improved human-impact forecasting beyond the persistence baseline;

recalibration of the nominal 90% prediction interval;

stronger temporal generalisation for escalation-risk classification;

reliable district-level human-impact targets;

GN-level forecasting only when suitable local target data become available;

additional rainfall, river-discharge and catchment features;

elevation, slope, drainage, soil and land-cover context;

population exposure and critical-infrastructure layers;

remote-sensing flood extent;

automated model-drift monitoring;

model version registry;

scheduled production updates;

automated parser-layout tests;

containerized deployment;

formal user studies with disaster-management users.

👤 Author

N. Yathursan
BSc (Hons) in Data Science
Sabaragamuwa University of Sri Lanka

📌 Disclaimer

FloodImpact-LK was developed as an academic Data Science capstone research project.

All forecasts, research indices, similarity outputs, SHAP explanations and escalation classes should be interpreted as decision-support research outputs.

Official disaster warnings, evacuation instructions and emergency-response decisions should always be obtained from and verified against the responsible Sri Lankan government agencies.

FloodImpact-LK does not provide official disaster warnings or operational certification.
