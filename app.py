from pathlib import Path
import json
import subprocess
import sys
import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


# FLOODIMPACT-LK INTERACTIVE DASHBOARD

st.set_page_config(
    page_title="FloodImpact-LK",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# PATHS

ROOT = Path(__file__).resolve().parent

PROCESSED = (
    ROOT
    / "data"
    / "processed"
)

FIGURES = (
    ROOT
    / "outputs"
    / "figures"
)

# CURRENT LIVE PIPELINE OUTPUTS


PAYLOAD_PATH = (
    PROCESSED
    / "dashboard_payload.json"
)

SIMILAR_PATH = (
    PROCESSED
    / "similar_events_dashboard.csv"
)

GLOBAL_SHAP_PATH = (
    PROCESSED
    / "shap_feature_importance.csv"
)

LATEST_SHAP_PATH = (
    PROCESSED
    / "shap_latest_deployment_contributions.csv"
)

TREND_PATH = (
    PROCESSED
    / "trend_history_dashboard.csv"
)

MAP_PATH = (
    PROCESSED
    / "floodimpact_lk_district_hazard_map.html"
)

UPDATE_LATEST_PATH = (
    ROOT
    / "update_latest.py"
)


SHAP_IMAGE_CANDIDATES = {

    "Global SHAP importance": [
        FIGURES
        / "shap_global_bar.png",
    ],

    "SHAP beeswarm": [
        FIGURES
        / "shap_beeswarm.png",
    ],

    "Latest prediction waterfall": [
        FIGURES
        / "shap_latest_deployment_waterfall.png",
    ],
}

# STYLE

st.markdown(
    """
    <style>
    /* =====================================
        CUSTOM TOP METRIC CARDS
        ===================================== */

    .custom-metric-card {
        border: 1px solid rgba(90, 170, 200, 0.35);
        border-radius: 16px;
        padding: 1rem 1.1rem;

        background: rgba(23, 42, 55, 0.96);

        min-height: 155px;
        height: 155px;

        display: flex;
        flex-direction: column;
        justify-content: center;

        box-sizing: border-box;
        color: white;
    }

    .custom-metric-label {
        font-size: 0.93rem;
        font-weight: 600;
        line-height: 1.35;
        color: rgba(255,255,255,0.82);

        white-space: normal;
        overflow: visible;
        text-overflow: unset;

        margin-bottom: 0.55rem;
    }

    .custom-metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.15;
    }

    .metric-delta-positive,
    .metric-delta-negative,
    .metric-delta-neutral {
        display: inline-block;
        width: fit-content;

        margin-top: 0.55rem;
        padding: 0.22rem 0.48rem;

        border-radius: 999px;

        font-size: 0.85rem;
        font-weight: 700;
    }

    .metric-delta-positive {
        color: #7ee2a8;
        background: rgba(40, 150, 90, 0.18);
    }

    .metric-delta-negative {
        color: #ff8b8b;
        background: rgba(200, 70, 70, 0.20);
    }

    .metric-delta-neutral {
        color: #d4d8dc;
        background: rgba(150, 150, 150, 0.15);
    }
    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #102f4c 0%,
                #164a66 100%
            );
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    .hero {
        padding: 1.35rem 1.55rem;
        border-radius: 18px;

        background:
            linear-gradient(
                120deg,
                #123c5a 0%,
                #176b87 58%,
                #1b8a8f 100%
            );

        color: white;

        margin-bottom: 1rem;

        box-shadow:
            0 8px 28px
            rgba(0,0,0,0.12);
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .hero-sub {
        margin-top: .45rem;
        opacity: .94;
        line-height: 1.55;
    }

    .section-card {
        border:
            1px solid
            rgba(49,77,101,0.13);

        border-radius: 14px;

        padding: 1rem 1.05rem;

        background:
            rgba(255,255,255,0.86);

        box-shadow:
            0 3px 14px
            rgba(23,52,73,0.055);

        margin-bottom: .7rem;
    }

    /* ================================
       DECISION SNAPSHOT CARDS
       ================================ */

    .plain-card {
        border: 1px solid rgba(90, 170, 200, 0.35);
        border-radius: 16px;

        padding: 1rem 1.15rem;

        background: rgba(23, 42, 55, 0.96);

        min-height: 175px;
        height: 175px;

        display: flex;
        flex-direction: column;
        justify-content: center;

        box-sizing: border-box;

        color: white;
    }

    .small-label {
        text-transform: uppercase;
        letter-spacing: .06em;
        font-size: .72rem;
        font-weight: 700;
        color: rgba(255,255,255,0.72);
        margin-bottom: .45rem;
    }

    .big-value {
        font-size: 1.55rem;
        font-weight: 800;
        line-height: 1.15;
        color: #ffffff;
    }

    .muted {
        color: rgba(255,255,255,0.72);
        font-size: .88rem;
        margin-top: .55rem;
        line-height: 1.45;
    }
    
    .badge {
        display: inline-block;
        padding: .28rem .58rem;
        border-radius: 999px;
        font-size: .78rem;
        font-weight: 700;
        margin: .12rem .15rem .12rem 0;
    }

    .badge-green {
        background:#e7f6ec;
        color:#176b37;
    }

    .badge-amber {
        background:#fff4d8;
        color:#8a5a00;
    }

    .badge-red {
        background:#fde8e8;
        color:#9d2525;
    }

    .badge-blue {
        background:#e8f1fb;
        color:#255e93;
    }

    .badge-gray {
        background:#eef1f4;
        color:#5d6670;
    }

    .meaning-box {
    padding: 1rem 1.15rem;

    background: #162b3a !important;

    border: 1px solid rgba(46, 196, 182, 0.45);
    border-left: 5px solid #2ec4b6;

    border-radius: 12px;

    margin: .6rem 0 1rem 0;

    color: #ffffff !important;

    font-size: 1rem;
    line-height: 1.6;
}

.meaning-box,
.meaning-box p,
.meaning-box b,
.meaning-box span {
    color: #ffffff !important;
}

.meaning-box b {
    color: #7ee7df !important;
    font-weight: 800;
}
    .warning-box {
    padding: 1rem 1.15rem;

    background: #162b3a !important;

    border: 1px solid rgba(245, 180, 45, 0.50) !important;
    border-left: 5px solid #f5b42d !important;

    border-radius: 12px;

    margin: .7rem 0;

    color: #ffffff !important;

    font-size: 1rem;
    font-weight: 500;
    line-height: 1.65;
}

.warning-box * {
    color: #ffffff !important;
}

.warning-box b {
    color: #ffd166 !important;
    font-weight: 800;
}
    .critical-box {
    padding: 1rem 1.15rem;

    background: #2b1d22 !important;

    border: 1px solid rgba(239, 90, 90, 0.45);
    border-left: 5px solid #ef5a5a;

    border-radius: 12px;

    margin: .7rem 0;

    color: #ffffff !important;

    font-size: 1rem;
    line-height: 1.6;
}

.critical-box,
.critical-box p,
.critical-box span {
    color: #ffffff !important;
}

.critical-box b {
    color: #ff8a8a !important;
    font-weight: 800;
}

    .footer-note {
        opacity: .7;
        font-size: .8rem;
        line-height: 1.5;
    }
    /* ================================
       TOP METRIC CARDS - SAME SIZE
       ================================ */

    div[data-testid="stMetric"] {
        border: 1px solid rgba(90, 170, 200, 0.35);
        border-radius: 16px;
        padding: 1rem 1.1rem;

        background: rgba(23, 42, 55, 0.96);

        min-height: 155px;
        height: 155px;

        display: flex;
        flex-direction: column;
        justify-content: center;

        box-sizing: border-box;
    }

    div[data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.82) !important;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
    }

    div[data-testid="stMetricDelta"] {
        font-weight: 700;
    }

    .custom-metric-card {
    transition: transform 0.2s ease,
                box-shadow 0.2s ease;
    }

    .custom-metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }

    /* Decision Snapshot cards */
    .plain-card {
        transition: transform 0.2s ease,
                    box-shadow 0.2s ease;
    }

    .plain-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.25);
    }

    # /* Uncertainty box - dark and clean */
    # div[data-testid="stAlert"] {
    #     background: rgba(23, 42, 55, 0.96) !important;
    #     border: 1px solid rgba(245, 180, 45, 0.55) !important;
    #     border-left: 4px solid #f5b42d !important;
    #     border-radius: 12px !important;
    # }

    div[data-testid="stAlert"] p {
        color: #f5f7fa !important;
        font-size: 0.98rem !important;
        line-height: 1.55 !important;
    }
    .shap-note {
        margin-top: 0.7rem;
        padding: 0.9rem 1.1rem;

        background: #162b3a;

        border: 1px solid rgba(46, 196, 182, 0.45);
        border-left: 5px solid #2ec4b6;

        border-radius: 10px;

        color: #ffffff !important;

        font-size: 0.95rem;
        line-height: 1.6;
    }

    .shap-note b {
        color: #7ee7df !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# HELPERS

def load_json(
    path
):

    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def load_csv(
    path
):

    if not path.exists():

        return pd.DataFrame()

    try:

        return pd.read_csv(
            path
        )

    except Exception:

        return pd.DataFrame()


def nfmt(
    value,
    decimals=0,
    fallback="Unavailable",
):

    try:

        if (
            value is None
            or
            pd.isna(
                value
            )
        ):

            return fallback

        return (
            f"{float(value):,.{decimals}f}"
        )

    except Exception:

        return fallback


def pct(
    value,
    decimals=1,
    already_percent=False,
    fallback="Unavailable",
):

    try:

        if (
            value is None
            or
            pd.isna(
                value
            )
        ):

            return fallback

        x = float(
            value
        )

        if not already_percent:

            if abs(
                x
            ) <= 1:

                x *= 100

        return (
            f"{x:.{decimals}f}%"
        )

    except Exception:

        return fallback


def clamp01(
    value
):

    try:

        if (
            value is None
            or
            pd.isna(
                value
            )
        ):

            return 0.0

        return max(
            0.0,
            min(
                1.0,
                float(
                    value
                ),
            ),
        )

    except Exception:

        return 0.0


def badge(
    text,
    kind="blue",
):

    return (
        f'<span class="badge badge-{kind}">'
        f'{text}'
        '</span>'
    )


def current_status_badge(
    is_current
):

    if is_current:

        return badge(
            "Current",
            "green",
        )

    return badge(
        "Stale / Unavailable",
        "red",
    )


def band_badge(
    band
):

    b = str(
        band
    )

    if b == "Low":

        return badge(
            b,
            "green",
        )

    if b == "Moderate":

        return badge(
            b,
            "amber",
        )

    if b in {
        "High",
        "Very High",
        "Critical",
    }:

        return badge(
            b,
            "red",
        )

    return badge(
        b,
        "gray",
    )


def find_first_existing(
    paths
):

    for path in paths:

        if path.exists():

            return path

    return None


def explanation(
    title,
    general,
    officer,
    evaluator,
    audience,
):

    text = {

        "General / Public":
            general,

        "Disaster Management / Relief Team":
            officer,

        "Evaluation Panel / Technical":
            evaluator,

    }[
        audience
    ]

    with st.expander(
        f"ℹ️ What does {title} mean?",
        expanded=False,
    ):

        st.markdown(
            text
        )


def section_intro(
    text
):

    st.markdown(

        (
            '<div class="meaning-box">'
            f'{text}'
            '</div>'
        ),

        unsafe_allow_html=True,
    )


def warning(
    text
):

    st.markdown(

        (
            '<div class="warning-box">'
            f'{text}'
            '</div>'
        ),

        unsafe_allow_html=True,
    )


def card(
    label,
    value,
    note="",
):

    card_html = (
        '<div class="plain-card">'
        '<div class="small-label">'
        f'{label}'
        '</div>'
        '<div class="big-value">'
        f'{value}'
        '</div>'
        '<div class="muted">'
        f'{note}'
        '</div>'
        '</div>'
    )

    st.markdown(
        card_html,
        unsafe_allow_html=True,
    )
def metric_card(
    label,
    value,
    delta=None,
):

    delta_html = ""

    if delta is not None:

        try:

            delta_value = float(delta)

            if delta_value > 0:
                delta_symbol = "↑"
                delta_class = "metric-delta-positive"

            elif delta_value < 0:
                delta_symbol = "↓"
                delta_class = "metric-delta-negative"

            else:
                delta_symbol = "→"
                delta_class = "metric-delta-neutral"

            delta_html = (
                f'<div class="{delta_class}">'
                f'{delta_symbol} {abs(delta_value):,.1f}'
                '</div>'
            )

        except Exception:
            pass


    metric_html = (
        '<div class="custom-metric-card">'
        '<div class="custom-metric-label">'
        f'{label}'
        '</div>'
        '<div class="custom-metric-value">'
        f'{value}'
        '</div>'
        f'{delta_html}'
        '</div>'
    )


    st.markdown(
        metric_html,
        unsafe_allow_html=True,
    )

def first_value(
    mapping,
    *keys,
    default=None,
):

    if not isinstance(
        mapping,
        dict,
    ):

        return default

    for key in keys:

        if (
            key in mapping
            and
            mapping[
                key
            ]
            is not None
        ):

            return mapping[
                key
            ]

    return default


def format_datetime(
    value
):

    if value is None:

        return "Unavailable"

    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(
        timestamp
    ):

        return str(
            value
        )

    return timestamp.strftime(
        "%d %b %Y, %H:%M"
    )


def friendly_feature_name(
    name
):

    mapping = {

        "prev_people_in_safety_centres":
            "Previous Safety-Centre Population",

        "safety_population_change_from_prev":
            "Safety-Centre Population Change",

        "report_dayofweek":
            "Report Day of Week",

        "affected_people_log_change_from_prev":
            "Affected-Population Log Change",

        "affected_people_change_from_prev":
            "Affected-Population Change",

        "current_heavy_rain_flag":
            "Heavy Rain Present",

        "dayofyear_cos":
            "Seasonal Timing",

        "landslide_districts_at_risk":
            "Landslide Districts at Risk",

        "landslide_level1_districts":
            "Level-1 Landslide Districts",

        "landslide_level2_districts":
            "Level-2 Landslide Districts",

        "landslide_level3_districts":
            "Level-3 Landslide Districts",

        "affected_families_change_from_prev":
            "Affected-Families Change",

        "weather_reports_24h":
            "Weather Reports in Previous 24h",

        "weather_max_warning_level_24h":
            "Maximum Weather Warning Level",

        "affected_families":
            "Current Affected Families",

        "previous_report_gap_hours":
            "Previous Report Gap",

        "log_current_people_in_safety_centres":
            "Current Safety Population",

        "latest_landslide_age_hours":
            "Landslide Bulletin Age",

        "log_current_affected_people":
            "Current Affected Population",

        "log_current_safety_centres":
            "Current Safety Centres",
    }

    raw = str(
        name
    )

    if raw in mapping:

        return mapping[
            raw
        ]

    return (
        raw
        .replace(
            "missingindicator_",
            "Missing: ",
        )
        .replace(
            "_",
            " ",
        )
        .title()
    )


def run_latest_update():

    # IMPORTANT:
    # - Calls update_latest.py only.
    # - Does NOT run the final notebook.
    # - Does NOT retrain predictive models.
    

    if not UPDATE_LATEST_PATH.exists():

        return (
            False,
            "update_latest.py is not available.",
        )

    try:

        result = subprocess.run(

            [
                sys.executable,
                str(
                    UPDATE_LATEST_PATH
                ),
                "--once",
            ],

            cwd=str(
                ROOT
            ),

            capture_output=True,

            text=True,

            timeout=
                30
                *
                60,
        )

        output = (

            (
                result.stdout
                or ""
            )

            +

            "\n"

            +

            (
                result.stderr
                or ""
            )

        ).strip()

        return (
            result.returncode
            ==
            0,
            output,
        )

    except subprocess.TimeoutExpired:

        return (
            False,
            "Latest-data check exceeded "
            "the 30-minute safety timeout.",
        )

    except Exception as exc:

        return (
            False,
            f"Latest-data check failed: {exc}",
        )


# LOAD CURRENT LIVE PAYLOAD


payload = load_json(
    PAYLOAD_PATH
)


if payload is None:

    st.error(
        "Dashboard data file is missing: "
        "data/processed/dashboard_payload.json"
    )

    st.info(
        "FloodImpact-LK uses dashboard_payload.json "
        "generated by the live inference pipeline. "
        "Do not create dashboard_summary.json."
    )

    st.stop()

# CURRENT PAYLOAD SCHEMA


current_snapshot = (
    payload.get(
        "current_snapshot",
        {},
    )
)

research_context = (
    payload.get(
        "research_context",
        {},
    )
)

raw_performance = (
    payload.get(
        "performance",
        {},
    )
)


hazard_context_raw = (
    research_context.get(
        "hazard_context",
        {},
    )
)

freshness_raw = (
    research_context.get(
        "freshness",
        {},
    )
)

trend_raw = (
    research_context.get(
        "trend",
        {},
    )
)

drpi_raw = (
    research_context.get(
        "DRPI",
        {},
    )
)

scpi_raw = (
    research_context.get(
        "SCPI",
        {},
    )
)


# PERFORMANCE SCHEMA NORMALIZATION
#
# Supports both:
#   test_mae / test_accuracy / ...
#
# and:
#   mae / accuracy / ...
#
# so older final performance metadata and current live
# payload are both handled safely.


uncertainty_raw = (

    raw_performance.get(

        "human_impact_uncertainty",

        raw_performance.get(
            "uncertainty",
            {},
        ),
    )
)


human_perf_raw = (

    raw_performance.get(
        "human_impact",
        {},
    )
)


safety_usage_perf_raw = (

    raw_performance.get(
        "safety_usage",
        {},
    )
)


safety_demand_perf_raw = (

    raw_performance.get(
        "safety_demand",
        {},
    )
)


risk_perf_raw = (

    raw_performance.get(

        "experimental_escalation_risk",

        raw_performance.get(
            "risk",
            {},
        ),
    )
)


performance = {

    "human_impact": {

        "mae":
            first_value(
                human_perf_raw,
                "test_mae",
                "mae",
            ),

        "rmse":
            first_value(
                human_perf_raw,
                "test_rmse",
                "rmse",
            ),

        "rmsle":
            first_value(
                human_perf_raw,
                "test_rmsle",
                "rmsle",
            ),

        "r2":
            first_value(
                human_perf_raw,
                "test_r2",
                "r2",
            ),

        "persistence_mae":
           first_value(
            human_perf_raw,
            "persistence_test_mae",
            "persistence_mae",
            "baseline_mae",
    ),
    },


    "safety_usage": {

        "accuracy":
            first_value(
                safety_usage_perf_raw,
                "test_accuracy",
                "accuracy",
            ),

        "balanced_accuracy":
            first_value(
                safety_usage_perf_raw,
                "test_balanced_accuracy",
                "balanced_accuracy",
            ),

        "f1":
            first_value(
                safety_usage_perf_raw,
                "test_f1",
                "f1",
            ),
    },


    "safety_demand": {

        "mae":
            first_value(
                safety_demand_perf_raw,
                "test_mae",
                "mae",
            ),

        "rmse":
            first_value(
                safety_demand_perf_raw,
                "test_rmse",
                "rmse",
            ),

        "rmsle":
            first_value(
                safety_demand_perf_raw,
                "test_rmsle",
                "rmsle",
            ),

        "r2":
            first_value(
                safety_demand_perf_raw,
                "test_r2",
                "r2",
            ),

        "persistence_mae":
            first_value(
                safety_demand_perf_raw,
                "persistence_mae",
                "baseline_mae",
            ),
    },


    "risk": {

        "accuracy":
            first_value(
                risk_perf_raw,
                "test_accuracy",
                "accuracy",
            ),

        "balanced_accuracy":
            first_value(
                risk_perf_raw,
                "test_balanced_accuracy",
                "balanced_accuracy",
            ),

        "macro_f1":
            first_value(
                risk_perf_raw,
                "test_macro_f1",
                "macro_f1",
            ),
    },
}



# ADAPT LIVE PAYLOAD TO DASHBOARD VIEW

# dashboard_payload.json structure:
#
#   current_snapshot
#   research_context
#   similar_events
#   explainability
#   performance

# We only transform those existing fields for display.
# No prediction is recalculated here.

summary = {

    "snapshot":

        first_value(
            current_snapshot,
            "snapshot_datetime",
            "report_datetime",
            "report_time",
            default="Unavailable",
        ),


    "reported_hazards":

        first_value(
            current_snapshot,
            "reported_hazards",
            "hazards",
            "disaster_types",
            default=[],
        ),

    # HUMAN IMPACT

    "human_impact": {

        "current_affected":

            first_value(
                current_snapshot,
                "current_affected_people",
                "current_affected",
            ),


        "predicted_next":

            first_value(
                current_snapshot,
                "predicted_next_affected_people",
                "predicted_affected_people",
            ),


        "predicted_change":

            first_value(
                current_snapshot,
                "predicted_change",
                "predicted_affected_change",
            ),


        "lower_90":

            first_value(
                current_snapshot,
                "prediction_interval_lower",
                "lower_90",
                "predicted_lower_90",
            ),


        "upper_90":

            first_value(
                current_snapshot,
                "prediction_interval_upper",
                "upper_90",
                "predicted_upper_90",
            ),


        "interval_empirical_coverage":

            first_value(
                uncertainty_raw,
                "held_out_empirical_coverage",
                "empirical_coverage",
            ),
    },

    # SAFETY CENTRES

    "safety_centres": {

        "current_population":

            first_value(
                current_snapshot,
                "current_safety_population",
                "current_people_in_safety_centres",
            ),


        "use_probability":

            first_value(
                current_snapshot,
                "safety_use_probability",
                "predicted_safety_use_probability",
            ),


        "predicted_next_population":

            first_value(
                current_snapshot,
                "predicted_next_safety_population",
                "predicted_safety_population",
            ),
    },

    # EXPERIMENTAL RISK

    "risk": {

        "predicted_class":

            first_value(
                current_snapshot,
                "experimental_escalation_risk",
                "escalation_risk",
                "predicted_escalation_risk",
                default="Unavailable",
            ),


        "probabilities":

            first_value(
                current_snapshot,
                "risk_probabilities",
                "escalation_risk_probabilities",
                default={},
            ),
    },

    # DRPI / SCPI

    "indices": {

        "drpi":

            first_value(
                drpi_raw,
                "score",
                "DRPI",
            ),


        "drpi_band":

            first_value(
                drpi_raw,
                "band",
                "level",
                "priority",
                default="Unavailable",
            ),


        "drpi_components":

            first_value(
                drpi_raw,
                "components",
                default={},
            ),


        "scpi":

            first_value(
                scpi_raw,
                "score",
                "SCPI",
            ),


        "scpi_band":

            first_value(
                scpi_raw,
                "band",
                "level",
                "priority",
                default="Unavailable",
            ),


        "scpi_components":

            first_value(
                scpi_raw,
                "components",
                default={},
            ),
    },

    # HAZARD CONTEXT

    "hazard_context": {

        "flood_score":

            first_value(
                hazard_context_raw,
                "flood_context_score",
                "Flood",
                default=50,
            ),


        "weather_score":

            first_value(
                hazard_context_raw,
                "weather_context_score",
                "Weather",
                default=50,
            ),


        "landslide_score":

            first_value(
                hazard_context_raw,
                "landslide_context_score",
                "Landslide",
                default=50,
            ),


        "combined_score":

            first_value(
                hazard_context_raw,
                "combined_context_score",
                "combined_hazard_context_score",
                "Combined",
                default=50,
            ),
    },

    # SOURCE FRESHNESS

    "freshness":
        freshness_raw,

    # TREND

    "trend": {

        "trend_label":

            first_value(
                trend_raw,
                "trend",
                "direction",
                default="Unavailable",
            ),


        "absolute_change":

            first_value(
                trend_raw,
                "change_from_previous",
                "absolute_change",
            ),


        "rapid_growth_score":

            first_value(
                trend_raw,
                "rapid_growth_score",
            ),


        "monitoring_status":

            first_value(
                trend_raw,
                "monitoring_status",
                "status",
                default="Unavailable",
            ),
    },


    "performance":
        performance,


    "system":
        payload.get(
            "project",
            {},
        ),
}

# ADDITIONAL DASHBOARD FILES

similar_df = load_csv(
    SIMILAR_PATH
)

shap_df = load_csv(
    GLOBAL_SHAP_PATH
)

latest_shap_df = load_csv(
    LATEST_SHAP_PATH
)

trend_df = load_csv(
    TREND_PATH
)

# DISPLAY VARIABLES

human = (
    summary[
        "human_impact"
    ]
)

safety = (
    summary[
        "safety_centres"
    ]
)

risk = (
    summary[
        "risk"
    ]
)

indices = (
    summary[
        "indices"
    ]
)

hazard = (
    summary[
        "hazard_context"
    ]
)

freshness = (
    summary[
        "freshness"
    ]
)

trend = (
    summary[
        "trend"
    ]
)

performance = (
    summary[
        "performance"
    ]
)

system = (
    summary[
        "system"
    ]
)

snapshot = (
    summary[
        "snapshot"
    ]
)

reported_hazards = (
    summary[
        "reported_hazards"
    ]
)

# SIDEBAR

with st.sidebar:

    st.markdown(
        "## 🌊 FloodImpact-LK"
    )

    st.caption(
        "Interactive research decision-support dashboard"
    )

    # AUDIENCE VIEW

    audience = st.radio(
    "Audience view",
    [
        "General / Public",
        "Disaster Management / Relief Team",
        "Evaluation Panel / Technical",
    ],
    index=0,
    help=(
        "Select the audience type to show the most relevant "
        "dashboard information and explanations."
    ),
)
    
    # AUDIENCE-SPECIFIC PAGE ACCESS

    AUDIENCE_PAGES = {

        "General / Public": [
            "🏠 Overview",
            "🌧️ Hazard Context",
            "🗺️ District Map",
            "📈 Trend & Anomalies",
            "📘 How to Read the System",
        ],

        "Disaster Management / Relief Team": [
            "🏠 Overview",
            "👥 Human Impact",
            "🏥 Safety Centres",
            "🌧️ Hazard Context",
            "🗺️ District Map",
            "🚦 Risk & Priority",
            "📈 Trend & Anomalies",
            "🕰️ Similar Events",
            "📘 How to Read the System",
        ],

        "Evaluation Panel / Technical": [
            "🏠 Overview",
            "👥 Human Impact",
            "🏥 Safety Centres",
            "🌧️ Hazard Context",
            "🗺️ District Map",
            "🚦 Risk & Priority",
            "📈 Trend & Anomalies",
            "🕰️ Similar Events",
            "🧠 Explainability",
            "📊 Model Performance",
            "📘 How to Read the System",
        ],
    }

    # PAGE BUTTONS FOR SELECTED AUDIENCE

    page = st.radio(
        "Explore",
        AUDIENCE_PAGES[audience],
    )

    # AUDIENCE DESCRIPTION

    if audience == "General / Public":

        st.caption(
            "Simple view: current situation, hazard context, "
            "district map and easy-to-understand interpretation."
        )

    elif audience == "Disaster Management / Relief Team":

        st.caption(
            "Operational view: impact, safety-centre demand, "
            "hazard freshness, priority and historical context."
        )

    else:

        st.caption(
            "Technical view: complete system outputs, "
            "explainability and held-out model evaluation."
        )

    # COMMON SIDEBAR AREA
    # Shown for ALL audience types
    
    st.markdown(
        "---"
    )


    st.caption(
        "Latest snapshot: "
        f"{format_datetime(snapshot)}"
    )

    # DATA REFRESH
    
    st.markdown(
        "### Data Refresh"
    )


    if st.button(
        "↻ Reload processed outputs",
        use_container_width=True,
        help=(
            "Reload the current processed "
            "dashboard JSON and CSV outputs."
        ),
    ):

        st.rerun()


    if UPDATE_LATEST_PATH.exists():

        if st.button(
            "🔄 Check latest official data",
            use_container_width=True,
            type="primary",
            help=(
                "Check official sources and update "
                "the dashboard only when the safe "
                "live pipeline allows it."
            ),
        ):

            with st.spinner(
                "Checking official sources and "
                "refreshing the latest valid outputs..."
            ):

                (
                    update_ok,
                    update_output,
                ) = run_latest_update()


            if update_ok:

                st.success(
                    "Latest official-data check completed."
                )

                st.rerun()

            else:

                st.error(
                    "Latest official-data check "
                    "did not complete."
                )

                if update_output:

                    st.code(
                        update_output[-5000:],
                        language="text",
                    )

    else:

        st.caption(
            "update_latest.py is unavailable."
        )

    # FOOTER
    
    st.markdown(
        "---"
    )

    st.caption(
        "Research prototype • "
        "Not an official warning system"
    )

# HERO

hero_html = (
    '<div class="hero">'
    '<div class="hero-title">'
    'FloodImpact-LK'
    '</div>'
    '<div class="hero-sub">'
    'Official-Report-Driven Near-Real-Time '
    'Multi-Source Spatio-Temporal Decision-Support '
    'Prototype for Floods and Rainfall-Induced '
    'Landslides in Sri Lanka.'
    '<br><br>'
    '<b>Latest system snapshot:</b> '
    f'{format_datetime(snapshot)}'
    '</div>'
    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)

# OVERVIEW


if page == "🏠 Overview":

    # REPORTED HAZARDS

    if reported_hazards:

        st.markdown(

            (
                "**Hazards mentioned in the latest "
                "Situation Report:** "
            )

            +

            " ".join(

                badge(
                    str(
                        item
                    ),
                    "blue",
                )

                for item
                in reported_hazards
            ),

            unsafe_allow_html=True,
        )

    # MAIN FORECAST METRICS
   
    c1, c2, c3, c4 = st.columns(4)


    with c1:

        metric_card(
            "Current Reported Affected People",
            nfmt(
                human.get("current_affected")
            ),
        )

        with st.popover(
            "ⓘ Details",
            use_container_width=True,
        ):

            st.markdown(
                """
                **Current Reported Affected People**

                Latest national number of people reported
                as affected in the DMC Situation Report.

                This is an observed reported value,
                not a prediction.
                """
            )


    with c2:

        metric_card(
            "Predicted Next Affected People",
            nfmt(
                human.get("predicted_next")
            ),
            human.get("predicted_change"),
        )

        with st.popover(
            "ⓘ Details",
            use_container_width=True,
        ):

            st.markdown(
                """
                **Predicted Next Affected People**

                National estimate for the next eligible
                DMC Situation Report, up to 48 hours.

                This is a national multi-hazard forecast.

                It is not a Flood-only, Landslide-only,
                or district-level prediction.
                """
            )


    with c3:

        metric_card(
            "Predicted Next Safety-Centre Population",
            nfmt(
                safety.get("predicted_next_population")
            ),
        )

        with st.popover(
            "ⓘ Details",
            use_container_width=True,
        ):

            st.markdown(
                """
                **Predicted Next Safety-Centre Population**

                Estimated national safety-centre population
                for the next eligible Situation Report.

                This is not safety-centre capacity.
                """
            )


    with c4:

        metric_card(
            "Safety-Centre Use Probability",
            pct(
                safety.get("use_probability")
            ),
        )

        with st.popover(
            "ⓘ Details",
            use_container_width=True,
        ):

            st.markdown(
                """
                **Safety-Centre Use Probability**

                Probability that the next eligible Situation
                Report will contain positive safety-centre use.

                This is not shelter occupancy or
                capacity utilization.
                """
            )

    # DECISION SNAPSHOT

    st.markdown(
        "### Decision snapshot"
    )


    d1, d2, d3, d4 = st.columns(
        4
    )


    with d1:

        card(

            "Experimental escalation risk",

            str(
                risk.get(
                    "predicted_class",
                    "Unavailable",
                )
            ),

            (
                "Research-derived classification; "
                "not an official warning."
            ),
        )


    with d2:

        card(

            "DRPI",

            (
                f'{nfmt(indices.get("drpi"), 2)}'
                "/100"
            ),

            (
                f'{indices.get("drpi_band", "Unavailable")} '
                "response-priority signal."
            ),
        )


    with d3:

        card(

            "SCPI",

            (
                f'{nfmt(indices.get("scpi"), 2)}'
                "/100"
            ),

            (
                f'{indices.get("scpi_band", "Unavailable")} '
                "relative safety-centre pressure."
            ),
        )


    with d4:

        card(

            "Rapid-growth status",

            str(
                trend.get(
                    "monitoring_status",
                    "Unavailable",
                )
            ),

            (
                "Score: "
                f'{nfmt(trend.get("rapid_growth_score"), 2)}'
                "/100"
            ),
        )

    # CURRENT INTERPRETATION

    st.markdown(
        "### Current interpretation"
    )


    current = human.get(
        "current_affected"
    )

    predicted = human.get(
        "predicted_next"
    )

    change = human.get(
        "predicted_change"
    )


    if change is None:

        impact_sentence = (
            "The latest forecast change "
            "is unavailable."
        )

    else:

        try:

            change_value = float(
                change
            )

            if abs(
                change_value
            ) < 1:

                impact_sentence = (
                    "The human-impact model currently "
                    "estimates **very little net change** "
                    "in the national affected population "
                    "for the next eligible report."
                )

            elif change_value > 0:

                impact_sentence = (
                    "The human-impact model currently "
                    "estimates an **increase** in the "
                    "national affected population for "
                    "the next eligible report."
                )

            else:

                impact_sentence = (
                    "The human-impact model currently "
                    "estimates a **decrease** in the "
                    "national affected population for "
                    "the next eligible report."
                )

        except Exception:

            impact_sentence = (
                "The latest forecast direction "
                "is unavailable."
            )


    st.markdown(

        f"""
        - Latest reported national affected population:
          **{nfmt(current)}**.

        - Next eligible Situation Report estimate:
          **{nfmt(predicted)}**.
          {impact_sentence}

        - Safety-centre use probability:
          **{pct(safety.get("use_probability"))}**.

        - River context:
          **{freshness.get("river_status", "Unavailable")}**;
          stations at Alert / Minor / Major Flood:
          **{freshness.get("river_alert_flood_stations", "Unavailable")}**.

        - Landslide context:
          **{freshness.get("landslide_status", "Unavailable")}**.
        """
    )
    # UNCERTAINTY

    lower = human.get(
        "lower_90"
    )

    upper = human.get(
        "upper_90"
    )


    warning(

        (
            "<b>Uncertainty matters:</b> "
            "the 90% model-based prediction range is "
        )

        +

        (
            f"<b>{nfmt(lower)} – "
            f"{nfmt(upper)}</b>. "
        )

        +

        (
            "Held-out empirical coverage was "
            f"<b>{pct(human.get('interval_empirical_coverage'))}</b>. "
            "This interval is not a guarantee."
        )
    )


    explanation(

        "the overview",

        """
        This page gives a simple summary of the latest
        disaster situation and the system's next-report
        estimates. It is designed to help a reader understand
        the situation without knowing machine learning.
        """,

        """
        Use this page as a triage view: current impact,
        likely next impact, safety-centre demand,
        hazard-data freshness and research-priority signals.
        Verify operational decisions against official warnings.
        """,

        """
        The overview combines frozen outputs from separate
        regression, classification, anomaly-detection,
        similarity, uncertainty and research-index components.
        Forecast targets are national DMC Situation Report
        totals. The test period remains a chronological
        held-out evaluation period.
        """,

        audience,
    )

# HUMAN IMPACT

elif page == "👥 Human Impact":

    section_intro(

        "<b>Purpose:</b> estimate the national "
        "affected population expected in the next "
        "eligible DMC Situation Report "
        "(maximum 48-hour gap), while showing "
        "uncertainty and the strong persistence baseline."
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(

        "Current reported affected",

        nfmt(
            human.get(
                "current_affected"
            )
        ),
    )


    impact_delta = human.get(
        "predicted_change"
    )

    impact_delta_text = None

    if impact_delta is not None:

        try:

            impact_delta_text = (
                f"{float(impact_delta):+,.1f}"
            )

        except Exception:

            impact_delta_text = None


    c2.metric(

        "Predicted next affected",

        nfmt(
            human.get(
                "predicted_next"
            )
        ),

        delta=
            impact_delta_text,
    )

    with c3:
        metric_card(
            "90% Model-Based Prediction Range",
            (
                f'{nfmt(human.get("lower_90"))}'
                ' – '
                f'{nfmt(human.get("upper_90"))}'
            ),
        )
    


    st.markdown(
        "#### How to interpret this forecast"
    )


    st.markdown(

        f"""
        The latest national Situation Report contains
        **{nfmt(human.get("current_affected"))}**
        affected people.

        The model estimates
        **{nfmt(human.get("predicted_next"))}**
        for the next eligible Situation Report.

        The 90% model-based prediction range is
        **{nfmt(human.get("lower_90"))}
        to
        {nfmt(human.get("upper_90"))}**.
        """
    )


    warning(

        "<b>Important target definition:</b> "
        "this is a national Situation Report total. "
        "It is not a Flood-only count, "
        "Landslide-only count, "
        "or district-level count."
    )

    # BASELINE COMPARISON

    st.markdown(
        "#### Forecast vs strong baseline"
    )


    human_performance = (
        performance.get(
            "human_impact",
            {},
        )
    )


    model_mae = (
        human_performance.get(
            "mae"
        )
    )

    persistence_mae = (
        human_performance.get(
            "persistence_mae"
        )
    )


    if (
        model_mae is not None
        and
        persistence_mae is not None
    ):

        comp = pd.DataFrame(

            {

                "Model": [
                    "Final ML model",
                    "Persistence baseline",
                ],

                "Test MAE": [
                    model_mae,
                    persistence_mae,
                ],
            }
        ).set_index(
            "Model"
        )


        st.bar_chart(
            comp
        )

    else:

        st.info(
            "Human-impact baseline comparison "
            "metadata is unavailable."
        )


    st.caption(
        "Lower MAE is better. On the final held-out "
        "test, persistence had slightly lower MAE "
        "than the ML model. This limitation is "
        "intentionally reported."
    )


    explanation(

        "Human Impact Forecast",

        """
        “Affected people” means the total number reported
        as affected in the national DMC Situation Report.
        The model estimates that corresponding total
        in the next eligible report.
        """,

        """
        Use the forecast as an early planning signal for
        the possible scale of relief needs. Always read
        the uncertainty interval and hazard context.
        Do not treat the point estimate as an exact headcount.
        """,

        """
        The final impact system forecasts a signed-log
        transformed change and adds it to the current
        national affected total. Model selection was
        completed before one-time chronological test
        evaluation. Persistence remains a strong benchmark
        because reports are cumulative and sequential.
        """,

        audience,
    )

# SAFETY CENTRES

elif page == "🏥 Safety Centres":

    section_intro(

        "<b>Purpose:</b> estimate whether "
        "safety-centre use will continue and "
        "how many people may be in safety centres "
        "in the next eligible DMC Situation Report."
    )


    c1, c2, c3 = st.columns(
        3
    )


    c1.metric(

        "Current safety-centre population",

        nfmt(
            safety.get(
                "current_population"
            )
        ),
    )


    c2.metric(

        "Use probability",

        pct(
            safety.get(
                "use_probability"
            )
        ),
    )


    c3.metric(

        "Predicted next population",

        nfmt(
            safety.get(
                "predicted_next_population"
            )
        ),
    )


    st.markdown(
        "#### Safety-centre use probability"
    )


    st.progress(

        clamp01(
            safety.get(
                "use_probability"
            )
        )
    )


    st.caption(
        "This is the model probability of positive "
        "safety-centre use in the next eligible report. "
        "It is not a percentage of shelter capacity."
    )


    c1, c2 = st.columns(
        2
    )


    with c1:

        card(

            "SCPI",

            (
                f'{nfmt(indices.get("scpi"), 2)}'
                "/100"
            ),

            (
                f'{indices.get("scpi_band", "Unavailable")} '
                "relative safety-centre pressure."
            ),
        )


    with c2:

        st.markdown(
            "#### SCPI components"
        )


        scpi_components = (
            indices.get(
                "scpi_components",
                {},
            )
        )


        if (
            isinstance(
                scpi_components,
                dict,
            )
            and
            scpi_components
        ):
            scpi_display_names = {
                    "Current Safety Population": "Current Population",
                    "Forecast Safety Population": "Forecast Pop.",
                    "Safety-Use Probability": "Use Probability",
                    "Predicted Safety Growth": "Predicted Growth",
                    "Hazard Context": "Hazard Context",

            }
            component_df = pd.DataFrame(
                {
                    "Component": [
                        scpi_display_names.get(
                            str(name),
                            str(name),
                        )
                        for name in scpi_components.keys()
                    ],
                    "Score": list(scpi_components.values()),
                }
            ).set_index("Component")
            st.bar_chart(
                component_df,
                horizontal=True,
            )

        else:

            st.info(
                "SCPI component values are unavailable."
            )


    warning(

        "<b>SCPI is not occupancy.</b> "
        "Actual safety-centre capacity data are not "
        "available, so SCPI must not be interpreted "
        "as a capacity-utilization percentage."
    )


    explanation(

        "Safety-Centre Demand",

        """
        This section estimates whether people are likely
        to remain in safety centres and the likely national
        number in the next eligible DMC Situation Report.
        """,

        """
        It can support preparation of food, bedding,
        sanitation, transport and staffing. It should be
        combined with official local shelter information
        before allocating resources.
        """,

        """
        A two-stage design is used: a classifier predicts
        whether safety-centre use is positive, then a
        regression model estimates the amount when use
        is predicted. The usage classifier generalised
        substantially better than the escalation-risk model.
        """,

        audience,
    )

# HAZARD CONTEXT

elif page == "🌧️ Hazard Context":

    section_intro(

        "<b>Purpose:</b> combine the Situation Report "
        "with weather, river and rainfall-induced "
        "landslide information while explicitly showing "
        "whether supporting sources are current or stale."
    )


    if reported_hazards:

        st.markdown(

            (
                "**Latest Situation Report mentions:** "
            )

            +

            " ".join(

                badge(
                    str(
                        item
                    ),
                    "blue",
                )

                for item
                in reported_hazards
            ),

            unsafe_allow_html=True,
        )


    c1, c2, c3, c4 = st.columns(
        4
    )


    with c1:

        card(

            "Flood context",

            (
                f'{nfmt(hazard.get("flood_score"), 1)}'
                "/100"
            ),

            (
                "Research context score, "
                "not an official flood-warning level."
            ),
        )


    with c2:

        card(

            "Weather / heavy-rain context",

            (
                f'{nfmt(hazard.get("weather_score"), 1)}'
                "/100"
            ),

            (
                "Uses Situation Report and "
                "weather-warning context."
            ),
        )


    with c3:

        card(

            "Landslide context",

            (
                f'{nfmt(hazard.get("landslide_score"), 1)}'
                "/100"
            ),

            (
                "Bulletin freshness is checked "
                "before current use."
            ),
        )


    with c4:

        card(

            "Combined hazard context",

            (
                f'{nfmt(hazard.get("combined_score"), 1)}'
                "/100"
            ),

            (
                "Research-derived contextual indicator."
            ),
        )

    # SOURCE FRESHNESS

    st.markdown(
        "### Source freshness"
    )


    freshness_table = pd.DataFrame(

        [

            {

                "Source":
                    "River observations",

                "Latest datetime":
                    freshness.get(
                        "river_latest_datetime"
                    ),

                "Age at snapshot (h)":
                    freshness.get(
                        "river_age_hours"
                    ),

                "Status":
                    freshness.get(
                        "river_status"
                    ),
            },


            {

                "Source":
                    "Landslide bulletin",

                "Latest datetime":
                    freshness.get(
                        "landslide_latest_datetime"
                    ),

                "Age at snapshot (h)":
                    freshness.get(
                        "landslide_age_hours"
                    ),

                "Status":
                    freshness.get(
                        "landslide_status"
                    ),
            },
        ]
    )


    st.dataframe(

        freshness_table,

        hide_index=True,

        use_container_width=True,
    )

    # RIVER CONTEXT
    
    st.markdown(
        "### River snapshot"
    )


    r1, r2 = st.columns(
        2
    )


    r1.metric(

        "Stations in snapshot-aligned report",

        freshness.get(
            "river_station_count",
            "Unavailable",
        ),
    )

    r2.metric(

        "Stations at Alert / Minor / Major Flood",

        freshness.get(
            "river_alert_flood_stations",
            "Unavailable",
        ),
    )

    warning(

        "<b>Unavailable ≠ safe.</b> "
        "A stale or missing hazard bulletin is "
        "deliberately kept separate from a verified "
        "“no warning” condition."
    )

    explanation(

        "Multi-Source Hazard Context",

        """
        This page shows Flood, Heavy-Rain and
        rainfall-induced landslide context and whether
        the supporting river and landslide information
        is fresh enough to use.
        """,

        """
        Check freshness before using any hazard layer.
        A stale landslide bulletin should trigger a
        data-quality caution, not a conclusion that
        the district is safe.
        """,

        """
        Snapshot alignment enforces supporting-source
        timestamps at or before the Situation Report time
        to avoid future leakage. Hazard scores are
        research-derived context indicators, not calibrated
        official warning probabilities.
        """,

        audience,
    )

# DISTRICT MAP

elif page == "🗺️ District Map":

    section_intro(

        "<b>Purpose:</b> visualize available "
        "district-level hazard context across Sri Lanka. "
        "The map intentionally does not fabricate "
        "district-level human-impact forecasts."
    )


    if MAP_PATH.exists():

        map_html = (
            MAP_PATH
            .read_text(
                encoding="utf-8"
            )
        )


        components.html(

            map_html,

            height=720,

            scrolling=True,
        )

    else:

        st.error(

            "District map not found: "
            "data/processed/"
            "floodimpact_lk_district_hazard_map.html"
        )


    warning(

        "<b>Map interpretation:</b> "
        "grey means stale/unavailable hazard information, "
        "not automatically a safe district. "
        "Human-impact and safety-demand predictions "
        "are national because the extracted model targets "
        "are national totals."
    )


    explanation(

        "the District Map",

        """
        The map lets you explore district hazard
        information spatially. Hover over or interact
        with districts to inspect available warning context.
        """,

        """
        Use the map to identify where verified district
        hazard information exists. Do not use it as a
        district-level affected-person forecast.
        """,

        """
        District-level landslide-warning information is
        available from NBRO-derived bulletin extraction,
        but freshness is enforced. National target
        predictions remain separate from spatial
        district hazard context.
        """,

        audience,
    )

# RISK & PRIORITY

elif page == "🚦 Risk & Priority":

    section_intro(

        "<b>Purpose:</b> summarize potential escalation "
        "and research-derived response priority without "
        "presenting these research outputs as official warnings."
    )


    c1, c2, c3 = st.columns(
        3
    )


    with c1:

        card(

            "Experimental escalation risk",

            str(
                risk.get(
                    "predicted_class",
                    "Unavailable",
                )
            ),

            (
                "Experimental model class for "
                "next-report impact escalation."
            ),
        )


    with c2:

        card(

            "DRPI",

            (
                f'{nfmt(indices.get("drpi"), 2)}'
                "/100"
            ),

            (
                f'{indices.get("drpi_band", "Unavailable")} '
                "research response priority."
            ),
        )


    with c3:

        card(

            "SCPI",

            (
                f'{nfmt(indices.get("scpi"), 2)}'
                "/100"
            ),

            (
                f'{indices.get("scpi_band", "Unavailable")} '
                "relative safety-centre pressure."
            ),
        )

    # RISK PROBABILITIES

    st.markdown(
        "### Escalation probabilities"
    )


    probs = risk.get(
        "probabilities",
        {},
    )


    if (
        isinstance(
            probs,
            dict,
        )
        and
        probs
    ):

        risk_order = [
            "Low",
            "Moderate",
            "High",
            "Critical",
        ]


        risk_rows = []


        for risk_class in risk_order:

            probability = (
                probs.get(
                    risk_class,
                    0,
                )
            )


            try:

                probability = float(
                    probability
                )

            except Exception:

                probability = 0.0


            risk_rows.append(

                {

                    "Risk class":
                        risk_class,

                    "Probability":
                        probability,
                }
            )


        prob_df = pd.DataFrame(
            risk_rows
        )

        risk_chart = (
            alt.Chart(
                prob_df
        )
            .mark_bar()
            .encode(
                x=alt.X(
                    "Probability:Q",
                    title="Probability",
                    axis=alt.Axis(
                        format=".0%"
                    ),
                ),
                y=alt.Y(
                    "Risk class:N",
                    title=None,
                    sort=[
                        "Low",
                        "Moderate",
                        "High",
                        "Critical",
                    ],
                ),
                tooltip=[
                    alt.Tooltip(
                        "Risk class:N",
                        title="Risk Class",
                    ),
                    alt.Tooltip(
                        "Probability:Q",
                        title="Probability",
                        format=".2%",
                    ),
                ],
            )
        )

        st.altair_chart(
            risk_chart,
            use_container_width=True,
        )
        

        display_prob_df = (
            prob_df
            .copy()
        )


        display_prob_df[
            "Probability"
        ] = (

            display_prob_df[
                "Probability"
            ]

            .apply(
                lambda value:
                    f"{value * 100:.2f}%"
            )
        )


        st.dataframe(

            display_prob_df,

            hide_index=True,

            use_container_width=True,
        )

    # COMPONENTS
    
    p1, p2 = st.columns(2)


    with p1:

        st.markdown(
            "#### DRPI components"
        )

        drpi_components = (
            indices.get(
                "drpi_components",
                {},
            )
        )

        drpi_display_names = {
            "Current Human Impact": "Current Impact",
            "Predicted Impact Growth": "Predicted Growth",
            "Predicted Safety Demand": "Safety Demand",
            "Hazard Context": "Hazard Context",
            "Rapid Growth": "Rapid Growth",
        }

        if (
            isinstance(
                drpi_components,
                dict,
            )
            and
            drpi_components
        ):

            drpi_component_df = pd.DataFrame(
                {
                    "Component": [
                        drpi_display_names.get(
                            str(name),
                            str(name),
                        )
                        for name in drpi_components.keys()
                    ],
                    "Score": list(
                        drpi_components.values()
                    ),
                }
            ).set_index(
                "Component"
            ).astype(
                float
            )

            st.bar_chart(
                drpi_component_df,
                horizontal=True,
            )

        else:

            st.info(
                "DRPI components are unavailable."
            )


    with p2:

        st.markdown(
            "#### SCPI components"
        )

        scpi_components = (
            indices.get(
                "scpi_components",
                {},
            )
        )

        scpi_display_names = {
            "Current Safety Population": "Current Population",
            "Forecast Safety Population": "Forecast Pop.",
            "Safety-Use Probability": "Use Probability",
            "Predicted Safety Growth": "Predicted Growth",
            "Hazard Context": "Hazard Context",
        }

        if (
            isinstance(
                scpi_components,
                dict,
            )
            and
            scpi_components
        ):

            scpi_component_df = pd.DataFrame(
                {
                    "Component": [
                        scpi_display_names.get(
                            str(name),
                            str(name),
                        )
                        for name in scpi_components.keys()
                    ],
                    "Score": list(
                        scpi_components.values()
                    ),
                }
            ).set_index(
                "Component"
            ).astype(
                float
            )

            st.bar_chart(
                scpi_component_df,
                horizontal=True,
            )

        else:

            st.info(
                "SCPI components are unavailable."
            )


    st.markdown(

        (
            '<div class="critical-box">'

            '<b>Operational caution:</b> '

            'the escalation-risk classifier showed '
            'weak held-out temporal generalisation '
            '(Macro-F1 about 0.337). '

            'It is retained as an '

            '<b>experimental research indicator</b>, '

            'not an operational warning.'

            '</div>'
        ),

        unsafe_allow_html=True,
    )


    explanation(

        "Risk & Priority",

        """
        Risk indicates the model's experimental estimate
        of whether impact may escalate. DRPI summarizes
        research response priority; SCPI summarizes
        relative safety-centre pressure.
        """,

        """
        Treat DRPI and SCPI as decision-support signals
        for investigation and preparation, not as automatic
        triggers. Verify against official alerts and
        field information.
        """,

        """
        Escalation labels use train-derived absolute
        increase thresholds. DRPI and SCPI are transparent
        research-derived indices. They are not official
        government warning levels.
        """,

        audience,
    )

# TREND & ANOMALIES

elif page == "📈 Trend & Anomalies":

    section_intro(

        "<b>Purpose:</b> distinguish ordinary changes "
        "from unusually rapid increases relative to the "
        "historical reporting pattern."
    )


    c1, c2, c3 = st.columns(3)
        
    c1.metric(

        "Current trend",

        trend.get(
            "trend_label",
            "Unavailable",
        ),
    )

    c2.metric(

        "Change from previous report",

        nfmt(
            trend.get(
                "absolute_change"
            )
        ),
    )

    c3.metric(

        "Rapid-growth score",

        (
            f'{nfmt(trend.get("rapid_growth_score"), 2)}'
            "/100"
        ),
    )

    st.markdown(

        (
            "**Monitoring status:** "
            f'{trend.get("monitoring_status", "Unavailable")}'
        )
    )

    # TREND CHART

    if not trend_df.empty:

        temp = (
            trend_df
            .copy()
        )


        if (
            "availability_datetime"
            in temp.columns
        ):

            temp[
                "availability_datetime"
            ] = pd.to_datetime(

                temp[
                    "availability_datetime"
                ],

                errors="coerce",
            )


            temp = (

                temp

                .dropna(
                    subset=[
                        "availability_datetime"
                    ]
                )

                .sort_values(
                    "availability_datetime"
                )

                .tail(
                    40
                )
            )

            # Observed affected population
        
            if (
                "affected_people"
                in temp.columns
            ):

                impact_chart = (

                    temp[
                        [
                            "availability_datetime",
                            "affected_people",
                        ]
                    ]

                    .copy()
                )


               


                impact_chart[
                    "affected_people"
                ] = pd.to_numeric(

                    impact_chart[
                        "affected_people"
                    ],

                    errors="coerce",
                )


                impact_chart = (

                    impact_chart

                    .dropna(
                        subset=[
                            "affected_people"
                        ]
                    )
                )


                st.markdown(
                    "#### Recent observed affected population"
                )

                st.line_chart(
                    impact_chart
                    .set_index(
                        "availability_datetime"
                    )[
                        [
                            "affected_people"
                        ]
                    ]
                    .rename(
                        columns={
                            "affected_people":
                                "Affected People"
                        }
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Points represent available DMC Situation Reports. "
                    "Report intervals are irregular, so connecting lines show "
                    "the sequence of observations rather than daily measurements."
                )
                                
            # Rapid-growth score

            if (
                "rapid_growth_score"
                in temp.columns
            ):

                rapid_chart = (

                    temp[
                        [
                            "availability_datetime",
                            "rapid_growth_score",
                        ]
                    ]

                    .copy()
                )


               


                rapid_chart[
                    "rapid_growth_score"
                ] = pd.to_numeric(

                    rapid_chart[
                        "rapid_growth_score"
                    ],

                    errors="coerce",
                )


                rapid_chart = (

                    rapid_chart

                    .dropna(
                        subset=[
                            "rapid_growth_score"
                        ]
                    )
                )


                st.markdown(
                    "#### Recent rapid-growth score"
                )

                st.line_chart(
                    rapid_chart
                    .set_index(
                        "availability_datetime"
                    )[
                        [
                            "rapid_growth_score"
                        ]
                    ]
                    .rename(
                        columns={
                            "rapid_growth_score":
                                "Rapid-Growth Score"
                        }
                    ),
                    use_container_width=True,
                )
                st.caption(
                    "Gaps in the rapid-growth line indicate reports where "
                    "there was insufficient comparable previous context to "
                    "calculate a valid rapid-growth score; they are not zero scores."
                )
                
    else:

        st.info(
            "Trend-history export is unavailable."
        )


    warning(

        "The trend section summarizes "
        "<b>observed reported history</b>. "
        "It is not the same as the next-report forecast."
    )


    explanation(

        "Rapid-Growth Detection",

        """
        This component asks whether the latest increase
        is unusually large compared with past positive
        changes. A low score means the current change
        is not unusually large.
        """,

        """
        Use anomalies as attention flags for sudden
        deterioration. They should prompt review of
        reports and hazard information rather than
        automatic decisions.
        """,

        """
        The final detector uses a bounded historical-
        percentile rule with a material increase
        threshold and both absolute and relative
        change information.
        """,

        audience,
    )


# =========================================================
# SIMILAR EVENTS
# =========================================================

elif page == "🕰️ Similar Events":

    section_intro(

        "<b>Purpose:</b> retrieve past situations "
        "that look most similar to the current state "
        "in the engineered feature space."
    )


    similar_payload = (
        payload.get(
            "similar_events",
            {},
        )
    )


    query_datetime = (
        similar_payload.get(
            "query_datetime"
        )
    )

    if query_datetime:

        st.caption(
            "Current similarity reference report: "
            f"{format_datetime(query_datetime)}"
        )

    if similar_df.empty:

        # Fallback to events already stored in JSON

        payload_events = (
            similar_payload.get(
                "events",
                [],
            )
        )

        if payload_events:

            similar_df = pd.DataFrame(
                payload_events
            )


    if similar_df.empty:

        st.info(
            "Similar-event export is unavailable."
        )

    else:

        display_similar = (
            similar_df
            .copy()
        )


        if (
            "historical_datetime"
            in display_similar.columns
        ):

            display_similar[
                "historical_datetime"
            ] = (

                display_similar[
                    "historical_datetime"
                ]

                .apply(
                    format_datetime
                )
            )


        rename_map = {

            "historical_datetime":
                "Historical Date",

            "similarity_score":
                "Similarity",

            "historical_affected_people":
                "Affected People",

            "affected_people":
                "Affected People",

            "historical_people_in_safety_centres":
                "Safety Population",

            "safety_population":
                "Safety Population",

            "historical_next_affected_people":
                "Next Affected",

            "next_affected_people":
                "Next Affected",

            "historical_next_safety_population":
                "Next Safety",

            "next_safety_population":
                "Next Safety",

            "historical_escalation_risk":
                "Historical Risk",

            "escalation_risk":
                "Historical Risk",
        }


        display_similar = (

            display_similar

            .rename(
                columns=
                    rename_map
            )
        )


        st.dataframe(

            display_similar,

            hide_index=True,

            use_container_width=True,
        )


        if {
            "Historical Date",
            "Similarity",
        }.issubset(
            display_similar.columns
        ):

            chart = (

                display_similar[
                    [
                        "Historical Date",
                        "Similarity",
                    ]
                ]

                .copy()
            )


            chart[
                "Similarity"
            ] = pd.to_numeric(

                chart[
                    "Similarity"
                ],

                errors="coerce",
            )


            chart = (

                chart

                .dropna(
                    subset=[
                        "Similarity"
                    ]
                )
            )


            if not chart.empty:

                st.bar_chart(

                    chart

                    .set_index(
                        "Historical Date"
                    )[
                        "Similarity"
                    ],
                    horizontal=True,
                )


    warning(

        "Similarity score is a "
        "distance-derived resemblance measure. "
        "It is <b>not</b> a probability, "
        "forecast confidence, or claim that "
        "history will repeat."
    )


    explanation(

        "the Similar Event Finder",

        """
        It answers: “Have we seen a historical
        situation that looked somewhat like this one?”
        The table shows the closest available
        historical matches.
        """,

        """
        Similar events can provide contextual
        reference for planning and briefing,
        but each event must still be checked
        individually because consequences may differ.
        """,

        """
        Similarity uses transformed engineered
        features and distance-based comparison.
        The similarity score is not calibrated
        probability or confidence.
        """,

        audience,
    )

# EXPLAINABILITY

elif page == "🧠 Explainability":

    section_intro(

        "<b>Purpose:</b> show which inputs "
        "influenced the human-impact model so "
        "the forecast is inspectable rather "
        "than a black box."
    )

    # LATEST SHAP CONTRIBUTIONS

    if latest_shap_df.empty:

        explainability_payload = (

            payload.get(
                "explainability",
                {},
            )
        )


        payload_contributors = (

            explainability_payload.get(
                "current_deployment_top_contributors",
                [],
            )
        )


        if payload_contributors:

            latest_shap_df = (
                pd.DataFrame(
                    payload_contributors
                )
            )


    if not latest_shap_df.empty:

        st.markdown(
            "### Latest prediction contributors"
        )


        latest_display = (
            latest_shap_df
            .copy()
        )


        if (
            "feature"
            in latest_display.columns
        ):

            latest_display[
                "feature"
            ] = (

                latest_display[
                    "feature"
                ]

                .apply(
                    friendly_feature_name
                )
            )


        if (
            "shap_value"
            in latest_display.columns
        ):

            latest_display[
                "_abs"
            ] = (

                pd.to_numeric(

                    latest_display[
                        "shap_value"
                    ],

                    errors="coerce",

                ).abs()
            )


            latest_display = (

                latest_display

                .sort_values(
                    "_abs",
                    ascending=False,
                )

                .drop(
                    columns=[
                        "_abs"
                    ],
                    errors="ignore",
                )

                .head(
                    10
                )
            )


        latest_display = (

            latest_display

            .drop(
                columns=[
                    "abs_shap"
                ],
                errors="ignore",
            )
        )


        latest_display = (

            latest_display

            .rename(
                columns={

                    "feature":
                        "Feature",

                    "feature_value":
                        "Current Value",

                    "shap_value":
                        "SHAP Contribution",

                    "direction":
                        "Effect",
                }
            )
        )


        st.dataframe(

            latest_display,

            hide_index=True,

            use_container_width=True,
        )

    else:

        st.info(
            "Latest SHAP contribution export "
            "is unavailable."
        )

    # GLOBAL SHAP
    
    if shap_df.empty:

        st.info(
            "Global SHAP importance export "
            "is unavailable."
        )

    else:

        st.markdown(
            "### Global feature influence"
        )


        plot_df = (
            shap_df
            .copy()
        )


        if {
            "feature",
            "mean_abs_shap",
        }.issubset(
            plot_df.columns
        ):

            plot_df[
                "mean_abs_shap"
            ] = pd.to_numeric(

                plot_df[
                    "mean_abs_shap"
                ],

                errors="coerce",
            )


            plot_df = (

                plot_df

                .dropna(
                    subset=[
                        "mean_abs_shap"
                    ]
                )

                .sort_values(
                    "mean_abs_shap",
                    ascending=False,
                )

                .head(
                    15
                )
            )


            plot_df[
                "feature"
            ] = (

                plot_df[
                    "feature"
                ]

                .apply(
                    friendly_feature_name
                )
            )


            feature_order = plot_df[
                "feature"
            ].tolist()

            global_shap_chart = (
                alt.Chart(
                    plot_df
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "mean_abs_shap:Q",
                        title="Mean |SHAP|",
                    ),
                    y=alt.Y(
                        "feature:N",
                        title=None,
                        sort=feature_order,
                        axis=alt.Axis(
                            labelLimit=300,
                            labelFontSize=12,
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "feature:N",
                            title="Feature",
                        ),
                        alt.Tooltip(
                            "mean_abs_shap:Q",
                            title="Mean |SHAP|",
                            format=".4f",
                        ),
                    ],
                )
                .properties(
                    height=440
                )
            )

            st.altair_chart(
                global_shap_chart,
                use_container_width=True,
            )


            st.dataframe(

                plot_df

                .rename(
                    columns={

                        "feature":
                            "Feature",

                        "mean_abs_shap":
                            "Mean |SHAP|",
                    }
                ),

                hide_index=True,

                use_container_width=True,
            )

    # SAVED SHAP FIGURES
    
    shown_any = False


    for (
        title,
        candidates,
    ) in SHAP_IMAGE_CANDIDATES.items():

        image_path = (
            find_first_existing(
                candidates
            )
        )

        if image_path:

            st.markdown(
                f"### {title}"
            )

            st.image(
                str(
                    image_path
                ),
                use_container_width=True,
            )

            if title == "Latest prediction waterfall":
                st.markdown(
                    (
                        '<div class="shap-note">'
                        '<b>How to read this waterfall:</b> '
                        "SHAP values explain the human-impact model's "
                        "<b>predicted signed-log change</b>, not the final "
                        "affected-people count. The model output is "
                        "inverse-transformed and combined with the current "
                        "affected population to obtain the final "
                        "next-report forecast."
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )
                
            shown_any = True
        


    if not shown_any:

        st.caption(
            "Optional SHAP PNGs were not found "
            "in outputs/figures. The SHAP tables "
            "above still provide the available "
            "explanation results."
        )


    warning(

        "<b>SHAP is not causality.</b> "
        "A feature with a large SHAP value "
        "influenced the model output; "
        "it does not prove that the feature "
        "caused disaster impact."
    )


    explanation(

        "SHAP Explainability",

        """
        SHAP helps explain why the model moved
        its forecast up or down by showing the
        contribution of individual inputs.
        """,

        """
        Use this page to challenge a prediction:
        check whether the forecast is being driven
        by recent impact changes, safety-centre
        conditions, heavy-rain context, reporting
        patterns or other features.
        """,

        """
        SHAP explains the human-impact model's
        predicted signed-log change. Missingness
        indicators can also appear because the
        preprocessing pipeline preserves information
        about source availability. Interpret SHAP as
        model behaviour, not causal effect.
        """,

        audience,
    )

# MODEL PERFORMANCE

elif page == "📊 Model Performance":

    section_intro(

        "<b>Purpose:</b> show the final "
        "held-out chronological test results "
        "honestly, including where simple "
        "baselines remain competitive."
    )


    rows = []

    # Human impact
   
    hp = (
        performance.get(
            "human_impact",
            {},
        )
    )


    rows.append(

        {

            "Output":
                "Human Impact Forecast",

            "Primary metrics":

                (
                    f"MAE {nfmt(hp.get('mae'), 2)} | "
                    f"RMSE {nfmt(hp.get('rmse'), 2)} | "
                    f"R² {nfmt(hp.get('r2'), 4)}"
                ),

            "Interpretation":

                (
                    "Persistence MAE "
                    f"{nfmt(hp.get('persistence_mae'), 2)}; "
                    "ML test MAE was slightly worse."
                ),
        }
    )

    # Safety usage

    su = (
        performance.get(
            "safety_usage",
            {},
        )
    )

    rows.append(

        {

            "Output":
                "Safety-Centre Usage",

            "Primary metrics":

                (
                    f"Accuracy {pct(su.get('accuracy'))} | "
                    "Balanced Acc. "
                    f"{pct(su.get('balanced_accuracy'))} | "
                    f"F1 {pct(su.get('f1'))}"
                ),

            "Interpretation":
                   "Strong held-out classification performance." ,
        }
    )

    # Safety demand
    
    sd = (
        performance.get(
            "safety_demand",
            {},
        )
    )


    rows.append(

        {

            "Output":
                "Safety-Centre Population",

            "Primary metrics":

                (
                    f"MAE {nfmt(sd.get('mae'), 2)} | "
                    f"RMSE {nfmt(sd.get('rmse'), 2)} | "
                    f"R² {nfmt(sd.get('r2'), 4)}"
                ),

            "Interpretation":
                "Mixed vs persistence; compare metrics separately.",
        }
    )

    # Prediction interval
   
    rows.append(
        {
            "Output":
                "90% Prediction Interval",

            "Primary metrics":
                (
                    "Nominal 90% | "
                    "Empirical Coverage "
                    f"{pct(human.get('interval_empirical_coverage'))}"
                ),

            "Interpretation":
                (
                    "Held-out coverage was below 90%; "
                    "the interval is not a guarantee."
                ),
        }
    )

    # Risk

    rr = (
        performance.get(
            "risk",
            {},
        )
    )


    rows.append(

        {

            "Output":
                "Experimental Escalation Risk",

            "Primary metrics":

                (
                    f"Accuracy {pct(rr.get('accuracy'))} | "
                    "Balanced Acc. "
                    f"{pct(rr.get('balanced_accuracy'))} | "
                    "Macro-F1 "
                    f"{pct(rr.get('macro_f1'))}"
                ),

            "Interpretation":
                (
                    "Weak temporal generalisation; "
                    "experimental only."
                ),
        }
    )


    st.dataframe(

        pd.DataFrame(
            rows
        ),

        hide_index=True,

        use_container_width=True,
    )

    # Honest interpretation
    
    st.markdown(
        "### Why there is no single “overall accuracy”"
    )


    st.info(

        "FloodImpact-LK contains both regression "
        "and classification tasks. Regression outputs "
        "are evaluated with MAE, RMSE, R² and RMSLE, "
        "while classifiers use Accuracy, Balanced "
        "Accuracy and F1. Combining them into one "
        "overall accuracy percentage would be "
        "statistically misleading."
    )


    st.markdown(
        "### Test-design integrity"
    )


    st.markdown(

        """
        - Chronological / event-aware splitting was
          used instead of random splitting.

        - The held-out test period was evaluated after
          model selection.

        - The final model was not retuned to improve
          the held-out test result.

        - Strong persistence baselines are reported
          instead of hidden.

        - Prediction-interval undercoverage and weak
          escalation-risk generalisation are disclosed.
        """
    )


    explanation(

        "Model Performance",

        """
        This page tells you how well each part of
        the system worked on data kept separate
        from model selection.
        """,

        """
        The safety-centre usage model is more
        dependable than the experimental risk
        classifier. The human-impact point forecast
        should be read together with the persistence
        baseline and uncertainty interval.
        """,

        """
        Evaluation follows task-appropriate metrics
        and a chronological holdout. R² is not
        presented as accuracy. The held-out test
        remains a final assessment rather than
        a tuning target.
        """,

        audience,
    )

# HOW TO READ

else:

    section_intro(

        "<b>Purpose:</b> make the dashboard "
        "understandable to a public audience, "
        "disaster-management users and "
        "technical evaluators."
    )


    st.markdown(

        """
        ### The dashboard answers these questions

        1. **What is happening now?**  
           Current reported affected people,
           safety-centre population and hazard context.

        2. **What may happen next?**  
           National estimates for the next eligible
           DMC Situation Report, up to a 48-hour gap.

        3. **How uncertain is the prediction?**  
           A model-based 90% prediction interval
           is displayed for human impact.

        4. **Is the situation escalating unusually fast?**  
           Experimental escalation risk and
           rapid-growth monitoring provide
           research signals.

        5. **What needs attention?**  
           DRPI and SCPI summarize research-derived
           response priority and relative
           safety-centre pressure.

        6. **Where is hazard information available?**  
           The interactive district map shows
           district hazard context and source freshness.

        7. **Have we seen something similar before?**  
           The similar-event finder retrieves
           historical reference situations.

        8. **Why did the model predict this?**  
           SHAP shows model features that influenced
           the human-impact forecast.

        9. **How well did the models work?**  
           The performance page reports held-out
           chronological test results and baselines.
        """
    )


    st.markdown(
        "### Terms to understand"
    )


    glossary = pd.DataFrame(

        [

            [
                "Affected people",

                (
                    "National total reported as affected "
                    "in the DMC Situation Report target "
                    "used by the model."
                ),
            ],


            [
                "Predicted next affected",

                (
                    "Model estimate for the next eligible "
                    "DMC Situation Report, with a maximum "
                    "48-hour gap."
                ),
            ],


            [
                "90% model-based range",

                (
                    "Model-based prediction interval "
                    "around the human-impact forecast; "
                    "not a guarantee."
                ),
            ],


            [
                "DRPI",

                (
                    "Research-derived Disaster Response "
                    "Priority Index; not an official warning."
                ),
            ],


            [
                "SCPI",

                (
                    "Research-derived relative Safety-Centre "
                    "Pressure Index; not capacity occupancy."
                ),
            ],


            [
                "Escalation risk",

                (
                    "Experimental Low / Moderate / High / "
                    "Critical research classification."
                ),
            ],


            [
                "Rapid-growth score",

                (
                    "Historical signal for unusually "
                    "rapid increases in reported impact."
                ),
            ],


            [
                "Similarity score",

                (
                    "Feature-space resemblance to a "
                    "historical event; not probability."
                ),
            ],


            [
                "SHAP",

                (
                    "Explanation of model feature "
                    "contributions; not causal proof."
                ),
            ],


            [
                "Stale / Unavailable",

                (
                    "The supporting source is too old "
                    "or missing. It does not mean "
                    "zero hazard."
                ),
            ],
        ],

        columns=[
            "Term",
            "Meaning",
        ],
    )


    st.dataframe(

        glossary,

        hide_index=True,

        use_container_width=True,
    )


    warning(

        "<b>Scope:</b> FloodImpact-LK is a "
        "research decision-support prototype. "
        "It does not replace official DMC, "
        "NBRO, Department of Meteorology "
        "or Irrigation warnings."
    )

# FOOTER

st.markdown(
    "---"
)


st.markdown(

    """
    <div class="footer-note">

    FloodImpact-LK •

    Official-report-driven near-real-time
    decision-support research prototype •

    National human-impact and safety-demand forecasts •

    District map used for hazard context only •

    Research-derived indices are not official
    warning classifications.

    </div>
    """,

    unsafe_allow_html=True,
)