from pathlib import Path
import argparse
import json
import math
import os
import re
import shutil
import tempfile
import joblib
import numpy as np
import pandas as pd
import pdfplumber
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


# =========================================================
# FloodImpact-LK - Safe Latest Inference
# =========================================================
# - DOES NOT execute the notebook.
# - DOES NOT retrain predictive models.
# - Loads frozen deployment .joblib models only.
# - Reads exact validated parser functions from notebook
#   source (definitions only; no notebook cell execution).
# - Uses only source information available at/before each
#   Situation Report timestamp.
# - Drought Situation Reports are excluded.
# - Forecasts remain NATIONAL; district map is hazard context.
# - Final dashboard outputs are staged before replacement.
# =========================================================

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGURES = ROOT / "outputs" / "figures"

SITUATION_DIR = RAW / "situation_reports"
WEATHER_DIR = RAW / "weather_reports"
RIVER_DIR = RAW / "river_reports"
LANDSLIDE_DIR = RAW / "landslide_reports"

SITUATION_METADATA = SITUATION_DIR / "dmc_situation_reports_metadata.csv"

SITUATION_CLEAN = PROCESSED / "situation_reports_cleaned.csv"
WEATHER_CLEAN = PROCESSED / "weather_reports_cleaned.csv"
RIVER_CLEAN = PROCESSED / "river_reports_cleaned.csv"
RIVER_STATIONS = PROCESSED / "river_station_observations.csv"
LANDSLIDE_CLEAN = PROCESSED / "landslide_reports_cleaned.csv"
LANDSLIDE_AREAS = PROCESSED / "landslide_warning_areas.csv"

ANALYTICAL_SCOPE = PROCESSED / "analytical_scope_dataset.csv"
SCOPE_AUDIT = PROCESSED / "situation_scope_classification.csv"
INTEGRATED = PROCESSED / "integrated_disaster_dataset.csv"

TREND_CSV = PROCESSED / "trend_history_dashboard.csv"
SIMILAR_CSV = PROCESSED / "similar_events_dashboard.csv"
LATEST_MODEL_JSON = PROCESSED / "latest_model_snapshot.json"
LATEST_RESEARCH_JSON = PROCESSED / "latest_research_context.json"
SHAP_CSV = PROCESSED / "shap_latest_deployment_contributions.csv"
SHAP_JSON = PROCESSED / "latest_deployment_shap.json"
MAP_HTML = PROCESSED / "floodimpact_lk_district_hazard_map.html"
PAYLOAD_JSON = PROCESSED / "dashboard_payload.json"

SHAP_IMAGE = FIGURES / "shap_latest_deployment_waterfall.png"
BOUNDARY_CACHE = PROCESSED / "sri_lanka_adm2_geoboundaries.geojson"

MODEL_PATHS = {
    "human": MODELS / "deployment_human_impact_model.joblib",
    "safety_use": MODELS / "deployment_safety_usage_model.joblib",
    "safety_demand": MODELS / "deployment_safety_demand_model.joblib",
    "risk": MODELS / "deployment_escalation_risk_model.joblib",
    "lower": MODELS / "deployment_lower_90_model.joblib",
    "upper": MODELS / "deployment_upper_90_model.joblib",
    "features": MODELS / "core_model_features.joblib",
}
FROZEN_INDEX_REFERENCE = (
    MODELS / "frozen_research_index_reference.csv"
)

FROZEN_REFERENCE_CUTOFF = pd.Timestamp("2026-08-14 10:00:00")
MAX_FORECAST_GAP_HOURS = 48.0
MODERATE_HIGH_THRESHOLD = 142.0
HIGH_CRITICAL_THRESHOLD = 5184.65

HISTORY_WINDOW = 120
MIN_POSITIVE_HISTORY = 20
MIN_MATERIAL_INCREASE = 142.0
EXTREME_PERCENTILE = 0.95


def now_iso():
    return pd.Timestamp.now(tz="Asia/Colombo").isoformat()


def jvalue(x):
    if x is None:
        return None
    if isinstance(x, pd.Timestamp):
        return None if pd.isna(x) else x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, float) and math.isnan(x):
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    return x


def clean_json(x):
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_json(v) for v in x]
    return jvalue(x)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def safe_num(x):
    return pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]


def signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def inverse_signed_log1p(x):
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.expm1(np.abs(x))


def filename_timestamp(name, prefix):
    m = re.search(
        rf"^{re.escape(prefix)}_(\d{{8}})_(\d{{4}}|unknown)_",
        str(name),
        flags=re.I,
    )
    if not m:
        return pd.NaT
    d, t = m.groups()
    if t.lower() == "unknown":
        t = "0000"
    return pd.to_datetime(d + t, format="%Y%m%d%H%M", errors="coerce")


def align_concat(old, new):
    if new.empty:
        return old.copy()
    cols = sorted(set(old.columns) | set(new.columns))
    old = old.copy()
    new = new.copy()
    for c in cols:
        if c not in old.columns:
            old[c] = np.nan
        if c not in new.columns:
            new[c] = np.nan
    return pd.concat([old[cols], new[cols]], ignore_index=True)


def require_files():
    required = [
        SITUATION_CLEAN, WEATHER_CLEAN, RIVER_CLEAN,
        RIVER_STATIONS, LANDSLIDE_CLEAN, LANDSLIDE_AREAS,
        ANALYTICAL_SCOPE,
        FROZEN_INDEX_REFERENCE,
        *MODEL_PATHS.values(),
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required frozen project files are missing:\n" + "\n".join(missing)
        )

# Exact validated parser/helper definitions copied from the frozen final notebook source. No notebook execution needed.

def extract_pdf_text(pdf_path):
    pages_text = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
        return '\n'.join(pages_text)
    except Exception as e:
        print(f'Error reading {pdf_path.name}: {e}')
        return ''

def clean_number(value):
    if value is None:
        return np.nan
    value = str(value).replace(',', '').strip()
    match = re.search('\\d+', value)
    if match:
        return int(match.group())
    return np.nan

def extract_date_from_filename(filename):
    match = re.search('(20\\d{2})[-_]?(\\d{2})[-_]?(\\d{2})', filename)
    if match:
        year, month, day = match.groups()
        try:
            return pd.Timestamp(year=int(year), month=int(month), day=int(day))
        except ValueError:
            return pd.NaT
    return pd.NaT

def normalize_text(text):
    text = text.replace('\xa0', ' ')
    text = re.sub('[ \\t]+', ' ', text)
    return text

def get_lines(text):
    text = normalize_text(text)
    return [line.strip() for line in text.splitlines() if line.strip()]

def detect_disaster_types(text):
    text = text.lower()
    disaster_types = []
    disaster_keywords = {'Flood': ['flood', 'flooding'], 'Landslide': ['landslide', 'land slide'], 'Heavy Rain': ['heavy rain', 'heavy rainfall'], 'Strong Wind': ['strong wind', 'high wind'], 'Cyclone': ['cyclone', 'cyclonic'], 'Lightning': ['lightning', 'thunderstorm']}
    for disaster, keywords in disaster_keywords.items():
        if any((keyword in text for keyword in keywords)):
            disaster_types.append(disaster)
    if not disaster_types:
        return 'Other'
    return ', '.join(disaster_types)

TOTAL_FIELDS = [
    "affected_families",
    "affected_people",
    "deaths",
    "injured",
    "missing",
    "fully_damaged_houses",
    "partially_damaged_houses",
    "enterprise_damage",
    "infrastructure_damage",
    "safety_centres",
    "families_in_safety_centres",
    "people_in_safety_centres",
]

def extract_grand_total(text):
    lines = get_lines(text)
    for line in lines:
        if 'grand total' in line.lower():
            part = re.split('grand\\s+total', line, flags=re.IGNORECASE)[-1]
            numbers = re.findall('(?<!\\d)\\d[\\d,]*(?!\\d)', part)
            values = [int(x.replace(',', '')) for x in numbers]
            if len(values) >= 12:
                return (values[:12], 'grand_total')
    for line in lines:
        line_lower = line.lower()
        if 'district total' in line_lower or 'province total' in line_lower:
            numbers = re.findall('(?<!\\d)\\d[\\d,]*(?!\\d)', line)
            values = [int(x.replace(',', '')) for x in numbers]
            if len(values) >= 12 and all((value == 0 for value in values[-12:])):
                return ([0] * 12, 'confirmed_zero_report')
    has_ref_error = any(('grand total' in line.lower() and '#ref!' in line.lower() for line in lines))
    if has_ref_error:
        province_candidates = []
        for line in lines:
            if 'province total' in line.lower():
                numbers = re.findall('(?<!\\d)\\d[\\d,]*(?!\\d)', line)
                values = [int(x.replace(',', '')) for x in numbers]
                if len(values) >= 12:
                    province_candidates.append(values[-12:])
        if len(province_candidates) == 1:
            return (province_candidates[0], 'grand_total_ref_fallback')
    has_grand_total = any(('grand total' in line.lower() for line in lines))
    nil_count = len(re.findall('\\b(?:nil|nill)\\b', text, flags=re.IGNORECASE))
    numeric_summary_exists = False
    for line in lines:
        numbers = re.findall('(?<!\\d)\\d[\\d,]*(?!\\d)', line)
        if len(numbers) >= 5:
            numeric_summary_exists = True
            break
    if has_grand_total and nil_count >= 3 and (not numeric_summary_exists):
        return ([0] * 12, 'confirmed_zero_report')
    return (None, 'failed')

def extract_modern_total(text):
    lines = get_lines(text)
    for line in lines:
        if not re.match('^\\s*total\\b', line, flags=re.IGNORECASE):
            continue
        numbers = re.findall('(?<!\\d)\\d[\\d,]*(?!\\d)', line)
        values = [int(x.replace(',', '')) for x in numbers]
        if len(values) == 18:
            return ({'peak_affected_families': values[0], 'peak_affected_people': values[1], 'affected_families': values[2], 'affected_people': values[3], 'deaths': values[4], 'injured': np.nan, 'missing': values[5], 'fully_damaged_houses': values[6], 'partially_damaged_houses': values[7], 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'peak_safety_centres': values[8], 'peak_families_in_safety_centres': values[9], 'peak_people_in_safety_centres': values[10], 'safety_centres': values[11], 'families_in_safety_centres': values[12], 'people_in_safety_centres': values[13], 'families_with_relatives': values[14], 'people_with_relatives': values[15], 'idp_families': values[16], 'idp_people': values[17]}, 'modern_18_column')
        if len(values) == 13:
            return ({'affected_families': values[0], 'affected_people': values[1], 'deaths': values[2], 'injured': np.nan, 'missing': values[3], 'fully_damaged_houses': values[4], 'partially_damaged_houses': values[5], 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'safety_centres': values[6], 'families_in_safety_centres': values[7], 'people_in_safety_centres': values[8], 'families_with_relatives': values[9], 'people_with_relatives': values[10], 'idp_families': values[11], 'idp_people': values[12]}, 'modern_13_column')
        if len(values) == 11:
            return ({'affected_families': values[0], 'affected_people': values[1], 'deaths': values[2], 'injured': np.nan, 'missing': values[3], 'fully_damaged_houses': values[4], 'partially_damaged_houses': values[5], 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'safety_centres': values[6], 'families_in_safety_centres': values[7], 'people_in_safety_centres': values[8], 'families_with_relatives': values[9], 'people_with_relatives': values[10]}, 'modern_11_column')
        if len(values) == 9:
            return ({'affected_families': values[0], 'affected_people': values[1], 'deaths': values[2], 'injured': np.nan, 'missing': values[3], 'fully_damaged_houses': values[4], 'partially_damaged_houses': values[5], 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'safety_centres': values[6], 'families_in_safety_centres': values[7], 'people_in_safety_centres': values[8]}, 'modern_9_column')
        if len(values) == 7:
            if values[1] >= values[0]:
                return ({'affected_families': values[0], 'affected_people': values[1], 'deaths': values[2], 'injured': np.nan, 'missing': values[3], 'fully_damaged_houses': np.nan, 'partially_damaged_houses': np.nan, 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'safety_centres': values[4], 'families_in_safety_centres': values[5], 'people_in_safety_centres': values[6]}, 'modern_7_column')
    total_markers = ['එකතුව', 'එ තුව']
    for line in lines:
        if not any((marker in line for marker in total_markers)):
            continue
        numbers = re.findall('(?<!\\d)\\d[\\d,]*(?!\\d)', line)
        values = [int(x.replace(',', '')) for x in numbers]
        if len(values) == 13:
            return ({'affected_families': values[1], 'affected_people': values[2], 'deaths': values[3], 'injured': values[4], 'missing': values[5], 'fully_damaged_houses': values[6], 'partially_damaged_houses': values[7], 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'families_with_relatives': values[8], 'people_with_relatives': values[9], 'safety_centres': values[10], 'families_in_safety_centres': values[11], 'people_in_safety_centres': values[12]}, 'modern_summary_13_column')
        if len(values) == 9:
            if values[1] >= values[0]:
                return ({'affected_families': values[0], 'affected_people': values[1], 'deaths': values[2], 'injured': np.nan, 'missing': values[3], 'fully_damaged_houses': values[4], 'partially_damaged_houses': values[5], 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'safety_centres': values[6], 'families_in_safety_centres': values[7], 'people_in_safety_centres': values[8]}, 'modern_summary_9_column')
        if len(values) == 8:
            return ({'affected_families': values[1], 'affected_people': values[2], 'deaths': values[3], 'injured': np.nan, 'missing': values[4], 'fully_damaged_houses': np.nan, 'partially_damaged_houses': np.nan, 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'safety_centres': values[5], 'families_in_safety_centres': values[6], 'people_in_safety_centres': values[7]}, 'modern_summary_8_column')
        if len(values) == 7:
            if values[1] >= values[0]:
                return ({'affected_families': values[0], 'affected_people': values[1], 'deaths': values[2], 'injured': np.nan, 'missing': values[3], 'fully_damaged_houses': np.nan, 'partially_damaged_houses': np.nan, 'enterprise_damage': np.nan, 'infrastructure_damage': np.nan, 'safety_centres': values[4], 'families_in_safety_centres': values[5], 'people_in_safety_centres': values[6]}, 'modern_summary_7_column')
    return (None, None)

def is_non_comparable_incident_report(text):
    text_lower = normalize_text(text).lower()
    return 'daily disaster incident report' in text_lower and 'situation report' not in text_lower

def is_non_situation_report(text):
    text_lower = normalize_text(text).lower()
    weather_markers = ['weather forecast', 'rainfall amount', 'condition of rain', 'state of sea', 'showers or thundershowers', 'department of meteorology', 'wind speed', 'sea areas']
    weather_score = sum((marker in text_lower for marker in weather_markers))
    if weather_score >= 2:
        return True
    river_markers = ['minor flood', 'major flood', 'river level', 'water level', 'rising', 'falling']
    river_score = sum((marker in text_lower for marker in river_markers))
    if river_score >= 2:
        return True
    return False

def extract_situation_report(pdf_path):
    text = extract_pdf_text(pdf_path)
    record = {'file_name': pdf_path.name, 'report_date': extract_date_from_filename(pdf_path.name), 'disaster_types': detect_disaster_types(text), 'text_length': len(text)}
    record['families_with_relatives'] = np.nan
    record['people_with_relatives'] = np.nan
    record['peak_affected_families'] = np.nan
    record['peak_affected_people'] = np.nan
    record['peak_safety_centres'] = np.nan
    record['peak_families_in_safety_centres'] = np.nan
    record['peak_people_in_safety_centres'] = np.nan
    record['idp_families'] = np.nan
    record['idp_people'] = np.nan
    totals, extraction_status = extract_grand_total(text)
    if totals is not None:
        for field, value in zip(TOTAL_FIELDS, totals):
            record[field] = value
        record['grand_total_extracted'] = True
        record['extraction_status'] = extraction_status
        return record
    modern_values, modern_status = extract_modern_total(text)
    if modern_values is not None:
        for field, value in modern_values.items():
            record[field] = value
        record['grand_total_extracted'] = True
        record['extraction_status'] = modern_status
        return record
    if is_non_comparable_incident_report(text):
        for field in TOTAL_FIELDS:
            record[field] = np.nan
        record['grand_total_extracted'] = False
        record['extraction_status'] = 'non_comparable_incident_report'
        return record
    if is_non_situation_report(text):
        for field in TOTAL_FIELDS:
            record[field] = np.nan
        record['grand_total_extracted'] = False
        record['extraction_status'] = 'non_situation_report'
        return record
    for field in TOTAL_FIELDS:
        record[field] = np.nan
    record['grand_total_extracted'] = False
    if len(text.strip()) < 500:
        record['extraction_status'] = 'insufficient_text'
    else:
        record['extraction_status'] = 'failed'
    return record

def normalize_text(text):
    if text is None:
        return ''
    text = str(text)
    text = text.replace('\xa0', ' ')
    text = re.sub('\\s+', ' ', text)
    return text.strip()

def get_lines(text):
    if text is None:
        return []
    return [normalize_text(line) for line in str(text).splitlines() if normalize_text(line)]

WEATHER_FILENAME_PATTERN = re.compile('^weather_(\\d{8})_(\\d{4}|unknown)_(.+)_([0-9a-f]{8})\\.pdf$', flags=re.IGNORECASE)

def parse_weather_filename(file_name):
    match = WEATHER_FILENAME_PATTERN.match(file_name)
    if not match:
        return {'report_date': pd.NaT, 'issue_time': np.nan, 'report_type': 'unknown'}
    date_text, time_text, report_type, _ = match.groups()
    return {'report_date': pd.to_datetime(date_text, format='%Y%m%d', errors='coerce'), 'issue_time': np.nan if time_text.lower() == 'unknown' else time_text, 'report_type': report_type.lower()}

def remove_weather_legends(text):
    cleaned_lines = []
    for line in get_lines(text):
        line_lower = normalize_text(line).lower()
        if 'rainfall amount (mm)' in line_lower:
            continue
        if 'spatial rainfall distribution' in line_lower:
            continue
        if 'precipitation rate categories' in line_lower:
            continue
        cleaned_lines.append(line)
    return ' '.join(cleaned_lines)

def extract_land_weather_text(text):
    text_lower = text.lower()
    land_markers = ['for land:', 'for land areas', 'for land area']
    start_index = 0
    for marker in land_markers:
        idx = text_lower.find(marker)
        if idx != -1:
            start_index = idx
            break
    land_text = text[start_index:]
    land_lower = land_text.lower()
    marine_markers = ['weather forecast for sea areas around the island', 'for the sea areas around the island', 'for sea areas around the island', 'for multi-day boats', 'for multi day boats']
    end_positions = []
    for marker in marine_markers:
        idx = land_lower.find(marker)
        if idx != -1:
            end_positions.append(idx)
    if end_positions:
        land_text = land_text[:min(end_positions)]
    return land_text

def extract_rainfall_threshold_mm(text):
    cleaned_text = remove_weather_legends(text)
    patterns = ['(?:above|over|about|around|approximately|more than)\\s*(\\d{2,3}(?:\\.\\d+)?)\\s*mm', '(\\d{2,3}(?:\\.\\d+)?)\\s*mm\\s*(?:or more|and above)']
    rainfall_values = []
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_text, flags=re.IGNORECASE)
        rainfall_values.extend((float(value) for value in matches))
    if rainfall_values:
        return max(rainfall_values)
    return np.nan

def extract_max_wind_kmph(text):
    cleaned_text = remove_weather_legends(text)
    wind_values = []
    ranges = re.findall('\\(?\\s*(\\d{1,3})\\s*[-–]\\s*(\\d{1,3})\\s*\\)?\\s*kmph', cleaned_text, flags=re.IGNORECASE)
    for low, high in ranges:
        wind_values.append(max(float(low), float(high)))
    singles = re.findall('(?:up to|about|around)\\s*\\(?\\s*(\\d{1,3})\\s*\\)?\\s*kmph', cleaned_text, flags=re.IGNORECASE)
    wind_values.extend((float(value) for value in singles))
    if wind_values:
        return max(wind_values)
    return np.nan

def extract_warning_color(text):
    match = re.search('\\bCOLOR\\s*:\\s*(Red|Amber|Green)\\b', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).title()
    return np.nan

def extract_forecast_horizon_hours(text):
    match = re.search('(?:next|for next)\\s*(\\d{1,3})\\s*hours?', text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search('(?:next|for next)\\s*(\\d{1,2})\\s*days?', text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1)) * 24
    return np.nan

def extract_weather_report(pdf_path):
    text = extract_pdf_text(pdf_path)
    metadata = parse_weather_filename(pdf_path.name)
    text_lower = normalize_text(text).lower()
    report_type = metadata['report_type']
    has_land_context = 'for land' in text_lower or 'land areas' in text_lower or 'weather forecast for sri lanka' in text_lower or ('general weather forecast for sri lanka' in text_lower)
    has_marine_context = 'multi-day boats' in text_lower or 'multi day boats' in text_lower or 'sea areas' in text_lower or ('fishing and naval' in text_lower)
    marine_only_flag = has_marine_context and (not has_land_context)
    land_text = extract_land_weather_text(text)
    rainfall_mm = extract_rainfall_threshold_mm(land_text)
    max_wind = extract_max_wind_kmph(land_text)
    if marine_only_flag:
        rainfall_mm = np.nan
        max_wind = np.nan
    heavy_rain_flag = not marine_only_flag and (report_type == 'heavy_rain' or (not pd.isna(rainfall_mm) and rainfall_mm >= 50))
    thunderstorm_flag = not marine_only_flag and (report_type == 'thunderstorm' or 'thundershower' in text_lower or 'thunderstorm' in text_lower or ('thunder shower' in text_lower))
    strong_wind_flag = not marine_only_flag and (report_type == 'strong_wind' or (not pd.isna(max_wind) and max_wind >= 40))
    low_pressure_flag = report_type == 'low_pressure' or 'low pressure area' in text_lower
    depression_flag = report_type == 'depression' or 'depression' in text_lower
    cyclone_flag = report_type == 'cyclone' or 'cyclonic storm' in text_lower or 'cyclone' in text_lower
    experimental_flag = 'experimental' in text_lower
    record = {'file_name': pdf_path.name, 'report_date': metadata['report_date'], 'issue_time': metadata['issue_time'], 'report_type': report_type, 'warning_color': extract_warning_color(text), 'forecast_horizon_hours': extract_forecast_horizon_hours(text), 'rainfall_threshold_mm': rainfall_mm, 'max_wind_kmph': max_wind, 'heavy_rain_flag': heavy_rain_flag, 'thunderstorm_flag': thunderstorm_flag, 'strong_wind_flag': strong_wind_flag, 'low_pressure_flag': low_pressure_flag, 'depression_flag': depression_flag, 'cyclone_flag': cyclone_flag, 'experimental_flag': experimental_flag, 'marine_only_flag': marine_only_flag, 'text_length': len(text), 'extraction_status': 'ok' if len(text.strip()) >= 100 else 'insufficient_text'}
    return record

RIVER_FILENAME_PATTERN = re.compile('^river_(\\d{8})_(\\d{4}|unknown)_(.+)_([0-9a-f]{8})\\.pdf$', flags=re.IGNORECASE)

def parse_river_filename(file_name):
    match = RIVER_FILENAME_PATTERN.match(file_name)
    if not match:
        return {'report_date': pd.NaT, 'issue_time': np.nan, 'report_type': 'unknown'}
    date_text, time_text, report_type, _ = match.groups()
    return {'report_date': pd.to_datetime(date_text, format='%Y%m%d', errors='coerce'), 'issue_time': np.nan if time_text.lower() == 'unknown' else time_text, 'report_type': report_type.lower()}

RIVER_ROW_PATTERN = re.compile('^(?P<label>.+?)\\s+(?P<unit>m|ft)\\s+(?P<alert>-?\\d+(?:\\.\\d+)?)\\s+(?P<minor>-?\\d+(?:\\.\\d+)?)\\s+(?P<major>-?\\d+(?:\\.\\d+)?)\\s+(?P<previous>-?\\d+(?:\\.\\d+)?)\\s+(?P<current>-?\\d+(?:\\.\\d+)?)\\s+(?P<status>Normal|Alert|Minor Flood|Major Flood)(?:\\s+(?P<trend>Rising|Falling))?\\s+(?P<rain>-|NA|-?\\d+(?:\\.\\d+)?)$', flags=re.IGNORECASE)

def extract_river_station_rows(text, pdf_path, metadata):
    rows = []
    for line in get_lines(text):
        match = RIVER_ROW_PATTERN.match(line.strip())
        if not match:
            continue
        values = match.groupdict()
        previous_level = float(values['previous'])
        current_level = float(values['current'])
        level_change = current_level - previous_level
        if level_change > 0:
            derived_trend = 'Rising'
        elif level_change < 0:
            derived_trend = 'Falling'
        else:
            derived_trend = 'Stable'
        rainfall_text = values['rain']
        rainfall_24h = np.nan if rainfall_text.upper() in ['-', 'NA'] else float(rainfall_text)
        alert_level = float(values['alert'])
        alert_ratio = current_level / alert_level if alert_level > 0 else np.nan
        row = {'file_name': pdf_path.name, 'report_date': metadata['report_date'], 'issue_time': metadata['issue_time'], 'report_type': metadata['report_type'], 'station_label_raw': normalize_text(values['label']), 'unit': values['unit'].lower(), 'alert_level': alert_level, 'minor_flood_level': float(values['minor']), 'major_flood_level': float(values['major']), 'water_level_previous': previous_level, 'water_level_current': current_level, 'level_change': level_change, 'status': values['status'].title(), 'reported_trend': values['trend'].title() if values['trend'] else np.nan, 'derived_trend': derived_trend, 'rainfall_24h_mm': rainfall_24h, 'alert_ratio': alert_ratio}
        rows.append(row)
    return rows

def is_river_warning_document(text):
    text_lower = normalize_text(text).lower()
    compact_text = re.sub('\\s+', '', text_lower)
    warning_markers = ['flood warning message', 'flood warning', 'major flood situation', 'high flood situation', 'early warning', 'warning is issued', 'warning issued']
    standard_warning = any((marker in text_lower for marker in warning_markers))
    dmc_forwarded_warning = 'disastermanagementcentre' in compact_text and ('dmc/08/irrigation' in compact_text or 'irrigation' in compact_text)
    return standard_warning or dmc_forwarded_warning

def extract_river_report(pdf_path):
    text = extract_pdf_text(pdf_path)
    metadata = parse_river_filename(pdf_path.name)
    text_lower = normalize_text(text).lower()
    station_rows = extract_river_station_rows(text, pdf_path, metadata)
    station_df_temp = pd.DataFrame(station_rows)
    report_type = metadata['report_type']
    warning_document = report_type in ['flood_warning', 'flood_advisory'] or (len(station_rows) == 0 and is_river_warning_document(text))
    withdrawal_markers = ['hereby withdrawn', 'withdrawal', 'withdrawn', 'removal', 'threat is over']
    warning_withdrawn_flag = warning_document and any((marker in text_lower for marker in withdrawal_markers))
    if warning_document and len(text.strip()) < 100:
        active_warning_flag = np.nan
    else:
        active_warning_flag = warning_document and (not warning_withdrawn_flag)
    if not station_df_temp.empty:
        status_lower = station_df_temp['status'].str.lower()
        normal_count = int((status_lower == 'normal').sum())
        alert_count = int((status_lower == 'alert').sum())
        minor_flood_count = int((status_lower == 'minor flood').sum())
        major_flood_count = int((status_lower == 'major flood').sum())
        rising_count = int((station_df_temp['derived_trend'] == 'Rising').sum())
        falling_count = int((station_df_temp['derived_trend'] == 'Falling').sum())
        max_rainfall = station_df_temp['rainfall_24h_mm'].max()
        mean_rainfall = station_df_temp['rainfall_24h_mm'].mean()
        max_alert_ratio = station_df_temp['alert_ratio'].max()
    else:
        normal_count = 0
        alert_count = 0
        minor_flood_count = 0
        major_flood_count = 0
        rising_count = 0
        falling_count = 0
        max_rainfall = np.nan
        mean_rainfall = np.nan
        max_alert_ratio = np.nan
    if len(text.strip()) < 100:
        extraction_status = 'insufficient_text'
    elif report_type in ['water_level', 'water_level_rainfall'] and len(station_rows) == 0 and is_river_warning_document(text):
        extraction_status = 'warning_document_no_station_table'
    elif report_type in ['water_level', 'water_level_rainfall'] and len(station_rows) == 0:
        extraction_status = 'no_station_rows'
    else:
        extraction_status = 'ok'
    return ({'file_name': pdf_path.name, 'report_date': metadata['report_date'], 'issue_time': metadata['issue_time'], 'report_type': report_type, 'station_rows_extracted': len(station_rows), 'normal_station_count': normal_count, 'alert_station_count': alert_count, 'minor_flood_station_count': minor_flood_count, 'major_flood_station_count': major_flood_count, 'rising_station_count': rising_count, 'falling_station_count': falling_count, 'max_rainfall_24h_mm': max_rainfall, 'mean_rainfall_24h_mm': mean_rainfall, 'max_alert_ratio': max_alert_ratio, 'warning_withdrawn_flag': warning_withdrawn_flag, 'active_warning_flag': active_warning_flag, 'text_length': len(text), 'extraction_status': extraction_status}, station_rows)


# VALIDATED LANDSLIDE PARSER
# Copied/adapted from frozen final notebook definitions.

LANDSLIDE_FILENAME_PATTERN = re.compile(
    r"^landslide_(\d{8})_(\d{4}|unknown)_(.+)_([0-9a-f]{8})\.pdf$",
    flags=re.IGNORECASE,
)


DISTRICT_VARIANTS = {
    "ampara": "Ampara",
    "anuradhapura": "Anuradhapura",
    "badulla": "Badulla",
    "batticaloa": "Batticaloa",
    "colombo": "Colombo",
    "galle": "Galle",
    "gampaha": "Gampaha",
    "hambantota": "Hambantota",
    "jaffna": "Jaffna",
    "kalutara": "Kalutara",
    "kandy": "Kandy",
    "kegalle": "Kegalle",
    "kilinochchi": "Kilinochchi",
    "kurunegala": "Kurunegala",
    "mannar": "Mannar",
    "matale": "Matale",
    "matara": "Matara",
    "monaragala": "Monaragala",
    "moneragala": "Monaragala",
    "mullaitivu": "Mullaitivu",
    "nuwara eliya": "Nuwara Eliya",
    "polonnaruwa": "Polonnaruwa",
    "puttalam": "Puttalam",
    "ratnapura": "Ratnapura",
    "rathnapura": "Ratnapura",
    "trincomalee": "Trincomalee",
    "vavuniya": "Vavuniya",
}


LEVEL_INFO = {
    1: {
        "color": "Yellow",
        "action": "Watch",
    },
    2: {
        "color": "Amber",
        "action": "Alert",
    },
    3: {
        "color": "Red",
        "action": "Evacuation",
    },
}


def parse_landslide_filename(file_name):

    match = LANDSLIDE_FILENAME_PATTERN.match(
        file_name
    )

    if not match:

        return {
            "report_date": pd.NaT,
            "issue_time": np.nan,
            "report_type": "unknown",
        }

    (
        date_text,
        time_text,
        report_type,
        _,
    ) = match.groups()

    return {

        "report_date":
            pd.to_datetime(
                date_text,
                format="%Y%m%d",
                errors="coerce",
            ),

        "issue_time":
            (
                np.nan
                if time_text.lower() == "unknown"
                else time_text
            ),

        "report_type":
            report_type.lower(),
    }


def normalize_district(value):

    if value is None:
        return None

    value = normalize_text(
        str(value)
    ).lower()

    if value in DISTRICT_VARIANTS:

        return DISTRICT_VARIANTS[
            value
        ]

    for (
        variant,
        canonical,
    ) in DISTRICT_VARIANTS.items():

        if re.search(
            rf"\b{re.escape(variant)}\b",
            value,
        ):

            return canonical

    return None


def detect_warning_level_from_line(
    line,
):

    line_lower = normalize_text(
        line
    ).lower()

    if "yellow" in line_lower:

        if (
            "level" in line_lower
            or "color" in line_lower
            or "colour" in line_lower
        ):

            return 1

    if "amber" in line_lower:

        if (
            "level" in line_lower
            or "color" in line_lower
            or "colour" in line_lower
            or "alert" in line_lower
        ):

            return 2

    if "red" in line_lower:

        if (
            "level" in line_lower
            or "color" in line_lower
            or "colour" in line_lower
            or "evacuation" in line_lower
        ):

            return 3

    return None


def extract_message_id(text):

    match = re.search(
        r"Message\s*ID\s*:?\s*"
        r"(LEWM-[A-Za-z0-9\-]+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return match.group(1)

    return np.nan


def extract_update_number(text):

    match = re.search(
        r"Update\s*:?\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:

        return int(
            match.group(1)
        )

    return np.nan


def extract_validity_period(text):

    pattern = re.compile(

        r"From\s+"
        r"(\d{1,2}:\d{2})\s*hrs\s*on\s*"
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})"
        r"\s+To\s+"
        r"(\d{1,2}:\d{2})\s*hrs\s*on\s*"
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})",

        flags=re.IGNORECASE,
    )

    match = pattern.search(
        normalize_text(
            text
        )
    )

    if not match:

        return (
            pd.NaT,
            pd.NaT,
        )

    (
        start_time,
        start_date,
        end_time,
        end_date,
    ) = match.groups()

    start_dt = pd.to_datetime(
        f"{start_date} {start_time}",
        dayfirst=True,
        errors="coerce",
    )

    end_dt = pd.to_datetime(
        f"{end_date} {end_time}",
        dayfirst=True,
        errors="coerce",
    )

    return (
        start_dt,
        end_dt,
    )

# MODERN LANDSLIDE TABLE EXTRACTION

def extract_modern_landslide_areas(
    pdf_path,
    metadata,
):

    rows = []

    with pdfplumber.open(
        pdf_path
    ) as pdf:

        for (
            page_no,
            page,
        ) in enumerate(
            pdf.pages,
            start=1,
        ):

            tables = (
                page.extract_tables()
            )

            for table in tables:

                for row in table:

                    if (
                        not row
                        or len(row) < 4
                    ):

                        continue

                    district = (
                        normalize_district(
                            row[0]
                        )
                    )

                    if district is None:

                        continue

                    for (
                        column_index,
                        level,
                    ) in [
                        (1, 1),
                        (2, 2),
                        (3, 3),
                    ]:

                        cell_text = (
                            normalize_text(
                                row[
                                    column_index
                                ]
                                if row[
                                    column_index
                                ]
                                else ""
                            )
                        )

                        if not cell_text:

                            continue

                        rows.append({

                            "file_name":
                                pdf_path.name,

                            "report_date":
                                metadata[
                                    "report_date"
                                ],

                            "issue_time":
                                metadata[
                                    "issue_time"
                                ],

                            "report_type":
                                metadata[
                                    "report_type"
                                ],

                            "district":
                                district,

                            "warning_level":
                                level,

                            "warning_color":
                                LEVEL_INFO[
                                    level
                                ][
                                    "color"
                                ],

                            "warning_action":
                                LEVEL_INFO[
                                    level
                                ][
                                    "action"
                                ],

                            "area_text_raw":
                                cell_text,

                            "page_number":
                                page_no,

                            "extraction_method":
                                "modern_table",
                        })

    return rows

# DSD -> DISTRICT SUPPORT FOR LEGACY REPORTS

def extract_dsd_names(
    area_text,
):

    if (
        area_text is None
        or pd.isna(
            area_text
        )
    ):

        return []

    text = str(
        area_text
    )

    text = (
        text
        .replace("↑", "")
        .replace("↓", "")
        .replace("*", "")
    )

    text = re.split(
        r"Divisional Secretariat",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    text = re.sub(
        r"\s+\band\b\s+",
        ",",
        text,
        flags=re.IGNORECASE,
    )

    parts = text.split(
        ","
    )

    cleaned = []

    for part in parts:

        part = normalize_text(
            part
        )

        part = re.sub(
            r"[^A-Za-z\s\-]",
            "",
            part,
        ).strip()

        if len(part) >= 3:

            cleaned.append(
                part
            )

    return cleaned


def build_dsd_to_district():

    mapping = {}

    print(
        "Building validated DSD -> District mapping..."
    )

    landslide_pdfs = sorted(
        LANDSLIDE_DIR.glob(
            "*.pdf"
        )
    )

    for pdf_path in landslide_pdfs:

        metadata = (
            parse_landslide_filename(
                pdf_path.name
            )
        )

        rows = (
            extract_modern_landslide_areas(
                pdf_path,
                metadata,
            )
        )

        for row in rows:

            district = row[
                "district"
            ]

            dsd_names = (
                extract_dsd_names(
                    row[
                        "area_text_raw"
                    ]
                )
            )

            for dsd in dsd_names:

                key = (
                    normalize_text(
                        dsd
                    )
                    .lower()
                )

                mapping[
                    key
                ] = district

    mapping.update({

        "kothmale":
            "Nuwara Eliya",

        "kotmale":
            "Nuwara Eliya",

        "ambagamuwa":
            "Nuwara Eliya",

        "kalawana":
            "Ratnapura",

        "nivithigala":
            "Ratnapura",

        "kuruwita":
            "Ratnapura",

        "pelmadulla":
            "Ratnapura",

        "eheliyagoda":
            "Ratnapura",

        "deraniyagala":
            "Kegalle",

        "dehiowita":
            "Kegalle",

        "yatiyanthota":
            "Kegalle",

        "aranayaka":
            "Kegalle",

        "agalawatta":
            "Kalutara",

        "bulathsinhala":
            "Kalutara",

        "walallawita":
            "Kalutara",

        "ingiriya":
            "Kalutara",

        "thawalama":
            "Galle",

        "nagoda":
            "Galle",

        "pasbage korale":
            "Kandy",
    })

    return mapping


_LANDSLIDE_DSD_TO_DISTRICT = None


def get_landslide_dsd_to_district():

    global _LANDSLIDE_DSD_TO_DISTRICT

    if (
        _LANDSLIDE_DSD_TO_DISTRICT
        is None
    ):

        _LANDSLIDE_DSD_TO_DISTRICT = (
            build_dsd_to_district()
        )

        print(
            "DSD -> District mappings:",
            len(
                _LANDSLIDE_DSD_TO_DISTRICT
            ),
        )

    return (
        _LANDSLIDE_DSD_TO_DISTRICT
    )

# LEGACY LANDSLIDE EXTRACTION

def extract_legacy_landslide_areas(
    pdf_path,
    metadata,
):

    rows = []

    current_level = None

    location_section_started = (
        False
    )

    dsd_to_district = (
        get_landslide_dsd_to_district()
    )

    with pdfplumber.open(
        pdf_path
    ) as pdf:

        for (
            page_no,
            page,
        ) in enumerate(
            pdf.pages,
            start=1,
        ):

            text = (
                page.extract_text(
                    layout=True,
                    x_tolerance=2,
                    y_tolerance=3,
                )
            )

            if not text:

                continue

            for line in (
                text.splitlines()
            ):

                clean_line = (
                    normalize_text(
                        line
                    )
                )

                if not clean_line:

                    continue

                line_lower = (
                    clean_line.lower()
                )

                detected_level = (
                    detect_warning_level_from_line(
                        clean_line
                    )
                )

                if (
                    detected_level
                    is not None
                ):

                    current_level = (
                        detected_level
                    )

                if (
                    (
                        "location"
                        in line_lower
                        and "risk"
                        in line_lower
                    )
                    or
                    (
                        "locrt"
                        in line_lower
                        and (
                            "risk"
                            in line_lower
                            or "ristr"
                            in line_lower
                        )
                    )
                    or
                    (
                        "divisional secretariat"
                        in line_lower
                    )
                ):

                    location_section_started = (
                        True
                    )

                if not location_section_started:

                    continue

                if current_level is None:

                    continue

                matched_districts = (
                    set()
                )

                # Method 1: direct district name

                for (
                    variant,
                    canonical,
                ) in (
                    DISTRICT_VARIANTS.items()
                ):

                    if re.search(
                        rf"\b{re.escape(variant)}\b",
                        line_lower,
                    ):

                        if (
                            canonical
                            == "Colombo"
                            and (
                                "jawatta"
                                in line_lower
                                or "colombo 05"
                                in line_lower
                            )
                        ):

                            continue

                        matched_districts.add(
                            canonical
                        )

                # Method 2: infer district from DSD

                english_line = re.sub(
                    r"[^a-z\s\-]",
                    " ",
                    line_lower,
                )

                english_line = (
                    normalize_text(
                        english_line
                    )
                )

                for (
                    dsd_name,
                    district,
                ) in (
                    dsd_to_district.items()
                ):

                    if (
                        len(
                            dsd_name
                        )
                        < 5
                    ):

                        continue

                    if re.search(
                        rf"\b{re.escape(dsd_name)}\b",
                        english_line,
                    ):

                        matched_districts.add(
                            district
                        )

                for district in (
                    matched_districts
                ):

                    rows.append({

                        "file_name":
                            pdf_path.name,

                        "report_date":
                            metadata[
                                "report_date"
                            ],

                        "issue_time":
                            metadata[
                                "issue_time"
                            ],

                        "report_type":
                            metadata[
                                "report_type"
                            ],

                        "district":
                            district,

                        "warning_level":
                            current_level,

                        "warning_color":
                            LEVEL_INFO[
                                current_level
                            ][
                                "color"
                            ],

                        "warning_action":
                            LEVEL_INFO[
                                current_level
                            ][
                                "action"
                            ],

                        "area_text_raw":
                            clean_line,

                        "page_number":
                            page_no,

                        "extraction_method":
                            "legacy_layout",
                    })

    return rows

# ONE LANDSLIDE PDF


def extract_landslide_report(
    pdf_path,
):

    metadata = (
        parse_landslide_filename(
            pdf_path.name
        )
    )

    text = extract_pdf_text(
        pdf_path
    )

    message_id = (
        extract_message_id(
            text
        )
    )

    update_number = (
        extract_update_number(
            text
        )
    )

    (
        validity_start,
        validity_end,
    ) = extract_validity_period(
        text
    )

    area_rows = (
        extract_modern_landslide_areas(
            pdf_path,
            metadata,
        )
    )

    extraction_method = (
        "modern_table"
    )

    if len(
        area_rows
    ) == 0:

        area_rows = (
            extract_legacy_landslide_areas(
                pdf_path,
                metadata,
            )
        )

        extraction_method = (
            "legacy_layout"
        )

    if area_rows:

        temp_df = pd.DataFrame(
            area_rows
        )

        temp_df = (
            temp_df
            .drop_duplicates(
                subset=[
                    "file_name",
                    "district",
                    "warning_level",
                ]
            )
        )

        area_rows = (
            temp_df.to_dict(
                "records"
            )
        )

    levels = [
        row[
            "warning_level"
        ]
        for row in area_rows
    ]

    districts = {
        row[
            "district"
        ]
        for row in area_rows
    }

    if levels:

        max_level = max(
            levels
        )

        max_color = (
            LEVEL_INFO[
                max_level
            ][
                "color"
            ]
        )

    else:

        max_level = np.nan
        max_color = np.nan

    level_1_count = sum(
        level == 1
        for level in levels
    )

    level_2_count = sum(
        level == 2
        for level in levels
    )

    level_3_count = sum(
        level == 3
        for level in levels
    )

    if (
        len(
            text.strip()
        )
        < 100
    ):

        extraction_status = (
            "insufficient_text"
        )

    elif (
        len(
            area_rows
        )
        == 0
    ):

        extraction_status = (
            "no_warning_areas"
        )

    else:

        extraction_status = (
            "ok"
        )

    report_record = {

        "file_name":
            pdf_path.name,

        "report_date":
            metadata[
                "report_date"
            ],

        "issue_time":
            metadata[
                "issue_time"
            ],

        "report_type":
            metadata[
                "report_type"
            ],

        "message_id":
            message_id,

        "update_number":
            update_number,

        "validity_start":
            validity_start,

        "validity_end":
            validity_end,

        "districts_extracted":
            len(
                districts
            ),

        "area_rows_extracted":
            len(
                area_rows
            ),

        "level_1_districts":
            level_1_count,

        "level_2_districts":
            level_2_count,

        "level_3_districts":
            level_3_count,

        "max_warning_level":
            max_level,

        "max_warning_color":
            max_color,

        "extraction_method":
            extraction_method,

        "text_length":
            len(
                text
            ),

        "extraction_status":
            extraction_status,
    }

    return (
        report_record,
        area_rows,
    )


def classify_unresolved_landslide_report(
    row,
):

    if (
        row[
            "extraction_status"
        ]
        == "ok"
    ):

        return "ok"

    if (
        row[
            "extraction_status"
        ]
        == "insufficient_text"
    ):

        return (
            "insufficient_text"
        )

    pdf_path = (
        LANDSLIDE_DIR
        / row[
            "file_name"
        ]
    )

    text = extract_pdf_text(
        pdf_path
    )

    text_lower = (
        normalize_text(
            text
        )
        .lower()
    )

    river_markers = (

        (
            "islandwide water level "
            "& rainfall situation "
            "in major rivers"
            in text_lower
        )

        or

        (
            "gauging station"
            in text_lower

            and "river basin"
            in text_lower

            and "water level"
            in text_lower
        )
    )

    if river_markers:

        return (
            "non_landslide_report"
        )

    return (
        "manual_review_legacy_layout"
    )

def hhmm_to_time_string(value):
    if pd.isna(value):
        return '0000'
    try:
        value = str(int(float(value)))
    except:
        value = str(value).strip()
    return value.zfill(4)

# Validated Landslide Parser
# Copied/adapted from frozen final notebook definitions.


LANDSLIDE_FILENAME_PATTERN = re.compile(
    r"^landslide_(\d{8})_(\d{4}|unknown)_(.+)_([0-9a-f]{8})\.pdf$",
    flags=re.IGNORECASE,
)


DISTRICT_VARIANTS = {
    "ampara": "Ampara",
    "anuradhapura": "Anuradhapura",
    "badulla": "Badulla",
    "batticaloa": "Batticaloa",
    "colombo": "Colombo",
    "galle": "Galle",
    "gampaha": "Gampaha",
    "hambantota": "Hambantota",
    "jaffna": "Jaffna",
    "kalutara": "Kalutara",
    "kandy": "Kandy",
    "kegalle": "Kegalle",
    "kilinochchi": "Kilinochchi",
    "kurunegala": "Kurunegala",
    "mannar": "Mannar",
    "matale": "Matale",
    "matara": "Matara",
    "monaragala": "Monaragala",
    "moneragala": "Monaragala",
    "mullaitivu": "Mullaitivu",
    "nuwara eliya": "Nuwara Eliya",
    "polonnaruwa": "Polonnaruwa",
    "puttalam": "Puttalam",
    "ratnapura": "Ratnapura",
    "rathnapura": "Ratnapura",
    "trincomalee": "Trincomalee",
    "vavuniya": "Vavuniya",
}


LEVEL_INFO = {
    1: {
        "color": "Yellow",
        "action": "Watch",
    },
    2: {
        "color": "Amber",
        "action": "Alert",
    },
    3: {
        "color": "Red",
        "action": "Evacuation",
    },
}


def parse_landslide_filename(file_name):

    match = LANDSLIDE_FILENAME_PATTERN.match(
        file_name
    )

    if not match:
        return {
            "report_date": pd.NaT,
            "issue_time": np.nan,
            "report_type": "unknown",
        }

    date_text, time_text, report_type, _ = (
        match.groups()
    )

    return {
        "report_date": pd.to_datetime(
            date_text,
            format="%Y%m%d",
            errors="coerce",
        ),
        "issue_time": (
            np.nan
            if time_text.lower() == "unknown"
            else time_text
        ),
        "report_type": report_type.lower(),
    }


def normalize_district(value):

    if value is None:
        return None

    value = normalize_text(
        str(value)
    ).lower()

    if value in DISTRICT_VARIANTS:
        return DISTRICT_VARIANTS[value]

    for variant, canonical in (
        DISTRICT_VARIANTS.items()
    ):

        if re.search(
            rf"\b{re.escape(variant)}\b",
            value,
        ):
            return canonical

    return None


def detect_warning_level_from_line(line):

    line_lower = normalize_text(
        line
    ).lower()

    if "yellow" in line_lower:
        if (
            "level" in line_lower
            or "color" in line_lower
            or "colour" in line_lower
        ):
            return 1

    if "amber" in line_lower:
        if (
            "level" in line_lower
            or "color" in line_lower
            or "colour" in line_lower
            or "alert" in line_lower
        ):
            return 2

    if "red" in line_lower:
        if (
            "level" in line_lower
            or "color" in line_lower
            or "colour" in line_lower
            or "evacuation" in line_lower
        ):
            return 3

    return None


def extract_message_id(text):

    match = re.search(
        r"Message\s*ID\s*:?\s*"
        r"(LEWM-[A-Za-z0-9\-]+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return np.nan


def extract_update_number(text):

    match = re.search(
        r"Update\s*:?\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match:
        return int(
            match.group(1)
        )

    return np.nan


def extract_validity_period(text):

    pattern = re.compile(
        r"From\s+"
        r"(\d{1,2}:\d{2})\s*hrs\s*on\s*"
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})"
        r"\s+To\s+"
        r"(\d{1,2}:\d{2})\s*hrs\s*on\s*"
        r"(\d{1,2}[./-]\d{1,2}[./-]\d{4})",
        flags=re.IGNORECASE,
    )

    match = pattern.search(
        normalize_text(text)
    )

    if not match:
        return pd.NaT, pd.NaT

    (
        start_time,
        start_date,
        end_time,
        end_date,
    ) = match.groups()

    start_dt = pd.to_datetime(
        f"{start_date} {start_time}",
        dayfirst=True,
        errors="coerce",
    )

    end_dt = pd.to_datetime(
        f"{end_date} {end_time}",
        dayfirst=True,
        errors="coerce",
    )

    return start_dt, end_dt


def extract_modern_landslide_areas(
    pdf_path,
    metadata,
):

    rows = []

    with pdfplumber.open(
        pdf_path
    ) as pdf:

        for page_no, page in enumerate(
            pdf.pages,
            start=1,
        ):

            tables = page.extract_tables()

            for table in tables:

                for row in table:

                    if (
                        not row
                        or len(row) < 4
                    ):
                        continue

                    district = normalize_district(
                        row[0]
                    )

                    if district is None:
                        continue

                    for column_index, level in [
                        (1, 1),
                        (2, 2),
                        (3, 3),
                    ]:

                        cell_text = normalize_text(
                            row[column_index]
                            if row[column_index]
                            else ""
                        )

                        if not cell_text:
                            continue

                        rows.append(
                            {
                                "file_name":
                                    pdf_path.name,

                                "report_date":
                                    metadata[
                                        "report_date"
                                    ],

                                "issue_time":
                                    metadata[
                                        "issue_time"
                                    ],

                                "report_type":
                                    metadata[
                                        "report_type"
                                    ],

                                "district":
                                    district,

                                "warning_level":
                                    level,

                                "warning_color":
                                    LEVEL_INFO[
                                        level
                                    ]["color"],

                                "warning_action":
                                    LEVEL_INFO[
                                        level
                                    ]["action"],

                                "area_text_raw":
                                    cell_text,

                                "page_number":
                                    page_no,

                                "extraction_method":
                                    "modern_table",
                            }
                        )

    return rows


def extract_dsd_names(
    area_text
):

    if (
        area_text is None
        or pd.isna(area_text)
    ):
        return []

    text = str(
        area_text
    )

    text = (
        text
        .replace("↑", "")
        .replace("↓", "")
        .replace("*", "")
    )

    text = re.split(
        r"Divisional Secretariat",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    text = re.sub(
        r"\s+\band\b\s+",
        ",",
        text,
        flags=re.IGNORECASE,
    )

    parts = text.split(",")

    cleaned = []

    for part in parts:

        part = normalize_text(
            part
        )

        part = re.sub(
            r"[^A-Za-z\s\-]",
            "",
            part,
        ).strip()

        if len(part) >= 3:
            cleaned.append(
                part
            )

    return cleaned


def build_live_dsd_to_district():

    """
    Reuse the validated historical district-warning
    observations instead of rebuilding the mapping by
    reparsing every historical PDF on each live update.
    """

    mapping = {}

    if LANDSLIDE_AREAS.exists():

        old_areas = pd.read_csv(
            LANDSLIDE_AREAS
        )

        required = {
            "district",
            "area_text_raw",
        }

        if required.issubset(
            old_areas.columns
        ):

            usable = (
                old_areas
                .dropna(
                    subset=[
                        "district",
                        "area_text_raw",
                    ]
                )
            )

            for _, row in usable.iterrows():

                district = normalize_district(
                    row["district"]
                )

                if district is None:
                    continue

                for dsd in extract_dsd_names(
                    row["area_text_raw"]
                ):

                    key = normalize_text(
                        dsd
                    ).lower()

                    mapping[key] = district

    mapping.update(
        {
            "kothmale": "Nuwara Eliya",
            "kotmale": "Nuwara Eliya",
            "ambagamuwa": "Nuwara Eliya",

            "kalawana": "Ratnapura",
            "nivithigala": "Ratnapura",
            "kuruwita": "Ratnapura",
            "pelmadulla": "Ratnapura",
            "eheliyagoda": "Ratnapura",

            "deraniyagala": "Kegalle",
            "dehiowita": "Kegalle",
            "yatiyanthota": "Kegalle",
            "aranayaka": "Kegalle",

            "agalawatta": "Kalutara",
            "bulathsinhala": "Kalutara",
            "walallawita": "Kalutara",
            "ingiriya": "Kalutara",

            "thawalama": "Galle",
            "nagoda": "Galle",

            "pasbage korale": "Kandy",
        }
    )

    return mapping


def extract_legacy_landslide_areas(
    pdf_path,
    metadata,
    dsd_to_district,
):

    rows = []

    current_level = None

    location_section_started = False

    with pdfplumber.open(
        pdf_path
    ) as pdf:

        for page_no, page in enumerate(
            pdf.pages,
            start=1,
        ):

            text = page.extract_text(
                layout=True,
                x_tolerance=2,
                y_tolerance=3,
            )

            if not text:
                continue

            for line in text.splitlines():

                clean_line = normalize_text(
                    line
                )

                if not clean_line:
                    continue

                line_lower = (
                    clean_line.lower()
                )

                detected_level = (
                    detect_warning_level_from_line(
                        clean_line
                    )
                )

                if detected_level is not None:
                    current_level = (
                        detected_level
                    )

                if (
                    (
                        "location"
                        in line_lower
                        and "risk"
                        in line_lower
                    )
                    or (
                        "locrt"
                        in line_lower
                        and (
                            "risk"
                            in line_lower
                            or "ristr"
                            in line_lower
                        )
                    )
                    or (
                        "divisional secretariat"
                        in line_lower
                    )
                ):
                    location_section_started = True

                if not location_section_started:
                    continue

                if current_level is None:
                    continue

                matched_districts = set()

                for variant, canonical in (
                    DISTRICT_VARIANTS.items()
                ):

                    if re.search(
                        rf"\b{re.escape(variant)}\b",
                        line_lower,
                    ):

                        if (
                            canonical == "Colombo"
                            and (
                                "jawatta"
                                in line_lower
                                or "colombo 05"
                                in line_lower
                            )
                        ):
                            continue

                        matched_districts.add(
                            canonical
                        )

                english_line = re.sub(
                    r"[^a-z\s\-]",
                    " ",
                    line_lower,
                )

                english_line = normalize_text(
                    english_line
                )

                for dsd_name, district in (
                    dsd_to_district.items()
                ):

                    if len(
                        dsd_name
                    ) < 5:
                        continue

                    if re.search(
                        rf"\b{re.escape(dsd_name)}\b",
                        english_line,
                    ):
                        matched_districts.add(
                            district
                        )

                for district in matched_districts:

                    rows.append(
                        {
                            "file_name":
                                pdf_path.name,

                            "report_date":
                                metadata[
                                    "report_date"
                                ],

                            "issue_time":
                                metadata[
                                    "issue_time"
                                ],

                            "report_type":
                                metadata[
                                    "report_type"
                                ],

                            "district":
                                district,

                            "warning_level":
                                current_level,

                            "warning_color":
                                LEVEL_INFO[
                                    current_level
                                ]["color"],

                            "warning_action":
                                LEVEL_INFO[
                                    current_level
                                ]["action"],

                            "area_text_raw":
                                clean_line,

                            "page_number":
                                page_no,

                            "extraction_method":
                                "legacy_layout",
                        }
                    )

    return rows


def extract_landslide_report(
    pdf_path,
    dsd_to_district,
):

    metadata = parse_landslide_filename(
        pdf_path.name
    )

    text = extract_pdf_text(
        pdf_path
    )

    message_id = extract_message_id(
        text
    )

    update_number = extract_update_number(
        text
    )

    validity_start, validity_end = (
        extract_validity_period(
            text
        )
    )

    area_rows = (
        extract_modern_landslide_areas(
            pdf_path,
            metadata,
        )
    )

    extraction_method = (
        "modern_table"
    )

    if len(
        area_rows
    ) == 0:

        area_rows = (
            extract_legacy_landslide_areas(
                pdf_path,
                metadata,
                dsd_to_district,
            )
        )

        extraction_method = (
            "legacy_layout"
        )

    if area_rows:

        temp_df = pd.DataFrame(
            area_rows
        )

        temp_df = (
            temp_df
            .drop_duplicates(
                subset=[
                    "file_name",
                    "district",
                    "warning_level",
                ]
            )
        )

        area_rows = (
            temp_df
            .to_dict(
                "records"
            )
        )

    levels = [
        row["warning_level"]
        for row in area_rows
    ]

    districts = {
        row["district"]
        for row in area_rows
    }

    if levels:

        max_level = max(
            levels
        )

        max_color = (
            LEVEL_INFO[
                max_level
            ]["color"]
        )

    else:

        max_level = np.nan
        max_color = np.nan

    level_1_count = sum(
        level == 1
        for level in levels
    )

    level_2_count = sum(
        level == 2
        for level in levels
    )

    level_3_count = sum(
        level == 3
        for level in levels
    )

    if len(
        text.strip()
    ) < 100:

        extraction_status = (
            "insufficient_text"
        )

    elif len(
        area_rows
    ) == 0:

        extraction_status = (
            "no_warning_areas"
        )

    else:

        extraction_status = "ok"

    report_record = {
        "file_name":
            pdf_path.name,

        "report_date":
            metadata["report_date"],

        "issue_time":
            metadata["issue_time"],

        "report_type":
            metadata["report_type"],

        "message_id":
            message_id,

        "update_number":
            update_number,

        "validity_start":
            validity_start,

        "validity_end":
            validity_end,

        "districts_extracted":
            len(districts),

        "area_rows_extracted":
            len(area_rows),

        "level_1_districts":
            level_1_count,

        "level_2_districts":
            level_2_count,

        "level_3_districts":
            level_3_count,

        "max_warning_level":
            max_level,

        "max_warning_color":
            max_color,

        "extraction_method":
            extraction_method,

        "text_length":
            len(text),

        "extraction_status":
            extraction_status,
    }

    return (
        report_record,
        area_rows,
    )


def classify_unresolved_landslide_report(
    row
):

    if (
        row["extraction_status"]
        == "ok"
    ):
        return "ok"

    if (
        row["extraction_status"]
        == "insufficient_text"
    ):
        return "insufficient_text"

    pdf_path = (
        LANDSLIDE_DIR
        / row["file_name"]
    )

    text = extract_pdf_text(
        pdf_path
    )

    text_lower = normalize_text(
        text
    ).lower()

    river_markers = (
        (
            "islandwide water level & rainfall "
            "situation in major rivers"
            in text_lower
        )
        or (
            "gauging station"
            in text_lower
            and "river basin"
            in text_lower
            and "water level"
            in text_lower
        )
    )

    if river_markers:
        return "non_landslide_report"

    return "manual_review_legacy_layout"

def create_report_datetime(df, date_col='report_date', time_col=None, filename_time=False):
    result = df.copy()
    
    result[date_col] = pd.to_datetime(
    result[date_col],
    errors="coerce",
    format="mixed",
    )
    if filename_time:
        extracted_time = result['file_name'].str.extract('_(\\d{8})_(\\d{4})_')[1]
        result['issue_time_standard'] = extracted_time.fillna('0000')
    elif time_col is not None:
        result['issue_time_standard'] = result[time_col].apply(hhmm_to_time_string)
    else:
        result['issue_time_standard'] = '0000'
    result['report_datetime'] = pd.to_datetime(result[date_col].dt.strftime('%Y-%m-%d') + ' ' + result['issue_time_standard'].str[:2] + ':' + result['issue_time_standard'].str[2:], errors='coerce')
    return result

# Incremental preprocessing


def latest_processed_timestamp(df, prefix):
    times = [
        filename_timestamp(x, prefix)
        for x in df["file_name"].dropna().astype(str)
    ]
    times = [x for x in times if pd.notna(x)]
    return max(times) if times else pd.Timestamp.min


def new_raw_files(folder, old_df, prefix):
    latest = latest_processed_timestamp(old_df, prefix)
    known = set(old_df["file_name"].dropna().astype(str))
    candidates = []
    for p in sorted(Path(folder).glob("*.pdf")):
        ts = filename_timestamp(p.name, prefix)
        if p.name not in known and pd.notna(ts) and ts > latest:
            candidates.append(p)
    return candidates


def append_new_situation(old_df):
    paths = new_raw_files(SITUATION_DIR, old_df, "situation")
    print("New Situation PDFs to preprocess:", len(paths))
    rows = []
    for p in paths:
        print("  Situation:", p.name)
        rows.append(extract_situation_report(p))
    new_df = pd.DataFrame(rows)
    combined = align_concat(old_df, new_df)
    if not combined.empty:
        combined = combined.drop_duplicates("file_name", keep="last").reset_index(drop=True)
    return combined, new_df


def append_new_weather(old_df):
    paths = new_raw_files(WEATHER_DIR, old_df, "weather")
    print("New Weather PDFs to preprocess:", len(paths))
    rows = []
    for p in paths:
        print("  Weather:", p.name)
        rows.append(extract_weather_report(p))
    new_df = pd.DataFrame(rows)
    combined = align_concat(old_df, new_df)
    if not combined.empty:
        combined = combined.drop_duplicates("file_name", keep="last").reset_index(drop=True)
    return combined, new_df


def append_new_river(old_reports, old_stations):
    paths = new_raw_files(RIVER_DIR, old_reports, "river")
    print("New River PDFs to preprocess:", len(paths))
    report_rows = []
    station_rows = []
    for p in paths:
        print("  River:", p.name)
        report, stations = extract_river_report(p)
        report_rows.append(report)
        station_rows.extend(stations)

    new_reports = pd.DataFrame(report_rows)
    new_stations = pd.DataFrame(station_rows)

    reports = align_concat(old_reports, new_reports)
    if not reports.empty:
        reports = reports.drop_duplicates("file_name", keep="last").reset_index(drop=True)

    stations = align_concat(old_stations, new_stations)
    if not stations.empty:
        dedupe = [
            c for c in
            ["file_name", "station_label_raw", "water_level_current"]
            if c in stations.columns
        ]
        if dedupe:
            stations = stations.drop_duplicates(dedupe, keep="last").reset_index(drop=True)

    return reports, stations, new_reports, new_stations


def append_new_landslide(
    old_reports,
    old_areas,
):

    paths = new_raw_files(
        LANDSLIDE_DIR,
        old_reports,
        "landslide",
    )

    print(
        "New Landslide PDFs to preprocess:",
        len(
            paths
        ),
    )

    report_rows = []
    area_rows = []

    # Nothing new to process: skip building the expensive DSD -> District mapping.

    if not paths:
        print("New Landslide report rows: 0")
        print("New Landslide district-warning rows: 0")

        return (
            old_reports,
            old_areas,
            pd.DataFrame(),
            pd.DataFrame(),
        )

    # Build mapping only when at least one new landslide PDF exists.

    dsd_to_district = get_landslide_dsd_to_district()


    for p in paths:

        print(
            "  Landslide:",
            p.name,
        )

        (
            report,
            areas,
        ) = extract_landslide_report(
            p,
            dsd_to_district,
        )

        report_rows.append(
            report
        )

        area_rows.extend(
            areas
        )

    new_reports = pd.DataFrame(
        report_rows
    )

    new_areas = pd.DataFrame(
        area_rows
    )

    # Apply exact frozen notebook review classification only to the newly parsed reports.
    
    if not new_reports.empty:

        new_reports[
            "extraction_status"
        ] = (
            new_reports.apply(
                classify_unresolved_landslide_report,
                axis=1,
            )
        )

    # Append report-level records
    
    reports = align_concat(
        old_reports,
        new_reports,
    )

    if not reports.empty:

        reports = (
            reports
            .drop_duplicates(
                subset=[
                    "file_name"
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

    # Append district-warning observations

    areas = align_concat(
        old_areas,
        new_areas,
    )

    if not areas.empty:

        dedupe_cols = [

            c
            for c in [
                "file_name",
                "district",
                "warning_level",
            ]

            if c in areas.columns
        ]

        if dedupe_cols:

            areas = (
                areas
                .drop_duplicates(
                    
                    dedupe_cols,
                    keep="last",
                )
                .reset_index(
                    drop=True
                )
            )

    print(
        "New Landslide report rows:",
        len(
            new_reports
        ),
    )

    print(
        "New Landslide district-warning rows:",
        len(
            new_areas
        ),
    )

    if (
        not new_reports.empty
        and "extraction_status"
        in new_reports.columns
    ):

        print(
            "\nNew Landslide extraction status:"
        )

        print(
            new_reports[
                "extraction_status"
            ].value_counts(
                dropna=False
            )
        )

    return (
        reports,
        areas,
        new_reports,
        new_areas,
    )

# Situation metadata title mapping

def situation_metadata_title_map():
    if not SITUATION_METADATA.exists():
        raise FileNotFoundError(f"Missing {SITUATION_METADATA}")

    import hashlib

    meta = pd.read_csv(SITUATION_METADATA)
    required = {"title", "date", "time", "pdf_url"}
    if not required.issubset(meta.columns):
        raise ValueError("DMC metadata is missing title/date/time/pdf_url.")

    out = {}
    for _, row in meta.iterrows():
        dt = pd.to_datetime(
            f'{row["date"]} {row["time"]}',
            errors="coerce",
        )
        if pd.isna(dt):
            continue
        url = str(row["pdf_url"])
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        name = f'situation_{dt.strftime("%Y%m%d_%H%M")}_{digest}.pdf'
        out[name] = str(row["title"]).strip()
    return out

# Standardize timestamps using exact notebook helper

def standardize_all(situation, weather, river, stations, landslide, areas):
    situation = create_report_datetime(situation, filename_time=True)
    weather = create_report_datetime(weather, time_col="issue_time")
    river = create_report_datetime(river, time_col="issue_time")
    stations = create_report_datetime(stations, time_col="issue_time")
    landslide = create_report_datetime(landslide, time_col="issue_time")
    areas = create_report_datetime(areas, time_col="issue_time")

    situation["availability_datetime"] = situation["report_datetime"]

    for df in [situation, weather, river, stations, landslide, areas]:
        df["report_datetime"] = pd.to_datetime(df["report_datetime"], errors="coerce")

    for c in ["validity_start", "validity_end"]:
        if c in landslide.columns:
            landslide[c] = pd.to_datetime(landslide[c], errors="coerce")

    return situation, weather, river, stations, landslide, areas

# Preserve frozen historical analytical scope exactly; classify only genuinely new Situation PDFs.


def build_live_scope(situation_int, new_situation_df):
    old_scope = pd.read_csv(ANALYTICAL_SCOPE)
    old_files = set(
        old_scope["file_name"]
        .dropna()
        .astype(str)
    )

    # Previously classified files, including reports that were
    # intentionally excluded from the analytical scope.
    old_audit = (
        pd.read_csv(SCOPE_AUDIT)
        if SCOPE_AUDIT.exists()
        else pd.DataFrame()
    )

    audited_files = set()

    if not old_audit.empty and "file_name" in old_audit.columns:
        audited_files = set(
            old_audit["file_name"]
            .dropna()
            .astype(str)
        )

    title_map = situation_metadata_title_map()

    # Files genuinely parsed during this run

    current_new_files = set(
        new_situation_df.get(
            "file_name",
            pd.Series(dtype=str)
        )
        .dropna()
        .astype(str)
    )

    # A previous interrupted/older run may already have written a Situation PDF to situation_reports_cleaned.csv without adding/classifying it in analytical scope.
   
    if "availability_datetime" in old_scope.columns:
        latest_scope_time = pd.to_datetime(
            old_scope["availability_datetime"],
            errors="coerce"
        ).max()

    elif "report_datetime" in old_scope.columns:
        latest_scope_time = pd.to_datetime(
            old_scope["report_datetime"],
            errors="coerce"
        ).max()

    else:
        latest_scope_time = pd.Timestamp.min

    processed_times = pd.to_datetime(
        situation_int["report_datetime"],
        errors="coerce"
    )

    recovery_mask = (
        ~situation_int["file_name"]
        .astype(str)
        .isin(old_files)
        &
        ~situation_int["file_name"]
        .astype(str)
        .isin(audited_files)
    )

    if pd.notna(latest_scope_time):
        recovery_mask &= (
            processed_times >= latest_scope_time
        )

    recovery_files = set(
        situation_int.loc[
            recovery_mask,
            "file_name"
        ]
        .dropna()
        .astype(str)
    )

    print(
        "Processed Situation PDFs awaiting scope classification:",
        len(recovery_files)
    )

    # Includes:
    # 1. genuinely new reports from this run
    # 2. previously processed but never classified reports
    new_file_names = current_new_files | recovery_files

    # IMPORTANT: always initialize these before the loop.
    retain_new = []
    audit_new = []

    # Classify candidate Situation PDFs
   
    for name in sorted(new_file_names):

        row_match = situation_int[
            situation_int["file_name"].astype(str) == name
        ]

        if row_match.empty:
            continue

        row = row_match.iloc[0]

        title = title_map.get(name, "")
        title_key = title.strip().casefold()

        extraction = str(
            row.get("extraction_status", "")
        )

        if title_key == "drought situation report":
            category = "drought_report"
            status = "exclude_drought"

        elif extraction == "non_situation_report":
            category = "non_situation_report"
            status = "exclude_non_situation"

        elif extraction == "non_comparable_incident_report":
            category = "non_comparable_incident_report"
            status = "exclude_non_comparable"

        elif title_key == "situation report":
            category = "general_situation_report"

            if pd.isna(
                row.get("affected_people", np.nan)
            ):
                raise RuntimeError(
                    f"New general Situation Report {name} "
                    "did not yield a comparable national "
                    "affected_people total."
                )

            status = "retain"
            retain_new.append(name)

        else:
            category = "other_heading"
            status = "exclude_unresolved_live"

        audit_new.append(
            {
                "file_name": name,
                "report_datetime": row["report_datetime"],
                "situation_report_category": category,
                "extraction_status": extraction,
                "scope_status": status,
            }
        )

    # Build updated analytical scope

    retain_files = old_files | set(retain_new)

    scope = situation_int[
        situation_int["file_name"]
        .astype(str)
        .isin(retain_files)
    ].copy()

    scope = (
        scope
        .sort_values("report_datetime")
        .reset_index(drop=True)
    )

    scope["availability_datetime"] = (
        scope["report_datetime"]
    )

    # Update classification audit
    
    audit_new_df = pd.DataFrame(audit_new)

    audit = align_concat(
        old_audit,
        audit_new_df
    )

    if (
        not audit.empty
        and "file_name" in audit.columns
    ):
        audit = (
            audit
            .drop_duplicates(
                "file_name",
                keep="last"
            )
            .reset_index(drop=True)
        )

    return scope, audit

# Timestamp-safe multi-source integration
# Exact feature names/rules from notebook cells 70-72.


def clean_bool_value(value):
    if pd.isna(value):
        return np.nan
    if value in [True, 1, "True", "true", "1"]:
        return True
    if value in [False, 0, "False", "false", "0"]:
        return False
    return np.nan


def integrate_sources(scope, weather_int, river_int, landslide_int):
    base = scope.sort_values("report_datetime").reset_index(drop=True).copy()

    # ---------------- Weather ----------------
    weather = weather_int.copy()
    weather_bool_cols = [
        "heavy_rain_flag", "thunderstorm_flag", "strong_wind_flag",
        "low_pressure_flag", "depression_flag", "cyclone_flag",
    ]
    for col in weather_bool_cols:
        weather[col] = weather[col].fillna(False).astype(bool)

    weather_color_map = {"Green": 1, "Amber": 2, "Red": 3}
    weather["weather_warning_level"] = (
        weather["warning_color"].map(weather_color_map).fillna(0)
    )
    weather = weather.sort_values("report_datetime").reset_index(drop=True)

    weather_rows = []
    for situation_time in base["report_datetime"]:
        right = weather["report_datetime"].searchsorted(situation_time, side="right")
        start = situation_time - pd.Timedelta(hours=24)
        left = weather["report_datetime"].searchsorted(start, side="left")
        recent = weather.iloc[left:right]
        previous = weather.iloc[:right]
        latest = previous.iloc[-1] if len(previous) else None

        if recent.empty:
            f = {
                "weather_reports_24h": 0,
                "weather_max_rainfall_threshold_24h": np.nan,
                "weather_max_wind_24h": np.nan,
                "weather_max_warning_level_24h": 0,
                "weather_heavy_rain_24h": 0,
                "weather_thunderstorm_24h": 0,
                "weather_strong_wind_24h": 0,
                "weather_low_pressure_24h": 0,
                "weather_depression_24h": 0,
                "weather_cyclone_24h": 0,
            }
        else:
            f = {
                "weather_reports_24h": len(recent),
                "weather_max_rainfall_threshold_24h": recent["rainfall_threshold_mm"].max(),
                "weather_max_wind_24h": recent["max_wind_kmph"].max(),
                "weather_max_warning_level_24h": recent["weather_warning_level"].max(),
                "weather_heavy_rain_24h": int(recent["heavy_rain_flag"].any()),
                "weather_thunderstorm_24h": int(recent["thunderstorm_flag"].any()),
                "weather_strong_wind_24h": int(recent["strong_wind_flag"].any()),
                "weather_low_pressure_24h": int(recent["low_pressure_flag"].any()),
                "weather_depression_24h": int(recent["depression_flag"].any()),
                "weather_cyclone_24h": int(recent["cyclone_flag"].any()),
            }

        if latest is not None:
            f["latest_weather_age_hours"] = (
                situation_time - latest["report_datetime"]
            ).total_seconds() / 3600
            f["latest_weather_report_type"] = latest["report_type"]
        else:
            f["latest_weather_age_hours"] = np.nan
            f["latest_weather_report_type"] = np.nan

        weather_rows.append(f)

    base = pd.concat([base, pd.DataFrame(weather_rows)], axis=1)

    # ---------------- River ----------------
    river = river_int.copy()
    river["active_warning_bool"] = river["active_warning_flag"].apply(clean_bool_value)
    river["warning_withdrawn_bool"] = river["warning_withdrawn_flag"].apply(clean_bool_value)
    river = river.sort_values("report_datetime").reset_index(drop=True)
    river_start = river["report_datetime"].min()

    river_rows = []
    for situation_time in base["report_datetime"]:
        if pd.isna(river_start) or situation_time < river_start:
            river_rows.append({
                "river_data_available": 0,
                "river_reports_24h": 0,
                "river_max_alert_stations_24h": np.nan,
                "river_max_minor_flood_stations_24h": np.nan,
                "river_max_major_flood_stations_24h": np.nan,
                "river_max_rising_stations_24h": np.nan,
                "river_max_rainfall_24h_mm": np.nan,
                "river_max_alert_ratio_24h": np.nan,
                "river_active_warning_24h": np.nan,
                "river_warning_withdrawn_24h": np.nan,
                "latest_river_age_hours": np.nan,
                "latest_river_report_type": np.nan,
            })
            continue

        right = river["report_datetime"].searchsorted(situation_time, side="right")
        start = situation_time - pd.Timedelta(hours=24)
        left = river["report_datetime"].searchsorted(start, side="left")
        recent = river.iloc[left:right]
        previous = river.iloc[:right]

        if len(previous):
            latest = previous.iloc[-1]
            latest_age = (situation_time - latest["report_datetime"]).total_seconds() / 3600
            latest_type = latest["report_type"]
        else:
            latest_age = np.nan
            latest_type = np.nan

        if recent.empty:
            f = {
                "river_data_available": 1,
                "river_reports_24h": 0,
                "river_max_alert_stations_24h": np.nan,
                "river_max_minor_flood_stations_24h": np.nan,
                "river_max_major_flood_stations_24h": np.nan,
                "river_max_rising_stations_24h": np.nan,
                "river_max_rainfall_24h_mm": np.nan,
                "river_max_alert_ratio_24h": np.nan,
                "river_active_warning_24h": np.nan,
                "river_warning_withdrawn_24h": np.nan,
                "latest_river_age_hours": latest_age,
                "latest_river_report_type": latest_type,
            }
        else:
            f = {
                "river_data_available": 1,
                "river_reports_24h": len(recent),
                # MAX, not SUM, because stations repeat across updates.
                "river_max_alert_stations_24h": recent["alert_station_count"].max(),
                "river_max_minor_flood_stations_24h": recent["minor_flood_station_count"].max(),
                "river_max_major_flood_stations_24h": recent["major_flood_station_count"].max(),
                "river_max_rising_stations_24h": recent["rising_station_count"].max(),
                "river_max_rainfall_24h_mm": recent["max_rainfall_24h_mm"].max(),
                "river_max_alert_ratio_24h": recent["max_alert_ratio"].max(),
                "river_active_warning_24h": int(
                    recent["active_warning_bool"].fillna(False).any()
                ),
                "river_warning_withdrawn_24h": int(
                    recent["warning_withdrawn_bool"].fillna(False).any()
                ),
                "latest_river_age_hours": latest_age,
                "latest_river_report_type": latest_type,
            }
        river_rows.append(f)

    base = pd.concat([base, pd.DataFrame(river_rows)], axis=1)

    # ---------------- Landslide ----------------
    landslide = landslide_int.copy()
    landslide["validity_start"] = pd.to_datetime(landslide["validity_start"], errors="coerce")
    landslide["validity_end"] = pd.to_datetime(landslide["validity_end"], errors="coerce")
    landslide = landslide[landslide["extraction_status"] == "ok"].copy()
    landslide = landslide.sort_values("report_datetime").reset_index(drop=True)

    ls_rows = []
    for situation_time in base["report_datetime"]:
        right = landslide["report_datetime"].searchsorted(situation_time, side="right")
        previous = landslide.iloc[:right]

        if previous.empty:
            ls_rows.append({
                "landslide_data_available": 0,
                "landslide_active_warning": 0,
                "landslide_max_warning_level": 0,
                "landslide_level1_districts": 0,
                "landslide_level2_districts": 0,
                "landslide_level3_districts": 0,
                "landslide_districts_at_risk": 0,
                "latest_landslide_age_hours": np.nan,
                "latest_landslide_report_type": np.nan,
            })
            continue

        latest = previous.iloc[-1]
        age = (situation_time - latest["report_datetime"]).total_seconds() / 3600
        vs, ve = latest["validity_start"], latest["validity_end"]

        if pd.notna(ve):
            start_ok = pd.isna(vs) or vs <= situation_time
            end_ok = ve >= situation_time
            active = bool(start_ok and end_ok)
        else:
            active = bool(age <= 24)

        if active:
            max_level = latest["max_warning_level"] if pd.notna(latest["max_warning_level"]) else 0
            level1 = int(latest["level_1_districts"])
            level2 = int(latest["level_2_districts"])
            level3 = int(latest["level_3_districts"])
            at_risk = int(latest["districts_extracted"])
        else:
            max_level = level1 = level2 = level3 = at_risk = 0

        ls_rows.append({
            "landslide_data_available": 1,
            "landslide_active_warning": int(active),
            "landslide_max_warning_level": max_level,
            "landslide_level1_districts": level1,
            "landslide_level2_districts": level2,
            "landslide_level3_districts": level3,
            "landslide_districts_at_risk": at_risk,
            "latest_landslide_age_hours": age,
            "latest_landslide_report_type": latest["report_type"],
        })

    base = pd.concat([base, pd.DataFrame(ls_rows)], axis=1)
    base["availability_datetime"] = base["report_datetime"]

    # Basic leakage safety.
    if (base["latest_weather_age_hours"].dropna() < 0).any():
        raise RuntimeError("Future weather leakage detected.")
    if (base["latest_river_age_hours"].dropna() < 0).any():
        raise RuntimeError("Future river leakage detected.")
    if (base["latest_landslide_age_hours"].dropna() < 0).any():
        raise RuntimeError("Future landslide leakage detected.")

    return base

# Canonical timeline + exact temporal/autoregressive features

def build_feature_timeline(integrated):
    timeline = integrated.copy()
    timeline["original_row_id"] = np.arange(len(timeline))
    timeline["target_available"] = timeline["affected_people"].notna().astype(int)

    timeline = timeline.sort_values(
        [
            "availability_datetime", "target_available",
            "affected_people", "affected_families",
            "people_in_safety_centres", "file_name",
        ],
        ascending=[True, False, False, False, False, True],
    )

    counts = timeline.groupby("availability_datetime").size().rename("reports_at_timestamp")

    forecast = (
        timeline
        .drop_duplicates("availability_datetime", keep="first")
        .merge(counts, on="availability_datetime", how="left")
        .sort_values("availability_datetime")
        .reset_index(drop=True)
    )

    forecast["hours_since_previous_report"] = (
        forecast["availability_datetime"].diff().dt.total_seconds() / 3600
    )

    forecast["next_report_datetime"] = forecast["availability_datetime"].shift(-1)
    forecast["hours_to_next_report"] = (
        (forecast["next_report_datetime"] - forecast["availability_datetime"])
        .dt.total_seconds() / 3600
    )

    forecast["target_next_affected_people"] = forecast["affected_people"].shift(-1)
    forecast["target_next_affected_families"] = forecast["affected_families"].shift(-1)
    forecast["target_next_safety_centres"] = forecast["safety_centres"].shift(-1)
    forecast["target_next_people_in_safety_centres"] = (
        forecast["people_in_safety_centres"].shift(-1)
    )

    forecast["valid_next_period_pair"] = (
        forecast["hours_to_next_report"].between(
            0, MAX_FORECAST_GAP_HOURS, inclusive="right"
        )
        & forecast["affected_people"].notna()
        & forecast["target_next_affected_people"].notna()
    )

    feature = forecast.copy().sort_values("availability_datetime").reset_index(drop=True)

    # Exact notebook transforms.
    feature["log_current_affected_people"] = np.log1p(feature["affected_people"])
    feature["log_current_affected_families"] = np.log1p(feature["affected_families"])
    feature["log_current_safety_centres"] = np.log1p(feature["safety_centres"])
    feature["log_current_people_in_safety_centres"] = np.log1p(
        feature["people_in_safety_centres"]
    )

    feature["previous_report_gap_hours"] = (
        feature["availability_datetime"].diff().dt.total_seconds() / 3600
    )
    feature["previous_context_valid"] = feature["previous_report_gap_hours"].between(
        0, 48, inclusive="right"
    )

    previous_cols = [
        "affected_people",
        "affected_families",
        "safety_centres",
        "people_in_safety_centres",
    ]
    for col in previous_cols:
        prev_col = f"prev_{col}"
        feature[prev_col] = feature[col].shift(1)
        feature.loc[~feature["previous_context_valid"], prev_col] = np.nan

    feature["affected_people_change_from_prev"] = (
        feature["affected_people"] - feature["prev_affected_people"]
    )
    feature["affected_families_change_from_prev"] = (
        feature["affected_families"] - feature["prev_affected_families"]
    )
    feature["safety_population_change_from_prev"] = (
        feature["people_in_safety_centres"] - feature["prev_people_in_safety_centres"]
    )

    # IMPORTANT: exact notebook log-difference definition, not signed-log absolute change.
    feature["affected_people_log_change_from_prev"] = (
        np.log1p(feature["affected_people"]) - np.log1p(feature["prev_affected_people"])
    )
    feature["safety_population_log_change_from_prev"] = (
        np.log1p(feature["people_in_safety_centres"])
        - np.log1p(feature["prev_people_in_safety_centres"])
    )

    feature["report_hour"] = feature["availability_datetime"].dt.hour
    feature["report_month"] = feature["availability_datetime"].dt.month
    feature["report_dayofyear"] = feature["availability_datetime"].dt.dayofyear
    feature["report_dayofweek"] = feature["availability_datetime"].dt.dayofweek

    feature["month_sin"] = np.sin(2 * np.pi * feature["report_month"] / 12)
    feature["month_cos"] = np.cos(2 * np.pi * feature["report_month"] / 12)
    feature["dayofyear_sin"] = np.sin(
        2 * np.pi * feature["report_dayofyear"] / 365.25
    )
    feature["dayofyear_cos"] = np.cos(
        2 * np.pi * feature["report_dayofyear"] / 365.25
    )

    # Hazard-label model features were created later in the notebook.
    disaster_text = feature["disaster_types"].fillna("").astype(str)
    feature["current_flood_flag"] = disaster_text.str.contains(
        "Flood", case=False, regex=False
    ).astype(int)
    feature["current_landslide_flag"] = disaster_text.str.contains(
        "Landslide", case=False, regex=False
    ).astype(int)
    feature["current_heavy_rain_flag"] = disaster_text.str.contains(
        "Heavy Rain", case=False, regex=False
    ).astype(int)

    # Historical risk labels for similar-event display only.
    change = feature["target_next_affected_people"] - feature["affected_people"]

    def assign_risk(x):
        if pd.isna(x):
            return np.nan
        if x <= 0:
            return "Low"
        if x <= MODERATE_HIGH_THRESHOLD:
            return "Moderate"
        if x <= HIGH_CRITICAL_THRESHOLD:
            return "High"
        return "Critical"

    feature["impact_escalation_risk"] = change.apply(assign_risk)

    return forecast, feature

# Frozen saved model inference - NO .fit() here.


def load_models():
    loaded = {k: joblib.load(p) for k, p in MODEL_PATHS.items()}
    features = list(loaded["features"])
    return loaded, features


def risk_classes_from_pipeline(model):
    try:
        return model.named_steps["model"].classes_
    except Exception:
        return model.classes_


def predict_latest(feature_df, models, core_features):
    latest_df = (
        feature_df.sort_values("availability_datetime").iloc[[-1]].copy()
    )

    missing = [c for c in core_features if c not in latest_df.columns]
    if missing:
        raise RuntimeError(
            "Latest feature row is missing frozen CORE features:\n"
            + "\n".join(missing)
        )

    X = latest_df[core_features].copy()
    for c in core_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    latest = latest_df.iloc[0]
    current = float(latest["affected_people"])

    human_log_delta = models["human"].predict(X)
    human_delta = float(inverse_signed_log1p(human_log_delta)[0])
    predicted = max(0.0, current + human_delta)

    lower_log = models["lower"].predict(X)
    upper_log = models["upper"].predict(X)
    lower_delta = float(inverse_signed_log1p(lower_log)[0])
    upper_delta = float(inverse_signed_log1p(upper_log)[0])

    lower = max(0.0, current + min(lower_delta, upper_delta))
    upper = max(0.0, current + max(lower_delta, upper_delta))

    safety_prob = float(models["safety_use"].predict_proba(X)[0, 1])
    safety_use = int(safety_prob >= 0.5)

    current_safety = float(latest["people_in_safety_centres"])
    safety_log_delta = models["safety_demand"].predict(X)
    safety_delta = float(inverse_signed_log1p(safety_log_delta)[0])
    safety_raw = max(0.0, current_safety + safety_delta)
    predicted_safety = safety_raw if safety_use == 1 else 0.0

    risk_pred = str(models["risk"].predict(X)[0])
    risk_probs = models["risk"].predict_proba(X)[0]
    risk_classes = risk_classes_from_pipeline(models["risk"])
    risk_dict = {
        str(label): float(prob)
        for label, prob in zip(risk_classes, risk_probs)
    }

    return {
        "latest_df": latest_df,
        "latest": latest,
        "X": X,
        "current_affected": current,
        "predicted_delta": human_delta,
        "predicted_affected": predicted,
        "lower_90": lower,
        "upper_90": upper,
        "current_safety": current_safety,
        "safety_probability": safety_prob,
        "safety_use": safety_use,
        "safety_delta": safety_delta,
        "predicted_safety": predicted_safety,
        "risk": risk_pred,
        "risk_probabilities": risk_dict,
    }

# Exact refined trend / rapid-growth anomaly logic


def build_refined_trend(feature_df):
    trend = feature_df.copy().sort_values("availability_datetime").reset_index(drop=True)
    trend["impact_absolute_change"] = pd.to_numeric(
        trend["affected_people_change_from_prev"], errors="coerce"
    )
    trend["impact_log_change"] = pd.to_numeric(
        trend["affected_people_log_change_from_prev"], errors="coerce"
    )

    def classify(row):
        if not bool(row["previous_context_valid"]):
            return "Insufficient Previous Context"
        change = row["impact_absolute_change"]
        if pd.isna(change):
            return "Insufficient Previous Context"
        if change > 0:
            return "Increasing"
        if change < 0:
            return "Decreasing"
        return "Stable"

    trend["impact_trend"] = trend.apply(classify, axis=1)

    refined = trend.copy()
    abs_pct = []
    rel_pct = []
    hist_count = []

    for i in range(len(refined)):
        current_row = refined.iloc[i]

        if (
            current_row["previous_context_valid"] != True
            or pd.isna(current_row["impact_absolute_change"])
            or pd.isna(current_row["impact_log_change"])
        ):
            abs_pct.append(np.nan)
            rel_pct.append(np.nan)
            hist_count.append(0)
            continue

        start = max(0, i - HISTORY_WINDOW)
        window = refined.iloc[start:i].copy()

        positive = window[
            (window["previous_context_valid"] == True)
            & (window["impact_absolute_change"] > 0)
            & window["impact_absolute_change"].notna()
            & window["impact_log_change"].notna()
        ]

        n = len(positive)
        hist_count.append(n)

        if n < MIN_POSITIVE_HISTORY:
            abs_pct.append(np.nan)
            rel_pct.append(np.nan)
            continue

        current_absolute = float(current_row["impact_absolute_change"])
        current_relative = float(current_row["impact_log_change"])

        abs_pct.append(
            float(np.mean(
                positive["impact_absolute_change"].to_numpy(dtype=float)
                <= current_absolute
            ))
        )
        rel_pct.append(
            float(np.mean(
                positive["impact_log_change"].to_numpy(dtype=float)
                <= current_relative
            ))
        )

    refined["absolute_growth_percentile"] = abs_pct
    refined["relative_growth_percentile"] = rel_pct
    refined["historical_positive_count"] = hist_count

    refined["rapid_growth_score"] = (
        np.minimum(
            refined["absolute_growth_percentile"],
            refined["relative_growth_percentile"],
        )
        * 100
    )

    refined["rapid_growth_anomaly"] = (
        (refined["previous_context_valid"] == True)
        & (refined["impact_absolute_change"] >= MIN_MATERIAL_INCREASE)
        & (refined["absolute_growth_percentile"] >= EXTREME_PERCENTILE)
        & (refined["relative_growth_percentile"] >= EXTREME_PERCENTILE)
    )

    def status(row):
        if row["previous_context_valid"] != True:
            return "Gap / No Comparable Previous Report"
        if pd.isna(row["rapid_growth_score"]):
            return "Insufficient Historical Baseline"
        if row["rapid_growth_anomaly"]:
            return "Rapid Growth Anomaly"
        if row["impact_absolute_change"] > 0:
            return "Normal Increase"
        if row["impact_absolute_change"] < 0:
            return "Decrease"
        return "Stable"

    refined["impact_monitoring_status"] = refined.apply(status, axis=1)
    return refined

# Refined Similar-Event Finder

SIMILARITY_COUNT_FEATURES = [
    "affected_people",
    "affected_families",
    "safety_centres",
    "people_in_safety_centres",
    "weather_reports_24h",
    "weather_max_rainfall_threshold_24h",
    "weather_max_wind_24h",
    "landslide_districts_at_risk",
    "landslide_level1_districts",
    "landslide_level2_districts",
    "landslide_level3_districts",
]

SIMILARITY_CHANGE_FEATURES = [
    "affected_people_change_from_prev",
    "affected_families_change_from_prev",
    "safety_population_change_from_prev",
]

SIMILARITY_STATE_FEATURES = [
    "weather_max_warning_level_24h",
    "weather_heavy_rain_24h",
    "current_heavy_rain_flag",
    "landslide_active_warning",
    "landslide_max_warning_level",
]


def build_similarity_matrix(df):
    transformed = pd.DataFrame(index=df.index)

    for col in SIMILARITY_COUNT_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        transformed[f"log_{col}"] = np.log1p(values.clip(lower=0))

    for col in SIMILARITY_CHANGE_FEATURES:
        values = pd.to_numeric(df[col], errors="coerce")
        transformed[f"signedlog_{col}"] = (
            np.sign(values) * np.log1p(np.abs(values))
        )

    for col in SIMILARITY_STATE_FEATURES:
        transformed[col] = pd.to_numeric(df[col], errors="coerce")

    return transformed


def current_similar_events(feature_df, top_k=5):
    query = feature_df.sort_values("availability_datetime").iloc[[-1]].copy()
    query_time = pd.to_datetime(query["availability_datetime"].iloc[0])

    # Only historical rows known before the current query,
    # with an observed next target. No future leakage.
    historical = feature_df[
        (feature_df["availability_datetime"] < query_time)
        & feature_df["target_next_affected_people"].notna()
    ].copy()

    if historical.empty:
        raise RuntimeError("No prior labelled rows for similar-event search.")

    history_matrix = build_similarity_matrix(historical)
    query_matrix = build_similarity_matrix(query)

    # This fit is only nearest-neighbour preprocessing,
    # not predictive model training.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    history_imputed = imputer.fit_transform(history_matrix)
    history_scaled = scaler.fit_transform(history_imputed)
    query_scaled = scaler.transform(imputer.transform(query_matrix))

    distances = euclidean_distances(query_scaled, history_scaled)[0]
    positions = np.argsort(distances)[:top_k]
    scores = 1 / (1 + distances[positions])
    selected = historical.iloc[positions].copy()

    result = pd.DataFrame({
        "historical_datetime": selected["availability_datetime"].to_numpy(),
        "similarity_score": scores,
        "historical_affected_people": selected["affected_people"].to_numpy(),
        "historical_people_in_safety_centres":
            selected["people_in_safety_centres"].to_numpy(),
        "historical_next_affected_people":
            selected["target_next_affected_people"].to_numpy(),
        "historical_next_safety_population":
            selected["target_next_people_in_safety_centres"].to_numpy(),
        "historical_escalation_risk":
            selected["impact_escalation_risk"].to_numpy(),
    })

    return query_time, (
        result
        .sort_values("similarity_score", ascending=False)
        .reset_index(drop=True)
    )

# DRPI / SCPI exact research-index formulas


def historical_percentile(value, historical_values,
                          log_transform=False, positive_only=False):
    values = pd.to_numeric(pd.Series(historical_values), errors="coerce").dropna()

    if positive_only:
        values = values[values > 0]
        if value <= 0:
            return 0.0

    if len(values) == 0:
        return np.nan

    if log_transform:
        values = np.log1p(np.clip(values, 0, None))
        value = np.log1p(max(float(value), 0.0))

    percentile = ((values <= value).mean() * 100)
    return float(np.clip(percentile, 0, 100))


def research_index_band(score):
    if pd.isna(score):
        return "Unavailable"
    if score < 25:
        return "Low"
    if score < 50:
        return "Moderate"
    if score < 75:
        return "High"
    return "Very High"


def build_research_indices(feature_df, refined_trend, pred):
    reference = pd.read_csv(
        FROZEN_INDEX_REFERENCE,
        parse_dates=["availability_datetime"],
    )

    required_reference_columns = [
        "availability_datetime",
        "affected_people",
        "target_next_affected_people",
        "people_in_safety_centres",
        "target_next_people_in_safety_centres",
    ]

    missing_columns = [
        col
        for col in required_reference_columns
        if col not in reference.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Frozen research-index reference is missing columns: "
            + ", ".join(missing_columns)
        )

    if len(reference) != 1281:
        raise RuntimeError(
            f"Frozen research-index reference has {len(reference)} rows; "
            "expected 1281."
        )

    print(
        "Frozen research-index reference rows:",
        len(reference)
    )

    latest = pred["latest"]

    current_flood = safe_num(latest.get("current_flood_flag", 0))
    current_landslide = safe_num(latest.get("current_landslide_flag", 0))
    current_heavy_rain = safe_num(latest.get("current_heavy_rain_flag", 0))

    current_flood = 0.0 if pd.isna(current_flood) else float(current_flood)
    current_landslide = 0.0 if pd.isna(current_landslide) else float(current_landslide)
    current_heavy_rain = 0.0 if pd.isna(current_heavy_rain) else float(current_heavy_rain)

    weather_level = safe_num(latest.get("weather_max_warning_level_24h", np.nan))
    weather_warning_score = (
        np.clip(weather_level / 3.0, 0, 1) * 100
        if pd.notna(weather_level) else 0.0
    )
    weather_context = float(max(50.0 * current_heavy_rain, weather_warning_score))

    landslide_level = safe_num(latest.get("landslide_max_warning_level", np.nan))
    landslide_warning_score = (
        np.clip(landslide_level / 3.0, 0, 1) * 100
        if pd.notna(landslide_level) else 0.0
    )
    landslide_context = float(max(50.0 * current_landslide, landslide_warning_score))

    river_warning = safe_num(latest.get("river_active_warning_24h", np.nan))
    river_warning_score = (
        np.clip(river_warning, 0, 1) * 100
        if pd.notna(river_warning) else 0.0
    )
    flood_context = float(max(50.0 * current_flood, river_warning_score))

    combined_context = float(np.mean(
        [flood_context, weather_context, landslide_context]
    ))

    latest_trend = refined_trend.iloc[-1]
    rapid = safe_num(latest_trend["rapid_growth_score"])
    rapid = 0.0 if pd.isna(rapid) else float(np.clip(rapid, 0, 100))

    current_impact_score = historical_percentile(
        pred["current_affected"], reference["affected_people"], log_transform=True
    )

    historical_impact_changes = (
        reference["target_next_affected_people"].astype(float)
        - reference["affected_people"].astype(float)
    )
    predicted_growth_score = historical_percentile(
        pred["predicted_delta"],
        historical_impact_changes,
        positive_only=True,
    )

    predicted_safety_score = historical_percentile(
        pred["predicted_safety"],
        reference["target_next_people_in_safety_centres"],
        log_transform=True,
    )

    drpi_components = {
        "Current Human Impact": current_impact_score,
        "Predicted Impact Growth": predicted_growth_score,
        "Predicted Safety Demand": predicted_safety_score,
        "Hazard Context": combined_context,
        "Rapid Growth": rapid,
    }

    drpi_vals = [float(x) for x in drpi_components.values() if pd.notna(x)]
    drpi = float(np.clip(np.mean(drpi_vals), 0, 100)) if drpi_vals else np.nan

    current_safety_score = historical_percentile(
        pred["current_safety"],
        reference["people_in_safety_centres"],
        log_transform=True,
    )
    forecast_safety_score = historical_percentile(
        pred["predicted_safety"],
        reference["target_next_people_in_safety_centres"],
        log_transform=True,
    )

    predicted_safety_increase = pred["predicted_safety"] - pred["current_safety"]
    historical_safety_changes = (
        reference["target_next_people_in_safety_centres"].astype(float)
        - reference["people_in_safety_centres"].astype(float)
    )
    safety_growth_score = historical_percentile(
        predicted_safety_increase,
        historical_safety_changes,
        positive_only=True,
    )

    scpi_components = {
        "Current Safety Population": current_safety_score,
        "Forecast Safety Population": forecast_safety_score,
        "Safety-Use Probability": pred["safety_probability"] * 100,
        "Predicted Safety Growth": safety_growth_score,
        "Hazard Context": combined_context,
    }

    scpi_vals = [float(x) for x in scpi_components.values() if pd.notna(x)]
    scpi = float(np.clip(np.mean(scpi_vals), 0, 100)) if scpi_vals else np.nan

    return {
        "flood_context_score": flood_context,
        "weather_context_score": weather_context,
        "landslide_context_score": landslide_context,
        "combined_context_score": combined_context,
        "rapid_growth_score": rapid,
        "latest_trend": latest_trend,
        "DRPI": drpi,
        "DRPI_BAND": research_index_band(drpi),
        "DRPI_COMPONENTS": drpi_components,
        "SCPI": scpi,
        "SCPI_BAND": research_index_band(scpi),
        "SCPI_COMPONENTS": scpi_components,
    }

# Current deployment SHAP - saved human model only

def build_current_shap(human_pipeline, X_latest, snapshot, stage_dir):
    try:
        import shap
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install shap and matplotlib for deployment explanation.") from exc

    if not hasattr(human_pipeline, "named_steps"):
        raise RuntimeError("Human deployment model is not the expected sklearn Pipeline.")

    imputer = human_pipeline.named_steps["imputer"]
    model = human_pipeline.named_steps["model"]

    X_trans = imputer.transform(X_latest)

    try:
        names = imputer.get_feature_names_out(X_latest.columns)
    except Exception:
        names = np.array([f"feature_{i}" for i in range(X_trans.shape[1])])

    explainer = shap.TreeExplainer(model)
    values = np.asarray(explainer.shap_values(X_trans))
    local = values[0] if values.ndim == 2 else values.reshape(-1)

    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        base = float(np.asarray(expected).reshape(-1)[0])
    else:
        base = float(expected)

    feature_values = np.asarray(X_trans)[0]

    table = pd.DataFrame({
        "feature": names,
        "feature_value": feature_values,
        "shap_value": local,
    })
    table["abs_shap"] = table["shap_value"].abs()
    table["direction"] = np.where(
        table["shap_value"] > 0,
        "upward",
        np.where(table["shap_value"] < 0, "downward", "neutral"),
    )
    table = table.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    transformed_prediction = float(model.predict(X_trans)[0])

    explanation = shap.Explanation(
        values=local,
        base_values=base,
        data=feature_values,
        feature_names=list(names),
    )

    image_path = stage_dir / "shap_latest_deployment_waterfall.png"
    shap.plots.waterfall(explanation, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(image_path, dpi=160, bbox_inches="tight")
    plt.close()

    shap_json = {
        "snapshot_datetime": str(snapshot),
        "explanation_target":
            "Predicted signed-log change in affected population",
        "base_value": base,
        "model_transformed_prediction": transformed_prediction,
        "top_contributors": table.head(10).to_dict(orient="records"),
        "note":
            "SHAP explains feature contributions to the modelled signed-log "
            "change; it does not represent causal effects.",
    }

    return table, shap_json, image_path

# Current source freshness

def current_freshness(snapshot, stations, landslide_areas):
    river_prior = stations[stations["report_datetime"] <= snapshot].copy()

    if river_prior.empty:
        river_dt = pd.NaT
        river_age = np.nan
        river_status = "Unavailable"
        river_count = 0
        river_alert_count = 0
        river_latest = pd.DataFrame()
    else:
        river_dt = river_prior["report_datetime"].max()
        river_age = (snapshot - river_dt).total_seconds() / 3600
        river_status = "Current" if river_age <= 24 else "Stale / Unavailable"
        river_latest = river_prior[river_prior["report_datetime"] == river_dt].copy()
        river_count = len(river_latest)
        states = river_latest["status"].fillna("").astype(str).str.lower()
        river_alert_count = int(
            states.isin(["alert", "minor flood", "major flood"]).sum()
        )

    ls_prior = landslide_areas[landslide_areas["report_datetime"] <= snapshot].copy()

    if ls_prior.empty:
        ls_dt = pd.NaT
        ls_age = np.nan
        ls_status = "Stale / Unavailable"
        ls_latest = pd.DataFrame()
    else:
        ls_dt = ls_prior["report_datetime"].max()
        ls_age = (snapshot - ls_dt).total_seconds() / 3600
        ls_status = "Current" if ls_age <= 24 else "Stale / Unavailable"
        ls_latest = ls_prior[ls_prior["report_datetime"] == ls_dt].copy()

    return {
        "river_latest_datetime": river_dt,
        "river_age_hours": river_age,
        "river_status": river_status,
        "river_station_count": int(river_count),
        "river_alert_flood_stations": int(river_alert_count),
        "river_latest": river_latest,
        "landslide_latest_datetime": ls_dt,
        "landslide_age_hours": ls_age,
        "landslide_status": ls_status,
        "landslide_latest": ls_latest,
    }

# District hazard table and Folium map

DISTRICTS = [
    "Ampara", "Anuradhapura", "Badulla", "Batticaloa", "Colombo",
    "Galle", "Gampaha", "Hambantota", "Jaffna", "Kalutara",
    "Kandy", "Kegalle", "Kilinochchi", "Kurunegala", "Mannar",
    "Matale", "Matara", "Monaragala", "Mullaitivu", "Nuwara Eliya",
    "Polonnaruwa", "Puttalam", "Ratnapura", "Trincomalee", "Vavuniya",
]

MAP_ALIASES = {
    "rathnapura": "Ratnapura",
    "moneragala": "Monaragala",
    "nuwara-eliya": "Nuwara Eliya",
    "nuwara eliya": "Nuwara Eliya",
    "mullativu": "Mullaitivu",
    "kilinochi": "Kilinochchi",
}


def normalize_map_district(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower().replace(".", "")
    if text.endswith(" district"):
        text = text[:-9].strip()
    if text in MAP_ALIASES:
        return MAP_ALIASES[text]
    for district in DISTRICTS:
        if district.lower() == text:
            return district
    return str(value).strip().title()


def district_hazard_table(freshness, pred, research):
    df = pd.DataFrame({"district": DISTRICTS})
    df["landslide_warning_level"] = np.nan
    df["landslide_warning"] = "Current bulletin unavailable / stale"
    df["landslide_priority_score"] = np.nan
    df["landslide_data_status"] = freshness["landslide_status"]
    df["landslide_data_age_hours"] = freshness["landslide_age_hours"]

    latest = freshness["landslide_latest"].copy()

    if freshness["landslide_status"] == "Current" and not latest.empty:
        district_col = next(
            (c for c in ["district", "district_clean"] if c in latest.columns),
            None,
        )
        level_col = next(
            (c for c in ["warning_level", "level", "max_warning_level"]
             if c in latest.columns),
            None,
        )

        if district_col and level_col:
            levels = latest[[district_col, level_col]].copy()
            levels["district"] = levels[district_col].apply(normalize_map_district)
            levels["level"] = pd.to_numeric(levels[level_col], errors="coerce")
            levels = levels.groupby("district", as_index=False)["level"].max()

            df = df.drop(columns=["landslide_warning_level"]).merge(
                levels, on="district", how="left"
            ).rename(columns={"level": "landslide_warning_level"})

            df["landslide_warning_level"] = (
                df["landslide_warning_level"].fillna(0).astype(int)
            )
            labels = {
                0: "No active warning in current bulletin",
                1: "Level 1 - Yellow / Watch",
                2: "Level 2 - Amber / Alert",
                3: "Level 3 - Red / Evacuation",
            }
            df["landslide_warning"] = df["landslide_warning_level"].map(labels)
            df["landslide_priority_score"] = (
                df["landslide_warning_level"] / 3.0 * 100
            )
            df["landslide_data_status"] = "Current"

    # NATIONAL values repeated only as contextual popup fields.
    df["national_current_affected"] = pred["current_affected"]
    df["national_predicted_next_affected"] = pred["predicted_affected"]
    df["national_experimental_risk"] = pred["risk"]
    df["national_DRPI"] = research["DRPI"]
    df["national_SCPI"] = research["SCPI"]
    return df


def load_boundaries():
    if BOUNDARY_CACHE.exists():
        return json.loads(BOUNDARY_CACHE.read_text(encoding="utf-8"))

    import requests

    api = "https://www.geoboundaries.org/api/current/gbOpen/LKA/ADM2/"
    response = requests.get(api, timeout=30)
    response.raise_for_status()
    meta = response.json()
    geo_url = meta["simplifiedGeometryGeoJSON"]

    geo_response = requests.get(geo_url, timeout=60)
    geo_response.raise_for_status()
    geo = geo_response.json()

    # Cache after successful download.
    temp = BOUNDARY_CACHE.with_suffix(".geojson.tmp")
    temp.write_text(json.dumps(geo), encoding="utf-8")
    os.replace(temp, BOUNDARY_CACHE)
    return geo


def generate_map(district_df, snapshot, freshness, pred, research, output_path):
    try:
        import folium
    except ImportError as exc:
        raise RuntimeError("Install folium for district map generation.") from exc

    geo = load_boundaries()

    if not geo.get("features"):
        raise RuntimeError("Sri Lanka ADM2 boundary GeoJSON has no features.")

    props = geo["features"][0].get("properties", {})
    if "shapeName" in props:
        name_field = "shapeName"
    elif "shape_name" in props:
        name_field = "shape_name"
    elif "NAME_2" in props:
        name_field = "NAME_2"
    elif "name" in props:
        name_field = "name"
    else:
        raise RuntimeError("Unable to identify district name field in GeoJSON.")

    lookup = district_df.set_index("district").to_dict(orient="index")

    for feature in geo["features"]:
        raw = feature.get("properties", {}).get(name_field)
        feature["properties"]["district_normalized"] = normalize_map_district(raw)

    m = folium.Map(
        location=[7.8731, 80.7718],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    def style(feature):
        district = feature["properties"].get("district_normalized")
        row = lookup.get(district, {})
        if row.get("landslide_data_status") != "Current":
            fill = "#9e9e9e"
        else:
            level = safe_num(row.get("landslide_warning_level", 0))
            level = 0 if pd.isna(level) else int(level)
            fill = {
                0: "#b7e4c7",
                1: "#f9e547",
                2: "#f0a34a",
                3: "#d9534f",
            }.get(level, "#9e9e9e")
        return {
            "fillColor": fill,
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.72,
        }

    folium.GeoJson(
        geo,
        name="District Hazard Context",
        style_function=style,
        tooltip=folium.GeoJsonTooltip(
            fields=["district_normalized"],
            aliases=["District:"],
            sticky=False,
        ),
    ).add_to(m)

    panel = f"""
    <div style="position:fixed;bottom:20px;left:20px;width:370px;
    z-index:9999;background:white;border:1px solid #777;border-radius:8px;
    padding:12px;font-size:12px;line-height:1.45;
    box-shadow:0 1px 8px rgba(0,0,0,.2)">
    <b>FloodImpact-LK hazard context</b><br>
    Snapshot: {snapshot}<br>
    National current affected: {pred["current_affected"]:,.0f}<br>
    National next estimate: {pred["predicted_affected"]:,.0f}<br>
    Experimental risk: {pred["risk"]}<br>
    DRPI: {research["DRPI"]:.2f}/100<br>
    SCPI: {research["SCPI"]:.2f}/100<br>
    River data: {freshness["river_status"]}<br>
    Landslide data: {freshness["landslide_status"]}<br>
    <hr style="margin:6px 0">
    District colours represent hazard/warning context only.
    National forecasts are not district-level forecasts.
    </div>
    """
    m.get_root().html.add_child(folium.Element(panel))
    m.save(str(output_path))

# Dashboard payload builders

def build_snapshot(pred):
    latest = pred["latest"]
    hazards = [
        x.strip()
        for x in str(latest.get("disaster_types", "")).split(",")
        if x.strip()
    ]

    return {
        "snapshot_datetime": str(latest["availability_datetime"]),
        "forecast_definition":
            "Next eligible DMC Situation Report within a maximum 48-hour gap",
        "geographic_scope": "Sri Lanka - national",
        "reported_hazards": hazards,

        "current_affected_people": pred["current_affected"],
        "predicted_next_affected_people": pred["predicted_affected"],
        "predicted_change": pred["predicted_delta"],
        "predicted_affected_change": pred["predicted_delta"],

        # Multiple aliases supported by the current Streamlit app.
        "prediction_interval_lower": pred["lower_90"],
        "prediction_interval_upper": pred["upper_90"],
        "lower_90": pred["lower_90"],
        "upper_90": pred["upper_90"],
        "prediction_interval_90": {
            "lower": pred["lower_90"],
            "upper": pred["upper_90"],
        },
        "prediction_interval_label": "90% model-based prediction interval",

        "current_safety_population": pred["current_safety"],
        "safety_use_probability": pred["safety_probability"],
        "safety_use_prediction": pred["safety_use"],
        "predicted_next_safety_population": pred["predicted_safety"],

        "experimental_escalation_risk": pred["risk"],
        "risk_probabilities": pred["risk_probabilities"],

        "hazard_flags": {
            "flood": int(latest["current_flood_flag"]),
            "landslide": int(latest["current_landslide_flag"]),
            "heavy_rain": int(latest["current_heavy_rain_flag"]),
        },

        "important_note":
            "Human-impact and safety-demand forecasts are national multi-hazard "
            "forecasts. They must not be interpreted as Flood-only, "
            "Landslide-only, or district-level forecasts.",
    }


def build_research_json(pred, research, freshness):
    t = research["latest_trend"]

    return {
        "snapshot_datetime": str(pred["latest"]["availability_datetime"]),

        "hazard_context": {
            "flood_context_score": research["flood_context_score"],
            "weather_context_score": research["weather_context_score"],
            "landslide_context_score": research["landslide_context_score"],
            "combined_context_score": research["combined_context_score"],
            "combined_hazard_context_score": research["combined_context_score"],
        },

        "trend": {
            "trend": str(t["impact_trend"]),
            "change_from_previous": safe_num(t["impact_absolute_change"]),
            "rapid_growth_score": research["rapid_growth_score"],
            "monitoring_status": str(t["impact_monitoring_status"]),
        },

        "DRPI": {
            "score": research["DRPI"],
            "band": research["DRPI_BAND"],
            "components": research["DRPI_COMPONENTS"],
            "note":
                "Research-derived FloodImpact-LK priority indicator; "
                "not an official warning.",
        },

        "SCPI": {
            "score": research["SCPI"],
            "band": research["SCPI_BAND"],
            "components": research["SCPI_COMPONENTS"],
            "note":
                "Research-derived relative safety-demand pressure indicator. "
                "It does not represent actual safety-centre capacity utilization.",
        },

        "freshness": {
            "river_age_hours": freshness["river_age_hours"],
            "river_status": freshness["river_status"],
            "river_latest_datetime": freshness["river_latest_datetime"],
            "river_station_count": freshness["river_station_count"],
            "river_alert_flood_stations": freshness["river_alert_flood_stations"],

            "landslide_age_hours": freshness["landslide_age_hours"],
            "landslide_status": freshness["landslide_status"],
            "landslide_latest_datetime": freshness["landslide_latest_datetime"],
        },
    }


def similar_payload(query_time, similar_df):
    events = []
    for _, row in similar_df.iterrows():
        events.append({
            "historical_datetime": row["historical_datetime"],
            "similarity_score": row["similarity_score"],
            "affected_people": row["historical_affected_people"],
            "safety_population": row["historical_people_in_safety_centres"],
            "next_affected_people": row["historical_next_affected_people"],
            "next_safety_population": row["historical_next_safety_population"],
            "escalation_risk": row["historical_escalation_risk"],
        })

    return {
        "query_datetime": str(query_time),
        "events": events,
        "note":
            "Similarity is derived from transformed feature-space distance; "
            "it is not a probability or forecast confidence.",
    }


def fallback_performance():
    return {
        "human_impact": {
            "mae": 21306.60,
            "rmse": 65900.17,
            "median_absolute_error": 40.76,
            "rmsle": 3.6472,
            "r2": 0.6481,
            "persistence_mae": 21283.91,
        },
        "safety_usage": {
            "accuracy": 0.886667,
            "balanced_accuracy": 0.886764,
            "f1": 0.884354,
        },
        "safety_demand": {
            "mae": 439.27,
            "rmse": 947.27,
            "rmsle": 2.4593,
            "r2": 0.9724,
            "persistence_mae": 381.37,
            "persistence_rmse": 988.76,
        },
        "risk": {
            "accuracy": 0.48,
            "balanced_accuracy": 0.33838,
            "macro_f1": 0.33726,
            "weighted_f1": 0.51472,
            "note": "Experimental only; weak held-out temporal generalisation.",
        },
        "uncertainty": {
            "nominal_interval": 0.90,
            "empirical_coverage": 0.8067,
            "average_width": 22664.99,
            "median_width": 7969.54,
        },
    }


def build_payload(snapshot, research_json, sim_payload, shap_json):
    old = read_json(PAYLOAD_JSON)

    project = old.get("project", {
        "name": "FloodImpact-LK",
        "scope": "Floods and rainfall-induced landslides in Sri Lanka",
        "prototype": True,
        "update_mode": "Official-report-driven near-real-time",
    })

    performance = old.get("performance")
    if not isinstance(performance, dict) or not performance:
        performance = fallback_performance()

    explainability = old.get("explainability", {})
    explainability["current_deployment_top_contributors"] = (
        shap_json.get("top_contributors", [])
    )
    explainability["current_explanation_target"] = shap_json.get(
        "explanation_target"
    )
    explainability["current_base_value"] = shap_json.get("base_value")
    explainability["note"] = shap_json.get("note")

    return {
        "project": project,
        "current_snapshot": snapshot,
        "research_context": research_json,
        "similar_events": sim_payload,
        "explainability": explainability,
        "performance": performance,
        "generated_at": now_iso(),
        "update_note":
            "Official-report-driven near-real-time update using frozen saved "
            "deployment models. No predictive model retraining was performed.",
    }


# Staging helpers

def write_json(path, obj):
    Path(path).write_text(
        json.dumps(clean_json(obj), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def stage_path(stage, target):
    return stage / target.name


def transactional_commit(pairs, backup_dir):
    """
    Replace all final outputs as one rollback-capable transaction.

    Individual os.replace operations are atomic. This wrapper additionally
    restores earlier files if any later replacement unexpectedly fails.
    """
    backup_dir.mkdir(parents=True, exist_ok=True)
    backups = {}
    committed = []

    try:
        for index, (staged, target) in enumerate(pairs):
            target.parent.mkdir(parents=True, exist_ok=True)

            if target.exists():
                backup = backup_dir / f"{index:03d}_{target.name}.bak"
                shutil.copy2(target, backup)
                backups[target] = backup
            else:
                backups[target] = None

            os.replace(staged, target)
            committed.append(target)

    except Exception:
        # Roll back every target that was already replaced.
        for target in reversed(committed):
            backup = backups.get(target)
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                elif backup.exists():
                    os.replace(backup, target)
            except Exception:
                pass
        raise

# Main


def run_once():
    print("\n" + "=" * 80)
    print("FLOODIMPACT-LK SAFE LATEST INFERENCE")
    print("=" * 80)
    print("Started:", now_iso())

    require_files()

    # 1. Load validated processed datasets.
    situation = pd.read_csv(SITUATION_CLEAN)
    weather = pd.read_csv(WEATHER_CLEAN)
    river = pd.read_csv(RIVER_CLEAN)
    stations = pd.read_csv(RIVER_STATIONS)
    landslide = pd.read_csv(LANDSLIDE_CLEAN)
    areas = pd.read_csv(LANDSLIDE_AREAS)

    # 2. Parse only newly downloaded PDFs.
    situation, new_situation = append_new_situation(situation)
    weather, new_weather = append_new_weather(weather)
    river, stations, new_river, new_stations = append_new_river(river, stations)
    (
        landslide,
        areas,
        new_landslide,
        new_landslide_areas,
    ) = append_new_landslide(
        landslide,
        areas,
    )
    

    # 3. Exact timestamp standardisation.
    situation_i, weather_i, river_i, stations_i, landslide_i, areas_i = (
        standardize_all(situation, weather, river, stations, landslide, areas)
    )

    # 4. Preserve frozen historical scope; classify only new DMC reports.
    scope, audit = build_live_scope(situation_i, new_situation)

    # The current automatic update must end on a normal general Situation Report.
    title_map = situation_metadata_title_map()
    latest_scope = scope.sort_values("availability_datetime").iloc[-1]
    latest_title = title_map.get(str(latest_scope["file_name"]), "")
    if latest_title.strip().casefold() != "situation report":
        raise RuntimeError(
            "Latest retained report is not metadata title 'Situation Report'. "
            "Safe stop prevents forecasting from a special/drought report."
        )

    print("\nLatest retained general Situation Report:")
    print("  File:", latest_scope["file_name"])
    print("  Time:", latest_scope["availability_datetime"])
    print("  Affected:", latest_scope["affected_people"])

    # 5. Rebuild timestamp-safe context features from cleaned sources.
    integrated = integrate_sources(scope, weather_i, river_i, landslide_i)

    # 6. Canonical timeline + exact feature engineering.
    forecast_timeline, feature_df = build_feature_timeline(integrated)

    # Expected historical parity checks.
    historical_before_live = feature_df[
        feature_df["availability_datetime"] <= FROZEN_REFERENCE_CUTOFF
    ]
    if historical_before_live.empty:
        raise RuntimeError("Historical feature timeline unexpectedly empty.")

    # 7. Frozen deployment inference.
    models, core_features = load_models()
    print("\nFrozen CORE feature count:", len(core_features))
    if len(core_features) != 46:
        raise RuntimeError(
            f"Saved core_model_features contains {len(core_features)} features; expected 46."
        )

    pred = predict_latest(feature_df, models, core_features)
    snapshot_dt = pd.to_datetime(pred["latest"]["availability_datetime"])

    print("Latest model row:", snapshot_dt)
    print("Latest model file:", pred["latest"]["file_name"])

    # 8. Trend, similarity and research indices.
    refined_trend = build_refined_trend(feature_df)
    query_time, similar_df = current_similar_events(feature_df, top_k=5)
    research = build_research_indices(feature_df, refined_trend, pred)

    # 9. Source freshness + district state.
    freshness = current_freshness(snapshot_dt, stations_i, areas_i)
    district_df = district_hazard_table(freshness, pred, research)

    # 10. Dynamic JSON sections.
    snapshot = build_snapshot(pred)
    research_json = build_research_json(pred, research, freshness)
    sim_json = similar_payload(query_time, similar_df)

    # 11. Stage every dashboard-facing output first.
    stage = Path(tempfile.mkdtemp(prefix="floodimpact_live_", dir=str(ROOT)))
    stage_processed = stage / "processed"
    stage_figures = stage / "figures"
    stage_processed.mkdir(parents=True, exist_ok=True)
    stage_figures.mkdir(parents=True, exist_ok=True)

    try:
        # SHAP must match current snapshot.
        shap_table, shap_json, shap_image_stage = build_current_shap(
            models["human"],
            pred["X"],
            snapshot_dt,
            stage_figures,
        )

        # Map must also match current snapshot.
        map_stage = stage_processed / MAP_HTML.name
        generate_map(
            district_df,
            snapshot_dt,
            freshness,
            pred,
            research,
            map_stage,
        )

        payload = build_payload(snapshot, research_json, sim_json, shap_json)

        # Final coherence checks BEFORE replacement.
        if pd.to_datetime(snapshot["snapshot_datetime"]) != snapshot_dt:
            raise RuntimeError("Snapshot timestamp mismatch.")
        if payload["current_snapshot"]["snapshot_datetime"] != snapshot["snapshot_datetime"]:
            raise RuntimeError("Dashboard payload current_snapshot mismatch.")
        if payload["research_context"]["snapshot_datetime"] != research_json["snapshot_datetime"]:
            raise RuntimeError("Dashboard payload research_context mismatch.")
        if not np.isfinite(pred["predicted_affected"]):
            raise RuntimeError("Non-finite human-impact prediction.")
        if pred["lower_90"] > pred["upper_90"]:
            raise RuntimeError("Prediction interval ordering failed.")

        # Stage processed CSVs.
        csv_outputs = {
            SITUATION_CLEAN: situation,
            WEATHER_CLEAN: weather,
            RIVER_CLEAN: river,
            RIVER_STATIONS: stations,
            LANDSLIDE_CLEAN: landslide,
            LANDSLIDE_AREAS: areas,
            ANALYTICAL_SCOPE: scope,
            SCOPE_AUDIT: audit,
            INTEGRATED: integrated,
            TREND_CSV: refined_trend[
                [
                    c for c in [
                        "availability_datetime",
                        "affected_people",
                        "rapid_growth_score",
                        "impact_trend",
                        "impact_monitoring_status",
                    ]
                    if c in refined_trend.columns
                ]
            ],
            SIMILAR_CSV: similar_df,
            SHAP_CSV: shap_table,
        }

        staged_pairs = []

        for target, df in csv_outputs.items():
            sp = stage_path(stage_processed, target)
            df.to_csv(sp, index=False)
            staged_pairs.append((sp, target))

        # Stage JSONs.
        json_outputs = {
            LATEST_MODEL_JSON: snapshot,
            LATEST_RESEARCH_JSON: research_json,
            SHAP_JSON: shap_json,
            PAYLOAD_JSON: payload,
        }

        for target, obj in json_outputs.items():
            sp = stage_path(stage_processed, target)
            write_json(sp, obj)
            staged_pairs.append((sp, target))

        staged_pairs.append((map_stage, MAP_HTML))
        staged_pairs.append((shap_image_stage, SHAP_IMAGE))

        # Only now commit final outputs. If any replacement fails,
        # prior valid outputs are restored.
        transactional_commit(
            staged_pairs,
            stage / "rollback_backups",
        )

    finally:
        shutil.rmtree(stage, ignore_errors=True)

    print("\n" + "=" * 80)
    print("LATEST INFERENCE COMPLETE")
    print("=" * 80)
    print("Snapshot:", snapshot_dt)
    print("Source file:", pred["latest"]["file_name"])

    print("\nHUMAN IMPACT")
    print("Current affected:", round(pred["current_affected"], 2))
    print("Predicted next:", round(pred["predicted_affected"], 2))
    print("Predicted change:", round(pred["predicted_delta"], 2))
    print(
        "90% model-based interval:",
        round(pred["lower_90"], 2),
        "-",
        round(pred["upper_90"], 2),
    )

    print("\nSAFETY-CENTRE DEMAND")
    print("Current:", round(pred["current_safety"], 2))
    print("Use probability:", round(pred["safety_probability"], 4))
    print("Predicted next:", round(pred["predicted_safety"], 2))

    print("\nEXPERIMENTAL ESCALATION RISK")
    print("Class:", pred["risk"])
    print("Probabilities:", pred["risk_probabilities"])

    print("\nRESEARCH INDICES")
    print("DRPI:", round(research["DRPI"], 2), research["DRPI_BAND"])
    print("SCPI:", round(research["SCPI"], 2), research["SCPI_BAND"])

    print("\nFRESHNESS")
    print(
        "River:", freshness["river_status"],
        "| age hours:", freshness["river_age_hours"],
        "| alert/flood stations:", freshness["river_alert_flood_stations"],
    )
    print(
        "Landslide:", freshness["landslide_status"],
        "| age hours:", freshness["landslide_age_hours"],
    )

    print("\nNo predictive model was retrained.")
    print("Notebook cells were not executed.")
    print("Updated dashboard payload:", PAYLOAD_JSON)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="FloodImpact-LK safe current-report inference."
    )
    parser.add_argument("--once", action="store_true")
    parser.parse_args()

    try:
        return run_once()
    except Exception as exc:
        print("\n" + "=" * 80)
        print("LATEST INFERENCE FAILED - SAFE STOP")
        print("=" * 80)
        print(str(exc))
        print("\nExisting valid dashboard outputs were preserved.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
