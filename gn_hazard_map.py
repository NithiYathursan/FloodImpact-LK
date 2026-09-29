from pathlib import Path
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent
BOUNDARY_DIR = ROOT / 'data' / 'boundaries'
DISTRICT_BOUNDARY_DIR = BOUNDARY_DIR / 'gn_by_district'
DISTRICT_INDEX = DISTRICT_BOUNDARY_DIR / 'index.csv'
PROCESSED = ROOT / 'data' / 'processed'
RIVER_DIR = ROOT / 'data' / 'raw' / 'river_reports'
LANDSLIDE_DIR = ROOT / 'data' / 'raw' / 'landslide_reports'
LANDSLIDE_AREAS = PROCESSED / 'landslide_warning_areas.csv'
WEATHER_CLEAN = PROCESSED / 'weather_reports_cleaned.csv'
RIVER_STATIONS = PROCESSED / 'river_station_observations.csv'
DASHBOARD_PAYLOAD = PROCESSED / 'dashboard_payload.json'


def _norm(v):
    if v is None or pd.isna(v):
        return ''
    s = str(v).strip().casefold()
    s = s.replace('&', ' and ')
    s = re.sub(r'[._\-]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def _dsd_key(v):
    """Canonical key for matching NBRO DSD names to Survey Dept boundaries."""
    s = '' if v is None or pd.isna(v) else str(v)

    # Remove warning-change symbols and list markers found in NBRO tables.
    s = re.sub(r'[↑↓*]+', ' ', s)

    # Remove administrative suffixes if they are present in the source text.
    s = re.sub(
        r'\bdivisional\s+secretariat\s+division(?:\(s\))?\b',
        ' ',
        s,
        flags=re.I,
    )
    s = re.sub(r'\bdsd\b', ' ', s, flags=re.I)

    key = _norm(s)

    # Known wording/spelling variants between NBRO bulletins and
    # the Survey Department administrative layer.
    aliases = {
        'deltota': 'delthota',
        'delthota': 'delthota',
        'gangawata korale': 'gangawata korale',
        'kandy four gravets and gangawata korale': 'gangawata korale',
        'kandy four gravets gangawata korale': 'gangawata korale',
        'pujapitiya': 'poojapitiya',
        'poojapitiya': 'poojapitiya',
    }

    return aliases.get(key, key)


def _extract_dsds_from_area_text(value):
    """Extract individual DSD names from an NBRO area_text_raw cell."""
    if value is None or pd.isna(value):
        return []

    s = str(value)
    # Keep only the location list before the standard administrative suffix.
    s = re.split(
        r'\bDivisional\s+Secretariat\s+Division',
        s,
        maxsplit=1,
        flags=re.I,
    )[0]

    # Remove arrows / asterisks used for change indicators.
    s = re.sub(r'[↑↓*]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,;:-')

    # NBRO rows normally separate multiple DSDs with commas and "and".
    parts = re.split(r'\s*,\s*|\s+and\s+|\s*&\s*', s, flags=re.I)

    out = []
    for part in parts:
        name = re.sub(r'\s+', ' ', part).strip(' ,;:-')
        if name:
            out.append(name)
    return out


def _clean(v):
    return re.sub(r'\s+', ' ', str(v)).strip(' ,;:-')


def _split_names(text):
    parts = re.split(r'\s*,\s*|\s+and\s+|\s*&\s*', _clean(text), flags=re.I)
    return [_clean(x) for x in parts if _clean(x)]


def _find_boundary_file():
    if not BOUNDARY_DIR.exists():
        return None
    for pat in ('*.shp', '*.geojson', '*.json', '*.gpkg'):
        files = sorted(BOUNDARY_DIR.glob(pat))
        if files:
            return files[0]
    return None


def _pick_field(columns, kind):
    cols = list(columns)
    n = {c: _norm(c) for c in cols}
    exact = {
        'district': {'district','district name','districtname','dist name','distname','dist n','name 1','adm1 en'},
        'dsd': {'dsd','ds division','divisional secretariat','divisional secretariat division','dsd name','dsdname','dsd n','ds name','name 2','adm2 en'},
        'gn': {'gn','gnd','gn division','grama niladhari','grama niladhari division','gn name','gnname','gn n','gnd name','gnd n','name 3','adm3 en'},
    }
    for c, v in n.items():
        if v in exact[kind]:
            return c
    keys = {
        'district': [('district',), ('dist','name')],
        'dsd': [('dsd',), ('divisional','secretariat'), ('ds','name')],
        'gn': [('grama','niladhari'), ('gn','name'), ('gnd',)],
    }
    for ks in keys[kind]:
        for c, v in n.items():
            if all(k in v for k in ks):
                return c
    return None


def _district_boundary_file(selected_district):
    if not DISTRICT_INDEX.exists():
        return None
    idx = pd.read_csv(DISTRICT_INDEX)
    if idx.empty or not {'district', 'file_name'}.issubset(idx.columns):
        return None
    key = _norm(selected_district)
    matches = idx[idx['district'].map(_norm) == key]
    if matches.empty:
        return None
    path = DISTRICT_BOUNDARY_DIR / str(matches.iloc[0]['file_name'])
    return path if path.exists() else None


def load_gn_boundaries(selected_district=None):
    try:
        import geopandas as gpd
    except ImportError as e:
        raise RuntimeError('Install GeoPandas: pip install geopandas') from e

    f = None
    if selected_district:
        f = _district_boundary_file(selected_district)

    if f is None:
        f = _find_boundary_file()

    if f is None:
        raise FileNotFoundError('No GN boundary SHP/GeoJSON found in data/boundaries.')

    g = gpd.read_file(f)
    if g.empty:
        raise RuntimeError(f'Boundary file is empty: {f}')
    if g.crs is None:
        raise RuntimeError('Boundary CRS is missing. Keep the .prj with the shapefile.')

    dc = _pick_field(g.columns, 'district')
    sc = _pick_field(g.columns, 'dsd')
    gc = _pick_field(g.columns, 'gn')
    if not all([dc, sc, gc]):
        raise RuntimeError('Could not detect district/DSD/GN columns. Columns: ' + ', '.join(map(str, g.columns)))

    g = g.to_crs(epsg=4326).copy()
    g['DISTRICT'] = g[dc].astype(str).str.strip()
    g['DSD'] = g[sc].astype(str).str.strip()
    g['GN'] = g[gc].astype(str).str.strip()
    g['district_key'] = g['DISTRICT'].map(_norm)
    g['dsd_key'] = g['DSD'].map(_dsd_key)
    g['gn_key'] = g['GN'].map(_norm)
    return g, f


def list_districts():
    if DISTRICT_INDEX.exists():
        idx = pd.read_csv(DISTRICT_INDEX)
        if 'district' in idx.columns:
            return sorted(
                x for x in idx['district'].dropna().astype(str).str.strip().unique()
                if x
            )

    g, _ = load_gn_boundaries()
    return sorted(x for x in g['DISTRICT'].dropna().unique() if str(x).strip())


def _filename_time(path):
    m = re.search(r'_(\d{8})_(\d{4})_', path.name)
    if not m:
        return pd.NaT
    return pd.to_datetime(f'{m.group(1)} {m.group(2)}', format='%Y%m%d %H%M', errors='coerce')


def _recent_pdfs(folder, snapshot, hours):
    snap = pd.to_datetime(snapshot, errors='coerce')
    if pd.isna(snap) or not folder.exists():
        return []
    cutoff = snap - pd.Timedelta(hours=hours)
    rows = []
    for p in folder.glob('*.pdf'):
        dt = _filename_time(p)
        if pd.notna(dt) and cutoff <= dt <= snap:
            rows.append((dt, p))
    return [p for _, p in sorted(rows, reverse=True)]


def _pdf_text(path):
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ''
            if t:
                pages.append(t)
    return re.sub(r'\s+', ' ', '\n'.join(pages)).strip()


def _river_level(text):
    t = text.casefold()
    if re.search(r'colou?r\s*[-:–]?\s*red|red\s+warning|high flood situation|major flood', t):
        return 3, 'Level 3 / high flood context'
    if re.search(r'colou?r\s*[-:–]?\s*(amber|orange)|amber\s+warning|flood warning', t):
        return 2, 'Level 2 / flood warning'
    if re.search(r'colou?r\s*[-:–]?\s*yellow|flood advisory', t):
        return 1, 'Level 1 / flood advisory'
    return None, 'Flood context'


def _extract_river_gns(text, file_name, dt):
    level, label = _river_level(text)
    recs = []
    patterns = [
        re.compile(r'(?P<gns>[A-Za-z][A-Za-z0-9 .\'\-/,&()]{1,220}?)\s+Grama\s+Niladhari\s+Divisions?\s+in\s+(?:the\s+)?(?P<dsd>[A-Za-z][A-Za-z0-9 .\'\-/&()]{1,100}?)\s+Divisional\s+Secretariat\s+Division', re.I),
        re.compile(r'(?P<gns>[A-Za-z][A-Za-z0-9 .\'\-/,&()]{1,220}?)\s+GN\s+Divisions?\s+in\s+(?P<dsd>[A-Za-z][A-Za-z0-9 .\'\-/&()]{1,100}?)\s+DSD', re.I),
    ]
    for pat in patterns:
        for m in pat.finditer(text):
            dsd = _clean(m.group('dsd'))
            for gn in _split_names(m.group('gns')):
                gn = re.sub(r'^.*?\b(?:covering|including)\b\s+', '', gn, flags=re.I)
                gn = _clean(gn)
                if gn:
                    recs.append({'hazard':'Flood','source':'Irrigation Department','resolution':'GN','gn':gn,'dsd':dsd,'level':level,'label':label,'report_datetime':dt,'report_file':file_name})
    return recs


def _extract_landslide_gns(text, file_name, dt):
    recs = []
    pat = re.compile(r'(?P<gns>[A-Za-z][A-Za-z0-9 .\'\-/,&()]{1,220}?)\s*,?\s*GN\s+Divisions?\s+in\s+(?P<dsd>[A-Za-z][A-Za-z0-9 .\'\-/&()]{1,100}?)\s+DSD', re.I)
    for m in pat.finditer(text):
        dsd = _clean(m.group('dsd'))
        for gn in _split_names(m.group('gns')):
            recs.append({'hazard':'Landslide','source':'NBRO','resolution':'GN named in bulletin','gn':_clean(gn),'dsd':dsd,'level':None,'label':'GN explicitly named','report_datetime':dt,'report_file':file_name})
    return recs


def explicit_gn_context(snapshot):
    recs = []
    for p in _recent_pdfs(RIVER_DIR, snapshot, 48):
        try:
            recs += _extract_river_gns(_pdf_text(p), p.name, _filename_time(p))
        except Exception:
            pass
    for p in _recent_pdfs(LANDSLIDE_DIR, snapshot, 24):
        try:
            recs += _extract_landslide_gns(_pdf_text(p), p.name, _filename_time(p))
        except Exception:
            pass
    if not recs:
        return pd.DataFrame(columns=['hazard','source','resolution','gn','dsd','level','label','report_datetime','report_file','gn_key','dsd_key'])
    d = pd.DataFrame(recs)
    d['gn_key'] = d['gn'].map(_norm)
    d['dsd_key'] = d['dsd'].map(_dsd_key)
    return d.sort_values('report_datetime').drop_duplicates(['hazard','source','gn_key','dsd_key'], keep='last')


def _detect_col(columns, names):
    nm = {c:_norm(c) for c in columns}
    for name in names:
        for c, v in nm.items():
            if v == name:
                return c
    for c, v in nm.items():
        if any(name in v for name in names):
            return c
    return None


def current_landslide_dsd_context(snapshot):
    """Return the latest current NBRO warning independently for each DSD.

    The processed landslide_warning_areas.csv stores one row per warning level
    and commonly keeps the DSD list inside area_text_raw rather than a separate
    DSD column. This function expands that list into one row per DSD, then
    selects the latest record for each district + DSD at/before the dashboard
    snapshot. A 24-hour freshness rule is retained.
    """
    if not LANDSLIDE_AREAS.exists():
        return pd.DataFrame()

    try:
        d = pd.read_csv(LANDSLIDE_AREAS)
    except Exception:
        return pd.DataFrame()

    if d.empty:
        return pd.DataFrame()

    snap = pd.to_datetime(snapshot, errors='coerce')
    if pd.isna(snap):
        return pd.DataFrame()

    dc = _detect_col(d.columns, ['district clean', 'district'])
    sc = _detect_col(d.columns, ['dsd clean', 'dsd'])
    ac = _detect_col(d.columns, ['area text raw', 'area text', 'location'])
    lc = _detect_col(
        d.columns,
        ['warning level', 'max warning level', 'level'],
    )

    if lc is None:
        return pd.DataFrame()

    # Reconstruct the real report timestamp. Do not parse numeric issue_time
    # alone because values such as 1600 would otherwise be interpreted as a
    # 1970 nanosecond timestamp by pandas.
    if 'report_datetime' in d.columns:
        d['report_datetime'] = pd.to_datetime(
            d['report_datetime'],
            errors='coerce',
        )
    else:
        date_col = _detect_col(d.columns, ['report date', 'date'])
        time_col = _detect_col(d.columns, ['issue time', 'time'])

        if date_col is None:
            return pd.DataFrame()

        dates = pd.to_datetime(d[date_col], errors='coerce')

        if time_col is not None:
            times = (
                d[time_col]
                .astype(str)
                .str.replace(r'\\.0$', '', regex=True)
                .str.replace(r'\\D', '', regex=True)
                .str.zfill(4)
            )

            d['report_datetime'] = pd.to_datetime(
                dates.dt.strftime('%Y-%m-%d') + ' ' + times,
                format='%Y-%m-%d %H%M',
                errors='coerce',
            )
        else:
            d['report_datetime'] = dates

    d = d[
        d['report_datetime'].notna()
        & (d['report_datetime'] <= snap)
    ].copy()

    if d.empty:
        return pd.DataFrame()

    d['district'] = (
        d[dc].astype(str).str.strip()
        if dc is not None
        else ''
    )
    d['level'] = pd.to_numeric(
        d[lc],
        errors='coerce',
    ).fillna(0)

    # Expand to one row per DSD.
    expanded_rows = []

    for _, row in d.iterrows():
        if sc is not None and str(row.get(sc, '')).strip():
            dsd_names = [str(row.get(sc, '')).strip()]
        elif ac is not None:
            dsd_names = _extract_dsds_from_area_text(
                row.get(ac)
            )
        else:
            dsd_names = []

        for dsd_name in dsd_names:
            if not dsd_name:
                continue

            expanded_rows.append({
                'district': row['district'],
                'dsd': dsd_name,
                'level': row['level'],
                'report_datetime': row['report_datetime'],
            })

    if not expanded_rows:
        return pd.DataFrame()

    x = pd.DataFrame(expanded_rows)
    x['district_key'] = x['district'].map(_norm)
    x['dsd_key'] = x['dsd'].map(_dsd_key)

    # Keep only plausible non-empty administrative names.
    x = x[
        x['dsd_key'].astype(str).str.len().gt(1)
    ].copy()

    if x.empty:
        return pd.DataFrame()

    # Latest row independently for each DSD at or before snapshot.
    x = x.sort_values('report_datetime')
    latest = (
        x.groupby(
            ['district_key', 'dsd_key'],
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    latest['age_hours'] = (
        snap - latest['report_datetime']
    ).dt.total_seconds() / 3600.0

    latest = latest[
        (latest['age_hours'] >= 0)
        & (latest['age_hours'] <= 24)
    ].copy()

    return latest[
        [
            'district',
            'dsd',
            'level',
            'report_datetime',
            'age_hours',
            'district_key',
            'dsd_key',
        ]
    ].drop_duplicates()



def _report_datetime(df):
    """Best-effort report timestamp for cleaned pipeline tables."""
    if df.empty:
        return pd.Series(dtype='datetime64[ns]')

    if 'report_datetime' in df.columns:
        return pd.to_datetime(df['report_datetime'], errors='coerce')

    date_col = _detect_col(df.columns, ['report date', 'date'])
    time_col = _detect_col(df.columns, ['issue time', 'time'])

    if date_col and time_col:
        return pd.to_datetime(
            df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip(),
            errors='coerce',
        )
    if date_col:
        return pd.to_datetime(df[date_col], errors='coerce')
    if time_col:
        return pd.to_datetime(df[time_col], errors='coerce')
    return pd.Series(pd.NaT, index=df.index)


def current_weather_context(snapshot):
    """Summarize recent weather reports without pretending they are GN-specific."""
    base = {
        'context': 'No matched current weather report',
        'resolution': 'Report-level context; not GN-specific',
        'latest_datetime': pd.NaT,
        'age_hours': None,
    }
    if not WEATHER_CLEAN.exists():
        return base

    try:
        d = pd.read_csv(WEATHER_CLEAN)
    except Exception:
        return base
    if d.empty:
        return base

    snap = pd.to_datetime(snapshot, errors='coerce')
    if pd.isna(snap):
        return base

    d = d.copy()
    d['__dt'] = _report_datetime(d)
    recent = d[
        d['__dt'].notna()
        & (d['__dt'] <= snap)
        & (d['__dt'] >= snap - pd.Timedelta(hours=24))
    ].copy()
    if recent.empty:
        return base

    latest_dt = recent['__dt'].max()
    age = (snap - latest_dt).total_seconds() / 3600.0

    flags = []
    flag_map = [
        ('heavy_rain_flag', 'Heavy rain'),
        ('thunderstorm_flag', 'Thunderstorm'),
        ('strong_wind_flag', 'Strong wind'),
        ('low_pressure_flag', 'Low pressure'),
        ('depression_flag', 'Depression'),
        ('cyclone_flag', 'Cyclone'),
    ]
    for col, label in flag_map:
        if col in recent.columns:
            vals = recent[col]
            if vals.dtype == bool:
                active = bool(vals.fillna(False).any())
            else:
                active = vals.astype(str).str.strip().str.casefold().isin(
                    ['true','1','yes','y']
                ).any()
            if active:
                flags.append(label)

    rain_col = _detect_col(recent.columns, ['rainfall threshold mm', 'rainfall threshold'])
    wind_col = _detect_col(recent.columns, ['max wind kmph', 'max wind'])
    extra = []
    if rain_col:
        mx = pd.to_numeric(recent[rain_col], errors='coerce').max()
        if pd.notna(mx):
            extra.append(f'max rainfall threshold {mx:g} mm')
    if wind_col:
        mx = pd.to_numeric(recent[wind_col], errors='coerce').max()
        if pd.notna(mx):
            extra.append(f'max wind {mx:g} km/h')

    parts = flags + extra
    if not parts:
        parts = [f'{len(recent)} weather report(s) in previous 24 h']

    return {
        'context': '; '.join(parts),
        'resolution': 'Report-level / national-regional context; not GN-specific',
        'latest_datetime': latest_dt,
        'age_hours': age,
    }


def current_river_station_context(snapshot):
    """Summarize the current river-station network without assigning it to a GN."""
    base = {
        'context': 'No matched current river-station observation',
        'resolution': 'Station-network context; not GN-resolved',
        'latest_datetime': pd.NaT,
        'age_hours': None,
    }
    if not RIVER_STATIONS.exists():
        return base

    try:
        d = pd.read_csv(RIVER_STATIONS)
    except Exception:
        return base
    if d.empty:
        return base

    snap = pd.to_datetime(snapshot, errors='coerce')
    if pd.isna(snap):
        return base

    d = d.copy()
    d['__dt'] = _report_datetime(d)
    d = d[d['__dt'].notna() & (d['__dt'] <= snap)].copy()
    if d.empty:
        return base

    latest_dt = d['__dt'].max()
    age = (snap - latest_dt).total_seconds() / 3600.0
    if age > 24:
        return {
            'context': f'Latest river-station observations are stale ({age:.1f} h old)',
            'resolution': 'Station-network context; not GN-resolved',
            'latest_datetime': latest_dt,
            'age_hours': age,
        }

    latest = d[d['__dt'] == latest_dt].copy()
    status_col = _detect_col(latest.columns, ['status'])
    if status_col:
        status = latest[status_col].fillna('').astype(str).str.strip().str.casefold()
        alert = int(status.isin(['alert','minor flood','major flood']).sum())
        major = int((status == 'major flood').sum())
        minor = int((status == 'minor flood').sum())
        context = (
            f'{len(latest)} station observation(s); '
            f'{alert} at Alert/Minor/Major Flood'
        )
        if major or minor:
            context += f' ({minor} minor, {major} major flood)'
    else:
        context = f'{len(latest)} river-station observation(s)'

    return {
        'context': context,
        'resolution': 'Station-network summary; not assigned to this GN',
        'latest_datetime': latest_dt,
        'age_hours': age,
    }

def build_gn_map(selected_district, snapshot):
    try:
        import folium
    except ImportError as e:
        raise RuntimeError('Install Folium: pip install folium') from e

    g, src = load_gn_boundaries(selected_district)
    district = g[g['district_key'] == _norm(selected_district)].copy()
    if district.empty:
        raise RuntimeError(f"No GN polygons matched district '{selected_district}' in {src.name}")

    explicit = explicit_gn_context(snapshot)
    ls = current_landslide_dsd_context(snapshot)
    weather_ctx = current_weather_context(snapshot)
    river_ctx = current_river_station_context(snapshot)

    exp_lookup = {}
    if not explicit.empty:
        explicit['match_key'] = explicit['gn_key'] + '||' + explicit['dsd_key']
        exp_lookup = {k:v.to_dict('records') for k,v in explicit.groupby('match_key')}

    exact, dsd_only = {}, {}
    if not ls.empty:
        for _, r in ls.iterrows():
            if r['district_key']:
                k = r['district_key'] + '||' + r['dsd_key']
                exact[k] = max(float(r['level']), float(exact.get(k, 0)))
            else:
                # Only use a DSD-name-only fallback when the source row itself
                # genuinely lacks district information. This avoids matching a
                # same-named DSD in the wrong district.
                dsd_only[r['dsd_key']] = max(
                    float(r['level']),
                    float(dsd_only.get(r['dsd_key'], 0)),
                )

    def add_context(r):
        ex = exp_lookup.get(r['gn_key'] + '||' + r['dsd_key'], [])
        floods = [x for x in ex if x['hazard'] == 'Flood']
        ls_gn = [x for x in ex if x['hazard'] == 'Landslide']
        lk = r['district_key'] + '||' + r['dsd_key']
        ls_level = exact.get(lk, dsd_only.get(r['dsd_key'], 0))

        common = {
            'river_context': river_ctx['context'],
            'river_resolution': river_ctx['resolution'],
            'weather_context': weather_ctx['context'],
            'weather_resolution': weather_ctx['resolution'],
        }

        if floods:
            x = max(
                floods,
                key=lambda z: -1 if pd.isna(z['level']) else float(z['level'])
            )
            lev = 2 if pd.isna(x['level']) else int(x['level'])
            values = {
                'category': 'Explicit GN flood warning',
                'map_level': lev,
                'flood_context': x['label'],
                'flood_resolution': 'GN',
                'landslide_context': (
                    f'NBRO Level {int(ls_level)} warning at DSD level'
                    if ls_level
                    else (
                        'GN explicitly named in NBRO bulletin'
                        if ls_gn else 'No matched current warning'
                    )
                ),
                'landslide_resolution': (
                    'DSD (applied to GN only for display)'
                    if ls_level
                    else ('GN named in bulletin' if ls_gn else 'No matched current warning')
                ),
            }
            values.update(common)
            return pd.Series(values)

        if ls_level:
            values = {
                'category': 'DSD-level landslide warning',
                'map_level': int(ls_level),
                'flood_context': 'No explicit GN flood warning matched',
                'flood_resolution': 'No matched GN warning',
                'landslide_context': f'NBRO Level {int(ls_level)} warning',
                'landslide_resolution': 'DSD (applied to GN only for display)',
            }
            values.update(common)
            return pd.Series(values)

        if ls_gn:
            values = {
                'category': 'GN named in NBRO bulletin',
                'map_level': 1,
                'flood_context': 'No explicit GN flood warning matched',
                'flood_resolution': 'No matched GN warning',
                'landslide_context': 'GN explicitly named in NBRO material',
                'landslide_resolution': 'GN named in bulletin',
            }
            values.update(common)
            return pd.Series(values)

        values = {
            'category': 'No matched current warning',
            'map_level': 0,
            'flood_context': 'No explicit GN flood warning matched',
            'flood_resolution': 'No matched GN warning',
            'landslide_context': 'No matched current warning',
            'landslide_resolution': 'No matched current warning',
        }
        values.update(common)
        return pd.Series(values)

    district = pd.concat([district, district.apply(add_context, axis=1)], axis=1)
    b = district.total_bounds
    m = folium.Map(
        location=[(b[1] + b[3]) / 2, (b[0] + b[2]) / 2],
        zoom_start=9,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    def style(feature):
        p = feature['properties']
        category = p.get('category','')
        level = int(p.get('map_level',0) or 0)
        colors = {0:'#f7f7f7',1:'#f9e547',2:'#f0a34a',3:'#d9534f'}
        fill = colors.get(level,'#9e9e9e')
        if category == 'Explicit GN flood warning':
            opacity, weight = .82, 1.4
        elif category == 'DSD-level landslide warning':
            opacity, weight = .42, .9
        elif category == 'GN named in NBRO bulletin':
            fill, opacity, weight = '#8e7cc3', .55, 1.0
        else:
            opacity, weight = .12, .5
        return {'fillColor':fill,'color':'#4a4a4a','weight':weight,'fillOpacity':opacity}

    view_cols = [
        'GN','DSD','DISTRICT','category',
        'flood_context','flood_resolution',
        'landslide_context','landslide_resolution',
        'river_context','river_resolution',
        'weather_context','weather_resolution',
        'map_level','geometry'
    ]
    folium.GeoJson(
        district[view_cols].to_json(),
        name='GN Hazard Context',
        style_function=style,
        tooltip=folium.GeoJsonTooltip(fields=['GN','DSD','category'], aliases=['GN:','DSD:','Context:'], sticky=False),
        popup=folium.GeoJsonPopup(
            fields=[
                'GN','DSD','DISTRICT','category',
                'flood_context','flood_resolution',
                'landslide_context','landslide_resolution',
                'river_context','river_resolution',
                'weather_context','weather_resolution',
            ],
            aliases=[
                'GN Division:','DS Division:','District:','Displayed map context:',
                'Flood:','Flood spatial resolution:',
                'Landslide:','Landslide spatial resolution:',
                'River stations:','River spatial resolution:',
                'Weather:','Weather spatial resolution:',
            ],
            localize=True,
            labels=True,
            sticky=False,
        )
    ).add_to(m)

    legend = '''<div style="position:fixed;bottom:22px;left:22px;z-index:9999;background:white;border:1px solid #777;border-radius:8px;padding:10px 12px;font-size:12px;line-height:1.5;box-shadow:0 1px 8px rgba(0,0,0,.18);max-width:410px"><b>GN / DSD hazard context</b><br><span style="display:inline-block;width:12px;height:12px;background:#d9534f"></span> Level 3 / high context<br><span style="display:inline-block;width:12px;height:12px;background:#f0a34a"></span> Level 2<br><span style="display:inline-block;width:12px;height:12px;background:#f9e547"></span> Level 1<br><span style="display:inline-block;width:12px;height:12px;background:#8e7cc3"></span> GN explicitly named in NBRO material<br><span style="display:inline-block;width:12px;height:12px;background:#f7f7f7;border:1px solid #777"></span> No matched current GN/DSD warning<br><br><b>Popup:</b> Flood and landslide retain their supported GN/DSD resolution. River and weather are shown separately as broader context and are not assigned to a GN unless a source explicitly supports that resolution.<br><b>Important:</b> lighter landslide fills are inherited from the parent DSD; they are not GN-specific forecasts.</div>'''
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=True).add_to(m)
    return m.get_root().render()


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--district', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--output', default='data/processed/floodimpact_lk_gn_hazard_map.html')
    a = p.parse_args()
    html = build_gn_map(a.district, a.snapshot)
    out = ROOT / a.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding='utf-8')
    print('GN map written to:', out)
