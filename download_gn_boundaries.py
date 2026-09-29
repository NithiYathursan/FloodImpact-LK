from pathlib import Path
import json
import math
import time

import requests

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "boundaries"
OUT_FILE = OUT_DIR / "sri_lanka_gn_boundaries.geojson"

SERVICE = (
    "https://lis.survey.gov.lk/server/rest/services/"
    "Admin_boundaries/FeatureServer/0"
)

APP_REFERER = (
    "https://lis.survey.gov.lk/portal/apps/webappviewer/"
    "index.html?id=1f452b1748244ecf8a34699f28b18f94"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/142 Safari/537.36"
    ),
    "Referer": APP_REFERER,
    "Accept": "application/json,text/plain,*/*",
}


def get_json(url, params=None, timeout=60):
    r = requests.get(
        url,
        params=params,
        headers=HEADERS,
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "error" in data:
        raise RuntimeError(
            "ArcGIS service error: "
            + json.dumps(data["error"], ensure_ascii=False)
        )
    return data


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FLOODIMPACT-LK | OFFICIAL GN BOUNDARY DOWNLOAD")
    print("=" * 72)
    print("Service:", SERVICE)

    # 1. Read public layer metadata.
    meta = get_json(
        SERVICE,
        params={"f": "json"},
    )

    print("Layer name:", meta.get("name"))
    print("Geometry type:", meta.get("geometryType"))

    object_id_field = (
        meta.get("objectIdField")
        or meta.get("objectIdFieldName")
        or "OBJECTID"
    )

    max_record_count = int(
        meta.get("maxRecordCount") or 1000
    )

    # Use conservative batches even if the server allows more.
    batch_size = min(max_record_count, 500)

    print("Object ID field:", object_id_field)
    print("Server max record count:", max_record_count)
    print("Batch size:", batch_size)

    # 2. Request all feature IDs first, which avoids truncation.
    ids_payload = get_json(
        SERVICE + "/query",
        params={
            "where": "1=1",
            "returnIdsOnly": "true",
            "f": "json",
        },
    )

    object_ids = ids_payload.get("objectIds") or []
    if not object_ids:
        raise RuntimeError(
            "No GN feature IDs were returned by the service."
        )

    object_ids = sorted(
        int(x) for x in object_ids
    )

    print("GN feature IDs returned:", len(object_ids))

    # 3. Download all geometry + attributes in batches as GeoJSON.
    all_features = []
    total_batches = math.ceil(len(object_ids) / batch_size)

    for i in range(0, len(object_ids), batch_size):
        batch_no = i // batch_size + 1
        ids = object_ids[i:i + batch_size]

        print(
            f"[{batch_no}/{total_batches}] "
            f"Downloading {len(ids)} GN polygons..."
        )

        params = {
            "objectIds": ",".join(map(str, ids)),
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "4326",
            "f": "geojson",
        }

        r = requests.get(
            SERVICE + "/query",
            params=params,
            headers=HEADERS,
            timeout=120,
        )
        r.raise_for_status()

        try:
            payload = r.json()
        except Exception as exc:
            raise RuntimeError(
                "The GN service did not return valid JSON/GeoJSON "
                f"for batch {batch_no}."
            ) from exc

        if "error" in payload:
            raise RuntimeError(
                "ArcGIS service error in batch "
                f"{batch_no}: "
                + json.dumps(
                    payload["error"],
                    ensure_ascii=False,
                )
            )

        features = payload.get("features") or []
        all_features.extend(features)

        time.sleep(0.15)

    # 4. Save one clean GeoJSON file for the project.
    feature_collection = {
        "type": "FeatureCollection",
        "name": "Sri_Lanka_GN_Boundaries",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            },
        },
        "features": all_features,
    }

    temp_file = OUT_FILE.with_suffix(".geojson.tmp")
    temp_file.write_text(
        json.dumps(
            feature_collection,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temp_file.replace(OUT_FILE)

    print()
    print("=" * 72)
    print("DOWNLOAD COMPLETE")
    print("=" * 72)
    print("Saved:", OUT_FILE)
    print("Downloaded polygons:", len(all_features))

    # 5. Lightweight attribute validation.
    props = [
        f.get("properties", {})
        for f in all_features
    ]

    required = [
        "PROVINCE_N",
        "DISTRICT_N",
        "DSD_N",
        "GND_N",
        "GND_NO",
    ]

    available = set()
    for p in props[:50]:
        available.update(p.keys())

    print("Required fields found:")
    for field in required:
        print(
            f"  {field}:",
            "YES" if field in available else "NO",
        )

    districts = sorted({
        str(p.get("DISTRICT_N")).strip()
        for p in props
        if p.get("DISTRICT_N") not in (None, "")
    })

    print("District count:", len(districts))
    print("Districts:", districts)

    if len(all_features) != len(object_ids):
        print(
            "WARNING: downloaded polygon count does not exactly "
            "match the returned feature-ID count."
        )

    if len(districts) != 25:
        print(
            "WARNING: expected 25 districts. "
            "Please inspect the downloaded attributes."
        )


if __name__ == "__main__":
    main()
