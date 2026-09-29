from pathlib import Path
import re

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "boundaries" / "sri_lanka_gn_boundaries.geojson"
OUT_DIR = ROOT / "data" / "boundaries" / "gn_by_district"
INDEX_FILE = OUT_DIR / "index.csv"


def safe_name(value):
    s = str(value).strip().upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return s.strip("_")


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading national GN boundary file once...")
    gdf = gpd.read_file(SOURCE)

    if "DISTRICT_N" not in gdf.columns:
        raise RuntimeError(
            "DISTRICT_N field was not found. "
            f"Available fields: {list(gdf.columns)}"
        )

    if gdf.crs is None:
        raise RuntimeError("Boundary CRS is missing.")

    gdf = gdf.to_crs(epsg=4326)

    district_values = (
        gdf["DISTRICT_N"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    districts = sorted(x for x in district_values.unique() if x)

    print(f"Districts found: {len(districts)}")

    rows = []
    total_written = 0

    for district in districts:
        part = gdf[
            gdf["DISTRICT_N"].astype(str).str.strip() == district
        ].copy()

        file_name = f"{safe_name(district)}.geojson"
        out = OUT_DIR / file_name

        print(
            f"Writing {district}: "
            f"{len(part)} GN polygons -> {out.name}"
        )

        part.to_file(
            out,
            driver="GeoJSON",
            index=False,
        )

        rows.append(
            {
                "district": district,
                "file_name": file_name,
                "gn_polygons": len(part),
            }
        )
        total_written += len(part)

    pd.DataFrame(rows).to_csv(INDEX_FILE, index=False)

    print()
    print("=" * 70)
    print("GN DISTRICT SPLIT COMPLETE")
    print("=" * 70)
    print("Output folder:", OUT_DIR)
    print("Index file:", INDEX_FILE)
    print("District files:", len(districts))
    print("Total polygons:", total_written)


if __name__ == "__main__":
    main()
