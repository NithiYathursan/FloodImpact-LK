from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import pandas as pd

# FloodImpact-LK
# SAFE INCREMENTAL OFFICIAL-DATA UPDATER
#
# IMPORTANT:
#
# - Does NOT retrain models
# - Does NOT run FloodImpact_LK_Final.ipynb
# - Does NOT overwrite dashboard outputs on scraper failure
# - Uses DMC metadata to distinguish:
#       Situation Report          -> INCLUDED
#       Drought Situation Report  -> EXCLUDED
# - Detects a newer valid national Situation Report
# - Runs latest_inference.py only when available


# 1. PROJECT PATHS

ROOT = Path(
    __file__
).resolve().parent


RAW_DIR = (
    ROOT
    / "data"
    / "raw"
)


PROCESSED_DIR = (
    ROOT
    / "data"
    / "processed"
)


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Dashboard

PAYLOAD_PATH = (
    PROCESSED_DIR
    / "dashboard_payload.json"
)


STATUS_PATH = (
    PROCESSED_DIR
    / "latest_update_status.json"
)

# Scrapers

DMC_SCRIPT = (
    ROOT
    / "scraping"
    / "scrape_dmc.py"
)


WEATHER_SCRIPT = (
    ROOT
    / "scraping"
    / "scrape_weather.py"
)


RIVER_SCRIPT = (
    ROOT
    / "scraping"
    / "scrape_river.py"
)


LANDSLIDE_SCRIPT = (
    ROOT
    / "scraping"
    / "scrape_landslide.py"
)

# DMC Situation metadata


DMC_METADATA_PATH = (
    RAW_DIR
    / "situation_reports"
    / "dmc_situation_reports_metadata.csv"
)

# Weather metadata

WEATHER_META = (
    RAW_DIR
    / "weather_reports"
    / "weather_reports_metadata.csv"
)
RIVER_META = (
    RAW_DIR
    / "river_reports"
    / "river_reports_metadata.csv"
)

LANDSLIDE_META = (
    RAW_DIR
    / "landslide_reports"
    / "landslide_reports_metadata.csv"
)

# Future inference pipeline

LATEST_INFERENCE_SCRIPT = (
    ROOT
    / "latest_inference.py"
)

# 2. GENERAL HELPERS

def now_iso():

    """
    Current Sri Lanka timestamp.
    """

    return (
        pd.Timestamp.now(
            tz="Asia/Colombo"
        )
        .isoformat()
    )


def write_status(
    state,
    message,
    **extra,
):

    """
    Write updater status.
    """

    data = {

        "updated_at":
            now_iso(),

        "state":
            state,

        "message":
            message,
    }


    data.update(
        extra
    )


    STATUS_PATH.write_text(

        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),

        encoding="utf-8",
    )

# 3. COMMAND RUNNER

def run_command(
    title,
    command,
    timeout_seconds,
):

    """
    Run a child process safely.

    stdout/stderr are shown only after the child finishes.
    """

    print(
        "\n"
        +
        "=" * 72
    )


    print(
        title
    )


    print(
        "=" * 72
    )


    print(
        "Command:",
        " ".join(
            str(x)
            for x
            in command
        )
    )


    try:

        result = subprocess.run(

            command,

            cwd=str(
                ROOT
            ),

            capture_output=True,

            text=True,

            timeout=
                timeout_seconds,
        )


    except subprocess.TimeoutExpired as exc:

        raise RuntimeError(

            f"{title} exceeded "
            f"{timeout_seconds} seconds."

        ) from exc

    # Show child output
    

    if result.stdout:

        print(
            "\n--- OUTPUT ---"
        )


        print(
            result.stdout[
                -12000:
            ]
        )

    if result.stderr:

        print(
            "\n--- STDERR ---"
        )


        print(
            result.stderr[
                -12000:
            ]
        )

    # Non-zero exit
    

    if result.returncode != 0:

        raise RuntimeError(

            f"{title} failed with "
            f"return code "
            f"{result.returncode}."
        )


    print(
        f"\n[OK] {title} completed"
    )

# 4. READ CURRENT DASHBOARD SNAPSHOT


def read_dashboard_snapshot():

    """
    Read current dashboard Situation Report timestamp.
    """

    if not PAYLOAD_PATH.exists():

        raise FileNotFoundError(

            "dashboard_payload.json "
            "was not found."
        )


    payload = json.loads(

        PAYLOAD_PATH.read_text(
            encoding="utf-8"
        )
    )


    snapshot = payload.get(
        "current_snapshot",
        {},
    )


    value = (

        snapshot.get(
            "snapshot_datetime"
        )

        or

        snapshot.get(
            "report_datetime"
        )

        or

        snapshot.get(
            "report_time"
        )
    )


    if value is None:

        raise ValueError(

            "Dashboard snapshot datetime "
            "was not found."
        )


    timestamp = pd.to_datetime(
        value,
        errors="coerce",
    )


    if pd.isna(
        timestamp
    ):

        raise ValueError(

            "Dashboard snapshot datetime "
            "could not be parsed."
        )

    # Make comparison timezone-neutral
    # because report timestamps are stored as local naive
    # datetimes.
    

    if timestamp.tzinfo is not None:

        timestamp = (
            timestamp
            .tz_localize(
                None
            )
        )


    return timestamp

# 5. LATEST VALID GENERAL DMC SITUATION REPORT


def latest_raw_situation_report():

    """
    Return latest GENERAL DMC Situation Report.

    CRITICAL:

    DMC may publish multiple reports with the same timestamp.

    Example:

        Situation Report
        Drought Situation Report

    The model scope must use the normal Situation Report
    and exclude Drought Situation Reports.

    Therefore selection is based on DMC metadata TITLE,
    not filename timestamp alone.
    """

    situation_folder = (
        RAW_DIR
        / "situation_reports"
    )

    # Metadata required

    if not DMC_METADATA_PATH.exists():

        raise FileNotFoundError(

            "DMC Situation Report metadata "
            "was not found:\n"
            f"{DMC_METADATA_PATH}"
        )


    metadata = pd.read_csv(
        DMC_METADATA_PATH
    )


    required_columns = {

        "title",
        "date",
        "time",
        "pdf_url",
    }


    missing_columns = (

        required_columns

        -

        set(
            metadata.columns
        )
    )


    if missing_columns:

        raise ValueError(

            "DMC metadata is missing "
            "required columns: "
            f"{sorted(missing_columns)}"
        )


    # 5.1 Keep ONLY exact normal Situation Report
    # Included:Situation Report
    # Excluded:Drought Situation Report, Daily Disaster Incident Report,Other archive documents
    

    titles = (

        metadata[
            "title"
        ]

        .fillna("")

        .astype(str)

        .str.strip()

        .str.casefold()
    )


    normal_reports = (

        metadata[
            titles
            ==
            "situation report"
        ]

        .copy()
    )


    if normal_reports.empty:

        return (
            pd.NaT,
            None,
        )

    # 5.2 Parse report timestamp

    normal_reports[
        "_datetime"
    ] = pd.to_datetime(

        (
            normal_reports[
                "date"
            ]
            .astype(str)
            .str.strip()

            +

            " "

            +

            normal_reports[
                "time"
            ]
            .astype(str)
            .str.strip()
        ),

        errors="coerce",
    )


    normal_reports = (

        normal_reports

        .dropna(
            subset=[
                "_datetime"
            ]
        )

        .sort_values(
            "_datetime"
        )
    )


    if normal_reports.empty:

        return (
            pd.NaT,
            None,
        )

    # Latest general report metadata row

    latest_row = (
        normal_reports.iloc[
            -1
        ]
    )


    latest_time = (
        latest_row[
            "_datetime"
        ]
    )


    pdf_url = str(
        latest_row[
            "pdf_url"
        ]
    )

    # 5.3 Recreate scraper filename
    # Local Situation Report filenames use:situation_YYYYMMDD_HHMM_<SHA1-8>.pdf

    url_hash = (

        hashlib.sha1(

            pdf_url.encode(
                "utf-8"
            )

        )

        .hexdigest()[
            :8
        ]
    )


    date_text = (

        latest_time

        .strftime(
            "%Y%m%d"
        )
    )


    time_text = (

        latest_time

        .strftime(
            "%H%M"
        )
    )


    filename = (

        f"situation_"
        f"{date_text}_"
        f"{time_text}_"
        f"{url_hash}.pdf"
    )


    path = (

        situation_folder
        / filename
    )

    # 5.4 Safety validation

    if not path.exists():

        raise FileNotFoundError(

            "Latest general Situation Report "
            "exists in metadata but its local PDF "
            "was not found:\n"
            f"{path}"
        )


    return (
        latest_time,
        path,
    )

# 6. RAW PDF SNAPSHOT

def pdf_snapshot():

    """
    Store current PDF paths and sizes.

    Used to identify new or changed raw PDFs.
    """

    output = {}


    if not RAW_DIR.exists():

        return output


    for path in RAW_DIR.rglob(
        "*.pdf"
    ):

        try:

            relative = str(

                path.relative_to(
                    ROOT
                )
            )


            output[
                relative
            ] = (
                path.stat()
                .st_size
            )


        except Exception:

            pass


    return output


def changed_pdf_files(
    before,
    after,
):

    """
    Compare raw PDF snapshots.
    """

    new_files = sorted(

        set(
            after
        )

        -

        set(
            before
        )
    )


    changed_files = sorted(

        path

        for path in (

            set(
                before
            )

            &

            set(
                after
            )
        )

        if (

            before[
                path
            ]

            !=

            after[
                path
            ]
        )
    )


    return (
        new_files,
        changed_files,
    )

# 7. RESTORE FILE HELPER

def restore_file(
    path,
    previous_bytes,
):

    """
    Restore a file after a failed temporary refresh.
    """

    if previous_bytes is None:

        path.unlink(
            missing_ok=True
        )


    else:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        path.write_bytes(
            previous_bytes
        )

# 8. SAFE RECENT WEATHER REFRESH

def refresh_weather_recent(
    dashboard_snapshot,
):

    """
    Refresh only recent Weather metadata/PDFs.

    The scraper's temporary recent metadata file is merged
    back with the previously stored full metadata archive.

    This avoids replacing the historical metadata dataset
    with only recent rows.
    """

    if not WEATHER_SCRIPT.exists():

        raise FileNotFoundError(

            f"Missing weather scraper: "
            f"{WEATHER_SCRIPT}"
        )

    # Backup full metadata

    old_bytes = (

        WEATHER_META.read_bytes()

        if WEATHER_META.exists()

        else None
    )


    old_df = pd.DataFrame()


    if WEATHER_META.exists():

        try:

            old_df = pd.read_csv(
                WEATHER_META
            )


        except Exception:

            old_df = (
                pd.DataFrame()
            )


    # =====================================================
    # Use a 14-day overlap before current dashboard report.
    #
    # This protects against:
    # - delayed publication
    # - late metadata changes
    # - reports slightly before snapshot
    # =====================================================

    from_date = (

        dashboard_snapshot

        -

        pd.Timedelta(
            days=14
        )

    ).strftime(
        "%Y-%m-%d"
    )


    try:

        
        # STEP 1
        # Scrape only recent metadata pages.

        run_command(

            "Weather Recent Metadata",

            [
                sys.executable,

                str(
                    WEATHER_SCRIPT
                ),

                "--metadata-only",

                "--max-pages",
                "40",
            ],

            timeout_seconds=
                10 * 60,
        )


        if not WEATHER_META.exists():

            raise RuntimeError(

                "Weather scraper completed "
                "but metadata CSV was not created."
            )


        recent_df = pd.read_csv(
            WEATHER_META
        )


        # STEP 2
        # Merge historical full metadata with recently scraped metadata.
      

        frames = [

            dataframe

            for dataframe in [

                old_df,
                recent_df,
            ]

            if not dataframe.empty
        ]


        if not frames:

            raise RuntimeError(

                "No weather metadata "
                "available after refresh."
            )


        merged = pd.concat(

            frames,

            ignore_index=True,
        )


        if "pdf_url" not in merged.columns:

            raise RuntimeError(

                "Weather metadata does not "
                "contain pdf_url."
            )


        merged = (

            merged

            .drop_duplicates(

                subset=[
                    "pdf_url"
                ],

                keep="last",
            )
        )

        # Sort newest first

        if "date" in merged.columns:

            merged[
                "_date_sort"
            ] = pd.to_datetime(

                merged[
                    "date"
                ],

                errors="coerce",
            )


            sort_columns = [
                "_date_sort"
            ]


            ascending = [
                False
            ]


            if "time" in merged.columns:

                sort_columns.append(
                    "time"
                )


                ascending.append(
                    False
                )


            merged = (

                merged

                .sort_values(
                    sort_columns,
                    ascending=ascending,
                )

                .drop(
                    columns=[
                        "_date_sort"
                    ]
                )
            )


        WEATHER_META.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        merged.to_csv(
            WEATHER_META,
            index=False,
        )


        print(
            "\nWeather metadata merged safely."
        )


        print(
            "Full metadata rows:",
            len(
                merged
            )
        )


        print(
            "Recent download start:",
            from_date
        )

        # STEP 3
        # Download only recent project-relevant PDFs.

        run_command(

            "Weather Recent PDF Download",

            [
                sys.executable,

                str(
                    WEATHER_SCRIPT
                ),

                "--download-only",

                "--from-date",
                from_date,
            ],

            timeout_seconds=
                10 * 60,
        )


    except Exception:

        print(
            "\nRestoring previous "
            "weather metadata after failure..."
        )


        restore_file(
            WEATHER_META,
            old_bytes,
        )


        raise

# 9. REFRESH ALL OFFICIAL SOURCES

def refresh_sources(
    dashboard_snapshot,
):

    """
    Refresh all four official-data sources.

    Full metadata coverage is checked so that reports
    available up to the refresh time are not intentionally
    excluded by a short date window.

    Existing valid PDFs are skipped by each scraper.
    Only new / missing PDFs are downloaded.

    Historical data is preserved.
    """

    required = [

        (
            DMC_SCRIPT,
            "DMC",
        ),

        (
            WEATHER_SCRIPT,
            "Weather",
        ),

        (
            RIVER_SCRIPT,
            "River",
        ),

        (
            LANDSLIDE_SCRIPT,
            "Landslide",
        ),
    ]


    for path, name in required:

        if not path.exists():

            raise FileNotFoundError(

                f"Missing {name} "
                f"scraper: {path}"
            )


    # DMC Situation Reports
    # Full metadata check + new/missing PDFs only

    run_command(

        "DMC Situation Reports",

        [
            sys.executable,

            str(
                DMC_SCRIPT
            ),
        ],

        timeout_seconds=
            25 * 60,
    )

    # Weather Reports
    # Full metadata check + relevant new/missing PDFs only

    run_command(

        "Weather Reports",

        [
            sys.executable,

            str(
                WEATHER_SCRIPT
            ),
        ],

        timeout_seconds=
            25 * 60,
    )

    # River Reports
    # Full metadata check + relevant new/missing PDFs only

    run_command(

        "River Reports",

        [
            sys.executable,

            str(
                RIVER_SCRIPT
            ),
        ],

        timeout_seconds=
            20 * 60,
    )

    # Landslide Reports
    # Full metadata check + English/project-relevant
    # new/missing PDFs only

    run_command(

        "Landslide Reports",

        [
            sys.executable,

            str(
                LANDSLIDE_SCRIPT
            ),
        ],

        timeout_seconds=
            20 * 60,
    )

# 10. LATEST INFERENCE

def run_latest_inference():

    """
    Run safe current-report inference.

    This script must use already-trained deployment models.

    It must NOT retrain models.
    """

    run_command(

        "Latest Safe Processing / Inference",

        [
            sys.executable,

            str(
                LATEST_INFERENCE_SCRIPT
            ),

            "--once",
        ],

        timeout_seconds=
            30 * 60,
    )

# 11. MAIN UPDATE

def update_once():

    print(
        "\n"
        +
        "=" * 72
    )


    print(
        "FLOODIMPACT-LK "
        "LATEST OFFICIAL-DATA CHECK"
    )


    print(
        "=" * 72
    )


    print(
        "Started:",
        now_iso()
    )


    try:

        # 11.1 Current dashboard snapshot

        dashboard_snapshot = (
            read_dashboard_snapshot()
        )


        # 11.2 Latest valid general DMC report BEFORE refreshing sources.
      

        latest_before, path_before = (
            latest_raw_situation_report()
        )


        print(
            "\nDashboard snapshot:",
            dashboard_snapshot,
        )


        print(
            "Latest raw GENERAL Situation Report "
            "before refresh:",
            latest_before,
        )


        if path_before is not None:

            print(
                "Latest raw file:",
                path_before.name,
            )

        # 11.3 Raw PDF snapshot before refresh

        before_pdfs = (
            pdf_snapshot()
        )


        write_status(

            state="running",

            message=(
                "Checking latest "
                "official data."
            ),

            dashboard_snapshot=
                dashboard_snapshot,

            latest_raw_before=
                latest_before,
        )


        # 11.4 Refresh official sources

        refresh_sources(
            dashboard_snapshot
        )

        # 11.5 Compare raw PDFs

        after_pdfs = (
            pdf_snapshot()
        )


        (
            new_pdfs,
            changed_pdfs,
        ) = changed_pdf_files(

            before_pdfs,
            after_pdfs,
        )


      
        # 11.6 Select latest valid GENERAL Situation Report
        # Drought report is automatically excluded.
    

        latest_after, path_after = (
            latest_raw_situation_report()
        )


        print(
            "\n"
            +
            "=" * 72
        )


        print(
            "LATEST-DATA AUDIT"
        )


        print(
            "=" * 72
        )


        print(
            "Dashboard snapshot:",
            dashboard_snapshot,
        )


        print(
            "Latest raw GENERAL Situation Report:",
            latest_after,
        )


        print(
            "New PDF files this run:",
            len(
                new_pdfs
            ),
        )


        print(
            "Changed PDF files this run:",
            len(
                changed_pdfs
            ),
        )


        if path_after is not None:

            print(
                "Latest raw GENERAL Situation file:",
                path_after.name,
            )


        # 11.7 Is a NEW national forecasting report waiting?
       
        # IMPORTANT:This comparison works even when the PDF was downloaded on a PREVIOUS updater run.
        
        pending_situation = (

            pd.notna(
                latest_after
            )

            and

            latest_after
            >
            dashboard_snapshot
        )


        if pending_situation:

            print(
                "\n[OK] A newer GENERAL Situation Report "
                "exists than the current "
                "dashboard snapshot."
            )


        # =================================================
        # 11.8 No changes
        # =================================================

        if (

            not pending_situation

            and

            not new_pdfs

            and

            not changed_pdfs
        ):

            write_status(

                state="no_new_data",

                message=(

                    "Official sources checked. "
                    "No newer general Situation Report "
                    "or new raw PDFs detected."
                ),

                dashboard_snapshot=
                    dashboard_snapshot,

                latest_raw=
                    latest_after,
            )


            print(
                "\nNO NEW OFFICIAL DATA "
                "REQUIRING A FORECAST UPDATE"
            )


            return 0

        # 11.9 New GENERAL Situation Report+ inference pipeline available
        

        if (

            pending_situation

            and

            LATEST_INFERENCE_SCRIPT.exists()
        ):

            run_latest_inference()


            write_status(

                state="success",

                message=(

                    "Newer general Situation Report "
                    "processed successfully."
                ),

                dashboard_snapshot_before=
                    dashboard_snapshot,

                latest_raw=
                    latest_after,

                latest_raw_file=(
                    path_after.name
                    if path_after is not None
                    else None
                ),

                new_pdf_count=
                    len(
                        new_pdfs
                    ),

                changed_pdf_count=
                    len(
                        changed_pdfs
                    ),
            )


            print(
                "\nLATEST DASHBOARD "
                "UPDATE COMPLETE"
            )


            return 0
        
        # 11.10 SAFE STOP
        #
        # Raw source refresh succeeded,
        # but latest inference is not yet connected.
        #
        # Preserve old dashboard.
        

        write_status(

            state=(
                "new_data_detected_"
                "awaiting_processing"
            ),

            message=(

                "New official raw data is available, "
                "but the safe latest inference pipeline "
                "is not connected yet. "
                "Previous dashboard preserved."
            ),

            dashboard_snapshot=
                dashboard_snapshot,

            latest_raw=
                latest_after,

            latest_raw_file=(

                path_after.name

                if path_after is not None

                else None
            ),

            pending_new_situation=
                bool(
                    pending_situation
                ),

            new_pdf_count=
                len(
                    new_pdfs
                ),

            changed_pdf_count=
                len(
                    changed_pdfs
                ),

            new_pdfs=
                new_pdfs[
                    :100
                ],

            changed_pdfs=
                changed_pdfs[
                    :100
                ],
        )


        print(
            "\n"
            +
            "=" * 72
        )


        print(
            "SAFE STOP"
        )


        print(
            "=" * 72
        )


        if pending_situation:

            print(
                "A newer GENERAL Situation Report "
                "is waiting to be processed."
            )


        else:

            print(
                "Supporting-source data changed, "
                "but no newer GENERAL Situation Report "
                "was detected."
            )


        print(
            "Previous valid dashboard "
            "was preserved."
        )


        if not LATEST_INFERENCE_SCRIPT.exists():

            print(
                "Next required component: "
                "latest_inference.py"
            )


        return 2

    # Any failure:
    # preserve current dashboard.

    except Exception as exc:

        write_status(

            state="failed",

            message=str(
                exc
            ),
        )


        print(
            "\n"
            +
            "=" * 72
        )


        print(
            "UPDATE FAILED"
        )


        print(
            "=" * 72
        )


        print(
            str(
                exc
            )
        )


        print(
            "\nPrevious valid dashboard "
            "outputs were preserved."
        )


        return 1

# 12. COMMAND LINE

def main():

    parser = argparse.ArgumentParser(

        description=(

            "Safely check FloodImpact-LK "
            "official sources for new data."
        )
    )


    parser.add_argument(

        "--once",

        action="store_true",

        help=(
            "Run one update check."
        ),
    )


    parser.parse_args()


    return (
        update_once()
    )

# 13. ENTRY POINT

if __name__ == "__main__":

    sys.exit(
        main()
    )