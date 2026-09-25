import argparse
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

# PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "landslide_reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_FILE = (
    OUTPUT_DIR
    / "landslide_reports_metadata.csv"
)

FAILED_FILE = (
    OUTPUT_DIR
    / "failed_downloads.csv"
)

# DMC LANDSLIDE WARNING ARCHIVE

BASE_URL = "https://www.dmc.gov.lk"

ARCHIVE_URL = (
    "https://www.dmc.gov.lk/index.php"
    "?Itemid=276"
    "&lang=en"
    "&option=com_dmcreports"
    "&report_type_id=5"
    "&view=reports"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(HEADERS)

# BASIC EXTRACTION

def extract_date(text):

    match = re.search(
        r"\b20\d{2}-\d{2}-\d{2}\b",
        text
    )

    if match:
        return match.group()

    return None


def extract_time(text):

    match = re.search(
        r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b",
        text
    )

    if match:
        return match.group()

    return None


# LANGUAGE DETECTION

def detect_language(title):

    text = title.lower()

    if (
        "tamil" in text
        or re.search(r"[\u0B80-\u0BFF]", title)
    ):
        return "Tamil"

    if (
        "sinhala" in text
        or re.search(r"[\u0D80-\u0DFF]", title)
    ):
        return "Sinhala"

    return "English"

# LANDSLIDE REPORT CLASSIFICATION

def classify_landslide_report(title):

    text = title.lower().strip()

    # Common typo found in archives
    normalized = (
        text
        .replace("lanslide", "landslide")
        .replace("land slide", "landslide")
    )

    if (
        "landslide early warning" in normalized
        or "landslide early-warning" in normalized
        or (
            "landslide" in normalized
            and "early warning" in normalized
        )
    ):
        return "early_warning"

    if (
        "landslide ew report" in normalized
        or (
            "landslide" in normalized
            and "ew report" in normalized
        )
    ):
        return "early_warning"

    if (
        "landslide warning" in normalized
        or (
            "landslide" in normalized
            and "warning" in normalized
        )
    ):
        return "warning"

    if "landslide" in normalized:
        return "landslide_related"

    return "other"

# PROJECT RELEVANCE

def is_project_relevant(title, report_type):

    text = title.lower()

    # Remove unrelated reports accidentally appearing
    # inside the DMC Landslide archive.
    exclusion_terms = [
        "air quality",
        "weather forecast",
        "weather forcast",
        "marine",
        "rough sea",
        "tsunami",
        "earthquake",
        "drought",
    ]

    if any(
        term in text
        for term in exclusion_terms
    ):
        return False

    relevant_types = {
        "early_warning",
        "warning",
        "landslide_related",
    }

    return report_type in relevant_types


# SCRAPE ONE PAGE

def scrape_page(offset):

    url = (
        ARCHIVE_URL
        + f"&limitstart={offset}"
    )

    response = session.get(
        url,
        timeout=40
    )

    response.raise_for_status()

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    records = []

    for row in soup.find_all("tr"):

        row_text = " ".join(
            row.stripped_strings
        )

        report_date = extract_date(
            row_text
        )

        if not report_date:
            continue

        # PDF URL

        pdf_url = None

        for link in row.find_all(
            "a",
            href=True
        ):

            href = link["href"]

            if ".pdf" in href.lower():

                pdf_url = urljoin(
                    BASE_URL,
                    href
                )

                break

        if not pdf_url:
            continue

        # Metadata

        title = row_text.split(
            report_date
        )[0].strip()

        report_time = extract_time(
            row_text
        )

        language = detect_language(
            title
        )

        report_type = classify_landslide_report(
            title
        )

        relevant = is_project_relevant(
            title,
            report_type
        )

        records.append(
            {
                "title": title,
                "date": report_date,
                "time": report_time,
                "language": language,
                "report_type": report_type,
                "project_relevant": relevant,
                "source": "Sri Lanka Disaster Management Centre",
                "pdf_url": pdf_url,
                "archive_offset": offset,
            }
        )

    return records

# SCRAPE FULL METADATA

def scrape_all_metadata(
    delay=0.3,
    max_pages=None
):

    print(
        "\nStarting DMC Landslide Warning "
        "metadata scraping...\n"
    )

    all_records = []
    progress_file = METADATA_FILE.with_name(
        "landslide_reports_metadata_progress.csv"
    )
    offset = 0
    page_number = 1

    PAGE_SIZE = 10

    consecutive_empty_pages = 0

    while True:

        if (
            max_pages is not None
            and page_number > max_pages
        ):
            break

        print(
            f"Page {page_number} | offset={offset}"
        )

        try:

            records = scrape_page(
                offset
            )

        except Exception as e:

            print(
                "Request failed:",
                e
            )

            print(
                "Retrying in 5 seconds..."
            )

            time.sleep(5)

            try:

                records = scrape_page(
                    offset
                )

            except Exception as retry_error:

                print(
                    "Retry failed:",
                    retry_error
                )

                raise RuntimeError(
                    "Landslide metadata scraping was interrupted "
                    f"at page {page_number}, offset={offset}. "
                    "Canonical metadata was NOT replaced."
                ) from retry_error

        if len(records) == 0:

            consecutive_empty_pages += 1

        else:

            consecutive_empty_pages = 0

        if consecutive_empty_pages >= 3:

            print(
                "\nEnd of landslide archive detected."
            )

            break

        all_records.extend(
            records
        )

        # Periodic save

        if (
            page_number % 50 == 0
            and all_records
        ):

            temp_df = pd.DataFrame(
                all_records
            )

            temp_df = temp_df.drop_duplicates(
                subset=["pdf_url"]
            )

            temp_df.to_csv(
                progress_file,
                index=False
            )

            print(
                f"\nProgress saved: "
                f"{len(temp_df)} records\n"
            )

        offset += PAGE_SIZE
        page_number += 1

        time.sleep(delay)

    # CLEAN METADATA

    df = pd.DataFrame(
        all_records
    )

    if df.empty:

        print(
            "\nNo landslide metadata collected."
        )

        return df

    df = (
        df
        .drop_duplicates(
            subset=["pdf_url"]
        )
        .copy()
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = (
        df
        .sort_values(
            ["date", "time"],
            ascending=[False, False]
        )
        .reset_index(drop=True)
    )
    temp_metadata_file = METADATA_FILE.with_name(
    "landslide_reports_metadata_tmp.csv"
    )

    df.to_csv(
        temp_metadata_file,
        index=False
    )

    temp_metadata_file.replace(
        METADATA_FILE
    )

    if progress_file.exists():
        progress_file.unlink()        
    

    print(
        "\n**********************************************"
    )

    print(
        "LANDSLIDE METADATA COLLECTION COMPLETED"
    )

    print(
        "**********************************************"
    )

    print(
        "Total reports:",
        len(df)
    )

    print(
        "English reports:",
        (
            df["language"] == "English"
        ).sum()
    )

    print(
        "Project-relevant reports:",
        df["project_relevant"].sum()
    )

    print(
        "Newest date:",
        df["date"].max()
    )

    print(
        "Oldest date:",
        df["date"].min()
    )

    print(
        "\nReport types:"
    )

    print(
        df["report_type"].value_counts()
    )

    print(
        "\nSaved to:"
    )

    print(
        METADATA_FILE
    )

    return df


# LOAD EXISTING METADATA

def load_existing_metadata():

    if not METADATA_FILE.exists():

        raise FileNotFoundError(
            "landslide_reports_metadata.csv "
            "not found. Run metadata scraping first."
        )

    df = pd.read_csv(
        METADATA_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    return df

# BUILD PDF FILE NAME

def build_filename(row):

    date_text = (
        row["date"]
        .strftime("%Y%m%d")
    )

    if pd.isna(
        row["time"]
    ):

        time_text = "unknown"

    else:

        time_text = (
            str(row["time"])
            .replace(":", "")
        )

    report_type = (
        str(row["report_type"])
        .replace(" ", "_")
    )

    url_hash = hashlib.sha1(
        row["pdf_url"].encode(
            "utf-8"
        )
    ).hexdigest()[:8]

    filename = (
        f"landslide_"
        f"{date_text}_"
        f"{time_text}_"
        f"{report_type}_"
        f"{url_hash}.pdf"
    )

    return filename

# DOWNLOAD PDFs

def download_pdfs(
    df,
    from_date=None,
    to_date=None,
    relevant_only=True,
    english_only=True,
    delay=0.25
):

    download_df = df.copy()

    # Date filtering

    if from_date:

        start = pd.to_datetime(
            from_date
        )

        download_df = download_df[
            download_df["date"] >= start
        ]

    if to_date:

        end = pd.to_datetime(
            to_date
        )

        download_df = download_df[
            download_df["date"] <= end
        ]

    # Project relevance

    if relevant_only:

        relevant_mask = (
            download_df[
                "project_relevant"
            ]
            .astype(str)
            .str.lower()
            .isin(
                ["true", "1"]
            )
        )

        download_df = download_df[
            relevant_mask
        ]

    
    # Language filtering
    

    if english_only:

        download_df = download_df[
            download_df["language"]
            == "English"
        ]

    download_df = (
        download_df
        .sort_values(
            ["date", "time"]
        )
        .reset_index(drop=True)
    )


# Identify only NEW / MISSING Landslide PDFs

    pending_downloads = []
    existing_count = 0

    for _, row in download_df.iterrows():

        filename = build_filename(
            row
        )

        file_path = (
            OUTPUT_DIR
            / filename
        )

        if (
            file_path.exists()
            and file_path.stat().st_size > 1000
        ):
            existing_count += 1
            continue

        pending_downloads.append(
            (
                row,
                filename,
                file_path,
            )
        )


    print(
        "\nLandslide PDF records in refreshed metadata:",
        len(download_df)
    )

    print(
        "Already available locally:",
        existing_count
    )

    print(
        "New / missing Landslide PDFs to download:",
        len(pending_downloads)
    )


    successful = existing_count
    failed = 0

    failed_records = []


    # Download only NEW / MISSING Landslide PDFs

    for index, (
        row,
        filename,
        file_path,
    ) in enumerate(
        pending_downloads,
        start=1,
    ):

        print(
            f"[{index}/{len(pending_downloads)}] "
            f"Downloading {filename}"
        )

        try:

            response = session.get(
                row["pdf_url"],
                timeout=60
            )

            response.raise_for_status()

            if not response.content.startswith(
                b"%PDF"
            ):

                print(
                    "  Invalid PDF response."
                )

                failed += 1

                failed_records.append(
                    {
                        "title": row["title"],
                        "date": row["date"],
                        "pdf_url": row["pdf_url"],
                        "reason": "Invalid PDF response",
                    }
                )

                continue

            file_path.write_bytes(
                response.content
            )

            successful += 1

        except Exception as e:

            print(
                "  Download failed:",
                e
            )

            failed += 1

            failed_records.append(
                {
                    "title": row["title"],
                    "date": row["date"],
                    "pdf_url": row["pdf_url"],
                    "reason": str(e),
                }
            )

        time.sleep(delay)

    # Failed downloads log

    if failed_records:

        pd.DataFrame(
            failed_records
        ).to_csv(
            FAILED_FILE,
            index=False
        )

    print(
        "\n**********************************************"
    )

    print(
        "LANDSLIDE PDF DOWNLOAD COMPLETED"
    )

    print(
        "**********************************************"
    )

    print(
        "Successful:",
        successful
    )

    print(
        "Failed:",
        failed
    )

    print(
        "Folder:",
        OUTPUT_DIR
    )

# MAIN

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Scrape Sri Lanka DMC "
            "Landslide Warning Reports"
        )
    )

    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Scrape metadata only "
            "without downloading PDFs"
        )
    )

    parser.add_argument(
        "--download-only",
        action="store_true",
        help=(
            "Use existing metadata CSV "
            "and download PDFs without "
            "scraping metadata again"
        )
    )

    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD"
    )

    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="End date YYYY-MM-DD"
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help=(
            "Optional archive page limit "
            "for testing"
        )
    )

    parser.add_argument(
        "--all-languages",
        action="store_true",
        help=(
            "Download Sinhala and Tamil "
            "reports also"
        )
    )

    parser.add_argument(
        "--all-landslide-reports",
        action="store_true",
        help=(
            "Download unrelated / other "
            "records from the archive also"
        )
    )

    args = parser.parse_args()

    # LOAD OR SCRAPE METADATA

    if args.download_only:

        metadata = load_existing_metadata()

    else:

        metadata = scrape_all_metadata(
            max_pages=args.max_pages
        )

    if metadata.empty:

        raise SystemExit(
            "No landslide metadata available."
        )

    # DOWNLOAD

    if not args.metadata_only:

        download_pdfs(
            metadata,
            from_date=args.from_date,
            to_date=args.to_date,
            relevant_only=(
                not args.all_landslide_reports
            ),
            english_only=(
                not args.all_languages
            ),
        )