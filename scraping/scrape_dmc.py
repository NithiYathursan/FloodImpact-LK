import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin
import pandas as pd
import requests
from bs4 import BeautifulSoup
import hashlib

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "situation_reports"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_FILE = (
    OUTPUT_DIR
    / "dmc_situation_reports_metadata.csv"
)


# =========================================================
# DMC CONFIGURATION
# =========================================================

BASE_URL = "https://www.dmc.gov.lk"

ARCHIVE_URL = (
    "https://www.dmc.gov.lk/index.php"
    "?Itemid=273"
    "&lang=en"
    "&option=com_dmcreports"
    "&report_type_id=1"
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


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def extract_date(text):
    """
    Extract YYYY-MM-DD from a DMC archive row.
    """

    match = re.search(
        r"\b20\d{2}-\d{2}-\d{2}\b",
        text
    )

    if match:
        return match.group()

    return None


def extract_time(text):
    """
    Extract HH:MM from DMC archive row.
    """

    match = re.search(
        r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b",
        text
    )

    if match:
        return match.group()

    return None


def is_situation_report(title):
    """
    Keep real Situation Reports and reject
    weather forecasts accidentally listed
    in the archive.
    """

    title = title.lower().strip()

    return (
        "situation report" in title
        and "weather forecast" not in title
        and "weather forcast" not in title
        and "dry weather" not in title
    )


# =========================================================
# SCRAPE ONE ARCHIVE PAGE
# =========================================================

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


        # ---------------------------------------------
        # Find PDF
        # ---------------------------------------------

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


        # ---------------------------------------------
        # Title
        # ---------------------------------------------

        title = row_text.split(
            report_date
        )[0].strip()


        if not is_situation_report(
            title
        ):
            continue


        report_time = extract_time(
            row_text
        )


        records.append({

            "title":
                title,

            "date":
                report_date,

            "time":
                report_time,

            "pdf_url":
                pdf_url,

            "archive_offset":
                offset

        })


    return records


# =========================================================
# SCRAPE ALL DMC METADATA
# =========================================================

def scrape_all_metadata(
    delay=0.4,
    max_pages=None
):

    print(
        "\nStarting DMC Situation Report metadata scraping...\n"
    )

    all_records = []

    offset = 0
    page_number = 1

    # DMC currently displays 10 records per page.
    PAGE_SIZE = 10


    while True:

        if (
            max_pages is not None
            and page_number > max_pages
        ):
            break


        print(
            f"Page {page_number} "
            f"| offset={offset}"
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
                "Waiting 5 seconds and retrying..."
            )

            time.sleep(5)

            try:

                records = scrape_page(
                    offset
                )

            except Exception as e2:

                print(
                    "Retry failed:",
                    e2
                )

                break


        # ---------------------------------------------
        # Stop when archive has no more report rows
        # ---------------------------------------------

        if not records:

            print(
                "\nNo Situation Reports found "
                "on this page."
            )

            # One archive page may contain an
            # incorrectly categorized document.
            # Check next page before stopping.

            if page_number > 2:

                try:

                    next_records = scrape_page(
                        offset + PAGE_SIZE
                    )

                except:
                    next_records = []

                if not next_records:

                    print(
                        "End of archive detected."
                    )

                    break


        all_records.extend(
            records
        )


        # ---------------------------------------------
        # Periodically save progress
        # ---------------------------------------------

        if (
            page_number % 20 == 0
            and all_records
        ):

            temp_df = pd.DataFrame(
                all_records
            )

            temp_df = (
                temp_df
                .drop_duplicates(
                    subset=["pdf_url"]
                )
            )

            temp_df.to_csv(
                METADATA_FILE,
                index=False
            )

            print(
                f"Progress saved: "
                f"{len(temp_df)} reports\n"
            )


        offset += PAGE_SIZE
        page_number += 1

        time.sleep(delay)


    # =====================================================
    # FINAL CLEANING
    # =====================================================

    df = pd.DataFrame(
        all_records
    )


    if df.empty:

        print(
            "\nNo metadata was collected."
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
        .reset_index(
            drop=True
        )
    )


    df.to_csv(
        METADATA_FILE,
        index=False
    )


    print(
        "\n======================================"
    )

    print(
        "DMC METADATA COLLECTION COMPLETED"
    )

    print(
        "======================================"
    )

    print(
        "Reports collected:",
        len(df)
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
        "\nSaved to:"
    )

    print(
        METADATA_FILE
    )


    return df


# =========================================================
# DOWNLOAD PDF FILES
# =========================================================

def download_pdfs(
    df,
    from_date=None,
    to_date=None,
    delay=0.3
):

    download_df = df.copy()


    if from_date:

        from_date = pd.to_datetime(
            from_date
        )

        download_df = download_df[
            download_df["date"]
            >= from_date
        ]


    if to_date:

        to_date = pd.to_datetime(
            to_date
        )

        download_df = download_df[
            download_df["date"]
            <= to_date
        ]


    download_df = (
        download_df
        .sort_values(
            ["date", "time"]
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\nPDFs selected for download:",
        len(download_df)
    )


    successful = 0
    failed = 0


    for index, row in download_df.iterrows():

        date_text = (
            row["date"]
            .strftime("%Y%m%d")
        )


        time_text = (
            str(
                row["time"]
            )
            .replace(
                ":",
                ""
            )
        )


        if (
            time_text.lower()
            in ["nan", "none", ""]
        ):
            time_text = "unknown"

        url_hash = hashlib.sha1(
            str(row["pdf_url"]).encode("utf-8")).hexdigest()[:8]


        filename = (
            f"situation_"
            f"{date_text}_"
            f"{time_text}_"
            f"{url_hash}.pdf"
        )

        file_path = (
            OUTPUT_DIR
            / filename
        )


        # ---------------------------------------------
        # Resume support
        # ---------------------------------------------

        if (
            file_path.exists()
            and file_path.stat().st_size > 1000
        ):

            print(
                f"[{index + 1}/{len(download_df)}] "
                f"Already exists: {filename}"
            )

            successful += 1

            continue


        print(
            f"[{index + 1}/{len(download_df)}] "
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
                    "  Response is not a valid PDF."
                )

                failed += 1

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


        time.sleep(delay)


    print(
        "\n======================================"
    )

    print(
        "PDF DOWNLOAD COMPLETED"
    )

    print(
        "======================================"
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


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Scrape Sri Lanka DMC "
            "Situation Reports."
        )
    )


    parser.add_argument(

        "--metadata-only",

        action="store_true",

        help=(
            "Collect report metadata "
            "without downloading PDFs."
        )

    )


    parser.add_argument(

        "--from-date",

        type=str,

        default=None,

        help="Download start date YYYY-MM-DD"

    )


    parser.add_argument(

        "--to-date",

        type=str,

        default=None,

        help="Download end date YYYY-MM-DD"

    )


    parser.add_argument(

        "--max-pages",

        type=int,

        default=None,

        help=(
            "Optional page limit for testing. "
            "Leave empty for full archive."
        )

    )


    args = parser.parse_args()


    metadata = scrape_all_metadata(

        max_pages=args.max_pages

    )


    if metadata.empty:

        raise SystemExit(
            "No Situation Report metadata found."
        )


    if not args.metadata_only:

        download_pdfs(

            metadata,

            from_date=args.from_date,

            to_date=args.to_date

        )