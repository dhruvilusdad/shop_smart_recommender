"""Report product image URL quality; optionally check remote URL accessibility."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from data_loader import (
    DATA_FILE,
    get_image_columns,
    inspect_image_value,
    normalize_column_name,
)


def _check_url_accessibility(url: str, timeout: float) -> bool:
    headers = {"User-Agent": "ShopSmartAI-image-diagnostic/1.0"}
    request = Request(url, headers=headers, method="HEAD")
    try:
        with urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except HTTPError as error:
        error.close()
        if error.code not in (403, 405, 501):
            return False
    except (URLError, TimeoutError, OSError, ValueError):
        return False

    request = Request(
        url,
        headers={**headers, "Range": "bytes=0-0"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            response.read(1)
            return 200 <= response.status < 400
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect product image fields in the local CSV. Remote accessibility "
            "checks are opt-in and disabled by default."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_FILE,
        help=f"CSV path (default: {DATA_FILE})",
    )
    parser.add_argument(
        "--check-accessibility",
        action="store_true",
        help="Send HTTP requests to test distinct valid-looking image URLs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-request timeout in seconds (default: 5; allowed: 0.1-30).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        choices=range(1, 9),
        default=4,
        help="Concurrent requests (1-8; default: 4).",
    )
    args = parser.parse_args()
    if not 0.1 <= args.timeout <= 30:
        parser.error("--timeout must be between 0.1 and 30 seconds.")
    return args


def main() -> int:
    args = _parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"Product CSV not found: {args.csv}")

    try:
        products = pd.read_csv(args.csv)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Product CSV is empty: {args.csv}") from error

    products.columns = [normalize_column_name(column) for column in products.columns]
    image_columns = get_image_columns(products.columns)
    if not image_columns:
        raise ValueError(
            "No image-related columns found. Expected names such as image, images, "
            "image_url, thumbnail, photo, or picture."
        )

    image_urls: set[str] = set()
    missing_records = 0
    malformed_values = 0
    records_with_valid_url = 0

    for values in products[image_columns].itertuples(index=False, name=None):
        record_urls: set[str] = set()
        has_image_data = False
        for value in values:
            urls, malformed_count, supplied = inspect_image_value(value)
            record_urls.update(urls)
            malformed_values += malformed_count
            has_image_data |= supplied

        if not has_image_data:
            missing_records += 1
        if record_urls:
            records_with_valid_url += 1
            image_urls.update(record_urls)

    print(f"Dataset: {args.csv}")
    print(f"Products scanned: {len(products):,}")
    print(f"Image columns: {', '.join(image_columns)}")
    print(f"Distinct valid-looking URLs: {len(image_urls):,}")
    print(f"Products with at least one valid-looking URL: {records_with_valid_url:,}")
    print(f"Products with missing image data: {missing_records:,}")
    print(f"Malformed URL/value entries: {malformed_values:,}")

    if args.check_accessibility:
        urls = sorted(image_urls)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            accessible_results = executor.map(
                lambda url: _check_url_accessibility(url, args.timeout),
                urls,
            )
            unreachable = sum(not accessible for accessible in accessible_results)
        print(
            f"Accessibility checks: {len(urls):,} URLs "
            f"(timeout {args.timeout:g}s, {args.workers} workers)"
        )
        print(f"Unreachable URLs: {unreachable:,}")
        print(
            "Unreachable results can reflect temporary server errors, "
            "anti-bot restrictions, or network limitations."
        )
    else:
        print("Unreachable URLs: not checked (use --check-accessibility to opt in)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
