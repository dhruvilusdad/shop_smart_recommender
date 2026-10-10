import ast
import json
from pathlib import Path
import re
from collections.abc import Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd


DATA_FILE = Path(__file__).resolve().parent / "data" / "products.csv"

_COLUMN_ALIASES = {
    "title": ("title", "product_title", "product_name", "name"),
    "description": ("description", "features", "details"),
    "category": ("category", "categories", "main_category"),
    "price": (
        "price",
        "final_price",
        "discount_price",
        "initial_price",
        "actual_price",
    ),
    "rating_rate": ("rating_rate", "rating", "ratings"),
    "rating_count": (
        "rating_count",
        "reviews_count",
        "no_of_ratings",
        "review_count",
    ),
}

_IMAGE_COLUMN_NAMES = (
    "image",
    "images",
    "image_url",
    "image_urls",
    "img",
    "img_url",
    "img_urls",
    "thumbnail",
    "primary_image",
    "product_image",
    "product_images",
    "product_img",
    "image_link",
    "image_path",
    "image_src",
    "featured_image",
    "picture",
    "picture_url",
    "photo",
    "photo_url",
    "product_photo",
)


def normalize_column_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def get_image_columns(columns: Iterable[object]) -> list[str]:
    normalized_columns = [
        normalize_column_name(column)
        for column in columns
    ]
    recognized = [
        name
        for name in _IMAGE_COLUMN_NAMES
        if name in normalized_columns
    ]
    recognized.extend(
        name
        for name in normalized_columns
        if name not in recognized
        and not name.startswith("_")
        and any(
            part in name
            for part in ("image", "photo", "picture", "thumbnail")
        )
    )
    return recognized


def _valid_image_url(value: str) -> bool:
    if (
        not value
        or any(character.isspace() for character in value)
        or re.search(r"%(?![0-9a-fA-F]{2})", value)
    ):
        return False
    try:
        parsed = urlparse(value)
        port = parsed.port
        return (
            parsed.scheme.lower() in {"http", "https"}
            and bool(parsed.netloc)
            and bool(parsed.hostname)
            and (port is None or 0 < port < 65536)
        )
    except ValueError:
        return False


def inspect_image_value(value: object) -> tuple[tuple[str, ...], int, bool]:
    """Return valid URLs, malformed-value count, and whether any data was supplied."""
    if value is None or value is pd.NA:
        return (), 0, False
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return (), 0, False

    if isinstance(value, dict):
        urls: list[str] = []
        malformed_count = 0
        has_data = False
        for nested_value in value.values():
            nested_urls, nested_malformed, nested_has_data = inspect_image_value(
                nested_value
            )
            urls.extend(nested_urls)
            malformed_count += nested_malformed
            has_data |= nested_has_data
        return tuple(dict.fromkeys(urls)), malformed_count, has_data

    if isinstance(value, (list, tuple)):
        urls = []
        malformed_count = 0
        has_data = False
        for nested_value in value:
            nested_urls, nested_malformed, nested_has_data = inspect_image_value(
                nested_value
            )
            urls.extend(nested_urls)
            malformed_count += nested_malformed
            has_data |= nested_has_data
        return tuple(dict.fromkeys(urls)), malformed_count, has_data

    if not isinstance(value, str):
        return (), 1, True

    candidate = value.strip()
    if not candidate:
        return (), 0, False

    if candidate.startswith(("[", "(", "{")):
        try:
            parsed_value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            try:
                parsed_value = ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                return (), 1, True
        if not isinstance(parsed_value, (list, tuple, dict)):
            return (), 1, True
        return inspect_image_value(parsed_value)

    if _valid_image_url(candidate):
        return (candidate,), 0, True
    return (), 1, True


def normalize_image_url(value: object) -> str:
    """Extract the first well-formed HTTP(S) URL from an image field."""
    urls, _, _ = inspect_image_value(value)
    return urls[0] if urls else ""


def product_image_urls(product: pd.Series) -> tuple[str, ...]:
    """Return image candidates for a product in field and list order."""
    candidates: list[str] = []
    saved_urls = product.get("_image_urls", ())
    if isinstance(saved_urls, (list, tuple)):
        candidates.extend(
            url for url in saved_urls if isinstance(url, str) and _valid_image_url(url)
        )

    for column in get_image_columns(product.index):
        if column in product.index:
            urls, _, _ = inspect_image_value(product[column])
            candidates.extend(urls)
    return tuple(dict.fromkeys(candidates))


def normalize_product_image(product: pd.Series) -> str:
    """Return the first valid image URL present in a product record."""
    urls = product_image_urls(product)
    return urls[0] if urls else ""


def _numeric_values(values: pd.Series) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    cleaned = (
        values.astype("string")
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9.+-]", "", regex=True)
    )
    fallback = pd.to_numeric(cleaned, errors="coerce")
    return parsed.fillna(fallback).astype("float64")


def normalize_category(value: object) -> str:
    if isinstance(value, (list, tuple)):
        values = value
    elif (
        value is None
        or value is pd.NA
        or (isinstance(value, (float, np.floating)) and np.isnan(value))
    ):
        return "Other"
    elif isinstance(value, str):
        value = value.strip()
        if not value:
            return "Other"
        if value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return "Other"
            if not isinstance(parsed, (list, tuple)):
                return "Other"
            values = parsed
        else:
            return value
    else:
        value = str(value).strip()
        return value or "Other"

    for item in values:
        if isinstance(item, str) and item.strip():
            return item.strip()
    return "Other"


def load_products() -> pd.DataFrame:
    """Load and normalize the local product catalog without making network requests."""
    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Product catalog not found at {DATA_FILE}. "
            "Add data/products.csv or explicitly run download_dataset.py to refresh it."
        )

    try:
        products = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError as error:
        raise ValueError(f"Product catalog at {DATA_FILE} is empty.") from error

    products.columns = [normalize_column_name(column) for column in products.columns]
    for canonical, aliases in _COLUMN_ALIASES.items():
        source = next(
            (name for name in aliases if name in products.columns),
            None,
        )
        if source is not None and source != canonical:
            products[canonical] = products[source]

    if "title" not in products.columns:
        raise ValueError(
            "Product catalog is missing the required title column. "
            "Expected a column named title, product_title, product_name, or name."
        )

    image_columns = get_image_columns(products.columns)

    for column in ("description", "category", "price", "rating_rate", "rating_count"):
        if column not in products.columns:
            products[column] = pd.NA

    products["_product_index"] = range(len(products))
    if not image_columns:
        products["image"] = ""

    for column in ("title", "description"):
        products[column] = products[column].fillna("").astype(str).str.strip()

    products["title"] = products["title"].str.strip()
    products = products.loc[products["title"].ne("")].copy()
    products["category"] = products["category"].map(normalize_category)
    image_values = []
    image_candidates = []
    if image_columns:
        image_rows = products[image_columns].itertuples(index=False, name=None)
        for values in image_rows:
            urls = tuple(
                dict.fromkeys(
                    url for value in values for url in inspect_image_value(value)[0]
                )
            )
            image_candidates.append(urls)
            image_values.append(urls[0] if urls else "")
    else:
        image_values = [""] * len(products)
        image_candidates = [()] * len(products)

    products["image"] = pd.Series(
        image_values,
        index=products.index,
        dtype="string",
    )
    products["_image_urls"] = pd.Series(
        image_candidates,
        index=products.index,
        dtype="object",
    )
    products = products.loc[products["_image_urls"].map(bool)].copy()

    products["price"] = _numeric_values(products["price"])
    products["price"] = products["price"].where(
        np.isfinite(products["price"]) & products["price"].ge(0)
    )

    products["rating_rate"] = _numeric_values(products["rating_rate"])
    products["rating_rate"] = products["rating_rate"].where(
        np.isfinite(products["rating_rate"])
        & products["rating_rate"].between(0, 5)
    )

    products["rating_count"] = _numeric_values(products["rating_count"])
    products["rating_count"] = products["rating_count"].where(
        np.isfinite(products["rating_count"]) & products["rating_count"].ge(0)
    )

    return products.reset_index(drop=True)
