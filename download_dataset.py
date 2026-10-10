
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_FILE = DATA_DIR / "products.csv"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "luminati-io/eCommerce-dataset-samples/main/"
)

DATASETS = [
    "amazon-products.csv",
    "walmart-products.csv",
    "shopee-products.csv",
    "lazada-products.csv",
    "shein-products.csv",
]

def find_column(df, possible_names):
    columns = {str(c).lower().strip(): c for c in df.columns}

    for name in possible_names:
        if name in columns:
            return df[columns[name]]

    return pd.Series("", index=df.index)

def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    for filename in DATASETS:
        url = BASE_URL + filename
        print(f"Downloading {filename}...")

        try:
            df = pd.read_csv(url, on_bad_lines="skip")
            df.columns = [
                str(c).lower().strip() for c in df.columns
            ]

            title = find_column(
                df, ["title", "name", "product_name"]
            )
            description = find_column(
                df, ["description", "features", "details"]
            )
            category = find_column(
                df, ["categories", "category", "main_category"]
            )
            price = find_column(
                df, [
                    "final_price", "discount_price",
                    "price", "initial_price", "actual_price"
                ]
            )
            image = find_column(
                df, [
                    "image_url", "image", "thumbnail",
                    "primary_image"
                ]
            )
            rating = find_column(df, ["rating", "ratings"])
            reviews = find_column(
                df, ["reviews_count", "no_of_ratings", "rating_count"]
            )

            clean = pd.DataFrame({
                "title": title,
                "description": description,
                "category": category,
                "price": price,
                "image": image,
                "rating_rate": rating,
                "rating_count": reviews,
            })

            clean["title"] = clean["title"].fillna("").astype(str)
            clean = clean[clean["title"].str.strip() != ""]
            frames.append(clean)
            print(f"  Loaded {len(clean)} products")

        except Exception as error:
            print(f"  Could not load {filename}: {error}")

    if not frames:
        raise RuntimeError(
            "No datasets downloaded. Check your internet connection."
        )

    products = pd.concat(frames, ignore_index=True)

    # Remove duplicate product titles
    products = products.drop_duplicates(
        subset=["title"], keep="first"
    )

    # Convert prices and ratings into numeric values
    for column in ["price", "rating_rate", "rating_count"]:
        products[column] = pd.to_numeric(
            products[column].astype(str).str.replace(
                r"[^0-9.]", "", regex=True
            ),
            errors="coerce",
        )

    products = products[
        products["price"].notna()
        & (products["price"] >= 0)
    ]

    # Keep up to 5,000 genuine records; do not fabricate products
    products = products.head(5000)

    products.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset saved successfully!")
    print(f"Total products: {len(products)}")
    print(f"Saved to: {OUTPUT_FILE.resolve()}")
    print(f"Products with image URLs: {products['image'].astype(str).str.startswith('http').sum()}")

if __name__ == "__main__":
    download_dataset()