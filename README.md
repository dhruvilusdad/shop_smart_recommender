# ShopSmart AI

ShopSmart AI is a product catalog browser and content-based recommendation demo built with Python and Streamlit. It loads a local CSV catalog and does not make API requests to fetch products when the app starts.

## Features

- Search product titles and descriptions.
- Filter by primary category, price range, and minimum rating.
- Sort by price, rating, or title.
- Browse products in pages of 24.
- Display product images from valid URLs in each product's own record. If one URL fails in the browser, the card tries the next available URL for that same product and then shows a placeholder.
- Exclude products without a valid-looking HTTP or HTTPS image URL from the displayed catalog.
- View ratings, review counts, and two-decimal prices when those values are available.
- Find up to five similar products using a content-based recommender.

## Product catalog and images

The primary data source is `data/products.csv`. The supplied dataset has approximately 4,320 rows. The loader normalizes supported column names and product fields in memory; it does not modify the CSV. Products without any valid-looking HTTP/HTTPS image URL are excluded from the app catalog. The current local dataset produces 2,389 products with valid-looking image URL candidates.

Image URLs may appear in fields such as `image`, `images`, or `thumbnail`, and a field may contain a JSON/Python-style list of URLs. The app parses those values safely and uses only URLs from the same product record. A syntactically valid URL can still fail to load because its remote server may be offline, return an error, or restrict automated requests.

The app does not invent products or images. This is a static sample catalog, not a store's live inventory.

### Image diagnostics

Run the offline image-field diagnostic from the project directory:

```powershell
python check_images.py
```

Optional URL accessibility checks send network requests only when explicitly requested:

```powershell
python check_images.py --check-accessibility --timeout 5 --workers 4
```

The checker deduplicates valid-looking URLs and limits concurrency to at most eight requests. An unreachable result may be caused by temporary server failures, anti-bot restrictions, or local network limitations.

## Recommendations

The recommender builds text from each product's title, category, and description, then transforms it with scikit-learn's TF-IDF vectorizer. It ranks products by cosine similarity, excludes the selected product, and returns up to five related products with positive similarity. Recommendations are drawn only from the image-qualified catalog. Similarity is a content-based score, not an evaluated accuracy metric.

## Technology stack

- Python 3.10+
- Streamlit
- pandas and NumPy
- scikit-learn and SciPy

## Installation and run

From the project directory, create and activate a virtual environment and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the app:

```powershell
streamlit run app.py
```

The CSV path is resolved relative to the project files. A refresh is optional and separate from app startup; `python download_dataset.py` downloads configured source files and replaces `data/products.csv`. Back up the existing local data before choosing to refresh it.

## Deployment

Include `app.py`, the supporting Python modules, `requirements.txt`, and `data/products.csv` in the deployment repository. For Streamlit Community Cloud, create an app from the repository and choose `app.py` as the entry point. Include the CSV in the deployed project and check the hosting provider's repository and application size limits. Normal app operation does not require API credentials.
