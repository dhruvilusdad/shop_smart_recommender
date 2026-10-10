import json
import math

import pandas as pd
import streamlit as st

from data_loader import (
    load_products,
    normalize_category,
    product_image_urls,
)
from recommender import ProductRecommender


PAGE_SIZE = 24
FILTER_KEYS = (
    "catalog_search",
    "catalog_category",
    "catalog_price",
    "catalog_rating",
    "catalog_sort",
    "catalog_page",
    "catalog_filter_signature",
    "selected_product_index",
)


st.set_page_config(
    page_title="ShopSmart AI | Product Discovery",
    page_icon="🛍️",
    layout="wide",
)


@st.cache_data(show_spinner="Loading the local product catalog...")
def get_products() -> pd.DataFrame:
    return load_products()


@st.cache_resource(show_spinner="Preparing product recommendations...")
def get_recommender(products: pd.DataFrame) -> ProductRecommender:
    return ProductRecommender(products)


def _format_price(value: float) -> str:
    return f"{value:,.2f}"


def _price_step(minimum: float, maximum: float) -> float:
    target = (maximum - minimum) / 100
    magnitude = 10 ** math.floor(math.log10(target))
    scaled = target / magnitude
    step_scale = 1 if scaled <= 1 else 2 if scaled <= 2 else 5 if scaled <= 5 else 10
    return float(step_scale * magnitude)


def _show_product_image(product: pd.Series) -> None:
    image_urls = product_image_urls(product)
    if not image_urls:
        st.markdown(
            '<div class="image-placeholder">Image unavailable</div>',
            unsafe_allow_html=True,
        )
        return

    serialized_urls = (
        json.dumps(image_urls)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    image_html = f"""
    <style>
        html, body {{ margin: 0; padding: 0; overflow: hidden; }}
        .product-image {{
            width: 100%;
            height: 190px;
            object-fit: contain;
            background: #fff;
            border-radius: 0.5rem;
        }}
        .image-placeholder {{
            height: 190px;
            display: none;
            align-items: center;
            justify-content: center;
            background: #f1f5f9;
            color: #64748b;
            border-radius: 0.5rem;
            font: 14px sans-serif;
        }}
    </style>
    <img class="product-image" alt="Product image">
    <div class="image-placeholder" role="img" aria-label="Image unavailable">
        Image unavailable
    </div>
    <script>
        const imageUrls = {serialized_urls};
        const productImage = document.querySelector(".product-image");
        const imagePlaceholder = document.querySelector(".image-placeholder");
        let imageIndex = 0;
        productImage.onerror = () => {{
            imageIndex += 1;
            if (imageIndex < imageUrls.length) {{
                productImage.src = imageUrls[imageIndex];
            }} else {{
                productImage.style.display = "none";
                imagePlaceholder.style.display = "flex";
            }}
        }};
        productImage.src = imageUrls[imageIndex];
    </script>
    """
    st.iframe(image_html, height=196)


def _format_count(value: float) -> str:
    return f"{value:,.0f}" if value.is_integer() else f"{value:,.1f}"


def _show_rating_and_reviews(product: pd.Series) -> None:
    rating = product["rating_rate"]
    review_count = product["rating_count"]
    if pd.notna(rating):
        rating_text = f"⭐ {float(rating):g}"
        if pd.notna(review_count):
            rating_text += f" · {_format_count(float(review_count))} reviews"
        st.caption(rating_text)
    elif pd.notna(review_count):
        st.caption(f"{_format_count(float(review_count))} reviews")


def _clear_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    div[data-testid="stImage"] img {
        height: 190px;
        object-fit: contain;
        background: #fff;
        padding: 0.35rem;
        border-radius: 0.5rem;
    }
    .image-placeholder {
        height: 190px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f1f5f9;
        color: #64748b;
        border-radius: 0.5rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("ShopSmart AI")
st.caption("Thoughtful product discovery, powered by content-based recommendations.")

try:
    products = get_products()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Unable to load the local product catalog: {error}")
    st.stop()

products["category"] = products["category"].map(normalize_category)

if products.empty:
    st.warning(
        "The local product catalog contains no products with valid image URLs."
    )
    st.stop()

recommender = get_recommender(products)

with st.sidebar:
    st.header("Find your next favorite")
    st.text_input(
        "Search products",
        placeholder="Search titles and descriptions",
        key="catalog_search",
    )

    categories = sorted(
        {
            category
            for category in products["category"].astype(str).str.strip()
            if category
        },
        key=lambda category: category.casefold(),
    )
    st.selectbox(
        "Category",
        ["All categories", *categories],
        key="catalog_category",
    )

    valid_prices = products.loc[
        products["price"].notna()
        & products["price"].map(math.isfinite)
        & products["price"].ge(0),
        "price",
    ]
    if valid_prices.empty:
        st.caption("Price filtering is unavailable because no valid prices were found.")
        price_range: tuple[float, float] | None = None
    else:
        minimum_price = float(valid_prices.min())
        maximum_price = float(valid_prices.max())
        if minimum_price == maximum_price:
            st.caption(f"All listed prices: { _format_price(minimum_price) }")
            price_range = (minimum_price, maximum_price)
        else:
            price_range = st.slider(
                "Price range",
                min_value=float(minimum_price),
                max_value=float(maximum_price),
                value=(float(minimum_price), float(maximum_price)),
                step=float(_price_step(minimum_price, maximum_price)),
                format="%g",
                key="catalog_price",
            )
            st.caption(
                f"Selected: {_format_price(price_range[0])}"
                f" – {_format_price(price_range[1])}"
            )

    valid_ratings = products["rating_rate"].dropna()
    if valid_ratings.empty:
        minimum_rating: float | None = None
        st.caption("Rating filtering is unavailable because no valid ratings were found.")
    else:
        rating_options: list[float | None] = [None]
        rating_options.extend(
            half_step / 2
            for half_step in range(0, math.floor(float(valid_ratings.max()) * 2) + 1)
        )
        minimum_rating = st.selectbox(
            "Minimum rating",
            rating_options,
            format_func=lambda value: (
                "Any rating" if value is None else f"{value:g} stars and up"
            ),
            key="catalog_rating",
        )

    sort_labels = {
        "Catalog order": None,
        "Price: low to high": ("price", True),
        "Price: high to low": ("price", False),
        "Rating: high to low": ("rating_rate", False),
        "Title: A to Z": ("title", True),
    }
    selected_sort = st.selectbox(
        "Sort by", list(sort_labels), key="catalog_sort"
    )

    st.button(
        "Clear filters",
        width="stretch",
        on_click=_clear_filters,
    )

search_text = st.session_state["catalog_search"].strip()
selected_category = st.session_state["catalog_category"]
filtered = products.copy()

if search_text:
    searchable_text = filtered["title"] + " " + filtered["description"]
    filtered = filtered.loc[
        searchable_text.str.contains(search_text, case=False, regex=False, na=False)
    ]

if selected_category != "All categories":
    filtered = filtered.loc[filtered["category"].eq(selected_category)]

if price_range is not None:
    price_min, price_max = price_range
    matches_price = filtered["price"].between(price_min, price_max, inclusive="both")
    is_full_price_range = (
        price_min == float(valid_prices.min())
        and price_max == float(valid_prices.max())
    )
    if is_full_price_range:
        matches_price |= filtered["price"].isna()
    filtered = filtered.loc[matches_price]

if minimum_rating is not None:
    filtered = filtered.loc[filtered["rating_rate"].ge(minimum_rating)]

sort_rule = sort_labels[selected_sort]
if sort_rule is not None:
    column, ascending = sort_rule
    filtered = filtered.sort_values(
        column,
        ascending=ascending,
        na_position="last",
        kind="mergesort",
    )

filter_signature = (
    search_text,
    selected_category,
    price_range,
    minimum_rating,
    selected_sort,
)
if st.session_state.get("catalog_filter_signature") != filter_signature:
    st.session_state["catalog_page"] = 1
    st.session_state["catalog_filter_signature"] = filter_signature

product_count = len(filtered)
page_count = max(1, math.ceil(product_count / PAGE_SIZE))
current_page = min(max(int(st.session_state.get("catalog_page", 1)), 1), page_count)
st.session_state["catalog_page"] = current_page

st.subheader(f"{product_count:,} products")
start = (current_page - 1) * PAGE_SIZE
page_products = filtered.iloc[start : start + PAGE_SIZE]

previous_column, info_column, next_column = st.columns([1, 3, 1])
with previous_column:
    if st.button(
        "← Previous",
        disabled=current_page <= 1,
        width="stretch",
    ):
        st.session_state["catalog_page"] = current_page - 1
        st.rerun()
with info_column:
    if product_count:
        st.markdown(
            f"<div style='text-align:center;padding:.45rem'>"
            f"Page {current_page} of {page_count} · "
            f"Showing {start + 1}–{min(start + PAGE_SIZE, product_count)}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='text-align:center;padding:.45rem'>Page 1 of 1</div>",
            unsafe_allow_html=True,
        )
with next_column:
    if st.button(
        "Next →",
        disabled=current_page >= page_count,
        width="stretch",
    ):
        st.session_state["catalog_page"] = current_page + 1
        st.rerun()

if product_count == 0:
    st.info("No products match these filters. Try changing your search or filters.")
else:
    for row_start in range(0, len(page_products), 4):
        columns = st.columns(4)
        for column, (_, product) in zip(
            columns, page_products.iloc[row_start : row_start + 4].iterrows()
        ):
            with column:
                with st.container(border=True):
                    _show_product_image(product)
                    st.markdown(f"**{product['title']}**")
                    st.caption(str(product["category"]))

                    price = product["price"]
                    st.markdown(
                        f"**Price:** {_format_price(float(price))}"
                        if pd.notna(price)
                        else "**Price:** Unavailable"
                    )

                    _show_rating_and_reviews(product)

                    if st.button(
                        "Find similar",
                        key=f"similar_{int(product['_product_index'])}",
                        width="stretch",
                    ):
                        st.session_state["selected_product_index"] = int(
                            product["_product_index"]
                        )
                        st.rerun()

selected_product_index = st.session_state.get("selected_product_index")
if selected_product_index is not None:
    selected_product = products.loc[
        products["_product_index"].eq(selected_product_index)
    ]
    if not selected_product.empty:
        selected_title = selected_product.iloc[0]["title"]
        recommendations = recommender.recommend(selected_product_index, top_n=5)

        st.divider()
        st.subheader("Recommended for you")
        st.caption(f"Similar to: {selected_title}")
        if recommendations.empty:
            st.info("No products with meaningful content similarity were found.")
        else:
            recommendation_columns = st.columns(min(5, len(recommendations)))
            for column, (_, product) in zip(
                recommendation_columns, recommendations.iterrows()
            ):
                with column:
                    with st.container(border=True):
                        _show_product_image(product)
                        st.markdown(f"**{product['title']}**")
                        st.caption(str(product["category"]))
                        if pd.notna(product["price"]):
                            st.markdown(
                                f"**Price:** {_format_price(float(product['price']))}"
                            )
                        else:
                            st.markdown("**Price:** Unavailable")

                        _show_rating_and_reviews(product)
                        similarity = float(product["similarity"])
                        st.caption(f"Content similarity: {similarity:.0%}")
