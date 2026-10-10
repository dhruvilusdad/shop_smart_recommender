import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ProductRecommender:
    """Content-based recommendations keyed to stable source-dataset row IDs."""

    def __init__(self, products: pd.DataFrame) -> None:
        self.products = products.copy().reset_index(drop=True)

        for column in ("title", "category", "description"):
            if column not in self.products.columns:
                self.products[column] = ""
            self.products[column] = (
                self.products[column].fillna("").astype(str)
            )

        if "_product_index" not in self.products.columns:
            self.products["_product_index"] = range(len(self.products))
        if self.products["_product_index"].duplicated().any():
            raise ValueError("Product dataset row IDs must be unique.")

        self._position_by_index = {
            int(product_index): position
            for position, product_index in enumerate(self.products["_product_index"])
        }

        features = (
            self.products["title"]
            + " "
            + self.products["category"]
            + " "
            + self.products["description"]
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        analyzer = self.vectorizer.build_analyzer()

        if any(analyzer(text) for text in features):
            self.tfidf_matrix = self.vectorizer.fit_transform(features)
            self._has_vocabulary = True
        else:
            self.tfidf_matrix = csr_matrix((len(self.products), 0), dtype=float)
            self._has_vocabulary = False

    def recommend(
        self, product_index: int, top_n: int = 5
    ) -> pd.DataFrame:
        """Return up to ``top_n`` positive-similarity products for a source row ID."""
        try:
            position = self._position_by_index[int(product_index)]
        except (KeyError, ValueError, TypeError) as error:
            raise IndexError("Invalid product index.") from error

        if top_n <= 0 or len(self.products) <= 1 or not self._has_vocabulary:
            recommendations = self.products.iloc[0:0].copy()
            recommendations["similarity"] = pd.Series(dtype="float64")
            return recommendations

        scores = cosine_similarity(
            self.tfidf_matrix.getrow(position), self.tfidf_matrix
        ).ravel()
        ranked = sorted(
            (
                (candidate_position, float(score))
                for candidate_position, score in enumerate(scores)
                if candidate_position != position and score > 0
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:top_n]

        if not ranked:
            recommendations = self.products.iloc[0:0].copy()
            recommendations["similarity"] = pd.Series(dtype="float64")
            return recommendations

        positions = [candidate_position for candidate_position, _ in ranked]
        recommendations = self.products.iloc[positions].copy()
        recommendations["similarity"] = [
            score for _, score in ranked
        ]
        return recommendations


if __name__ == "__main__":
    from data_loader import load_products

    products = load_products()
    recommender = ProductRecommender(products)

    if products.empty:
        print("The local product catalog is empty.")
    else:
        selected_index = int(products.iloc[0]["_product_index"])
        recommendations = recommender.recommend(selected_index)
        print("Selected product:")
        print(products.iloc[0][["title", "category", "price"]].to_string())
        print("\nRecommended products:")
        print(
            recommendations[
                ["title", "category", "price", "similarity"]
            ].to_string(index=False)
        )