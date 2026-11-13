"""Shared data preparation and model definitions for product classification."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


DATA_FILE: Final = Path(__file__).with_name("IMLP6_TASK_03-products.csv")
MODEL_FILE: Final = Path(__file__).with_name("product_category_model.pkl")
TITLE_COLUMN: Final = "Product Title"
CATEGORY_COLUMN: Final = "Category Label"

_CATEGORY_ALIASES: Final = {
    "fridge freezers": "Fridge Freezers",
    "washing machines": "Washing Machines",
    "mobile phones": "Mobile Phones",
    "mobile phone": "Mobile Phones",
    "cpus": "CPUs",
    "cpu": "CPUs",
    "tvs": "TVs",
    "fridges": "Fridges",
    "fridge": "Fridges",
    "dishwashers": "Dishwashers",
    "digital cameras": "Digital Cameras",
    "microwaves": "Microwaves",
    "freezers": "Freezers",
}


def load_products(csv_path: str | Path = DATA_FILE) -> pd.DataFrame:
    """Load labeled product titles and normalize known category-name variants."""
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()

    missing_columns = {TITLE_COLUMN, CATEGORY_COLUMN}.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "CSV must contain 'Product Title' and 'Category Label'. "
            f"Missing: {', '.join(sorted(missing_columns))}"
        )

    products = data[[TITLE_COLUMN, CATEGORY_COLUMN]].copy()
    products[TITLE_COLUMN] = (
        products[TITLE_COLUMN]
        .astype("string")
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.casefold()
    )
    normalized_labels = products[CATEGORY_COLUMN].astype("string").str.strip().str.casefold()
    products[CATEGORY_COLUMN] = normalized_labels.map(_CATEGORY_ALIASES)
    products = products.dropna(subset=[TITLE_COLUMN, CATEGORY_COLUMN])
    products = products.loc[products[TITLE_COLUMN].ne("")].reset_index(drop=True)

    if products.empty or products[CATEGORY_COLUMN].nunique() < 2:
        raise ValueError("The dataset must contain labeled titles from at least two categories.")

    return products


def _word_features() -> TfidfVectorizer:
    """Capture product words and short phrases."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )


def _word_and_character_features() -> FeatureUnion:
    """Combine word meaning with character patterns such as model numbers."""
    return FeatureUnion(
        [
            ("word", _word_features()),
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=150_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def make_candidate_models() -> dict[str, Pipeline]:
    """Return comparable baselines and a richer word/character model."""
    return {
        "Logistic Regression (word TF-IDF)": Pipeline(
            [("tfidf", _word_features()), ("classifier", LogisticRegression(max_iter=1500))]
        ),
        "Multinomial Naive Bayes (word TF-IDF)": Pipeline(
            [("tfidf", _word_features()), ("classifier", MultinomialNB(alpha=0.1))]
        ),
        "Linear SVC (word + character TF-IDF)": Pipeline(
            [
                ("features", _word_and_character_features()),
                ("classifier", LinearSVC(C=2.0, random_state=42, max_iter=5000)),
            ]
        ),
    }


def make_final_model() -> Pipeline:
    """Build the selected word-and-character TF-IDF classifier."""
    return Pipeline(
        [
            ("features", _word_and_character_features()),
            ("classifier", LinearSVC(C=2.0, random_state=42, max_iter=5000)),
        ]
    )
