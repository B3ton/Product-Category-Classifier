"""Train the product-title classifier and save it for interactive predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from product_model import DATA_FILE, MODEL_FILE, load_products, make_final_model


def train_and_save(
    csv_path: str | Path = DATA_FILE,
    model_path: str | Path = MODEL_FILE,
) -> None:
    """Fit the final classifier on all cleaned labeled rows and serialize it."""
    products = load_products(csv_path)
    model = make_final_model()
    model.fit(products["Product Title"], products["Category Label"])
    joblib.dump(model, model_path)

    print(f"Trained using {len(products):,} product titles and {products['Category Label'].nunique()} categories.")
    print(f"Saved model to: {Path(model_path).resolve()}")
    print("Categories:")
    for category, count in products["Category Label"].value_counts().items():
        print(f"  {category}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILE, help="Path to the products CSV.")
    parser.add_argument("--output", type=Path, default=MODEL_FILE, help="Path for the trained .pkl model.")
    args = parser.parse_args()
    train_and_save(args.data, args.output)


if __name__ == "__main__":
    main()
