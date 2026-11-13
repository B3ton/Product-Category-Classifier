"""Interactively predict product categories from product titles."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from product_model import MODEL_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_FILE, help="Path to a trusted trained .pkl model.")
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(
            f"Model not found: {args.model}. Run train_model.py first to create it."
        )

    # Only load model files you trust: pickle-based files can execute code when loaded.
    model = joblib.load(args.model)
    print("Enter a product title to predict its category. Type 'exit' to quit.")

    while True:
        try:
            title = input("Product title: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if title.casefold() in {"exit", "quit", "q"}:
            print("Exiting.")
            break
        if not title:
            print("Please enter a non-empty product title.")
            continue

        category = model.predict([title])[0]
        print(f"Suggested category: {category}")


if __name__ == "__main__":
    main()
