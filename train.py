"""Train the TF-IDF + LinearSVC sentiment model and save it for the Streamlit app."""

from __future__ import annotations

import argparse
import sys

from sentiment_core import (
    ColumnDetectionError,
    UnclearLabelError,
    format_report,
    run_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a customer-review sentiment model with TF-IDF and LinearSVC."
    )
    parser.add_argument("--data", required=True, help="Path to the labeled CSV file.")
    parser.add_argument("--text-col", default=None, help="Review text column. Detected automatically when omitted.")
    parser.add_argument("--label-col", default=None, help="Sentiment label column. Detected automatically when omitted.")
    parser.add_argument("--model-dir", default="models", help="Folder for the saved model, vectorizer, and metrics.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of cleaned rows reserved for testing.")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        metrics = run_training(
            data_path=args.data,
            text_col=args.text_col,
            label_col=args.label_col,
            model_dir=args.model_dir,
            test_size=args.test_size,
        )
    except (ColumnDetectionError, UnclearLabelError, FileNotFoundError, ValueError) as exc:
        print(exc)
        return 1
    print(format_report(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
