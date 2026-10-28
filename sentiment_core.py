"""Shared cleaning and LinearSVC training for customer-review sentiment.

TF-IDF is fitted only inside a scikit-learn Pipeline, and only on the
training split, so test reviews never influence the vocabulary.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TEXT_PRIORITY = [
    "sentence",
    "review_text",
    "reviewtext",
    "review_body",
    "review",
    "customer_review",
    "tweet",
    "comment",
    "comments",
    "feedback",
    "text",
    "content",
    "body",
    "clean_text",
    "reviews",
]
LABEL_PRIORITY = [
    "sentiment",
    "sentiment_label",
    "polarity",
    "label",
    "emotion",
    "target",
    "class",
]
TEXT_LABELS = {
    "positive": "Positive",
    "pos": "Positive",
    "negative": "Negative",
    "neg": "Negative",
    "neutral": "Neutral",
    "neu": "Neutral",
}
# The supplied dataset README defines these codes.
CODE_LABELS_012 = {0: "Negative", 1: "Positive", 2: "Neutral"}
PIPELINE_NAME = "sentiment_pipeline.joblib"
VECTORIZER_NAME = "tfidf_vectorizer.joblib"
MODEL_NAME = "svm_model.joblib"
METRICS_NAME = "metrics.json"
ACCURACY_GOAL = 0.80


class ColumnDetectionError(Exception):
    """Raised when the review or sentiment column cannot be chosen safely."""


class UnclearLabelError(Exception):
    """Raised when label values do not match a known sentiment encoding."""


def read_dataset(path) -> pd.DataFrame:
    """Read a CSV, including files saved with a UTF-8 BOM or semicolon separator."""
    path = Path(path)
    if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
        raise ValueError("Please provide the dataset as a CSV file.")
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            frame = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        if frame.shape[1] == 1:
            sample = path.read_text(encoding=encoding, errors="ignore")[:4000]
            if sample.count(";") > sample.count(","):
                frame = pd.read_csv(path, encoding=encoding, sep=";")
            elif sample.count("\t") > sample.count(","):
                frame = pd.read_csv(path, encoding=encoding, sep="\t")
        frame.columns = [str(column).strip() for column in frame.columns]
        if frame.columns.duplicated().any():
            duplicates = frame.columns[frame.columns.duplicated()].tolist()
            raise ValueError(f"The CSV has duplicate column names: {duplicates}")
        if frame.empty:
            raise ValueError("The CSV file has no rows.")
        return frame
    raise ValueError(f"Could not read {path.name} as text. Last error: {last_error}")


def detect_columns(frame: pd.DataFrame, text_col=None, label_col=None):
    """Return the review-text column and the sentiment column."""
    if text_col or label_col:
        if not text_col or not label_col:
            raise ColumnDetectionError(
                "Provide both the text column and the label column, or leave both empty for automatic detection."
            )
        missing = [name for name in (text_col, label_col) if name not in frame.columns]
        if missing:
            available = ", ".join(map(str, frame.columns))
            raise ColumnDetectionError(
                f"Column(s) not found: {', '.join(missing)}. Available columns: {available}"
            )
        if text_col == label_col:
            raise ColumnDetectionError("The text column and the label column must be different.")
        return text_col, label_col

    text_hits = _priority_hits(frame.columns, TEXT_PRIORITY)
    label_hits = _priority_hits(frame.columns, LABEL_PRIORITY)
    if len(text_hits) >= 1 and len(label_hits) >= 1 and text_hits[0] != label_hits[0]:
        return text_hits[0], label_hits[0]
    raise ColumnDetectionError(_column_help(frame))


def detect_text_column(frame: pd.DataFrame):
    """Return a likely review column, or None when the upload needs a manual choice."""
    hits = _priority_hits(frame.columns, TEXT_PRIORITY)
    return hits[0] if hits else None


def detect_label_column(frame: pd.DataFrame, text_col=None):
    """Return a likely sentiment column, or None when the upload has no label."""
    hits = [column for column in _priority_hits(frame.columns, LABEL_PRIORITY) if column != text_col]
    return hits[0] if hits else None


def clean_reviews(frame: pd.DataFrame, text_col: str, label_col: str):
    """Drop empty, duplicate, and contradictory rows and normalize sentiment labels."""
    report = {"rows_loaded": int(len(frame))}
    work = frame[[text_col, label_col]].copy()
    work.columns = ["text", "label_raw"]
    report["raw_label_counts"] = {
        _json_key(key): int(value)
        for key, value in work["label_raw"].value_counts(dropna=False).items()
    }
    work["text"] = work["text"].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    raw_label_text = work["label_raw"].astype("string").str.strip()
    missing = work["text"].isna() | (work["text"] == "") | work["label_raw"].isna() | (raw_label_text == "")
    report["dropped_missing"] = int(missing.sum())
    work = work.loc[~missing].copy()
    if work.empty:
        raise ValueError("Every row was removed because the review text or the label was missing.")

    work["sentiment"] = normalize_labels(work["label_raw"])
    before_duplicates = len(work)
    work = work.drop_duplicates(subset=["text", "sentiment"])
    report["dropped_duplicates"] = int(before_duplicates - len(work))

    sentiment_per_text = work.groupby("text")["sentiment"].nunique()
    conflict_texts = sentiment_per_text[sentiment_per_text > 1].index
    conflict_rows = work["text"].isin(conflict_texts)
    report["dropped_conflicts"] = int(conflict_rows.sum())
    work = work.loc[~conflict_rows].copy()
    if work.empty:
        raise ValueError("Every row was removed because the same review had more than one sentiment label.")
    if work["sentiment"].nunique() < 2:
        raise ValueError("Cleaning left fewer than two sentiment classes, so there is nothing to train.")

    report["rows_clean"] = int(len(work))
    report["class_counts"] = {
        str(key): int(value) for key, value in work["sentiment"].value_counts().items()
    }
    return work[["text", "sentiment"]].reset_index(drop=True), report


def normalize_labels(raw: pd.Series) -> pd.Series:
    """Map known words and the dataset's 0/1/2 codes onto Positive, Negative, and Neutral."""
    text_mapped = raw.map(_map_text_label)
    if text_mapped.notna().all():
        return text_mapped

    numeric = pd.to_numeric(raw, errors="coerce")
    integer_like = numeric.notna() & (numeric == numeric.round())
    if integer_like.all():
        values = {int(value) for value in numeric.tolist()}
        table = _numeric_table(values)
        return numeric.map(lambda value: table[int(value)])

    unknown = raw.loc[text_mapped.isna()].astype(str).value_counts().head(12)
    examples = ", ".join(f"{label} ({count})" for label, count in unknown.items())
    raise UnclearLabelError(
        "Some label values are not a known sentiment encoding. "
        f"Examples: {examples}. "
        "Use Positive, Negative, and Neutral, or the dataset codes 0, 1, and 2."
    )


def train_and_evaluate(clean_frame: pd.DataFrame, test_size=0.2, random_state=42):
    """Fit TF-IDF and LinearSVC on the training split, then score the untouched test split."""
    label_counts = clean_frame["sentiment"].value_counts()
    if int(label_counts.min()) < 2:
        smallest = label_counts.idxmin()
        raise ValueError(
            f"The class '{smallest}' has only {int(label_counts.min())} row(s). "
            "Stratified splitting needs at least 2 rows in every class."
        )
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")

    features = clean_frame["text"]
    labels = clean_frame["sentiment"]
    features_train, features_test, labels_train, labels_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    pipeline = build_pipeline(len(features_train), random_state)
    started = time.perf_counter()
    try:
        pipeline.fit(features_train, labels_train)
    except ValueError as exc:
        raise ValueError(
            "The vectorizer could not build features from the training text. "
            "Check that the text column contains the reviews. "
            f"Original error: {exc}"
        ) from exc
    fit_seconds = time.perf_counter() - started

    test_predictions = pipeline.predict(features_test)
    train_predictions = pipeline.predict(features_train)
    class_names = [str(name) for name in pipeline.named_steps["svm"].classes_]
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_test,
        test_predictions,
        labels=class_names,
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(
        labels_test,
        test_predictions,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        labels_test,
        test_predictions,
        average="weighted",
        zero_division=0,
    )
    matrix = confusion_matrix(labels_test, test_predictions, labels=class_names)
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["svm"]
    test_accuracy = float(accuracy_score(labels_test, test_predictions))

    per_class = {}
    for index, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }

    metrics = {
        "algorithm": "LinearSVC",
        "vectorizer_type": "TfidfVectorizer",
        "accuracy": test_accuracy,
        "train_accuracy": float(accuracy_score(labels_train, train_predictions)),
        "accuracy_goal": ACCURACY_GOAL,
        "meets_accuracy_goal": bool(test_accuracy >= ACCURACY_GOAL),
        "labels": class_names,
        "per_class": per_class,
        "macro_avg": _average_block(macro),
        "weighted_avg": _average_block(weighted),
        "confusion_matrix": matrix.astype(int).tolist(),
        "train_size": int(len(features_train)),
        "test_size": int(len(features_test)),
        "test_fraction": float(test_size),
        "random_state": int(random_state),
        "fit_seconds": round(fit_seconds, 2),
        "vectorizer": {
            "ngram_range": list(vectorizer.ngram_range),
            "min_df": vectorizer.min_df,
            "max_df": vectorizer.max_df,
            "max_features": vectorizer.max_features,
            "stop_words": "english, keeping negation words",
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
        },
        "model": {
            "algorithm": "LinearSVC",
            "C": float(classifier.C),
            "class_weight": "balanced",
            "max_iter": int(classifier.max_iter),
        },
    }
    return pipeline, metrics


def finalize_metrics(metrics, source_path, text_col, label_col, clean_report):
    """Attach dataset facts to the evaluation numbers before they are saved."""
    source_path = Path(source_path)
    metrics["source_file"] = source_path.name
    metrics["sklearn_version"] = sklearn.__version__
    metrics["columns"] = {"text": text_col, "label": label_col}
    metrics["dataset"] = clean_report
    metrics["data_notes"] = data_notes(source_path)
    return metrics


def save_artifacts(pipeline, metrics, model_dir="models"):
    """Save the paired pipeline plus separate vectorizer and SVM files."""
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    pipeline_path = folder / PIPELINE_NAME
    vectorizer_path = folder / VECTORIZER_NAME
    model_path = folder / MODEL_NAME
    metrics_path = folder / METRICS_NAME
    joblib.dump(pipeline, pipeline_path)
    joblib.dump(pipeline.named_steps["tfidf"], vectorizer_path)
    joblib.dump(pipeline.named_steps["svm"], model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metrics["saved_files"] = [
        pipeline_path.as_posix(),
        vectorizer_path.as_posix(),
        model_path.as_posix(),
        metrics_path.as_posix(),
    ]
    return metrics


def load_artifacts(model_dir="models"):
    """Load the saved pipeline and the metrics written at training time."""
    folder = Path(model_dir)
    pipeline_path = folder / PIPELINE_NAME
    metrics_path = folder / METRICS_NAME
    if not pipeline_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "The trained model is not in the models folder yet. "
            "Run python train.py --data data/balanced_sentiment_dataset.csv"
        )
    pipeline = joblib.load(pipeline_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return pipeline, metrics


def run_training(data_path, text_col=None, label_col=None, model_dir="models", test_size=0.2, random_state=42):
    """Read, clean, train, evaluate, and save. This is the command-line entry path."""
    frame = read_dataset(data_path)
    text_name, label_name = detect_columns(frame, text_col, label_col)
    clean_frame, report = clean_reviews(frame, text_name, label_name)
    pipeline, metrics = train_and_evaluate(clean_frame, test_size=test_size, random_state=random_state)
    finalize_metrics(metrics, data_path, text_name, label_name, report)
    save_artifacts(pipeline, metrics, model_dir)
    return metrics


def format_report(metrics: dict) -> str:
    """Plain-text report of the real cleaning steps and held-out test metrics."""
    dataset = metrics["dataset"]
    lines = [
        f"Source file: {metrics['source_file']}",
        f"Text column: {metrics['columns']['text']}",
        f"Label column: {metrics['columns']['label']}",
        f"Rows loaded: {dataset['rows_loaded']}",
        f"Dropped missing review or label: {dataset['dropped_missing']}",
        f"Dropped duplicate rows: {dataset['dropped_duplicates']}",
        f"Dropped reviews with conflicting labels: {dataset['dropped_conflicts']}",
        f"Rows after cleaning: {dataset['rows_clean']}",
        "Raw label counts:",
    ]
    for key, value in dataset["raw_label_counts"].items():
        lines.append(f"  {key}: {value}")
    lines.append("Class counts after cleaning:")
    for label in metrics["labels"]:
        lines.append(f"  {label}: {dataset['class_counts'].get(label, 0)}")
    lines.extend(
        [
            "",
            (
                f"Split: stratified {1 - metrics['test_fraction']:.0%} train / "
                f"{metrics['test_fraction']:.0%} test, random_state={metrics['random_state']}"
            ),
            f"Train rows: {metrics['train_size']}",
            f"Test rows: {metrics['test_size']}",
            "Model: TF-IDF fitted on the training split only, then LinearSVC.",
            f"Vocabulary size: {metrics['vectorizer']['vocabulary_size']}",
            f"Fit time: {metrics['fit_seconds']} seconds",
            "",
            f"Test accuracy: {metrics['accuracy']:.4f}",
            f"Training accuracy, for comparison: {metrics['train_accuracy']:.4f}",
            f"Accuracy goal: {metrics['accuracy_goal']:.2f}",
        ]
    )
    if metrics["meets_accuracy_goal"]:
        lines.append("The held-out test accuracy meets the 80% goal.")
    else:
        lines.append(
            "The held-out test accuracy is below the 80% goal. The number above is the measured test result."
        )
    lines.extend(["", "Per-class test metrics:"])
    for label in metrics["labels"]:
        scores = metrics["per_class"][label]
        lines.append(
            f"  {label}: precision {scores['precision']:.4f}, "
            f"recall {scores['recall']:.4f}, f1 {scores['f1']:.4f}, support {scores['support']}"
        )
    lines.append(
        f"  Macro average: precision {metrics['macro_avg']['precision']:.4f}, "
        f"recall {metrics['macro_avg']['recall']:.4f}, f1 {metrics['macro_avg']['f1']:.4f}"
    )
    lines.extend(["", "Confusion matrix (rows are actual labels, columns are predicted labels):"])
    header = "  " + " ".join(f"{label:>12}" for label in metrics["labels"])
    lines.append(header)
    for label, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        lines.append(f"  {label:>12} " + " ".join(f"{value:>12}" for value in row))
    if metrics.get("data_notes"):
        lines.extend(["", "Dataset notes:"])
        lines.extend(f"  - {note}" for note in metrics["data_notes"])
    lines.extend(["", "Saved files:"])
    lines.extend(f"  {path}" for path in metrics.get("saved_files", []))
    return "\n".join(lines)


def predict_reviews(pipeline, reviews) -> pd.DataFrame:
    """Predict sentiment for a list or series of reviews. Empty rows are skipped."""
    original = pd.Series(reviews, dtype="string").fillna("").astype(str)
    cleaned = original.str.replace(r"\s+", " ", regex=True).str.strip()
    classes = [str(name) for name in pipeline.named_steps["svm"].classes_]
    result = pd.DataFrame({"review": original.tolist()})
    result["predicted_sentiment"] = pd.NA
    result["decision_score"] = pd.NA
    for class_name in classes:
        result[f"score_{class_name}"] = pd.NA
    result["note"] = ""

    valid_positions = np.flatnonzero(cleaned.str.len().to_numpy() > 0)
    if len(valid_positions) == 0:
        result["note"] = "Empty review skipped"
        return result

    texts = cleaned.iloc[valid_positions].tolist()
    predicted = pipeline.predict(texts)
    matrix = _score_matrix(pipeline, texts)
    class_index = {name: index for index, name in enumerate(classes)}
    for row_number, position in enumerate(valid_positions):
        position = int(position)
        label = str(predicted[row_number])
        result.at[position, "predicted_sentiment"] = label
        result.at[position, "decision_score"] = round(float(matrix[row_number, class_index[label]]), 4)
        for class_name in classes:
            result.at[position, f"score_{class_name}"] = round(
                float(matrix[row_number, class_index[class_name]]),
                4,
            )
    empty_positions = np.flatnonzero(cleaned.str.len().to_numpy() == 0)
    for position in empty_positions:
        result.at[int(position), "note"] = "Empty review skipped"
    return result


def data_notes(source_path: Path) -> list[str]:
    """Notes taken from the README shipped with this dataset."""
    name = Path(source_path).name.lower()
    code_note = "Label codes from the dataset README: 0 = Negative, 1 = Positive, 2 = Neutral."
    if name == "balanced_sentiment_dataset.csv":
        return [
            "The saved model uses balanced_sentiment_dataset.csv. The dataset README recommends this subset for SVM experiments.",
            code_note,
            "The dataset README states that neutral messages are synthetic, while positive and negative rows are sampled from the original training data.",
            "expanded_train_data.csv was not used to fit the saved model.",
        ]
    if name == "expanded_train_data.csv":
        return [
            "The saved model uses expanded_train_data.csv. The dataset README says this file keeps the original rows and adds generated neutral messages.",
            code_note,
        ]
    return [code_note]


# These stay in the text. The English stop list would otherwise delete "not"
# and turn "not bad" into "bad".
_NEGATION_WORDS = frozenset(
    {
        "not",
        "no",
        "nor",
        "never",
        "neither",
        "none",
        "nobody",
        "nothing",
        "nowhere",
        "cannot",
        "without",
    }
)
_NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|cannot|isn.?t|wasn.?t|weren.?t|aren.?t|don.?t|doesn.?t|didn.?t|"
    r"won.?t|can.?t|couldn.?t|wouldn.?t|shouldn.?t)\s+([a-z]+)",
    re.IGNORECASE,
)


def mark_negation(text: str) -> str:
    """Join a negation to the next word, so 'not bad' is one token instead of 'bad'."""
    lowered = str(text).lower()
    return _NEGATION_PATTERN.sub(lambda match: "not_" + match.group(1), lowered)


def sentiment_stop_words() -> list[str]:
    """English stop words with negation words left in the review text."""
    return sorted(ENGLISH_STOP_WORDS - _NEGATION_WORDS)


def build_pipeline(train_rows: int, random_state: int) -> Pipeline:
    """Create an unfitted TF-IDF + LinearSVC pipeline sized to the training split."""
    min_df = 2 if train_rows >= 200 else 1
    max_df = 0.95 if train_rows >= 200 else 1.0
    # The vocabulary cap is 30,000 features. Larger training sets use the primal SVM form.
    dual = train_rows < 30000
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=mark_negation,
                    stop_words=sentiment_stop_words(),
                    token_pattern=r"(?u)\b\w\w+\b",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_df=max_df,
                    max_features=30000,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "svm",
                LinearSVC(
                    C=1.0,
                    class_weight="balanced",
                    random_state=random_state,
                    max_iter=10000,
                    dual=dual,
                ),
            ),
        ]
    )


def _priority_hits(columns, priority):
    lookup = {_normalize_name(column): column for column in columns}
    return [lookup[name] for name in priority if name in lookup]


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _map_text_label(value):
    if pd.isna(value):
        return pd.NA
    text = re.sub(r"[^a-z]+", " ", str(value).strip().lower())
    text = re.sub(r"\s+", " ", text).strip()
    return TEXT_LABELS.get(text, pd.NA)


def _numeric_table(values: set[int]) -> dict[int, str]:
    if values <= {0, 1}:
        return {0: "Negative", 1: "Positive"}
    if values <= {0, 1, 2}:
        return CODE_LABELS_012
    if values <= {-1, 0, 1}:
        return {-1: "Negative", 0: "Neutral", 1: "Positive"}
    shown = ", ".join(str(value) for value in sorted(values))
    raise UnclearLabelError(
        f"The label column contains the numbers {shown}. "
        "Star ratings are left unchanged until you say how they should map. "
        "This dataset's README uses 0 = Negative, 1 = Positive, and 2 = Neutral."
    )


def _score_matrix(pipeline, texts):
    raw_scores = pipeline.decision_function(texts)
    if raw_scores.ndim == 1:
        column = np.asarray(raw_scores, dtype=float).reshape(-1, 1)
        return np.column_stack([-column, column])
    return np.asarray(raw_scores, dtype=float)


def _average_block(scores) -> dict:
    precision, recall, f1, _support = scores
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def _column_help(frame: pd.DataFrame) -> str:
    lines = [
        "I could not tell which column is the review text and which column is the sentiment label.",
        "Columns in the file:",
    ]
    for column in frame.columns:
        sample = frame[column].dropna().astype(str).head(3).tolist()
        preview = "; ".join(value[:80] for value in sample) if sample else "(empty)"
        lines.append(f"- {column} ({frame[column].dtype}): {preview}")
    lines.append(
        "Run the trainer again with both names, for example: "
        "python train.py --data your.csv --text-col sentence --label-col sentiment"
    )
    return "\n".join(lines)


def _json_key(value) -> str:
    if pd.isna(value):
        return "missing"
    return str(value)
