"""Streamlit dashboard for customer-review sentiment.

The page loads the saved TF-IDF + LinearSVC pipeline. It does not retrain.
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from sentiment_core import detect_text_column, load_artifacts, predict_reviews

STYLES = {
    "Positive": {"color": "#0B8A4B", "bg": "#E7F6EE"},
    "Negative": {"color": "#C4372F", "bg": "#FDECEC"},
    "Neutral": {"color": "#9A6B12", "bg": "#FFF6DE"},
}
UPLOAD_COLORS = {
    "Positive": "#0B8A4B",
    "Negative": "#C4372F",
    "Neutral": "#2F6FDB",
}
CARD_ORDER = ["Positive", "Negative", "Neutral"]
EXAMPLES = [
    "I love these headphones. The sound is excellent and they arrived quickly.",
    "The package arrived on Tuesday.",
    "This broke on the second day and support never replied.",
]


def main():
    site = website_name()
    st.set_page_config(
        page_title=site or "Customer Review Sentiment",
        page_icon="◆",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    try:
        pipeline, metrics = get_artifacts()
    except FileNotFoundError as exc:
        st.title("Customer Review Sentiment")
        st.error(str(exc))
        st.markdown(
            "Train the model once, then refresh this page. "
            "A single review prediction does not need a CSV upload."
        )
        st.code(
            "python train.py --data data/balanced_sentiment_dataset.csv",
            language="text",
        )
        return

    render_sidebar(metrics)
    render_header(metrics)
    render_cards(metrics)
    prediction_column, chart_column = st.columns([1.15, 0.85], gap="large")
    with prediction_column:
        render_single_prediction(pipeline)
    with chart_column:
        st.pyplot(distribution_figure(metrics), clear_figure=True)
        st.caption("Counts from the cleaned dataset saved during training.")
        st.pyplot(confusion_figure(metrics), clear_figure=True)
        st.caption("Rows are the actual test labels. Columns are the predicted labels.")
    render_bulk(pipeline)
    render_footer(metrics)


@st.cache_resource
def get_artifacts():
    return load_artifacts(BASE_DIR / "models")


def render_sidebar(metrics):
    with st.sidebar:
        st.header("Model")
        st.write("TF-IDF + LinearSVC")
        st.write(f"Source file: `{metrics.get('source_file', 'unknown')}`")
        st.write(f"Test accuracy: **{metrics['accuracy'] * 100:.2f}%**")
        st.write(f"Train rows: {metrics['train_size']:,}")
        st.write(f"Test rows: {metrics['test_size']:,}")
        st.write(f"Vocabulary: {metrics['vectorizer']['vocabulary_size']:,} features")
        saved_version = metrics.get("sklearn_version")
        if saved_version and saved_version != sklearn.__version__:
            st.warning(
                f"This model was saved with scikit-learn {saved_version}. "
                f"The app is running {sklearn.__version__}."
            )
        st.divider()
        st.header("Dataset notes")
        for note in metrics.get("data_notes", []):
            st.write(note)


def website_name() -> str:
    """Return the website name saved from the text field, if the user entered one."""
    return str(st.session_state.get("website_name", "")).strip()


def render_header(metrics):
    site = website_name()
    safe_site = html.escape(site)
    title = safe_site or "Customer Review Sentiment"
    detail = (
        f"Sentiment for reviews on {safe_site}. "
        "Type one review and the saved SVM model returns Positive, Negative, or Neutral."
        if site
        else "Type one review and the saved SVM model returns Positive, Negative, or Neutral."
    )
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p>{detail}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.text_input(
        "Name of your website",
        key="website_name",
        placeholder="Enter the name of your website",
    )
    goal = "meets" if metrics["meets_accuracy_goal"] else "is below"
    st.caption(
        f"Held-out test accuracy is {metrics['accuracy'] * 100:.2f}% "
        f"({metrics['accuracy']:.4f}), which {goal} the 80% goal. "
        "The model is loaded from disk and is not retrained when this page opens."
    )


def render_cards(metrics):
    counts = metrics["dataset"]["class_counts"]
    total = int(metrics["dataset"]["rows_clean"])
    visible = [label for label in CARD_ORDER if label in counts]
    extras = [label for label in counts if label not in CARD_ORDER]
    labels = ["Total"] + visible + extras
    values = [total] + [int(counts[label]) for label in visible + extras]
    columns = st.columns(len(labels))
    for column, label, value in zip(columns, labels, values):
        color = STYLES.get(label, {}).get("color", "#1B2421")
        column.markdown(
            f"""
            <div class="stat-card">
              <p class="stat-label">{html.escape(label)}</p>
              <p class="stat-value" style="color:{color}">{value:,}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_single_prediction(pipeline):
    st.subheader("Predict one review")
    st.write("CSV upload is optional and lives further down the page.")
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""
    example_columns = st.columns(3)
    example_names = ["Positive example", "Neutral example", "Negative example"]
    for column, name, example in zip(example_columns, example_names, EXAMPLES):
        if column.button(name, use_container_width=True):
            st.session_state.review_text = example
            st.session_state.pop("last_result", None)
            st.rerun()

    with st.form("single-review"):
        review = st.text_area(
            "Customer review",
            key="review_text",
            height=150,
            placeholder="Paste a customer review here.",
        )
        submitted = st.form_submit_button("Predict sentiment", type="primary", use_container_width=True)

    if submitted:
        message = validate_review(review)
        if message:
            st.error(message)
        else:
            prediction = predict_reviews(pipeline, [review]).iloc[0]
            st.session_state.last_result = prediction.to_dict()

    result = st.session_state.get("last_result")
    if result:
        render_result(result)


def render_result(result):
    label = str(result["predicted_sentiment"])
    style = STYLES.get(label, {"color": "#334155", "bg": "#F1F5F9"})
    score = result["decision_score"]
    st.markdown(
        f"""
        <div class="result-banner" style="background:{style['bg']}; border-color:{style['color']}">
          <p class="result-kicker">Predicted sentiment</p>
          <p class="result-label" style="color:{style['color']}">{html.escape(label)}</p>
          <p class="result-score">Decision score {html.escape(str(score))}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    score_columns = {
        column.removeprefix("score_"): result[column]
        for column in result
        if str(column).startswith("score_") and pd.notna(result[column])
    }
    if score_columns:
        score_frame = pd.DataFrame(
            [{"Sentiment": name, "Decision score": value} for name, value in score_columns.items()]
        ).sort_values("Decision score", ascending=False)
        st.dataframe(score_frame, use_container_width=True, hide_index=True)
        st.caption(
            "A decision score is the SVM margin for that class. "
            "A larger score means the review sits farther toward that class. "
            "These numbers are not probabilities."
        )


def render_bulk(pipeline):
    st.subheader("Your reviews")
    st.write(
        "Upload your own CSV. The download is that same file with one added column, "
        "`predicted_sentiment`, labeling each review Positive, Negative, or Neutral. "
        "The circle uses those labels: green for Positive, red for Negative, and blue for Neutral."
    )
    upload = st.file_uploader("Upload a CSV of reviews", type=["csv"])
    if upload is None:
        return
    if st.session_state.get("bulk_name") != upload.name:
        st.session_state.bulk_name = upload.name
        st.session_state.pop("bulk_labeled", None)
    frame = read_upload(upload)
    if frame is None:
        return
    text_column = choose_text_column(frame)
    if text_column is None:
        return
    if st.button("Label my CSV", type="primary"):
        with st.spinner("Labeling every review..."):
            predictions = predict_reviews(pipeline, frame[text_column])
        st.session_state.bulk_labeled = attach_sentiment(frame, predictions)
        st.session_state.bulk_download_name = labeled_filename(upload.name)

    labeled = st.session_state.get("bulk_labeled")
    if labeled is None:
        return
    counts = sentiment_counts(labeled["predicted_sentiment"])
    render_upload_summary(counts)
    chart_column, table_column = st.columns([0.85, 1.15], gap="large")
    with chart_column:
        st.pyplot(upload_donut(counts), clear_figure=True)
    with table_column:
        st.dataframe(labeled.head(200), use_container_width=True, hide_index=True)
        if len(labeled) > 200:
            st.caption("Showing the first 200 rows here. The download contains every row from your file.")
    st.download_button(
        "Download your labeled CSV",
        data=labeled.to_csv(index=False).encode("utf-8-sig"),
        file_name=st.session_state.get("bulk_download_name", "reviews_with_sentiment.csv"),
        mime="text/csv",
    )


def attach_sentiment(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Return the uploaded rows unchanged, plus one sentiment label per review."""
    labeled = frame.copy()
    labeled["predicted_sentiment"] = predictions["predicted_sentiment"].astype("string").tolist()
    notes = predictions["note"].astype(str)
    if notes.str.len().gt(0).any():
        labeled["prediction_note"] = notes.tolist()
    return labeled


def labeled_filename(upload_name: str) -> str:
    stem = Path(upload_name).stem or "reviews"
    return f"{stem}_with_sentiment.csv"


def sentiment_counts(labels: pd.Series) -> dict[str, int]:
    present = labels.dropna().astype(str)
    present = present[present.str.len() > 0]
    counts = {label: int((present == label).sum()) for label in CARD_ORDER}
    for label, count in present.value_counts().items():
        if label not in counts:
            counts[str(label)] = int(count)
    return {label: count for label, count in counts.items() if count > 0}


def render_upload_summary(counts: dict[str, int]):
    total = sum(counts.values())
    winners = majority_labels(counts)
    if len(winners) == 1:
        st.markdown(f"**Most of these reviews are {winners[0]}.** {total:,} reviews labeled.")
    else:
        joined = " and ".join(winners)
        st.markdown(f"**{joined} are tied for the most reviews.** {total:,} reviews labeled.")
    columns = st.columns(len(counts))
    for column, label in zip(columns, counts):
        color = UPLOAD_COLORS.get(label, "#1B2421")
        column.markdown(
            f"""
            <div class="stat-card">
              <p class="stat-label">{html.escape(label)}</p>
              <p class="stat-value" style="color:{color}">{counts[label]:,}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def majority_labels(counts: dict[str, int]) -> list[str]:
    if not counts:
        return []
    highest = max(counts.values())
    return [label for label in counts if counts[label] == highest]


def upload_donut(counts: dict[str, int]):
    """Circle of the uploaded reviews. The largest sentiment fills most of the ring."""
    labels = list(counts)
    values = [counts[label] for label in labels]
    colors = [UPLOAD_COLORS.get(label, "#64748B") for label in labels]
    figure, axis = plt.subplots(figsize=(5.4, 5.4))
    axis.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 2},
    )
    total = sum(values) or 1
    winners = majority_labels(counts)
    center = winners[0] if len(winners) == 1 else "Tied"
    share = counts[winners[0]] / total * 100
    axis.text(0, 0.08, center, ha="center", va="center", fontsize=16, fontweight="bold", color="#16312B")
    axis.text(0, -0.14, f"{share:.0f}%", ha="center", va="center", fontsize=14, color="#3E514B")
    axis.legend(
        [f"{label}  {counts[label]:,}" for label in labels],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        frameon=False,
        ncol=min(len(labels), 3),
    )
    axis.set_title("Your uploaded reviews", pad=12)
    figure.patch.set_facecolor("#FFFFFF")
    figure.tight_layout()
    return figure


def render_footer(metrics):
    columns = metrics.get("columns", {})
    text_name = columns.get("text", "text")
    label_name = columns.get("label", "label")
    st.divider()
    st.caption(
        f"Pipeline: TF-IDF + LinearSVC. Text column `{text_name}`, label column `{label_name}`. "
        f"Fit time {metrics.get('fit_seconds', 'unknown')} seconds. "
        "Decision scores are margins, not probabilities."
    )


def choose_text_column(frame):
    detected = detect_text_column(frame)
    options = list(frame.columns)
    if not options:
        st.error("The uploaded CSV has no columns.")
        return None
    index = options.index(detected) if detected in options else 0
    if detected is None:
        st.info("Choose the column that contains the review text.")
    return st.selectbox("Review text column", options, index=index)


def read_upload(upload):
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        upload.seek(0)
        try:
            frame = pd.read_csv(upload, encoding=encoding)
            break
        except UnicodeDecodeError:
            frame = None
    else:
        st.error("The file could not be read as text. Save it as a UTF-8 CSV and try again.")
        return None
    if frame is None or frame.empty:
        st.error("The uploaded CSV has no rows.")
        return None
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def validate_review(review: str):
    cleaned = " ".join(str(review).split())
    letters = sum(character.isalpha() for character in cleaned)
    if cleaned == "":
        return "Enter a customer review before predicting."
    if letters < 3:
        return "That review is too short to classify. Add a few more words."
    return ""


def distribution_figure(metrics):
    labels = [label for label in metrics["labels"] if label in metrics["dataset"]["class_counts"]]
    values = [metrics["dataset"]["class_counts"][label] for label in labels]
    colors = [STYLES.get(label, {}).get("color", "#334155") for label in labels]
    figure, axis = plt.subplots(figsize=(6.2, 3.4))
    bars = axis.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.62)
    axis.bar_label(bars, padding=4, fontsize=9)
    axis.set_title("Sentiment distribution", loc="left", fontsize=13, pad=12)
    axis.set_xlabel("Reviews")
    style_axis(figure, axis)
    return figure


def confusion_figure(metrics):
    labels = metrics["labels"]
    matrix = metrics["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    image = axis.imshow(matrix, cmap="Greens")
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Test confusion matrix", loc="left", fontsize=13, pad=12)
    peak = max(max(row) for row in matrix) if matrix else 1
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            color = "white" if value > peak * 0.55 else "#1B2421"
            axis.text(column_index, row_index, f"{value:,}", ha="center", va="center", color=color, fontsize=9)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    figure.tight_layout()
    return figure


def style_axis(figure, axis):
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#FFFFFF")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(length=0)
    figure.tight_layout()


def inject_css():
    st.markdown(
        """
        <style>
          .stApp { background: #F3F6F4; }
          .hero h1 { margin: 0; font-size: 2.4rem; letter-spacing: -0.03em; color: #16312B; }
          .hero p { color: #3E514B; font-size: 1.05rem; }
          .stat-card { background: white; border: 1px solid #E1E8E4; border-radius: 16px; padding: 0.9rem 1rem 0.7rem; min-height: 92px; }
          .stat-label { margin: 0; color: #5C6F69; font-size: 0.86rem; }
          .stat-value { margin: 0.15rem 0 0; font-size: 1.7rem; font-weight: 750; letter-spacing: -0.03em; }
          .result-banner { border-left: 8px solid; border-radius: 16px; padding: 0.9rem 1.1rem 0.7rem; margin: 0.8rem 0 0.6rem; }
          .result-kicker { margin: 0; color: #5C6F69; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.06em; }
          .result-label { margin: 0.1rem 0; font-size: 2rem; font-weight: 750; }
          .result-score { margin: 0; color: #3E514B; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
