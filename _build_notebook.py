"""One-off helper that writes the Colab notebook from the project source files."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def writefile_cell(filename: str) -> dict:
    body = (ROOT / filename).read_text(encoding="utf-8").replace("\r\n", "\n")
    if not body.endswith("\n"):
        body += "\n"
    return code("%%writefile " + filename + "\n" + body)


cells = [
    markdown(
        """# Customer Review Sentiment Analysis

Portfolio model: **TF-IDF + LinearSVC**. The notebook trains on your labeled CSV, saves the model, and opens a Streamlit app.

Run the cells in order. The app predicts one review from a text box. CSV upload is optional and is not required for that prediction.

`balanced_sentiment_dataset.csv` is the file used here. Its columns are `sentence` and `sentiment`. The dataset README defines `0` as Negative, `1` as Positive, and `2` as Neutral. Neutral rows in that file are synthetic examples.
"""
    ),
    markdown(
        """## 1. Install dependencies

The next code cell installs the libraries. The cell after it writes `sentiment_core.py`, which the cleaning and training cells import. The Streamlit file is created later, after the model has been evaluated.
"""
    ),
    code(
        """%pip install -q "pandas>=2.0" "numpy>=1.24" "scikit-learn>=1.4" "matplotlib>=3.7" "joblib>=1.3" "streamlit>=1.32"
"""
    ),
    writefile_cell("sentiment_core.py"),
    markdown(
        """## 2. Upload or load the dataset

If `data/balanced_sentiment_dataset.csv` is already in the working directory, this cell uses it. Otherwise it asks you to upload a CSV. In Colab, choose `balanced_sentiment_dataset.csv` from the supplied zip.

Leave `TEXT_COLUMN` and `LABEL_COLUMN` empty to detect `sentence` and `sentiment`. Fill both only when detection stops and asks you to.
"""
    ),
    code(
        """from pathlib import Path

DATA_PATH = "data/balanced_sentiment_dataset.csv"
TEXT_COLUMN = ""  # set both names only if detection cannot tell the columns apart
LABEL_COLUMN = ""

Path("data").mkdir(exist_ok=True)
if Path(DATA_PATH).exists():
    print("Using", DATA_PATH)
else:
    try:
        from google.colab import files
    except ImportError:
        raise SystemExit(
            "Put the CSV in the data folder, or open this notebook in Google Colab to upload it."
        )
    print("Upload balanced_sentiment_dataset.csv from the dataset zip.")
    uploaded = files.upload()
    if not uploaded:
        raise SystemExit("No file was uploaded.")
    name = next(iter(uploaded))
    destination = Path("data") / Path(name).name
    destination.write_bytes(uploaded[name])
    DATA_PATH = str(destination)
    print("Saved", DATA_PATH)
"""
    ),
    markdown(
        """## 3. Inspect and preprocess

This cell prints the columns, missing values, and label counts, then removes incomplete rows, duplicate rows, and reviews that carry more than one sentiment.
"""
    ),
    code(
        """from sentiment_core import clean_reviews, detect_columns, read_dataset

frame = read_dataset(DATA_PATH)
print("Shape:", frame.shape)
print("Columns:", list(frame.columns))
display(frame.head())
print("Missing values:")
display(frame.isna().sum().to_frame("missing"))

text_col, label_col = detect_columns(frame, TEXT_COLUMN or None, LABEL_COLUMN or None)
print("Text column:", text_col)
print("Label column:", label_col)
print("Raw label counts:")
display(frame[label_col].value_counts(dropna=False).to_frame("rows"))

clean_df, clean_report = clean_reviews(frame, text_col, label_col)
print("Cleaning report:")
for key, value in clean_report.items():
    print(f"  {key}: {value}")
display(clean_df.head())
"""
    ),
    markdown(
        """## 4. Train and evaluate the SVM

TF-IDF and `LinearSVC` sit in one pipeline. The vectorizer is fitted on the training split only. The test split is stratified and is not used during fitting. The printed accuracy is the measured test result.
"""
    ),
    code(
        """import pandas as pd
from sentiment_core import finalize_metrics, train_and_evaluate

pipeline, metrics = train_and_evaluate(clean_df)
finalize_metrics(metrics, DATA_PATH, text_col, label_col, clean_report)

print(f"Test accuracy: {metrics['accuracy']:.4f}")
print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
print(f"Meets 80% goal: {metrics['meets_accuracy_goal']}")
display(pd.DataFrame(metrics["per_class"]).T)
labels = metrics["labels"]
display(
    pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels],
    )
)
"""
    ),
    markdown(
        """## 5. Save the model and vectorizer

The app loads `models/sentiment_pipeline.joblib`, which keeps the fitted TF-IDF vectorizer and the LinearSVC model together. The vectorizer and SVM are also saved as separate files.
"""
    ),
    code(
        """from sentiment_core import format_report, save_artifacts

save_artifacts(pipeline, metrics, "models")
print(format_report(metrics))
"""
    ),
    markdown(
        """## 6. Create the Streamlit application

This cell writes `app.py`. The page loads the saved pipeline and does not train again.
"""
    ),
    writefile_cell("app.py"),
    markdown(
        """## 7. Install and configure a public tunnel

This step is for Google Colab. It downloads Cloudflare's quick-tunnel program and writes a headless Streamlit config. No Cloudflare account is required. On your own computer, skip the last two cells and run `streamlit run app.py`.
"""
    ),
    code(
        """import subprocess
import sys
from pathlib import Path

try:
    import google.colab  # noqa: F401
except ImportError:
    raise SystemExit("Tunnel setup is for Google Colab. Locally, run: streamlit run app.py")

Path(".streamlit").mkdir(exist_ok=True)
Path(".streamlit/config.toml").write_text(
    \"\"\"
[server]
headless = true
address = "0.0.0.0"
port = 8501
enableCORS = false
enableXsrfProtection = false
fileWatcherType = "none"
maxUploadSize = 50

[browser]
gatherUsageStats = false
\"\"\".strip()
    + "\\n",
    encoding="utf-8",
)
if not Path("cloudflared").exists():
    subprocess.run(
        [
            "wget",
            "-q",
            "-O",
            "cloudflared",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        ],
        check=True,
    )
    subprocess.run(["chmod", "+x", "cloudflared"], check=True)
print("Tunnel program is ready.")
"""
    ),
    markdown(
        """## 8. Launch the app and display its public URL

Wait until the cell prints a `trycloudflare.com` address, then open it. The first load can take a few seconds. If you run this cell again, it replaces the previous Streamlit process.
"""
    ),
    code(
        """import re
import subprocess
import time
from pathlib import Path

try:
    import google.colab  # noqa: F401
except ImportError:
    raise SystemExit("Launch the local app with: streamlit run app.py")

if not Path("models/sentiment_pipeline.joblib").exists():
    raise SystemExit("Save the model in the previous cells before launching the app.")

subprocess.run("pkill -f 'streamlit run app.py' || true", shell=True)
subprocess.run("pkill -f 'cloudflared tunnel' || true", shell=True)
time.sleep(1)

streamlit_log = open("streamlit.log", "w", encoding="utf-8")
subprocess.Popen(
    ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"],
    stdout=streamlit_log,
    stderr=subprocess.STDOUT,
)
tunnel_log = open("tunnel.log", "w", encoding="utf-8")
subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8501", "--no-autoupdate"],
    stdout=tunnel_log,
    stderr=subprocess.STDOUT,
)

public_url = None
for _ in range(40):
    time.sleep(1)
    tunnel_text = Path("tunnel.log").read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"https://[-0-9a-zA-Z]+\\.trycloudflare\\.com", tunnel_text)
    if match:
        public_url = match.group(0)
        break

if public_url:
    print("Public URL:", public_url)
else:
    print("The public URL did not appear. Tunnel log:")
    print(Path("tunnel.log").read_text(encoding="utf-8", errors="ignore")[-2000:])
    print("Streamlit log:")
    print(Path("streamlit.log").read_text(encoding="utf-8", errors="ignore")[-2000:])
"""
    ),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"provenance": []},
    },
    "cells": cells,
}

destination = ROOT / "Customer_Review_Sentiment_Analysis.ipynb"
destination.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(destination, "cells", len(cells))
