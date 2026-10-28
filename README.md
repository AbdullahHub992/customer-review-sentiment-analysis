# Customer Review Sentiment

A portfolio app that classifies a customer review as Positive, Negative, or Neutral with TF-IDF and a linear support vector machine (`LinearSVC`). The Streamlit page loads the saved model. It does not retrain when it opens.

## Dataset

The file used for the saved model is `data/balanced_sentiment_dataset.csv`, copied from `sentiment_dataset_files.zip`.

| Column | Role |
| --- | --- |
| `sentence` | Review text |
| `sentiment` | Label code |

The dataset README defines the codes: `0` = Negative, `1` = Positive, `2` = Neutral.

`balanced_sentiment_dataset.csv` is the file the dataset notes recommend for SVM experiments. `expanded_train_data.csv` (about 1.5 million rows) is still in the original zip and was not used to fit the saved model. Neutral messages in this package are synthetic examples, according to that README. Positive and negative rows are sampled from the original training data.

The measured test accuracy is written to `models/metrics.json` when you train. The app reads that file. It does not use a placeholder score.

## Run locally on Windows

From this folder:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py --data data\balanced_sentiment_dataset.csv
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

Type a review and press **Predict sentiment**. You do not need to upload a CSV for that. The bulk upload further down the page is optional.

If automatic column detection stops, name the columns yourself:

```powershell
python train.py --data data\balanced_sentiment_dataset.csv --text-col sentence --label-col sentiment
```

To train on another CSV from the zip, pass that path instead. Training the 1.5 million row file needs much more time and memory.

## Run in Google Colab

1. Upload `Customer_Review_Sentiment_Analysis.ipynb` to Colab, or upload this whole folder and open the notebook there.
2. Run the cells from top to bottom.
3. When the dataset cell asks for a file, upload `balanced_sentiment_dataset.csv`. If you uploaded the whole folder and `data/balanced_sentiment_dataset.csv` is already present, that cell uses it.
4. The last cell prints a public `trycloudflare.com` URL. Open that URL to view the app.

The notebook installs the libraries, writes `sentiment_core.py`, loads the CSV, cleans it, trains `LinearSVC`, saves the pipeline and vectorizer, writes `app.py`, then starts Streamlit behind a Cloudflare quick tunnel.

## Saved files

After training, `models/` contains:

- `sentiment_pipeline.joblib` — TF-IDF and LinearSVC together. The app loads this file.
- `tfidf_vectorizer.joblib` — the fitted vectorizer on its own.
- `svm_model.joblib` — the fitted LinearSVC on its own.
- `metrics.json` — the real test accuracy, precision, recall, F1, and confusion matrix.

TF-IDF is fitted only on the training split, inside a scikit-learn `Pipeline`.
