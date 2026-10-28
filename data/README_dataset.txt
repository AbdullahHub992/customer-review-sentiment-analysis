Customer Sentiment Dataset - Expanded

Files:
1. expanded_train_data.csv
   - Preserves all 1,523,975 original training rows and adds 20,000 generated neutral messages.
   - Labels: 0 = Negative, 1 = Positive, 2 = Neutral.
   - Note: the full dataset is highly imbalanced because the original dataset has over 1.5 million binary-labeled rows.

2. balanced_sentiment_dataset.csv
   - A 60,000-row balanced subset for easier initial experimentation.
   - 20,000 Negative (0), 20,000 Positive (1), and 20,000 Neutral (2).
   - Positive and negative rows are sampled from the original training dataset; neutral rows are generated examples.
   - Recommended for initial SVM experiments, but evaluate on a separate representative test set.

3. added_neutral_messages.csv
   - The 20,000 newly generated neutral messages only.

Important:
- The neutral messages are synthetic examples, not original Kaggle records.
- The examples are intended as a starting point and may contain some template-like phrasing.
- Do not use the same rows for both training and testing. Create a held-out test set, and ideally validate on real neutral messages that resemble the intended application.
