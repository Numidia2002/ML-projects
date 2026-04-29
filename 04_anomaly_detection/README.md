# Anomaly Detection

Unsupervised outlier detection on text data using two algorithms (Isolation Forest and Local Outlier Factor) over the 20 Newsgroups dataset.

## Problem

In a large collection of text documents, some items may be off-topic, malformed, or substantially different from the rest. Anomaly detection methods aim to identify these unusual documents without using any labels. This project applies and compares two classical outlier detection algorithms on a real text corpus.

## Dataset

[20 Newsgroups](http://qwone.com/~jason/20Newsgroups/)

- ~20,000 documents
- 20 categories (e.g., sci.crypt, soc.religion.christian, talk.politics.mideast)
- Raw text format with headers, signatures, and Usenet-specific metadata

## Approach

1. **Preprocessing:**
   - Lowercasing, URL and email removal, punctuation and number stripping
   - Tokenization (NLTK)
   - Stopword removal
   - POS tagging
   - POS-aware lemmatization (WordNet)
2. **Feature extraction:** TF-IDF (`max_features=10000`)
3. **Models:**
   - **Isolation Forest:** isolates anomalies by recursively partitioning the feature space; documents that get isolated quickly are flagged as outliers.
   - **Local Outlier Factor (LOF):** measures the local density deviation of each document relative to its neighbors.
4. **Visualization:** distribution of outlierness scores and LOF scores, with examples of the most anomalous documents.

## Conclusion

Both methods identify a small proportion of documents as anomalous, suggesting that most texts in the dataset follow similar patterns. Isolation Forest and LOF produce broadly consistent results when using the same contamination value. See the notebook for the score distributions and qualitative examples of detected outliers.

## Tech Stack

- scikit-learn (`IsolationForest`, `LocalOutlierFactor`, `TfidfVectorizer`, `CountVectorizer`, `load_files`)
- NLTK (tokenization, POS tagging, WordNet lemmatization)
- matplotlib
- pandas, NumPy
