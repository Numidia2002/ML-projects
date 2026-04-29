# Multi-label Classification

Multi-label text classification of toxic online comments using a One-vs-Rest strategy with Logistic Regression. Each comment can simultaneously belong to one or more toxicity categories.

## Problem

Online comments often contain multiple types of harmful content at the same time: a single message can be toxic, obscene, and insulting all at once. This project tackles this as a multi-label classification problem, where each comment can be assigned to one or more categories independently.

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) (Wikipedia talk page comments labeled by human raters).

- ~159,000 comments
- 6 binary labels (not mutually exclusive): `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`
- Highly imbalanced: most comments are non-toxic, and rare classes (`threat`, `identity_hate`) appear in less than 1% of the data

## Approach

1. **Preprocessing:**
   - Lowercasing, URL and number removal, punctuation stripping
   - Tokenization
   - Stopword removal (using scikit-learn's English stopword list)
   - POS tagging (NLTK)
   - POS-aware lemmatization (WordNet)
2. **Feature extraction:** TF-IDF with unigrams and bigrams (`max_features=20000`, `min_df=2`)
3. **Model:** `OneVsRestClassifier` wrapping a `LogisticRegression` — trains one independent binary classifier per label
4. **Evaluation:** F1 (micro and macro), Hamming loss, per-label classification report

## Conclusion

The model performs reasonably well given the strong class imbalance, achieving stronger results on frequent labels (`toxic`, `obscene`, `insult`) than on rare ones (`threat`, `identity_hate`). See the notebook for the full evaluation report and example predictions.

## Tech Stack

- scikit-learn (`TfidfVectorizer`, `OneVsRestClassifier`, `LogisticRegression`, `f1_score`, `hamming_loss`, `classification_report`)
- NLTK (tokenization, POS tagging, WordNet lemmatization)
- pandas, NumPy

## How to Run

The notebook was developed in [Google Colab](https://colab.research.google.com/) and loads the dataset from Google Drive. To reproduce, download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data), upload it to your own Drive (or local environment), and adjust the path in the notebook accordingly.
