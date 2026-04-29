# Semi-Supervised Learning

Comparison of two semi-supervised learning approaches (Self-Training and Clustering + Labeling) against a fully supervised baseline, on the SMS Spam Collection dataset.

## Problem

Labeled data is expensive to obtain, but unlabeled data is often abundant. Can we leverage unlabeled examples to improve classification performance when only a small portion of the data is labeled? This project explores that question by simulating a low-label scenario and comparing two semi-supervised strategies against a standard supervised baseline.

## Dataset

[SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (loaded directly from a public GitHub mirror).

- 5,572 SMS messages
- Binary classification: `ham` vs. `spam`
- Imbalanced: ~87% ham, ~13% spam

## Approach

To simulate a semi-supervised scenario, the training set is split so that only **30% of examples keep their labels** while the remaining **70% are treated as unlabeled**. Three models are then compared on the same held-out test set:

1. **Supervised baseline:** Logistic Regression trained only on the 30% labeled subset.
2. **Self-Training:** `SelfTrainingClassifier` (scikit-learn) wrapping a Logistic Regression, with a high confidence threshold (0.9) to add pseudo-labels iteratively.
3. **Clustering + Labeling:** `MiniBatchKMeans` clusters the full training data (labeled + unlabeled) into 2 clusters, each cluster is assigned a class via majority vote of its labeled members, and a Logistic Regression is then trained on these pseudo-labels.

All models share the same TF-IDF representation and the same train/test split for fair comparison.

## Conclusion

The Clustering + Labeling approach achieves the best overall performance, slightly outperforming the supervised baseline and clearly outperforming Self-Training. Self-Training underperforms because the high confidence threshold combined with class imbalance leads to very low recall on the spam class. See the notebook for the full classification reports of all three models.

## Tech Stack

- scikit-learn (`LogisticRegression`, `SelfTrainingClassifier`, `MiniBatchKMeans`, `TfidfVectorizer`, `train_test_split`)
- NLTK (tokenization, POS tagging, WordNet lemmatization)
- pandas, NumPy, SciPy
