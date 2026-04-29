# Statistical Model Comparison

Paired statistical comparison of two classifiers (Logistic Regression vs. Gaussian Naive Bayes) on the Spambase dataset, using repeated cross-validation and a paired t-test.

## Problem

Given two candidate classifiers for the same task, how can we tell whether one is genuinely better, or whether the observed accuracy difference could simply be due to random variation in the cross-validation splits? This project addresses that question with a rigorous paired statistical comparison.

## Dataset

[Spambase](https://www.openml.org/d/44) (loaded directly from OpenML via scikit-learn).

- 4,601 emails
- 57 numerical features
- Binary classification: spam vs. non-spam

## Approach

1. **Two classifiers** are compared:
   - Logistic Regression (with `StandardScaler` in a pipeline)
   - Gaussian Naive Bayes
2. **Evaluation:** 10-fold stratified cross-validation, repeated 20 times with different random seeds.
3. **Paired design:** in every run, both classifiers are evaluated on the *same* CV splits (same seed), so each pair of accuracy scores comes from identical partitions.
4. **Normality check:** Shapiro-Wilk test on the differences to decide which statistical test is appropriate.
5. **Statistical test:** paired t-test (since normality is not rejected).

## Conclusion

Logistic Regression consistently outperforms Gaussian Naive Bayes on this dataset, and the difference is statistically significant. See the notebook for full numerical results.

## Tech Stack

- scikit-learn (`StratifiedKFold`, `cross_val_score`, `LogisticRegression`, `GaussianNB`, `StandardScaler`, `Pipeline`)
- SciPy (`shapiro`, `ttest_rel`)
- pandas, NumPy

## How to Run

Open `statistical_comparison.ipynb` in [Google Colab](https://colab.research.google.com/) environment. 
